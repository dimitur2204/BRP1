#!/usr/bin/env python3
"""S12 -- fine-tune the DINO ViT with a Cox loss (D6), or train the same ViT from scratch.

The question: does DINO pretraining help when the whole network learns from the cancer
labels (not only a linear probe on frozen features, S10)? Two runs, same script and recipe:

  finetune/dino_ep39   backbone = TEACHER of runs/dino/small_150ep/stable, dino_epoch=039
  finetune/scratch     same backbone, random weights (seed 0, = S9's "random"),
                       --layer-decay 1.0 --lr-backbone 1e-3 (layer decay only exists to
                       protect pretrained weights; on random weights it would freeze the
                       early blocks)

Network
  backbone   rebuilt from the pretraining run's launch.json (s9_dino_features.load_model),
             so architecture and weights always match: 8^3 patch embedding, CLS token,
             12 EVA blocks (384-dim, 4 heads, RoPE), final norm; drop-path / attention
             dropout as in pretraining (0.2 / 0.2). The DINO / iBOT heads, the student and
             the unused up_projection decoder are not part of it (up_projection is built
             but gets no gradient and is left out of the optimiser).
  input      the whole 144x128x128 S2 volume (2.5 mm) -> 18x16x16 = 4,608 tokens + CLS,
             position embedding resampled from the 12^3 pretraining grid (as in S9)
  head       [CLS, mean patch token] (2 x 384, the S10 probe features) -> dropout ->
             Linear(768 -> 1) = eta. Zero-initialised, so eta = 0 at the start and the first
             updates do not push a random head's noise into the pretrained blocks. No bias:
             the Cox loss does not depend on a constant shift of eta.

Recipe (S4 where possible: same data, batches, augment, cox_loss, schedule shape, outputs)
  loss       Cox partial likelihood over each batch of 32 patients (the whole risk set must
             be in one forward pass, so no gradient accumulation; block activations are
             recomputed in the backward pass (gradient checkpointing, default on), because
             32 whole-lung volumes do not fit in an L40S otherwise)
  optimiser  AdamW, weight decay 0.05 (0 for biases, norms, gammas, cls/pos embedding)
  lr         head peak --lr-head 1e-3; backbone peak --lr-backbone 1e-4 x layer decay:
             block i of 12 (0-based) gets 0.75^(12 - i), embeddings 0.75^13, final norm 1.
             Linear warm-up over 3 epochs, then cosine to 0, per batch (s4_train.lr_at)
  clip       gradient norm 1.0
  length     30 epochs (~7-8 min each on an L40S), last epoch kept (--select last)
  precision  bf16 autocast on GPU
  augment    model.augment (rotation +-10 deg axial, scale +-10%, shift +-6 voxels)

Outputs, runs/{run_name}/ -- the S4 format, so s5_evaluate.py runs unchanged:
  config.json, history.csv (+ lr_backbone column), model.pt (backbone + head state dict),
  pred.csv (eta for every cohort patient; test rows are only read by s5_evaluate.py),
  curves.png

Runs in env/.venv-dino (needs the repo's Primus). That venv has no scikit-survival, so
Harrell C is computed here (harrell_c_np, checked against model.harrell_c / sksurv).

Usage:
    sbatch src/submit_s12_dino_finetune.sh --run-name finetune/dino_ep39 \\
        --dino-run small_150ep/stable --ckpt "dino_epoch=039"
    sbatch src/submit_s12_dino_finetune.sh --run-name finetune/scratch \\
        --dino-run small_150ep/stable --ckpt random --layer-decay 1.0 --lr-backbone 1e-3
    env/.venv-dino/bin/python src/s12_dino_finetune.py --smoke --device cpu \\
        --run-name x --dino-run small_150ep/stable --ckpt "dino_epoch=039" \\
        --cohort <scratch>/cohort.csv --runs-dir <scratch>/runs        # CPU check
An existing run directory with a model.pt is never overwritten (new --run-name or --overwrite).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from s9_dino_features import REPO, load_model  # also puts the repo on sys.path
from s4_train import epoch_batches, lr_at, plot_curves, predict, val_cox
from model import augment, cox_loss, to_input
from nlst import CACHE_NPY, COHORT_CSV, RUNS_DIR


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="e.g. finetune/dino_ep39")
    ap.add_argument("--dino-run", default="small_150ep/stable", help="pretraining run under runs/dino/")
    ap.add_argument("--ckpt", required=True, help='checkpoint without .ckpt (e.g. "dino_epoch=039") or random')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3, help="peak lr of the Cox head")
    ap.add_argument("--lr-backbone", type=float, default=1e-4, help="peak lr of the top backbone layer")
    ap.add_argument("--layer-decay", type=float, default=0.75, help="lr factor per block going down; 1 = off")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0, help="max gradient norm; 0 = off")
    ap.add_argument("--dropout", type=float, default=0.3, help="before the Cox head (as LungCNN)")
    ap.add_argument("--select", choices=["val_c", "last"], default="last",
                    help="save the best-val-C epoch or the final epoch")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="recompute block activations in the backward pass (default on: without it batch 32 "
                         "needs > 44 GB, out of memory on an L40S, job 6287322)")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--n-train-eval", type=int, default=2000,
                    help="train patients scored in eval mode each epoch (train C)")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--cache", default=str(CACHE_NPY))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--overwrite", action="store_true", help="replace an existing finished run")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, batch 4, central 64^3 crop of each volume: a CPU check of the code path")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch_size, a.n_train_eval, a.warmup_epochs, a.eval_batch = 2, 4, 4, 1.0, 4
    if not 0 < a.layer_decay <= 1:
        ap.error("--layer-decay must be in (0, 1]")
    return a


def harrell_c_np(eta, time, event) -> float:
    """Harrell's C as in sksurv.concordance_index_censored: an event at t_i is comparable
    with every patient with t_j > t_i and with censored patients at t_j == t_i; concordant
    if eta_i > eta_j, eta ties count 1/2. O(n^2) in blocks (fine for a few thousand)."""
    eta, time, event = (np.asarray(v, dtype=float) for v in (eta, time, event))
    ev = np.flatnonzero(event == 1)
    if len(ev) == 0:
        return float("nan")
    conc = pairs = 0.0
    for s in range(0, len(ev), 256):
        i = ev[s:s + 256]
        comp = (time[None, :] > time[i, None]) | ((time[None, :] == time[i, None]) & (event[None, :] == 0))
        d = eta[i, None] - eta[None, :]
        conc += ((d > 0) & comp).sum() + 0.5 * ((d == 0) & comp).sum()
        pairs += comp.sum()
    return float(conc / pairs) if pairs else float("nan")


class CoxViT(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float):
        super().__init__()
        self.backbone = backbone
        d = backbone.cls_token.shape[-1]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, 1, bias=False))
        nn.init.zeros_(self.head[1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, 1, 144, 128, 128) in [0, 1]
        tok = self.backbone(x)                                    # (B, 1 + 4608, D)
        f = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], dim=1)    # [CLS, mean patch] as in S9/S10
        return self.head(f.float()).squeeze(1)                    # (B,) log-risk eta


def checkpoint_blocks(vit: nn.Module) -> None:
    """Recompute each block's activations in the backward pass (training only). The repo's
    set_grad_checkpointing() calls the reentrant checkpoint with rope= as a keyword, which
    it rejects; the non-reentrant one accepts it. Patching forward keeps parameter names."""
    from torch.utils.checkpoint import checkpoint

    def wrap(blk):
        fwd = blk.forward

        def forward(x, rope=None):
            if blk.training and torch.is_grad_enabled():
                return checkpoint(fwd, x, rope=rope, use_reentrant=False)
            return fwd(x, rope=rope)
        blk.forward = forward

    for blk in vit.blocks:
        wrap(blk)


def param_groups(model: CoxViT, a) -> list:
    """AdamW groups with a peak lr each (layer-wise decay) and weight decay on matrices only."""
    n_blocks = len(model.backbone.vit.blocks)
    groups, skipped = {}, []
    for name, p in model.named_parameters():
        if name.startswith("backbone.up_projection") or "rope" in name:
            p.requires_grad_(False)     # decoder is never called; RoPE periods stay fixed
            skipped.append(name)
            continue
        if name.startswith("head."):
            peak = a.lr_head
        else:
            n = name[len("backbone."):]
            m = re.match(r"vit\.blocks\.(\d+)\.", n)
            if m:
                depth = n_blocks - int(m.group(1))           # top block 1 ... first block 12
            elif n.startswith("vit.norm"):
                depth = 0
            elif n.startswith(("cls_token", "vit.pos_embed", "down_projection")):
                depth = n_blocks + 1
            else:
                raise SystemExit(f"no lr rule for parameter {name}")
            peak = a.lr_backbone * a.layer_decay ** depth
        no_wd = p.ndim <= 1 or "gamma" in name or name.endswith(("cls_token", "pos_embed"))
        wd = 0.0 if no_wd else a.weight_decay
        g = groups.setdefault((peak, wd), {"params": [], "peak": peak, "weight_decay": wd, "lr": 0.0})
        g["params"].append(p)
    print(f"optimiser: {len(groups)} groups; not trained: {sorted(set(s.split('.')[1] for s in skipped))}"
          f" ({len(skipped)} tensors); peak lr head {a.lr_head:g}, backbone top block "
          f"{a.lr_backbone * a.layer_decay:.3g}, first block {a.lr_backbone * a.layer_decay ** n_blocks:.3g}",
          flush=True)
    return list(groups.values())


def main() -> None:
    a = parse_args()
    device = torch.device(a.device)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext
    out = Path(a.runs_dir) / a.run_name
    if (out / "model.pt").exists() and not a.overwrite:
        raise SystemExit(f"{out} already holds a finished run -- use a new --run-name or --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    from utils.imports import import_module_from_path
    import_module_from_path("project", str(REPO))
    backbone, meta = load_model(a.dino_run, a.ckpt, "cpu")       # seeds 0 for the random control
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.grad_checkpoint:
        checkpoint_blocks(backbone.vit)
    model = CoxViT(backbone, a.dropout).to(device)
    groups = param_groups(model, a)
    opt = torch.optim.AdamW(groups)
    trained = [p for g in groups for p in g["params"]]

    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    cache = np.load(a.cache, mmap_mode="r")
    tr = cohort[cohort.split == "train"].sort_values("row")
    va = cohort[cohort.split == "val"].sort_values("row")
    t0 = time.time()
    # --smoke: central 64^3 crop (8^3 tokens), because whole-lung attention on CPU needs more
    # memory than the login node allows; the full-size check is a short GPU job
    view = (lambda x: np.ascontiguousarray(x[:, 40:104, 32:96, 32:96])) if a.smoke else np.asarray
    x_tr = view(cache[tr["row"].to_numpy()])                  # sorted rows = sequential read
    x_va = view(cache[va["row"].to_numpy()])
    print(f"loaded train {x_tr.shape} + val {x_va.shape} in {time.time() - t0:.0f}s", flush=True)
    t_tr = tr["time"].to_numpy(np.float32); e_tr = tr["event"].to_numpy(np.float32)
    t_va = va["time"].to_numpy(np.float32); e_va = va["event"].to_numpy(np.float32)
    sub = np.sort(rng.choice(len(tr), size=min(a.n_train_eval, len(tr)), replace=False))

    steps_per_epoch = len(tr) // a.batch_size        # batch slots, incl. skipped ones
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = int(round(a.warmup_epochs * steps_per_epoch))
    cfg = {**vars(a), "init": meta, "n_params_trained": sum(p.numel() for p in trained),
           "n_train": len(tr), "events_train": int(e_tr.sum()),
           "n_val": len(va), "events_val": int(e_va.sum())}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2), flush=True)

    hist, best_c, best_epoch, best_state, step = [], -1.0, -1, None, 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses, skipped, gnorms = [], 0, []
        for b in epoch_batches(rng, e_tr, a.batch_size, 0):
            f = lr_at(step, total_steps, warmup_steps, 1.0, "cosine")   # schedule factor in [0, 1]
            step += 1  # skipped batches advance the schedule too, so it ends on time
            if e_tr[b].sum() == 0:
                skipped += 1
                continue
            for g in opt.param_groups:
                g["lr"] = g["peak"] * f
            xb = to_input(torch.from_numpy(x_tr[b]).to(device, non_blocking=True))
            if not a.no_aug:
                xb = augment(xb)
            tb = torch.from_numpy(t_tr[b]).to(device)
            eb = torch.from_numpy(e_tr[b]).to(device)
            with amp():
                eta = model(xb)
            loss = cox_loss(eta.float(), tb, eb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip:
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(trained, a.clip)))
            opt.step()
            losses.append(loss.item())

        eta_va = predict(model, x_va, device, amp, batch=a.eval_batch)
        eta_sub = predict(model, x_tr, device, amp, idx=sub, batch=a.eval_batch)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val_cox(eta_va, t_va, e_va),
            "val_loss_null": val_cox(np.zeros_like(eta_va), t_va, e_va),
            "train_c": harrell_c_np(eta_sub, t_tr[sub], e_tr[sub]),
            "val_c": harrell_c_np(eta_va, t_va, e_va),
            "val_eta_sd": float(eta_va.std()),
            "lr": f * a.lr_head,
            "lr_backbone": f * a.lr_backbone * a.layer_decay,   # top block
            "grad_norm": float(np.median(gnorms)) if gnorms else float("nan"),  # before clipping
            "skipped_batches": skipped,
            "gpu_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else 0.0,
            "sec": round(time.time() - t0, 1),
        }
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        improved = row["val_c"] > best_c  # NaN (no val events, smoke) never improves
        if improved or best_epoch < 0:
            best_c, best_epoch = (row["val_c"] if improved else best_c), epoch
        if a.select == "last" or improved or best_state is None:
            sel_c, sel_epoch = row["val_c"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
              + (" *" if improved else ""), flush=True)

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "epoch": sel_epoch, "val_c": sel_c, "config": cfg},
               out / "model.pt")
    te = cohort[cohort.split == "test"].sort_values("row")
    rows = [df[["pid", "split", "time", "event"]].assign(eta=predict(model, x, device, amp, batch=a.eval_batch))
            for df, x in ((tr, x_tr), (va, x_va), (te, view(cache[te["row"].to_numpy()])))]
    pd.concat(rows).sort_values("pid").to_csv(out / "pred.csv", index=False)
    plot_curves(pd.DataFrame(hist), out / "curves.png",
                f"{a.run_name} (init {a.ckpt}): selected epoch {sel_epoch} ({a.select}), val C {sel_c:.3f}"
                f" (best {best_c:.3f} at ep {best_epoch})", sel_epoch)
    print(f"done: selected epoch {sel_epoch} ({a.select}), val C {sel_c:.4f};"
          f" best val C {best_c:.4f} at epoch {best_epoch} -> {out}")


if __name__ == "__main__":
    main()
