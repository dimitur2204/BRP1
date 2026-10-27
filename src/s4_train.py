#!/usr/bin/env python3
"""S4 -- train the lungs 3D CNN with a Cox loss (one run = one config, one seed).

Recipe (all defaults; every one is a CLI flag):
  data       data/cohort.csv train split; the uint8 cache rows are read into
             host RAM once (~27 GB) and sent to the GPU batch by batch
  model      model.LungCNN (~3.5 M parameters), dropout 0.3
  loss       Cox partial likelihood over each batch of 32 shuffled patients
             (~6% events, so ~2 events per batch; the rare batch without
             events is skipped). --events-per-batch N instead builds every
             batch from N events + (32 - N) non-events (events are seen ~2x
             per epoch, no batch is skipped)
  optimiser  AdamW, weight decay 1e-2
  lr         --schedule cosine (default): linear warm-up from 0 to the peak lr
             1e-3 over the first 3 epochs, then cosine decay to --min-lr
             (default 0) at the last epoch, updated every batch.
             --schedule constant keeps lr fixed.
  length     80 epochs, run to the end (no early stopping, --patience 0), so
             the cosine schedule completes. --patience N re-enables early
             stopping after N epochs without val-C improvement.
  precision  bf16 autocast on GPU
  augment    rotation +-10 deg (axial plane), scale +-10%, shift +-6 voxels
  EMA        --ema D (default 0 = off): keep an exponential moving average of
             the weights (ema = D*ema + (1-D)*w after every optimiser step);
             all evaluation, selection and saving then use the EMA weights
  selection  after every epoch: Harrell C on the whole val split; keep the
             best-val-C weights (--select val_c, default) or the final
             weights (--select last)
  clinical   --clinical age,sex,... (default off): the Cox loss uses
             eta = eta_img + z @ beta (model.ClinicalCox), z = the cohort.csv
             covariates standardised on train (sex -> male 0/1). beta starts
             at a clinical-only Cox fit on train; --clinical-mode offset
             (default) keeps it fixed, so the CNN learns only the risk the
             covariates do not explain; joint trains it with the CNN (no
             weight decay). Selection and val/train C then use the total eta
No hyper-parameter search: this is the baseline to build on.
The first run (runs/baseline, 2026-10-05) used the old defaults:
  --schedule constant --lr 3e-4 --epochs 60 --patience 12

Outputs, runs/{run_name}/:
  config.json   the arguments + cohort sizes
  history.csv   per epoch: train loss, val loss (whole-split risk sets) and its
                eta=0 baseline, train C (eval mode, fixed 2,000-patient
                subset), val C (and val_c_raw, the non-EMA weights, with
                --ema), learning rate at the end of the epoch, peak GPU
                memory, seconds
  model.pt      the selected weights (EMA weights with --ema)
  pred.csv      eta for EVERY cohort patient (all splits) from model.pt.
                Test rows are only read by s5_evaluate.py.
                With --clinical: eta = the image part only (what s5 tests
                for added value over clinical), plus eta_clin (z @ beta)
                and eta_total (their sum, the trained score)
  clinical.json with --clinical: covariates, train mean/SD, beta (start and
                selected), clinical-only val C and val loss
  curves.png    loss and C curves

Usage:
    sbatch src/submit_s4_train.sh --run-name cosine     # current defaults, seed 0
    sbatch src/submit_s4_train.sh --run-name cosine_s1 --seed 1
    sbatch src/submit_s4_train.sh --run-name ema_epb4 --ema 0.999 --select last --events-per-batch 4
    sbatch src/submit_s4_train.sh --run-name clinical_cox/offset --clinical age,sex,cigsmok,lung_ml,lung_mean_hu
An existing run directory with a model.pt is never overwritten (pick a new
--run-name, or pass --overwrite).
    env/.venv/bin/python src/s4_train.py --smoke --device cpu \\
        --cohort <scratch>/cohort.csv --cache <scratch>/lungs_u8.npy --runs-dir <scratch>/runs
"""
from __future__ import annotations

import argparse
import json
import math
import time
import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import ClinicalCox, LungCNN, augment, cox_loss, harrell_c, to_input
from nlst import CACHE_NPY, COHORT_CSV, GRID, RUNS_DIR
from s5_evaluate import covariates


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N epochs without val-C improvement; 0 = run all epochs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    ap.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--min-lr", type=float, default=0.0, help="cosine ends here instead of 0")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--events-per-batch", type=int, default=0,
                    help="build every batch from N events + (batch - N) non-events; 0 = plain shuffle")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay of the weights (e.g. 0.999); evaluate/select/save the EMA. 0 = off")
    ap.add_argument("--select", choices=["val_c", "last"], default="val_c",
                    help="save the best-val-C epoch or the final epoch")
    ap.add_argument("--clinical", default="",
                    help="comma-separated cohort.csv covariates added to eta in the Cox loss; empty = off")
    ap.add_argument("--clinical-mode", choices=["offset", "joint"], default="offset",
                    help="offset: beta fixed at the train Cox fit; joint: beta trained with the CNN")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--n-train-eval", type=int, default=2000,
                    help="train patients scored in eval mode each epoch (train C)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--cache", default=str(CACHE_NPY))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--overwrite", action="store_true", help="replace an existing finished run")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, batch 4, for a CPU check")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch_size, a.n_train_eval, a.warmup_epochs = 3, 4, 16, 1.0
        a.events_per_batch = min(a.events_per_batch, 1)
    if not 0 <= a.events_per_batch < a.batch_size:
        ap.error("--events-per-batch must be in [0, batch size)")
    if not 0 <= a.ema < 1:
        ap.error("--ema must be in [0, 1)")
    a.clinical = [c for c in a.clinical.split(",") if c]
    return a


def clinical_design(tr: pd.DataFrame, others: list, cols: list) -> tuple:
    """Standardised covariates (float32 arrays, train mean/SD) for tr and each
    frame in `others`, and the clinical-only Cox coefficients fitted on tr."""
    from lifelines import CoxPHFitter
    X = covariates(tr, cols)
    mu, sd = X.mean(), X.std().replace(0, 1.0)
    z = [((covariates(d, cols) - mu) / sd).to_numpy(np.float32) for d in [tr, *others]]
    cph = CoxPHFitter().fit(pd.DataFrame(z[0], columns=X.columns)
                            .assign(time=tr["time"].to_numpy(), event=tr["event"].to_numpy()),
                            "time", "event")
    info = {"columns": list(X.columns), "mean": mu.tolist(), "sd": sd.tolist(),
            "beta_start": cph.params_[X.columns].tolist()}
    return z, info


def lr_at(step: int, total: int, warmup: int, peak: float, schedule: str, floor: float = 0.0) -> float:
    """Learning rate for optimiser step `step` (0-based) of `total`."""
    if schedule == "constant":
        return peak
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def epoch_batches(rng, event: np.ndarray, bs: int, events_per_batch: int) -> list:
    """Index arrays of one epoch's len(event) // bs batches (last partial batch dropped).

    events_per_batch 0: a plain shuffle (the original recipe, same RNG use).
    events_per_batch k: k events + (bs - k) non-events per batch. Both groups
    are drawn from reshuffled cycles, so every non-event is used about once
    per epoch and every event about (n_batches * k / n_events) times.
    """
    n_batches = len(event) // bs
    if not events_per_batch:
        perm = rng.permutation(len(event))
        return [np.sort(perm[i:i + bs]) for i in range(0, n_batches * bs, bs)]
    k = events_per_batch

    def cycle(idx, n):  # n draws from idx, reshuffled each time it runs out
        return np.concatenate([rng.permutation(idx) for _ in range(-(-n // len(idx)))])[:n]

    ev = cycle(np.flatnonzero(event == 1), n_batches * k).reshape(n_batches, k)
    ct = cycle(np.flatnonzero(event == 0), n_batches * (bs - k)).reshape(n_batches, bs - k)
    return [np.sort(np.concatenate([e, c])) for e, c in zip(ev, ct)]


@torch.no_grad()
def predict(model, x_u8: np.ndarray, device, amp, idx=None, batch: int = 32) -> np.ndarray:
    """eta for x_u8[idx] (all rows if idx is None), in eval mode, without copying x_u8."""
    model.eval()
    idx = np.arange(len(x_u8)) if idx is None else idx
    out = []
    for i in range(0, len(idx), batch):
        xb = to_input(torch.from_numpy(np.ascontiguousarray(x_u8[idx[i:i + batch]])).to(device))
        with amp():
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


def val_cox(eta: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    return float(cox_loss(torch.from_numpy(eta), torch.from_numpy(t), torch.from_numpy(e)))


def plot_curves(hist: pd.DataFrame, path, title: str, selected: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    a1.plot(hist.epoch, hist.train_loss, label="train (batch risk sets, aug)")
    a1.plot(hist.epoch, hist.val_loss - hist.val_loss_null, label="val - val(eta=0)")
    if "val_loss_clin" in hist:
        a1.plot(hist.epoch, hist.val_loss_clin - hist.val_loss_null, ls="--", label="val, clinical part only")
    a1.axhline(0, color="grey", lw=0.8)
    a1.set_xlabel("epoch"); a1.set_ylabel("Cox loss per event"); a1.legend(fontsize=8)
    a1.set_title("loss (val shown relative to a constant score)", fontsize=9)
    a2.plot(hist.epoch, hist.train_c, label="train C (eval mode, subset)")
    a2.plot(hist.epoch, hist.val_c, label="val C (EMA)" if "val_c_raw" in hist else "val C")
    if "val_c_raw" in hist:
        a2.plot(hist.epoch, hist.val_c_raw, lw=0.8, alpha=0.6, label="val C (raw weights)")
    if "val_c_img" in hist:
        a2.plot(hist.epoch, hist.val_c_img, label="val C, image part only")
        a2.plot(hist.epoch, hist.val_c_clin, ls="--", label="val C, clinical part only")
    a2.axvline(selected, color="grey", ls="--", lw=0.8, label=f"selected: ep {selected}")
    a2.axhline(0.5, color="grey", lw=0.8)
    a2.set_xlabel("epoch"); a2.set_ylabel("Harrell C"); a2.legend(fontsize=8)
    if "lr" in hist:
        a3.plot(hist.epoch, hist.lr)
    a3.set_xlabel("epoch"); a3.set_ylabel("learning rate (end of epoch)")
    a3.set_title("learning-rate schedule", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out = Path(a.runs_dir) / a.run_name
    if (out / "model.pt").exists() and not a.overwrite:
        raise SystemExit(f"{out} already holds a finished run -- use a new --run-name or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    cache = np.load(a.cache, mmap_mode="r")
    if cache.shape[1:] != GRID and not a.smoke:  # the 2.5 mm lungs cache, not e.g. lungs_1p4mm_u8.npy
        raise SystemExit(f"{a.cache} has volumes {cache.shape[1:]}, expected {GRID} (2.5 mm grid)")
    tr, va = cohort[cohort.split == "train"], cohort[cohort.split == "val"]
    te = cohort[cohort.split == "test"].sort_values("row")

    t0 = time.time()
    x_tr = np.asarray(cache[np.sort(tr["row"].to_numpy())])  # sorted rows = sequential read
    tr = tr.sort_values("row")
    x_va = np.asarray(cache[np.sort(va["row"].to_numpy())])
    va = va.sort_values("row")
    print(f"loaded train {x_tr.shape} + val {x_va.shape} in {time.time() - t0:.0f}s", flush=True)
    t_tr = tr["time"].to_numpy(np.float32); e_tr = tr["event"].to_numpy(np.float32)
    t_va = va["time"].to_numpy(np.float32); e_va = va["event"].to_numpy(np.float32)
    sub = np.sort(rng.choice(len(tr), size=min(a.n_train_eval, len(tr)), replace=False))

    clin = bool(a.clinical)
    z_tr = z_va = z_te = None
    model = LungCNN(dropout=a.dropout)
    if clin:
        (z_tr, z_va, z_te), clin_info = clinical_design(tr, [va, te], a.clinical)
        model = ClinicalCox(model, clin_info["beta_start"], train_beta=a.clinical_mode == "joint")
        lp_va = z_va @ np.asarray(clin_info["beta_start"], np.float32)
        clin_info.update({"val_c": harrell_c(lp_va, t_va, e_va), "val_loss": val_cox(lp_va, t_va, e_va)})
        print("clinical-only Cox (train fit): " + json.dumps(clin_info), flush=True)
    model = model.to(device)
    ema_model = copy.deepcopy(model) if a.ema else None  # GroupNorm: parameters only, no buffers
    eval_model = ema_model if a.ema else model           # what is evaluated, selected and saved

    def score(m, x, z, idx=None) -> tuple:
        """(total eta, image eta) of model m; equal without --clinical."""
        e_img = predict(m.image if clin else m, x, device, amp, idx)
        if not clin:
            return e_img, e_img
        e_clin = (z if idx is None else z[idx]) @ m.beta.detach().cpu().numpy()
        return e_img + e_clin, e_img

    n_updates = 0
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p is not getattr(model, "beta", None)]}]
    if clin and a.clinical_mode == "joint":
        groups.append({"params": [model.beta], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    n_params = sum(p.numel() for p in params)
    steps_per_epoch = len(tr) // a.batch_size        # batch slots, incl. skipped ones
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = int(round(a.warmup_epochs * steps_per_epoch))
    step = 0
    cfg = {**vars(a), "n_params": n_params,
           "n_train": len(tr), "events_train": int(e_tr.sum()),
           "n_val": len(va), "events_val": int(e_va.sum())}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2), flush=True)

    hist, best_c, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses, skipped = [], 0
        for b in epoch_batches(rng, e_tr, a.batch_size, a.events_per_batch):
            lr = lr_at(step, total_steps, warmup_steps, a.lr, a.schedule, a.min_lr)
            step += 1  # skipped batches advance the schedule too, so it ends on time
            if e_tr[b].sum() == 0:
                skipped += 1
                continue
            for g in opt.param_groups:
                g["lr"] = lr
            xb = to_input(torch.from_numpy(x_tr[b]).to(device, non_blocking=True))
            if not a.no_aug:
                xb = augment(xb)
            tb = torch.from_numpy(t_tr[b]).to(device)
            eb = torch.from_numpy(e_tr[b]).to(device)
            with amp():
                eta = model(xb, torch.from_numpy(z_tr[b]).to(device)) if clin else model(xb)
            loss = cox_loss(eta.float(), tb, eb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if a.ema:  # bias-corrected start: the decay ramps up to a.ema over the first steps
                n_updates += 1
                d = min(a.ema, (1 + n_updates) / (10 + n_updates))
                with torch.no_grad():
                    for pe, p in zip(ema_model.parameters(), model.parameters()):
                        pe.lerp_(p, 1.0 - d)

        eta_va, img_va = score(eval_model, x_va, z_va)
        eta_sub, _ = score(eval_model, x_tr, z_tr, idx=sub)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val_cox(eta_va, t_va, e_va),
            "val_loss_null": val_cox(np.zeros_like(eta_va), t_va, e_va),
            "train_c": harrell_c(eta_sub, t_tr[sub], e_tr[sub]),
            "val_c": harrell_c(eta_va, t_va, e_va),
            "val_eta_sd": float(eta_va.std()),
            "lr": opt.param_groups[0]["lr"],
            "skipped_batches": skipped,
            "gpu_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else 0.0,
            "sec": round(time.time() - t0, 1),
        }
        if clin:  # the image part alone, and the clinical-only references (constant if offset)
            row["val_c_img"] = harrell_c(img_va, t_va, e_va)
            row["val_c_clin"] = harrell_c(eta_va - img_va, t_va, e_va)
            row["val_loss_clin"] = val_cox(eta_va - img_va, t_va, e_va)
            row["val_img_sd"] = float(img_va.std())
        if a.ema:
            row["val_c_raw"] = harrell_c(score(model, x_va, z_va)[0], t_va, e_va)
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        improved = row["val_c"] > best_c  # NaN (no val events, smoke) never improves
        if improved or best_epoch < 0:  # best val C so far (for early stopping and logging)
            best_c, best_epoch = (row["val_c"] if improved else best_c), epoch
        if a.select == "last" or improved or best_state is None:
            sel_c, sel_epoch = row["val_c"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
              + (" *" if improved else ""), flush=True)
        if a.patience and epoch - best_epoch >= a.patience:
            print(f"early stop: no val C improvement for {a.patience} epochs")
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "epoch": sel_epoch, "val_c": sel_c, "config": cfg},
               out / "model.pt")
    rows = []
    for df, x, z in ((tr, x_tr, z_tr), (va, x_va, z_va), (te, cache[te["row"].to_numpy()], z_te)):
        eta, e_img = score(model, x, z)
        d = df[["pid", "split", "time", "event"]].assign(eta=e_img)  # eta = image part (s5)
        rows.append(d.assign(eta_clin=eta - e_img, eta_total=eta) if clin else d)
    pd.concat(rows).sort_values("pid").to_csv(out / "pred.csv", index=False)
    if clin:
        clin_info["beta_selected"] = model.beta.detach().cpu().tolist()
        (out / "clinical.json").write_text(json.dumps(clin_info, indent=2))
    plot_curves(pd.DataFrame(hist), out / "curves.png",
                f"{a.run_name}: selected epoch {sel_epoch} ({a.select}), val C {sel_c:.3f}"
                f" (best {best_c:.3f} at ep {best_epoch})", sel_epoch)
    print(f"done: selected epoch {sel_epoch} ({a.select}), val C {sel_c:.4f};"
          f" best val C {best_c:.4f} at epoch {best_epoch} -> {out}")


if __name__ == "__main__":
    main()
