#!/usr/bin/env python3
"""S4 -- train the lungs 3D CNN with a Cox loss (one run = one config, one seed).

Recipe (all defaults; every one is a CLI flag):
  data       data/cohort.csv train split; the uint8 cache rows are read into
             host RAM once (~27 GB) and sent to the GPU batch by batch
  model      model.LungCNN (~3.5 M parameters), dropout 0.3
  loss       Cox partial likelihood over each batch of 32 shuffled patients
             (~6% events, so ~2 events per batch; the rare batch without
             events is skipped)
  optimiser  AdamW, lr 3e-4, weight decay 1e-2, constant learning rate
  precision  bf16 autocast on GPU
  augment    rotation +-10 deg (axial plane), scale +-10%, shift +-6 voxels
  selection  after every epoch: Harrell C on the whole val split; keep the
             best-val-C weights, stop after `patience` epochs without
             improvement (max 60 epochs)
No hyper-parameter search: this is the baseline to build on.

Outputs, runs/{run_name}/:
  config.json   the arguments + cohort sizes
  history.csv   per epoch: train loss, val loss (whole-split risk sets) and its
                eta=0 baseline, train C (eval mode, fixed 2,000-patient
                subset), val C, seconds
  model.pt      best-val-C weights
  pred.csv      eta for EVERY cohort patient (all splits) from model.pt.
                Test rows are only read by s5_evaluate.py.
  curves.png    loss and C curves

Usage:
    sbatch src/submit_s4_train.sh                       # baseline, seed 0
    sbatch src/submit_s4_train.sh --seed 1 --run-name baseline_s1
    env/.venv/bin/python src/s4_train.py --smoke --device cpu \\
        --cohort <scratch>/cohort.csv --cache <scratch>/lungs_u8.npy --runs-dir <scratch>/runs
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import LungCNN, augment, cox_loss, harrell_c, to_input
from nlst import CACHE_NPY, COHORT_CSV, RUNS_DIR


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--n-train-eval", type=int, default=2000,
                    help="train patients scored in eval mode each epoch (train C)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--cache", default=str(CACHE_NPY))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--smoke", action="store_true", help="2 epochs, batch 4, for a CPU check")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch_size, a.n_train_eval = 2, 4, 16
    return a


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


def plot_curves(hist: pd.DataFrame, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist.epoch, hist.train_loss, label="train (batch risk sets, aug)")
    a1.plot(hist.epoch, hist.val_loss - hist.val_loss_null, label="val - val(eta=0)")
    a1.axhline(0, color="grey", lw=0.8)
    a1.set_xlabel("epoch"); a1.set_ylabel("Cox loss per event"); a1.legend(fontsize=8)
    a1.set_title("loss (val shown relative to a constant score)", fontsize=9)
    a2.plot(hist.epoch, hist.train_c, label="train C (eval mode, subset)")
    a2.plot(hist.epoch, hist.val_c, label="val C")
    best = hist.loc[hist.val_c.idxmax()]
    a2.axvline(best.epoch, color="grey", ls="--", lw=0.8, label=f"selected: ep {int(best.epoch)}")
    a2.axhline(0.5, color="grey", lw=0.8)
    a2.set_xlabel("epoch"); a2.set_ylabel("Harrell C"); a2.legend(fontsize=8)
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
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    cache = np.load(a.cache, mmap_mode="r")
    tr, va = cohort[cohort.split == "train"], cohort[cohort.split == "val"]

    t0 = time.time()
    x_tr = np.asarray(cache[np.sort(tr["row"].to_numpy())])  # sorted rows = sequential read
    tr = tr.sort_values("row")
    x_va = np.asarray(cache[np.sort(va["row"].to_numpy())])
    va = va.sort_values("row")
    print(f"loaded train {x_tr.shape} + val {x_va.shape} in {time.time() - t0:.0f}s", flush=True)
    t_tr = tr["time"].to_numpy(np.float32); e_tr = tr["event"].to_numpy(np.float32)
    t_va = va["time"].to_numpy(np.float32); e_va = va["event"].to_numpy(np.float32)
    sub = np.sort(rng.choice(len(tr), size=min(a.n_train_eval, len(tr)), replace=False))

    model = LungCNN(dropout=a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    cfg = {**vars(a), "n_params": n_params,
           "n_train": len(tr), "events_train": int(e_tr.sum()),
           "n_val": len(va), "events_val": int(e_va.sum())}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2), flush=True)

    hist, best_c, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        losses, skipped = [], 0
        for i in range(0, len(perm) - a.batch_size + 1, a.batch_size):  # drop last partial batch
            b = np.sort(perm[i:i + a.batch_size])
            if e_tr[b].sum() == 0:
                skipped += 1
                continue
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
            opt.step()
            losses.append(loss.item())

        eta_va = predict(model, x_va, device, amp)
        eta_sub = predict(model, x_tr, device, amp, idx=sub)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val_cox(eta_va, t_va, e_va),
            "val_loss_null": val_cox(np.zeros_like(eta_va), t_va, e_va),
            "train_c": harrell_c(eta_sub, t_tr[sub], e_tr[sub]),
            "val_c": harrell_c(eta_va, t_va, e_va),
            "val_eta_sd": float(eta_va.std()),
            "skipped_batches": skipped,
            "sec": round(time.time() - t0, 1),
        }
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        improved = row["val_c"] > best_c  # NaN (no val events, smoke) never improves
        if improved or best_state is None:
            best_c, best_epoch = (row["val_c"] if improved else best_c), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
              + (" *" if improved else ""), flush=True)
        if epoch - best_epoch >= a.patience:
            print(f"early stop: no val C improvement for {a.patience} epochs")
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_c": best_c, "config": cfg},
               out / "model.pt")
    te = cohort[cohort.split == "test"].sort_values("row")
    rows = [df[["pid", "split", "time", "event"]].assign(eta=predict(model, x, device, amp))
            for df, x in ((tr, x_tr), (va, x_va), (te, cache[te["row"].to_numpy()]))]
    pd.concat(rows).sort_values("pid").to_csv(out / "pred.csv", index=False)
    plot_curves(pd.DataFrame(hist), out / "curves.png",
                f"{a.run_name}: selected epoch {best_epoch}, val C {best_c:.3f}")
    print(f"done: best epoch {best_epoch}, val C {best_c:.4f} -> {out}")


if __name__ == "__main__":
    main()
