#!/usr/bin/env python3
"""C2 training curves + BCE-spike diagnostics, from runs/curves/*/history.json.

The `curves` run set (c2_train.py --set curves) re-trains, one seed each, the
configs whose curves we show: new_main, new_bce, null and legacy_scale per
organ, for the full epoch budget (no early stop; the epoch early stopping
would have picked is marked). Every curve uses ONE convention per panel:

  (a) optimisation loss -- what the optimiser saw: online mean over the
      epoch's batches, train-mode BatchNorm, augmentation, weights changing.
      Cox: batch-local risk sets (32 pts, zero-score loss ~3.2); BCE:
      pos_weight-weighted. Dashed = the same loss for a constant score.
  (b) evaluation loss minus its no-information baseline, model frozen in
      eval() mode, no augmentation, same definition on train and val.
      Cox: risk sets over the whole split, minus the loss of a constant score
      (removes the log risk-set-size baseline, ~7.5 for either split).
      BCE: UNweighted, minus the binary entropy of the split's prevalence
      (the best constant). Below 0 = better than knowing nothing.
  (c) discrimination: train C from the same frozen eval pass (comparable to
      val C), the old online train C (labelled), val C, val AUC.

Diagnostics figure (*_diag.png) per run: val logit quantiles per epoch (logit
shift vs. ranking), BCE loss contributions from positives / negatives and the
share from the worst 1% of patients, BatchNorm running statistics, and val
loss / C with BN running stats (eval) vs. BN batch statistics. Spiking epochs
of every BCE run are tabulated in data/c2_curves_spikes.csv.

c2_train.py imports plot_curves() for each run's own runs/<set>/<run>/curves.png.

Usage:
    env/.venv/bin/python src/c2_plot_curves.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"
FIG_DIR = PROJECT / "figs" / "c2_training_curves"
SPIKES_CSV = PROJECT / "data" / "c2_curves_spikes.csv"
FIG_NAME = {"main": "new_main", "bce": "new_bce", "null": "null", "legacy_scale": "legacy_scale"}


def n_headline_seeds(cfg: dict, runs_dir: Path) -> int:
    """How many seeds the headline (C4) ensemble of this run's group has."""
    src = "grid" if cfg["tag"] == "legacy_scale" else "final"
    return len(list((runs_dir / src).glob(f"{src}__{cfg['organ']}__{cfg['tag']}__*/result.json")))


def _arr(hist, k):
    return np.asarray(hist[k], dtype=float)


def plot_curves(hist: dict, cfg: dict, res: dict, out: Path, n_seeds: int | None = None) -> None:
    cox = cfg["loss"] == "cox"
    ep = np.arange(len(hist["cindex_val"]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(ep, hist["opt_loss"], color="tab:blue", label="train (online)")
    ax[0].plot(ep, hist["opt_loss_null"], color="tab:blue", ls="--", lw=0.8, label="constant score")
    ax[0].set_title("(a) optimisation loss\n" + (f"Cox, batch-local risk sets ({cfg['batch_size']} pts)" if cox
                    else f"BCE weighted by pos_weight {res.get('pos_weight', float('nan')):.2f}"), fontsize=9)

    for split, col in (("train", "tab:blue"), ("val", "tab:orange")):
        ax[1].plot(ep, _arr(hist, f"eval_loss_{split}") - _arr(hist, f"eval_null_{split}"), color=col, label=split)
    ax[1].axhline(0, color="gray", ls="--", lw=0.8)
    ax[1].set_title("(b) evaluation loss - no-information baseline\n" +
                    ("Cox, whole-split risk sets; 0 = constant score" if cox else
                     "unweighted BCE - H(split prevalence); 0 = best constant"), fontsize=9)

    ax[2].plot(ep, hist["cindex_train"], color="tab:blue", label="train C (eval mode, end of epoch)")
    ax[2].plot(ep, hist["cindex_train_online"], color="tab:blue", ls=":", lw=0.9, alpha=0.7,
               label="train C (online: train-mode BN, " + ("aug, " if cfg["augment"] else "") + "changing weights)")
    ax[2].plot(ep, hist["cindex_val"], color="tab:orange", label="val C")
    ax[2].plot(ep, hist["auc_val"], color="tab:green", alpha=0.6, label="val AUC")
    ax[2].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[2].set_title("(c) discrimination", fontsize=9)
    for a in ax:
        a.axvline(res["best_epoch"], color="k", ls=":", lw=0.8, label=f"selected ep {res['best_epoch']}")
        if res.get("stop_epoch") is not None and res["stop_epoch"] < len(ep) - 1:
            a.axvline(res["stop_epoch"], color="r", ls=":", lw=0.8, label=f"early stop ep {res['stop_epoch']}")
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    seeds = (f"one training run (seed {cfg['seed']}); headline test results use the "
             f"{n_seeds}-seed ensemble" if n_seeds else f"one training run (seed {cfg['seed']})")
    fig.suptitle(f"{cfg['run_id']}\n{seeds}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def plot_diagnostics(hist: dict, cfg: dict, res: dict, out: Path) -> None:
    ep = np.arange(len(hist["cindex_val"]))
    q = np.asarray(hist["val_eta_q"])  # quantiles 0, 1, 5, 25, 50, 75, 95, 99, 100
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))

    a = ax[0, 0]
    a.fill_between(ep, q[:, 2], q[:, 6], color="tab:orange", alpha=0.2, label="val 5-95%")
    a.fill_between(ep, q[:, 3], q[:, 5], color="tab:orange", alpha=0.4, label="val 25-75%")
    a.plot(ep, q[:, 4], color="tab:orange", label="val median")
    a.plot(ep, q[:, 0], color="tab:orange", ls=":", lw=0.8, label="val min / max")
    a.plot(ep, q[:, 8], color="tab:orange", ls=":", lw=0.8)
    a.plot(ep, hist["train_eta_mean"], color="tab:blue", label="train mean (eval mode)")
    a.set_title("(a) logit eta per epoch, eval mode", fontsize=9)
    a.set_ylabel("eta")

    a = ax[0, 1]
    if cfg["loss"] == "bce":
        a.plot(ep, hist["val_loss_pos"], color="tab:red", label="from positives (cancers)")
        a.plot(ep, hist["val_loss_neg"], color="tab:blue", label="from negatives")
        a.set_title("(b) val unweighted BCE = positives + negatives (each / n_val)", fontsize=9)
        b = a.twinx()
        b.plot(ep, hist["val_loss_top1pct_share"], color="k", lw=0.8, ls="--", label="share from worst 1%")
        b.set_ylim(0, 1)
        b.set_ylabel("share of val loss from worst 1% of patients", fontsize=8)
        b.legend(fontsize=7, loc="upper right")
    else:
        a.plot(ep, _arr(hist, "eval_loss_val") - _arr(hist, "eval_null_val"), color="tab:orange", label="val")
        a.axhline(0, color="gray", ls="--", lw=0.8)
        a.set_title("(b) val Cox loss - constant-score loss", fontsize=9)

    a = ax[1, 0]
    rv = np.asarray(hist["bn_running_var_mean"])
    rm = np.asarray(hist["bn_running_mean_absmean"])
    for j in range(rv.shape[1]):
        a.plot(ep, rv[:, j], color=f"C{j}", label=f"BN{j + 1} mean running var")
        a.plot(ep, rm[:, j], color=f"C{j}", ls="--", lw=0.8, label=f"BN{j + 1} mean |running mean|")
    a.set_yscale("log")
    a.set_title("(c) BatchNorm running statistics (end of epoch)", fontsize=9)
    a.legend(fontsize=6, ncol=2)

    a = ax[1, 1]
    a.plot(ep, _arr(hist, "eval_loss_val") - _arr(hist, "eval_null_val"), color="tab:orange",
           label="val loss excess, BN running stats (eval)")
    if "eval_loss_val_bnbatch" in hist:
        a.plot(ep, _arr(hist, "eval_loss_val_bnbatch") - _arr(hist, "eval_null_val"), color="tab:purple",
               label="val loss excess, BN batch stats")
    a.axhline(0, color="gray", ls="--", lw=0.8)
    a.set_ylabel("loss - no-information baseline")
    b = a.twinx()
    b.plot(ep, hist["cindex_val"], color="tab:orange", ls=":", label="val C, running stats")
    if "cindex_val_bnbatch" in hist:
        b.plot(ep, hist["cindex_val_bnbatch"], color="tab:purple", ls=":", label="val C, batch stats")
    b.set_ylabel("val C (dotted)")
    b.legend(fontsize=7, loc="lower right")
    a.set_title("(d) BN running stats vs batch stats on val", fontsize=9)
    for a in ax.flat:
        a.axvline(res["best_epoch"], color="k", ls=":", lw=0.8)
        a.set_xlabel("epoch")
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{cfg['run_id']} -- diagnostics", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def spike_rows(hist: dict, cfg: dict, k: int = 5) -> list[dict]:
    """The k epochs with the largest val evaluation loss (BCE runs), with what
    each candidate explanation predicts: a global logit shift (median moves,
    C stable), a few extreme patients (top-1% share high), BN running stats
    (batch-stat loss stays low)."""
    loss = _arr(hist, "eval_loss_val")
    q = np.asarray(hist["val_eta_q"])
    rows = []
    for e in np.argsort(-loss)[:k]:
        rows.append({"run_id": cfg["run_id"], "epoch": int(e), "val_bce": loss[e],
                     "val_bce_pos": hist["val_loss_pos"][e], "val_bce_neg": hist["val_loss_neg"][e],
                     "top1pct_share": hist["val_loss_top1pct_share"][e],
                     "val_bce_bnbatch": hist.get("eval_loss_val_bnbatch", [np.nan] * len(loss))[e],
                     "val_c": hist["cindex_val"][e],
                     "val_c_bnbatch": hist.get("cindex_val_bnbatch", [np.nan] * len(loss))[e],
                     "eta_min": q[e, 0], "eta_median": q[e, 4], "eta_max": q[e, 8],
                     "median_val_bce_all_epochs": float(np.median(loss))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--set", default="curves")
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    ap.add_argument("--spikes-csv", default=str(SPIKES_CSV))
    args = ap.parse_args()
    runs_dir, fig_dir = Path(args.runs_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    spikes = []
    for res_f in sorted((runs_dir / args.set).glob("*/result.json")):
        res = json.loads(res_f.read_text())
        hist = json.loads((res_f.parent / "history.json").read_text())
        name = f"{res['organ']}_{FIG_NAME.get(res['tag'], res['tag'])}"
        plot_curves(hist, res, res, fig_dir / f"{name}.png", n_headline_seeds(res, runs_dir))
        plot_diagnostics(hist, res, res, fig_dir / f"{name}_diag.png")
        if res["loss"] == "bce":
            spikes += spike_rows(hist, res)
        print(f"wrote {fig_dir}/{name}{{,_diag}}.png")
    if spikes:
        df = pd.DataFrame(spikes)
        df.to_csv(args.spikes_csv, index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(df.drop(columns="run_id").round(3).to_string(index=False))
        print(f"wrote {args.spikes_csv}")


if __name__ == "__main__":
    main()
