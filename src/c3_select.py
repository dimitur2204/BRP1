#!/usr/bin/env python3
"""C3 -- pick one new-recipe config per organ from the C2 grid, on VALIDATION
Harrell C only (test predictions are never read here).

Outputs:
  data/c3_val_grid.csv            every grid run: lr, wd, dropout, best epoch, val C/AUC
  data/c3_chosen_configs.json     {organ: {lr, weight_decay, dropout, val_cindex}}
  figs/c3_val_grid.png            val C heatmap per organ (lr x wd, one panel per dropout)

Winner's curse: the chosen config's val C is the MAXIMUM of 12 noisy
estimates, so it is optimistic. That is why c2 --set final retrains the chosen
config with fresh seeds (1-5) and c4 reports test C only.

Usage:
    env/.venv/bin/python src/c3_select.py
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
RUNS = PROJECT / "runs" / "grid"
OUT_CSV = PROJECT / "data" / "c3_val_grid.csv"
OUT_JSON = PROJECT / "data" / "c3_chosen_configs.json"
OUT_FIG = PROJECT / "figs" / "c3_val_grid.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.runs_dir).glob("*/result.json")):
        r = json.loads(f.read_text())
        if r.get("kind") != "train":
            continue
        rows.append({"organ": r["organ"], "tag": r["tag"], "seed": r["seed"],
                     "lr": r["lr"], "weight_decay": r["weight_decay"], "dropout": r["dropout"],
                     "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                     "val_cindex": r["val"]["cindex"], "val_auc": r["val"]["auc"],
                     "seconds": r["seconds"], "run_id": r["run_id"]})
    df = pd.DataFrame(rows)
    df.sort_values(["organ", "tag", "val_cindex"], ascending=[True, True, False]).to_csv(OUT_CSV, index=False)
    grid = df[df["tag"] == "grid"]
    print(grid.drop(columns=["run_id", "tag", "seed"]).sort_values(["organ", "val_cindex"],
                                                                     ascending=[True, False]).to_string(index=False))
    print("\nlegacy-recipe runs (val):")
    print(df[df["tag"] != "grid"][["organ", "tag", "seed", "val_cindex", "val_auc", "best_epoch"]].to_string(index=False))

    chosen = {}
    for organ, g in grid.groupby("organ"):
        n_expected = 12
        if len(g) < n_expected:
            print(f"WARNING: {organ} has only {len(g)}/{n_expected} grid runs finished")
        b = g.loc[g["val_cindex"].idxmax()]
        chosen[organ] = {"lr": float(b.lr), "weight_decay": float(b.weight_decay),
                         "dropout": float(b.dropout), "val_cindex": round(float(b.val_cindex), 4),
                         "grid_val_cindex_range": [round(float(g.val_cindex.min()), 4),
                                                   round(float(g.val_cindex.max()), 4)],
                         "n_grid_runs": int(len(g))}
        print(f"\n{organ}: chosen lr={b.lr:g} wd={b.weight_decay:g} dropout={b.dropout:g} "
              f"(val C {b.val_cindex:.3f}; grid range {g.val_cindex.min():.3f}-{g.val_cindex.max():.3f})")
    Path(args.out_json).write_text(json.dumps(chosen, indent=1))
    print(f"wrote {args.out_json}, {OUT_CSV}")

    organs = sorted(grid["organ"].unique())
    dos = sorted(grid["dropout"].unique())
    fig, axes = plt.subplots(len(organs), len(dos), figsize=(4.6 * len(dos) + 1.2, 3.6 * len(organs)),
                             squeeze=False, layout="constrained")
    vmin, vmax = 0.5, max(0.7, grid.val_cindex.max())
    for i, organ in enumerate(organs):
        for j, do in enumerate(dos):
            g = grid[(grid.organ == organ) & (grid.dropout == do)]
            piv = g.pivot(index="weight_decay", columns="lr", values="val_cindex")
            ax = axes[i, j]
            im = ax.imshow(piv.values, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index])
            for (yy, xx), v in np.ndenumerate(piv.values):
                light = (v - vmin) / (vmax - vmin) > 0.6  # dark text on the bright end of viridis
                ax.text(xx, yy, f"{v:.3f}", ha="center", va="center", color="k" if light else "w", fontsize=9)
            ax.set_xlabel("learning rate")
            ax.set_ylabel("weight decay")
            ax.set_title(f"{organ}, dropout {do:g}: val Harrell C", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.7, label="val Harrell C")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
