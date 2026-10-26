#!/usr/bin/env python3
"""S7 -- linear Cox probe on frozen features (S6), evaluated by s5_evaluate.py.

The image encoder stays frozen; only a linear log-risk eta = x.w is fitted.
This is the "linear readout" protocol of DrasCLR / 3DINO / DINOv2, with the
Cox loss of S4 instead of a classifier:
  features   one S6 feature set (cls | patch | cls_patch | last4), or none;
             --clinical appends age, sex (M=1) and current smoking
             (cigsmok=1). "none --clinical" is the clinical-only reference.
             Standardised with train-split mean/SD.
  fit        full-batch Cox partial likelihood (model.cox_loss, Breslow)
             + l2 * |w|^2, L-BFGS, on the train split
  selection  l2 from a fixed grid by Harrell C on val. Test is not read.
Writes runs/{run_name}/ in the S4 layout, so s5_evaluate.py works unchanged:
  config.json, probe_grid.csv (val C per l2), pred.csv (eta for all splits).

Usage (login node, CPU, ~minutes):
    env/.venv/bin/python src/s7_probe.py --features data/embeddings/3dino.npz \\
        --feature-set cls --run-name probe_3dino_cls
    env/.venv/bin/python src/s7_probe.py --feature-set none --clinical --run-name probe_clinical
    env/.venv/bin/python src/s5_evaluate.py --run probe_3dino_cls
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import cox_loss, harrell_c
from nlst import COHORT_CSV, DATA, RUNS_DIR

FEATURE_SETS = ("cls", "patch", "cls_patch", "last4", "none")
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(DATA / "embeddings" / "3dino.npz"))
    ap.add_argument("--feature-set", choices=FEATURE_SETS, default="cls")
    ap.add_argument("--clinical", action="store_true", help="append age, sex, current smoking")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    return ap.parse_args()


def design(cohort: pd.DataFrame, a) -> tuple[np.ndarray, list[str]]:
    """Feature matrix in cohort row order + column names."""
    blocks, names = [], []
    if a.feature_set != "none":
        z = np.load(a.features, allow_pickle=True)
        pos = pd.Series(np.arange(len(z["pid"])), index=z["pid"].astype(str))
        missing = set(cohort["pid"]) - set(pos.index)
        assert not missing, f"{len(missing)} cohort pids have no features"
        take = pos.loc[cohort["pid"]].to_numpy()
        keys = {"cls_patch": ("cls", "patch")}.get(a.feature_set, (a.feature_set,))
        for k in keys:
            blocks.append(z[k][take])
            names += [f"{k}_{i}" for i in range(z[k].shape[1])]
    if a.clinical:
        blocks.append(np.stack([cohort["age"].to_numpy(float),
                                (cohort["sex"] == "M").to_numpy(float),
                                (cohort["cigsmok"] == 1).to_numpy(float)], axis=1))
        names += ["age", "male", "current_smoker"]
    assert blocks, "no features: use a feature set or --clinical"
    return np.concatenate(blocks, axis=1).astype(np.float64), names


def fit_cox(x: torch.Tensor, t: torch.Tensor, e: torch.Tensor, l2: float) -> torch.Tensor:
    """argmin_w cox_loss(x w) + l2 |w|^2 (convex), by L-BFGS."""
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=500, tolerance_grad=1e-7,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = cox_loss(x @ w, t, e) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def main() -> None:
    a = parse_args()
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    X, names = design(cohort, a)
    tr = (cohort["split"] == "train").to_numpy()
    va = (cohort["split"] == "val").to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1.0)

    x = torch.from_numpy(X)
    t = torch.from_numpy(cohort["time"].to_numpy(float))
    e = torch.from_numpy(cohort["event"].to_numpy(float))
    trt = torch.from_numpy(tr)

    grid, best = [], None
    for l2 in L2_GRID:
        w = fit_cox(x[trt], t[trt], e[trt], l2)
        eta = (x @ w).numpy()
        row = {"l2": l2,
               "train_c": harrell_c(eta[tr], cohort["time"][tr], cohort["event"][tr]),
               "val_c": harrell_c(eta[va], cohort["time"][va], cohort["event"][va])}
        grid.append(row)
        print(f"l2 {l2:g}: train C {row['train_c']:.4f}  val C {row['val_c']:.4f}", flush=True)
        if best is None or row["val_c"] > best[0]["val_c"]:
            best = (row, eta)

    out = Path(a.runs_dir) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(out / "probe_grid.csv", index=False)
    cfg = {**vars(a), "n_features": len(names), "l2": best[0]["l2"], "val_c": best[0]["val_c"],
           "n_train": int(tr.sum()), "events_train": int(cohort.loc[tr, "event"].sum())}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    cohort[["pid", "split", "time", "event"]].assign(eta=best[1]) \
        .sort_values("pid").to_csv(out / "pred.csv", index=False)
    print(f"chose l2 {cfg['l2']:g} (val C {cfg['val_c']:.4f}); wrote {out}")


if __name__ == "__main__":
    main()
