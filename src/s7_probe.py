#!/usr/bin/env python3
"""S7 -- linear Cox probe on frozen features (S6), evaluated by s5_evaluate.py.

The image encoder stays frozen; only a linear log-risk eta = x.w is fitted.
This is the "linear readout" protocol of DrasCLR / 3DINO / DINOv2, with the
Cox loss of S4 instead of a classifier:
  features   one S6 feature set (cls | patch | cls_patch | last4), or none;
             --clinical appends age, sex (M=1) and current smoking
             (cigsmok=1). "none --clinical" is the clinical-only reference.
             Standardised with train-split mean/SD.
  fit        elastic-net Cox path (sksurv CoxnetSurvivalAnalysis, Breslow,
             l1_ratio 0.5, 50 penalties down to 1% of alpha_max) on train.
             Not ridge: on synthetic data (1 informative column among 1,023
             noise columns) ridge could not find the signal at any l2.
  selection  the penalty with the highest Harrell C on val. Test is not read.
Writes runs/{run_name}/ in the S4 layout, so s5_evaluate.py works unchanged:
  config.json, probe_grid.csv (val C per penalty), pred.csv (eta for all splits).

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
from model import harrell_c
from nlst import COHORT_CSV, DATA, RUNS_DIR

FEATURE_SETS = ("cls", "patch", "cls_patch", "last4", "none")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(DATA / "embeddings" / "3dino.npz"))
    ap.add_argument("--feature-set", choices=FEATURE_SETS, default="cls")
    ap.add_argument("--clinical", action="store_true", help="append age, sex, current smoking")
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--n-alphas", type=int, default=50)
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


def fit_path(x: np.ndarray, t: np.ndarray, e: np.ndarray, l1_ratio: float, n_alphas: int):
    """Elastic-net Cox regularisation path (glmnet-style, Breslow ties) on train."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=n_alphas, alpha_min_ratio=0.01,
                               max_iter=200000, fit_baseline_model=False)
    m.fit(x, Surv.from_arrays(e.astype(bool), t))
    return m


def main() -> None:
    a = parse_args()
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    X, names = design(cohort, a)
    tr = (cohort["split"] == "train").to_numpy()
    va = (cohort["split"] == "val").to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1.0)

    t, e = cohort["time"].to_numpy(float), cohort["event"].to_numpy(int)
    m = fit_path(X[tr], t[tr], e[tr], a.l1_ratio, a.n_alphas)

    grid, best = [], None
    for alpha, coef in zip(m.alphas_, m.coef_.T):
        eta = X @ coef
        row = {"alpha": float(alpha), "nonzero": int((coef != 0).sum()),
               "train_c": harrell_c(eta[tr], t[tr], e[tr]),
               "val_c": harrell_c(eta[va], t[va], e[va])}
        grid.append(row)
        if row["nonzero"] and (best is None or row["val_c"] > best[0]["val_c"]):
            best = (row, eta)
    assert best is not None, "every penalty on the path gave an all-zero model"
    for row in grid[::max(1, len(grid) // 10)]:
        print(f"alpha {row['alpha']:.3g}: {row['nonzero']:4d} non-zero  train C {row['train_c']:.4f}  "
              f"val C {row['val_c']:.4f}", flush=True)

    out = Path(a.runs_dir) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(out / "probe_grid.csv", index=False)
    cfg = {**vars(a), "n_features": len(names), "alpha": best[0]["alpha"],
           "nonzero": best[0]["nonzero"], "val_c": best[0]["val_c"], "train_c": best[0]["train_c"],
           "n_train": int(tr.sum()), "events_train": int(cohort.loc[tr, "event"].sum())}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    cohort[["pid", "split", "time", "event"]].assign(eta=best[1]) \
        .sort_values("pid").to_csv(out / "pred.csv", index=False)
    print(f"chose alpha {cfg['alpha']:.3g}: {cfg['nonzero']} non-zero, val C {cfg['val_c']:.4f}; wrote {out}")


if __name__ == "__main__":
    main()
