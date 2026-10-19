#!/usr/bin/env python3
"""S10 -- linear Cox probe on frozen DINOv2 features (D5).

The question: do the self-supervised features carry information about time to
lung cancer? A penalised linear Cox model is fitted on them; nothing in the
backbone is trained. Same cohort, split, time and event as S4 (data/cohort.csv).

  features   data/features/{run}/{ckpt}.npy (S9): [CLS, mean patch token] per patient;
             --use cls | mean | both (default)
  scaling    z-score with train mean/SD; constant dims dropped
  model      elastic-net Cox (sksurv CoxnetSurvivalAnalysis, l1_ratio 0.5,
             50 penalties from alpha_max down to 1% of it), fitted on TRAIN
  selection  the penalty with the highest Harrell C on VAL
  output     runs/{name}/pred.csv in the S4 format (eta for every cohort patient),
             so s5_evaluate.py runs unchanged. This script never computes a test
             metric; test rows are only read by s5_evaluate.py.

Outputs, runs/{name}/ (name default: probe_{run}_{ckpt}_{use}):
  config.json   arguments, chosen penalty, non-zero coefficients, train / val C
  path.csv      per penalty: alpha, non-zero coefficients, train C, val C
  pred.csv      pid, split, time, event, eta

Usage (login node, env/.venv, ~1-5 min per probe):
    env/.venv/bin/python src/s10_dino_probe.py --run d2_m --ckpt "dino_epoch=039"
    env/.venv/bin/python src/s10_dino_probe.py --summary      # val C of every probe_* run
    env/.venv/bin/python src/s5_evaluate.py --run probe_d2_m_dino_epoch=039_both  # chosen one only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from nlst import COHORT_CSV, DATA, RUNS_DIR


def cindex(eta: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    if event.sum() == 0:
        return float("nan")
    return float(concordance_index_censored(event.astype(bool), time, eta)[0])


def load_features(run: str, ckpt: str, use: str, cohort: pd.DataFrame) -> np.ndarray:
    d = DATA / "features" / run
    f = np.load(d / f"{ckpt}.npy")
    pids = pd.read_csv(d / "pids.csv", dtype={"pid": str})["pid"]
    if pids.tolist() != cohort["pid"].tolist():
        raise SystemExit(f"{d}/pids.csv != cohort.csv order -- re-run s9")
    half = f.shape[1] // 2
    return {"cls": f[:, :half], "mean": f[:, half:], "both": f}[use]


def probe(a) -> None:
    name = a.name or f"probe_{a.run}_{a.ckpt}_{a.use}"
    out = Path(RUNS_DIR) / name
    if (out / "pred.csv").exists() and not a.overwrite:
        raise SystemExit(f"{out} exists (new --name or --overwrite)")
    cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    X = load_features(a.run, a.ckpt, a.use, cohort).astype(np.float64)
    split = cohort["split"].to_numpy()
    time, event = cohort["time"].to_numpy(float), cohort["event"].to_numpy(int)
    tr, va = split == "train", split == "val"

    mu, sd = X[tr].mean(0), X[tr].std(0)
    keep = sd > 1e-8
    X = (X[:, keep] - mu[keep]) / sd[keep]
    y_tr = np.array(list(zip(event[tr].astype(bool), time[tr])), dtype=[("e", bool), ("t", float)])
    print(f"{name}: {keep.sum()}/{len(keep)} non-constant dims, train {tr.sum()} "
          f"({event[tr].sum()} events), val {va.sum()} ({event[va].sum()} events)", flush=True)

    m = CoxnetSurvivalAnalysis(l1_ratio=a.l1_ratio, n_alphas=a.n_alphas, alpha_min_ratio=0.01,
                               max_iter=100000, fit_baseline_model=False)
    m.fit(X[tr], y_tr)
    rows = []
    for j, alpha in enumerate(m.alphas_):
        b = m.coef_[:, j]
        rows.append({"alpha": alpha, "nonzero": int((b != 0).sum()),
                     "train_c": cindex(X[tr] @ b, time[tr], event[tr]),
                     "val_c": cindex(X[va] @ b, time[va], event[va])})
    path = pd.DataFrame(rows)
    best = int(path["val_c"].idxmax())
    eta = X @ m.coef_[:, best]

    out.mkdir(parents=True, exist_ok=True)
    path.to_csv(out / "path.csv", index=False)
    cohort[["pid", "split", "time", "event"]].assign(eta=eta).sort_values("pid").to_csv(
        out / "pred.csv", index=False)
    meta = json.loads((DATA / "features" / a.run / f"{a.ckpt}.json").read_text())
    cfg = {"run": a.run, "ckpt": a.ckpt, "epoch": meta.get("epoch"), "use": a.use,
           "dims": int(keep.sum()), "l1_ratio": a.l1_ratio, **path.iloc[best].to_dict()}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(path.iloc[::5].round(4).to_string(index=False))
    print(f"chosen alpha {cfg['alpha']:.4g}: {cfg['nonzero']} non-zero, train C "
          f"{cfg['train_c']:.3f}, val C {cfg['val_c']:.3f} -> {out}/pred.csv", flush=True)


def summary() -> None:
    rows = [json.loads(p.read_text()) | {"name": p.parent.name}
            for p in sorted(Path(RUNS_DIR).glob("probe_*/config.json"))]
    ref = Path(RUNS_DIR) / "cosine" / "pred.csv"
    if ref.exists():  # the supervised CNN, val C from its own predictions
        p = pd.read_csv(ref)
        p = p[p.split == "val"]
        rows.append({"name": "cosine (supervised CNN, reference)",
                     "val_c": cindex(p["eta"].to_numpy(), p["time"].to_numpy(float),
                                     p["event"].to_numpy(int))})
    df = pd.DataFrame(rows)
    cols = [c for c in ["name", "epoch", "use", "dims", "nonzero", "train_c", "val_c"] if c in df]
    print(df[cols].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="pretraining run (data/features/{run}/)")
    ap.add_argument("--ckpt", help='feature file name without .npy, e.g. "dino_epoch=039" or random')
    ap.add_argument("--use", choices=["cls", "mean", "both"], default="both")
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--n-alphas", type=int, default=50)
    ap.add_argument("--name")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if not (a.run and a.ckpt):
        ap.error("give --run and --ckpt, or --summary")
    probe(a)


if __name__ == "__main__":
    main()
