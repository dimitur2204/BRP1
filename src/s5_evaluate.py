#!/usr/bin/env python3
"""S5 -- evaluate a trained run on the held-out test split (and val, for reference).

This is the only script that reads test predictions. Metrics per split:
  c_index         Harrell C of eta, with a 95% patient-bootstrap CI
  c_after_1y      Harrell C among patients still event-free and in follow-up
                  at 1 year. Drops the cancers already visible on the T0 scan
                  (27% of events are diagnosed within a year), so this is the
                  "predicts future cancer" number.
  auc_{1,2,3,5}y  cumulative/dynamic time-dependent AUC (sksurv, IPCW with the
                  train split's censoring distribution)
  c_age           Harrell C of age alone (reference for how much the image adds)
KM figure: test patients in eta tertiles, cut points taken from val.

Usage:
    env/.venv/bin/python src/s5_evaluate.py --run baseline [--n-boot 1000]
-> runs/{run}/eval/metrics.csv, runs/{run}/eval/km_test.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import harrell_c
from nlst import COHORT_CSV, RUNS_DIR

YEARS = (1, 2, 3, 5)


def surv(df: pd.DataFrame):
    from sksurv.util import Surv
    return Surv.from_arrays(df["event"].astype(bool).to_numpy(), df["time"].to_numpy(float))


def split_metrics(df: pd.DataFrame, train: pd.DataFrame, n_boot: int, rng) -> dict:
    from sksurv.metrics import cumulative_dynamic_auc
    eta, t, e = df["eta"].to_numpy(), df["time"].to_numpy(), df["event"].to_numpy()
    c = harrell_c(eta, t, e)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(df), len(df))
        boot.append(harrell_c(eta[i], t[i], e[i]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
    late = df[df["time"] > 365]
    times = [y * 365.25 for y in YEARS]
    auc = []
    for tp in times:  # NaN where a horizon is outside the split's follow-up (tiny smoke splits)
        try:
            auc.append(float(cumulative_dynamic_auc(surv(train), surv(df), eta, [tp])[0][0]))
        except ValueError:
            auc.append(np.nan)
    out = {"n": len(df), "events": int(e.sum()),
           "c_index": c, "c_lo": lo, "c_hi": hi,
           "c_after_1y": harrell_c(late["eta"], late["time"], late["event"]),
           "events_after_1y": int(late["event"].sum()),
           "c_age": harrell_c(df["age"], t, e)}
    out.update({f"auc_{y}y": a for y, a in zip(YEARS, auc)})
    return out


def km_figure(test: pd.DataFrame, cuts, path: Path, title: str) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    grp = np.digitize(test["eta"], cuts)  # 0 low, 1 mid, 2 high
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for g, name in enumerate(["low", "mid", "high"]):
        d = test[grp == g]
        KaplanMeierFitter(label=f"{name} risk (n={len(d)}, {int(d.event.sum())} events)") \
            .fit(d["time"] / 365.25, d["event"]).plot_survival_function(ax=ax, ci_show=True)
    p = multivariate_logrank_test(test["time"], grp, test["event"]).p_value
    ax.set_xlabel("years since T0 scan"); ax.set_ylabel("lung-cancer-free probability")
    ax.set_title(f"{title}\ntest split, eta tertiles from val, log-rank p = {p:.2g}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    run = Path(a.runs_dir) / a.run
    pred = pd.read_csv(run / "pred.csv", dtype={"pid": str})
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})[["pid", "age"]]
    pred = pred.merge(cohort, on="pid", how="left")
    train = pred[pred.split == "train"]
    rng = np.random.default_rng(0)

    rows = []
    for s in ("val", "test"):
        rows.append({"split": s, **split_metrics(pred[pred.split == s], train, a.n_boot, rng)})
    res = pd.DataFrame(rows)
    out = run / "eval"
    out.mkdir(exist_ok=True)
    cuts = np.quantile(pred.loc[pred.split == "val", "eta"], [1 / 3, 2 / 3])
    res["km_logrank_p"] = [np.nan, km_figure(pred[pred.split == "test"], cuts, out / "km_test.png", a.run)]
    res.to_csv(out / "metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res.round(4).to_string(index=False))
    sel = json.loads((run / "config.json").read_text())
    print(f"\n{a.run}: n_train {sel['n_train']} ({sel['events_train']} events); wrote {out}")


if __name__ == "__main__":
    main()
