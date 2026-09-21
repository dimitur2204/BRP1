#!/usr/bin/env python3
"""Stage 4 -- rank organs by held-out (test) AUC from Stage 3, WITH confidence
intervals. A bare AUC point estimate is meaningless with only 18 positive
cases in the test set (see PROGRESS.md) -- the CI is the actual result here,
not decoration.

Uses the Hanley-McNeil (1982) approximate standard error for AUC, computed
from the test set's true class counts (n0, n1 -- recovered from the stored
confusion matrix's row sums, which are threshold-independent).

Usage:
    env/.venv/bin/python src/stage4_ranking.py
"""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def hanley_mcneil_se(auc: float, n1: int, n0: int) -> float:
    """Standard error of an AUC estimate (Hanley & McNeil, 1982).
    n1 = number of positive (event) cases, n0 = number of negative cases."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return float(np.sqrt(max(var, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default=str(PROJECT / "data" / "stage3_results.csv"))
    parser.add_argument("--out", default=str(PROJECT / "figs" / "stage4_ranking.png"))
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.results_csv)))

    organs, aucs, los, his, ns = [], [], [], [], []
    for row in rows:
        cm = ast.literal_eval(row["test_confusion"])
        n0 = cm[0][0] + cm[0][1]  # true negatives + false positives = all true-label-0
        n1 = cm[1][0] + cm[1][1]  # all true-label-1
        auc = float(row["test_auc"])
        se = hanley_mcneil_se(auc, n1, n0)
        organs.append(row["organ"])
        aucs.append(auc)
        los.append(auc - 1.96 * se)
        his.append(auc + 1.96 * se)
        ns.append((n1, n0))

    order = np.argsort(aucs)[::-1]
    organs = [organs[i] for i in order]
    aucs = [aucs[i] for i in order]
    los = [los[i] for i in order]
    his = [his[i] for i in order]
    ns = [ns[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(organs))
    err_lo = [max(0, a - lo) for a, lo in zip(aucs, los)]
    err_hi = [max(0, hi - a) for a, hi in zip(aucs, his)]
    ax.errorbar(aucs, y, xerr=[err_lo, err_hi], fmt="o", capsize=4, color="tab:blue")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUC=0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{o}  (n1={n1}, n0={n0})" for o, (n1, n0) in zip(organs, ns)])
    ax.set_xlabel("test AUC (95% CI, Hanley-McNeil)")
    ax.set_title(f"Stage 4: per-organ CNN test AUC ranking{args.title_suffix}\n"
                  "(subset n=400, test n=127 with only 18 positive cases)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    print(f"wrote {out_png}")
    print("\norgan                    AUC    95% CI            CI excludes 0.5?")
    for o, a, lo, hi in zip(organs, aucs, los, his):
        excludes = "yes" if lo > 0.5 or hi < 0.5 else "no"
        print(f"{o:22s} {a:.3f}  [{lo:.3f}, {hi:.3f}]   {excludes}")


if __name__ == "__main__":
    main()
