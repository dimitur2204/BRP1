#!/usr/bin/env python3
"""E2 -- radiologist yr0 reads (ctab): calcium validity + smoking-burden proxy test (protocol.md, H5-H6).

B1 calcium / TH vs 'significant CV abnormality' (ctab 60) and emphysema (ctab 59)
B2 Cox attenuation of calcium / TH log-HR when emphysema is added (bootstrap CI)
B3 held-out test: does imaging add C-index beyond clinical + emphysema read?

Usage: env/.venv/bin/python experiments/E2_radiologist_reads/code/run_e2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
from aging_data import (BLUE, CLIN, INK, INK2, ORANGE, SURFACE, fit_cox, hr_row,  # noqa: E402
                        load_frame, logit, paired_bootstrap_c, style)

OUT = EXP / "results"
SEED = 0


def b1_validity(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for y in ("cv0", "emph0"):
        for x in ("calc_z", "th_z", "fat_z"):
            auc = roc_auc_score(df[y], df[x])
            boots = []
            for _ in range(1000):
                i = rng.integers(0, len(df), len(df))
                boots.append(roc_auc_score(df[y].iloc[i], df[x].iloc[i]))
            lr = logit(df, y, CLIN + [x])
            rows.append({"outcome": y, "marker": x, "n_pos": int(df[y].sum()), "n": len(df),
                         "auc": auc, "auc_lo": np.percentile(boots, 2.5), "auc_hi": np.percentile(boots, 97.5),
                         "or_adj": np.exp(lr.loc[x, "coef"]), "or_lo": np.exp(lr.loc[x, "coef"] - 1.96 * lr.loc[x, "se"]),
                         "or_hi": np.exp(lr.loc[x, "coef"] + 1.96 * lr.loc[x, "se"]), "p": lr.loc[x, "p"]})
    return pd.DataFrame(rows)


def attenuation(d: pd.DataFrame, marker: str) -> float:
    b0 = fit_cox(d, CLIN + [marker]).params_[marker]
    b1 = fit_cox(d, CLIN + ["emph0", marker]).params_[marker]
    return 1 - b1 / b0


def b2_proxy(df: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    rows = []
    emph = fit_cox(df, CLIN + ["emph0"])
    rows.append({"model": "clinical + emphysema", **hr_row(emph, "emph0")})
    for m in ("calc_z", "th_z"):
        rows.append({"model": "clinical + marker", **hr_row(fit_cox(df, CLIN + [m]), m)})
        rows.append({"model": "clinical + emphysema + marker", **hr_row(fit_cox(df, CLIN + ["emph0", m]), m)})
    joint = fit_cox(df, CLIN + ["emph0", "th_z", "calc_z"])
    rows += [{"model": "clinical + emphysema + TH + calcium", **hr_row(joint, v)} for v in ("emph0", "th_z", "calc_z")]
    res = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(df), len(df)) for _ in range(n_boot)]
    for m in ("calc_z", "th_z"):
        point = attenuation(df, m)
        boots = Parallel(n_jobs=4)(delayed(attenuation)(df.iloc[i].reset_index(drop=True), m) for i in idx)
        boots = np.array(boots)
        res.attrs[f"atten_{m}"] = {"point": float(point), "lo": float(np.percentile(boots, 2.5)),
                                   "hi": float(np.percentile(boots, 97.5))}
    return res


MODELS = {"clinical": CLIN, "clinical+emph": CLIN + ["emph0"],
          "clinical+emph+calcium": CLIN + ["emph0", "calc_z"],
          "clinical+emph+TH+calcium": CLIN + ["emph0", "th_z", "calc_z"]}


def b3_test(df: pd.DataFrame) -> pd.DataFrame:
    tv, te = df[df["split"] != "test"], df[df["split"] == "test"].reset_index(drop=True)
    risks = {}
    for m, cols in MODELS.items():
        cph = fit_cox(tv, cols)
        risks[m] = te[cols].to_numpy(float) @ cph.params_[cols].to_numpy()
    a = paired_bootstrap_c(te, risks, "clinical", 1000, SEED)
    b = paired_bootstrap_c(te, risks, "clinical+emph", 1000, SEED)
    a["dc_vs_clinical+emph"], a["dc2_lo"], a["dc2_hi"], a["dc2_p"] = (
        b["dc_vs_clinical+emph"].values, b["dc_lo"].values, b["dc_hi"].values, b["dc_p"].values)
    a["test_n"], a["test_events"] = len(te), int(te["event"].sum())
    return a


def fig_validity(df: pd.DataFrame, b1: pd.DataFrame) -> plt.Figure:
    """Prevalence of radiologist flags by calcium decile and TH category."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    dec = pd.qcut(df["log_calc"].rank(method="first"), 10, labels=False)
    for ax, (var, lab, xs, xl) in zip(axes, [
            ("cv0", "Significant CV abnormality (ctab 60)", dec, "Cardiac calcium decile (1 = lowest)"),
            ("emph0", "Emphysema (ctab 59)", dec, "Cardiac calcium decile (1 = lowest)")]):
        style(ax)
        p = df.groupby(xs)[var].mean()
        ax.bar(p.index + 1, p.values, color=ORANGE, width=0.8, zorder=2)
        r = b1[(b1["outcome"] == var) & (b1["marker"] == "calc_z")].iloc[0]
        ax.set_title(f"{lab}: prevalence at yr0\ncalcium AUC {r['auc']:.2f} [{r['auc_lo']:.2f}, {r['auc_hi']:.2f}], "
                     f"adj OR/SD {r['or_adj']:.2f}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(xl, fontsize=8, color=INK2)
        ax.set_ylabel("Fraction flagged by radiologist", fontsize=8, color=INK2)
        ax.set_xticks(range(1, 11))
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, att = load_frame()
    print(att, "emph0:", int(df["emph0"].sum()), "cv0:", int(df["cv0"].sum()))

    b1 = b1_validity(df)
    b1.to_csv(OUT / "b1_validity.csv", index=False)
    print(b1.round(4).to_string())
    fig_validity(df, b1).savefig(OUT / "fig_validity.png", dpi=160)

    b2 = b2_proxy(df)
    b2.to_csv(OUT / "b2_emphysema_adjustment.csv", index=False)
    print(b2.round(4).to_string(), "\n", b2.attrs)

    b3 = b3_test(df)
    b3.to_csv(OUT / "b3_test_cindex.csv", index=False)
    print(b3.round(4).to_string())

    json.dump({**att, **b2.attrs}, open(OUT / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
