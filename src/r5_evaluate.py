#!/usr/bin/env python3
"""R5 -- evaluate R4's held-out test risk scores (test split: n~2,040, 154 events).

Per model (data/r4_test_predictions.csv):
  - Harrell's C and Uno's C (IPCW, censoring distribution from train+val,
    truncated at TAU years), 95% bootstrap CIs (N_BOOT resamples of the test set)
  - dC vs the clinical-only model: paired bootstrap (same resamples), CI and
    two-sided bootstrap p -- the incremental-value number
  - time-dependent AUC at 2 / 4 / 6 years (cumulative/dynamic, IPCW)
  - HR per SD of the risk score, unadjusted and adjusted for age/sex/smoking
    (for heart-only models: does the heart score add anything beyond clinical?)
  - Kaplan-Meier by risk tertile (tie-safe rank -> pd.qcut, same convention as
    stage5 / the mentor's km_analysis.py) + multivariate log-rank trend test,
    with number-at-risk tables. The raw calcium feature gets its own panel
    (no model at all -- the most interpretable version of the R3 finding).
No Brier score / calibration: the enriched cohort (all 1,061 NLST cases, a
fraction of controls) makes absolute risk meaningless; rankings are valid.

Usage:
    env/.venv/bin/python src/r5_evaluate.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

PROJECT = Path(__file__).resolve().parents[1]
PRED_CSV = PROJECT / "data" / "r4_test_predictions.csv"
DATA_CSV = PROJECT / "data" / "radiomics" / "heart_features_v1_r3.csv"
OUT_METRICS = PROJECT / "data" / "r5_test_metrics.csv"
OUT_FIG = PROJECT / "figs" / "r5_evaluation"

TAU = 7.0
AUC_TIMES = [2.0, 4.0, 6.0]
N_BOOT = 1000
SEED = 0
CLIN = ["age", "male", "cigsmok"]
CALCIUM = "card_calc_mass_proxy"
GROUPS = ["Low risk", "Mid risk", "High risk"]

LABELS = {
    "clinical": "Clinical (age, sex, smoking)",
    "clinical_calcium": "Clinical + heart calcium",
    "calcium_raw": "Heart calcium alone (raw feature)",
    "coxen_heart": "Elastic-net Cox: heart only",
    "coxen_all": "Elastic-net Cox: heart + clinical",
    "rsf_all": "Random survival forest: heart + clinical",
    "deepsurv_heart": "DeepSurv: heart only",
    "deepsurv_all": "DeepSurv: heart + clinical",
    "deepsurv_null": "DeepSurv on shuffled labels (null control)",
}
KM_PANELS = ["clinical", "clinical_calcium", "calcium_raw", "coxen_all", "deepsurv_all", "deepsurv_heart"]

# dataviz reference palette: ordinal blue ramp for low -> high risk, light surface
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ab"
TERTILE_COLORS = ["#86b6ef", "#2a78d6", "#104281"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def harrell(t, e, r) -> float:
    return concordance_index(t, -np.asarray(r), e)


def uno(s_train, t, e, r) -> float:
    return concordance_index_ipcw(s_train, Surv.from_arrays(e.astype(bool), t), r, tau=TAU)[0]


def metrics(test: pd.DataFrame, s_train, models: list[str]) -> pd.DataFrame:
    t, e = test["time_yr"].values, test["event"].values
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]

    point, bh, bu = {}, {}, {}
    for m in models:
        r = test[m].values
        point[m] = (harrell(t, e, r), uno(s_train, t, e, r))
        bh[m] = np.array([harrell(t[b], e[b], r[b]) for b in boots])
        bu[m] = np.array([uno(s_train, t[b], e[b], r[b]) for b in boots])

    s_test = Surv.from_arrays(e.astype(bool), t)
    rows = []
    for m in models:
        d_h = bh[m] - bh["clinical"]
        auc, _ = cumulative_dynamic_auc(s_train, s_test, test[m].values, AUC_TIMES)
        z = (test[m] - test[m].mean()) / test[m].std()
        d = test[["time_yr", "event", *CLIN]].assign(score=z)
        hr_u = CoxPHFitter().fit(d[["time_yr", "event", "score"]], "time_yr", "event").summary.loc["score"]
        hr_a = CoxPHFitter().fit(d, "time_yr", "event").summary.loc["score"]
        lo_h, hi_h = np.percentile(bh[m], [2.5, 97.5])
        rows.append({
            "model": m, "label": LABELS[m],
            "harrell_c": point[m][0], "harrell_lo": lo_h, "harrell_hi": hi_h,
            "uno_c": point[m][1], "uno_lo": np.percentile(bu[m], 2.5), "uno_hi": np.percentile(bu[m], 97.5),
            "ci_spans_0.5": bool(lo_h <= 0.5),
            "delta_c_vs_clinical": point[m][0] - point["clinical"][0],
            "delta_lo": np.percentile(d_h, 2.5), "delta_hi": np.percentile(d_h, 97.5),
            "delta_p": min(1.0, 2 * min((d_h <= 0).mean(), (d_h >= 0).mean())) if m != "clinical" else np.nan,
            **{f"auc_{int(tt)}y": a for tt, a in zip(AUC_TIMES, auc)},
            "hr_per_sd": hr_u["exp(coef)"], "hr_p": hr_u["p"],
            "hr_per_sd_adj_clin": hr_a["exp(coef)"], "hr_adj_lo": hr_a["exp(coef) lower 95%"],
            "hr_adj_hi": hr_a["exp(coef) upper 95%"], "hr_adj_p": hr_a["p"],
        })
    return pd.DataFrame(rows)


def km_panel(ax, test: pd.DataFrame, score: str) -> float:
    d = test.assign(group=pd.qcut(test[score].rank(method="first"), 3, labels=GROUPS))
    fitters = []
    for g, col in zip(GROUPS, TERTILE_COLORS):
        sub = d[d.group == g]
        kmf = KaplanMeierFitter().fit(sub.time_yr, sub.event,
                                      label=f"{g} ({int(sub.event.sum())}/{len(sub)} events)")
        kmf.plot_survival_function(ax=ax, color=col, ci_show=False, lw=2)
        fitters.append(kmf)
    p = multivariate_logrank_test(d.time_yr, d.group, d.event).p_value
    style(ax)
    ax.set_ylim(0.80, 1.005)
    ax.set_xlim(0, 8)
    ax.set_xlabel("years since baseline CT", fontsize=8, color=INK2)
    ax.set_ylabel("lung-cancer-free", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"], fontsize=6.5, labels=["Low", "Mid", "High"])
    return p


def fig_km(test: pd.DataFrame, met: pd.DataFrame, n: int, ev: int) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), facecolor=SURFACE)
    rows = []
    for ax, m in zip(axes.ravel(), KM_PANELS):
        p = km_panel(ax, test, m)
        r = met.set_index("model").loc[m]
        caution = "  — CI spans 0.5, interpret with caution" if r["ci_spans_0.5"] else ""
        ax.set_title(f"{LABELS[m]}\nC={r.harrell_c:.3f} [{r.harrell_lo:.3f}, {r.harrell_hi:.3f}], "
                     f"log-rank trend p={p:.2g}{caution}", fontsize=9, color=INK, loc="left")
        rows.append({"model": m, "logrank_trend_p": p})
    fig.suptitle(f"Held-out test set: lung-cancer-free survival by risk tertile "
                 f"(n={n}, {ev} events, ~{ev // 3} per tertile)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Enriched cohort (all NLST lung-cancer cases, subsampled controls): curve heights are "
             "not population risk; the separation between tertiles is what's interpretable.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=3.5)
    fig.savefig(OUT_FIG / "r5_km_tertiles.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_cindex(met: pd.DataFrame, n: int, ev: int) -> None:
    order = ["clinical", "clinical_calcium", "calcium_raw", "coxen_heart", "coxen_all", "rsf_all",
             "deepsurv_heart", "deepsurv_all", "deepsurv_null"]
    m = met.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(m))[::-1]
    col = [GRAY if k == "deepsurv_null" else (INK2 if k == "clinical" else BLUE) for k in m.index]
    for yy, (k, r), c in zip(y, m.iterrows(), col):
        ax.errorbar(r.harrell_c, yy, xerr=[[r.harrell_c - r.harrell_lo], [r.harrell_hi - r.harrell_c]],
                    fmt="o", ms=6, color=c, elinewidth=1.6, capsize=0, zorder=3)
        dc = "" if k == "clinical" else f"   ΔC {r.delta_c_vs_clinical:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
        ax.text(r.harrell_hi + 0.004, yy, f"{r.harrell_c:.3f}{dc}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.axvline(m.loc["clinical", "harrell_c"], color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y, [LABELS[k] for k in m.index], fontsize=8)
    ax.set_xlim(0.44, 0.86)
    ax.set_xlabel("Harrell's C-index on held-out test (95% bootstrap CI);  dashed = clinical model, "
                  "dotted = chance", fontsize=8, color=INK2)
    ax.set_title(f"Discrimination on the test set (n={n}, {ev} events)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "r5_cindex.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_CSV, dtype={"pid": str})
    data = pd.read_csv(DATA_CSV, dtype={"pid": str})
    test = pred.merge(data[["pid", *CLIN, CALCIUM]], on="pid").rename(columns={CALCIUM: "calcium_raw"})
    trva = data[data.split != "test"]
    s_train = Surv.from_arrays(trva.event.astype(bool).values, trva.time_yr.values)
    models = [c for c in LABELS if c in test.columns]
    n, ev = len(test), int(test.event.sum())
    print(f"test n={n}, events={ev}; models: {models}")

    met = metrics(test, s_train, models)
    km = fig_km(test, met, n, ev)
    met = met.merge(km, on="model", how="left")
    met.to_csv(OUT_METRICS, index=False)
    fig_cindex(met, n, ev)

    show = ["model", "harrell_c", "harrell_lo", "harrell_hi", "uno_c", "delta_c_vs_clinical", "delta_lo",
            "delta_hi", "delta_p", "auc_2y", "auc_4y", "auc_6y", "hr_per_sd_adj_clin", "hr_adj_p",
            "logrank_trend_p"]
    with pd.option_context("display.width", 250):
        print(met[show].round(4).to_string(index=False))
    print(f"\nwrote {OUT_METRICS}, figs/r5_evaluation/r5_km_tertiles.png, r5_cindex.png")


if __name__ == "__main__":
    main()
