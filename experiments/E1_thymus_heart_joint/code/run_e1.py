#!/usr/bin/env python3
"""E1 -- thymic health x cardiac calcium -> lung-cancer incidence (protocol.md, H1-H4).

A1 TH replication (full cohort, Cox strata=split) + sensitivities
A2 partial Spearman between TH, calcium, pericardial fat (bootstrap CI)
A3 joint Cox: independence / attenuation / interaction
A4 held-out test: Cox fit on train+val, Harrell C + paired bootstrap dC; 2x2 KM

Usage: env/.venv/bin/python experiments/E1_thymus_heart_joint/code/run_e1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from scipy.stats import spearmanr

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
from aging_data import (AQUA, BLUE, CLIN, GRAY, INK, INK2, ORANGE, SURFACE, fit_cox,  # noqa: E402
                        hr_row, load_frame, paired_bootstrap_c, style)

OUT = EXP / "results"
N_BOOT = 1000
SEED = 0


def a1_replication(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name, cph, var):
        rows.append({"analysis": name, **hr_row(cph, var), "n": cph._n_examples,
                     "events": int(cph.event_observed.sum())})
    add("TH per SD, unadjusted", fit_cox(df, ["th_z"]), "th_z")
    adj = fit_cox(df, CLIN + ["th_z"])
    add("TH per SD, adj age/sex/smoking", adj, "th_z")
    cat = fit_cox(df, CLIN + ["th_avg", "th_high"])
    add("TH average vs low, adj", cat, "th_avg")
    add("TH high vs low, adj", cat, "th_high")
    # sensitivity (i): administrative censoring at 6 yr
    d6 = df.assign(event=((df["event"] == 1) & (df["time_yr"] <= 6)).astype(int),
                   time_yr=df["time_yr"].clip(upper=6))
    add("TH per SD, adj, censored at 6 yr", fit_cox(d6, CLIN + ["th_z"]), "th_z")
    c6 = fit_cox(d6, CLIN + ["th_avg", "th_high"])
    add("TH high vs low, adj, censored at 6 yr", c6, "th_high")
    # sensitivity (ii): strata split x sex x 5-yr age bin (paper's PH handling)
    ds = df.assign(agebin=(df["age"] // 5).astype(int))
    add("TH per SD, strata split x sex x age5", fit_cox(ds, ["cigsmok", "th_z"], strata=["split", "male", "agebin"]), "th_z")
    res = pd.DataFrame(rows)
    ph = proportional_hazard_test(adj, df[CLIN + ["th_z", "time_yr", "event", "split"]], time_transform="rank")
    res.attrs["schoenfeld_p_th"] = float(ph.summary.loc["th_z", "p"])
    return res


def resid(df: pd.DataFrame, col: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df[CLIN].to_numpy(float)])
    y = df[col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def a2_correlations(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [("th_pct", "log_calc", "confirmatory"), ("th_pct", "card_fat_volume_ml", "confirmatory"),
             ("log_calc", "card_fat_volume_ml", "exploratory"), ("th_pct", "card_heart_volume_ml", "exploratory"),
             ("th_pct", "card_fat_mean_hu", "exploratory")]
    R = {c: resid(df, c) for c in {p for a, b, _ in pairs for p in (a, b)}}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(df), len(df)) for _ in range(N_BOOT)]
    rows = []
    for a, b, kind in pairs:
        rho = spearmanr(R[a], R[b])[0]
        boots = [spearmanr(R[a][i], R[b][i])[0] for i in idx]
        rows.append({"x": a, "y": b, "kind": kind, "raw_rho": spearmanr(df[a], df[b])[0],
                     "partial_rho": rho, "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
    return pd.DataFrame(rows)


def a3_joint(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    th1 = fit_cox(df, CLIN + ["th_z"])
    ca1 = fit_cox(df, CLIN + ["calc_z"])
    joint = fit_cox(df, CLIN + ["th_z", "calc_z"])
    inter = fit_cox(df.assign(th_x_calc=df["th_z"] * df["calc_z"]), CLIN + ["th_z", "calc_z", "th_x_calc"])
    ca_cb = fit_cox(df, CLIN + ["th_z", "calc_combat_z"])  # sensitivity: harmonized calcium
    rows = [{"model": "clinical + TH", **hr_row(th1, "th_z")},
            {"model": "clinical + calcium", **hr_row(ca1, "calc_z")},
            {"model": "clinical + TH + calcium", **hr_row(joint, "th_z")},
            {"model": "clinical + TH + calcium", **hr_row(joint, "calc_z")},
            {"model": "joint + TH x calcium", **hr_row(inter, "th_x_calc")},
            {"model": "clinical + TH + calcium(ComBat)", **hr_row(ca_cb, "th_z")},
            {"model": "clinical + TH + calcium(ComBat)", **hr_row(ca_cb, "calc_combat_z")}]
    summ = {"atten_th": 1 - joint.params_["th_z"] / th1.params_["th_z"],
            "atten_calc": 1 - joint.params_["calc_z"] / ca1.params_["calc_z"],
            "interaction_p": float(inter.summary.loc["th_x_calc", "p"]),
            "lr_test_joint_vs_th": float(joint.log_likelihood_ - th1.log_likelihood_),
            "lr_test_joint_vs_calc": float(joint.log_likelihood_ - ca1.log_likelihood_)}
    for name, cph in {"clinical": fit_cox(df, CLIN), "clinical+TH": th1, "clinical+calcium": ca1,
                      "clinical+TH+calcium": joint}.items():
        summ[f"aic_{name}"] = float(cph.AIC_partial_)
    return pd.DataFrame(rows), summ


MODELS = {"clinical": CLIN, "clinical+TH": CLIN + ["th_z"], "clinical+calcium": CLIN + ["calc_z"],
          "clinical+TH+calcium": CLIN + ["th_z", "calc_z"]}


def a4_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tv, te = df[df["split"] != "test"], df[df["split"] == "test"].reset_index(drop=True)
    risks, coefs = {}, []
    for m, cols in MODELS.items():
        cph = fit_cox(tv, cols)
        risks[m] = te[cols].to_numpy(float) @ cph.params_[cols].to_numpy()
        coefs += [{"model": m, **hr_row(cph, c)} for c in cols]
    c_clin = paired_bootstrap_c(te, risks, "clinical", N_BOOT, SEED)
    c_ca = paired_bootstrap_c(te, risks, "clinical+calcium", N_BOOT, SEED)
    c_clin["dc_vs_clinical+calcium"] = c_ca["dc_vs_clinical+calcium"].values
    c_clin["dc2_lo"], c_clin["dc2_hi"], c_clin["dc2_p"] = c_ca["dc_lo"].values, c_ca["dc_hi"].values, c_ca["dc_p"].values
    c_clin["test_n"], c_clin["test_events"] = len(te), int(te["event"].sum())
    pred = te[["pid", "time_yr", "event"]].assign(**{f"risk_{m}": r for m, r in risks.items()})
    return c_clin, pd.DataFrame(coefs), pred


GROUPS = [("neither", GRAY), ("low TH only", BLUE), ("high calcium only", ORANGE), ("low TH + high calcium", AQUA)]


def add_groups(df: pd.DataFrame, q75: float) -> pd.DataFrame:
    low_th = df["th_cat"] == 0
    hi_ca = df["log_calc"] > q75
    g = np.select([low_th & hi_ca, hi_ca, low_th], ["low TH + high calcium", "high calcium only", "low TH only"], "neither")
    return df.assign(group=g)


def place_labels(ax, ends: list[tuple[float, str]]) -> None:
    """Direct end-of-line labels, nudged apart so two-line labels never overlap."""
    ends = sorted(ends)
    lo, hi = ax.get_ylim()
    gap = 0.095 * (hi - lo)
    ys = []
    for y, _ in ends:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    for y, (_, txt) in zip(ys, ends):
        ax.text(8.05, y, txt, color=INK2, fontsize=7, va="center")
    ax.set_ylim(top=max(ys[-1] + gap * 0.6, ax.get_ylim()[1]))


def km_2x2(df: pd.DataFrame, q75: float) -> tuple[pd.DataFrame, plt.Figure]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    rows = []
    for ax, (name, d) in zip(axes, [("Full cohort (strata-free KM)", df), ("Held-out test split", df[df["split"] == "test"])]):
        d = add_groups(d, q75)
        style(ax)
        ends = []
        for g, col in GROUPS:
            s = d[d["group"] == g]
            km = KaplanMeierFitter().fit(s["time_yr"], s["event"], label=g)
            ax.step(km.survival_function_.index, 1 - km.survival_function_[g], where="post", color=col, lw=2)
            ends.append((1 - km.survival_function_[g].iloc[-1], f"{g}\n{int(s['event'].sum())}/{len(s)}"))
            rows.append({"panel": name, "group": g, "n": len(s), "events": int(s["event"].sum()),
                         "cum_inc_6y": float(1 - km.predict(6.0))})
        place_labels(ax, ends)
        lr = multivariate_logrank_test(d["time_yr"], d["group"], d["event"])
        ax.set_title(f"{name}\nlog-rank p = {lr.p_value:.1e}", fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, 10.2)
        ax.set_xlabel("Years since baseline CT", fontsize=8, color=INK2)
        ax.set_ylabel("Cumulative lung-cancer incidence", fontsize=8, color=INK2)
    fig.suptitle("Low thymic health (≤25th pct) × high cardiac calcium (top quartile)\n"
                 "enriched cohort: absolute incidence is not population risk; compare curves only",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    return pd.DataFrame(rows), fig


def group_hrs(df: pd.DataFrame, q75: float) -> pd.DataFrame:
    d = add_groups(df, q75)
    for g, _ in GROUPS[1:]:
        d[g] = (d["group"] == g).astype(int)
    cols = [g for g, _ in GROUPS[1:]]
    cph = fit_cox(d, CLIN + cols)
    return pd.DataFrame([{"group": g, **hr_row(cph, g), "n": int(d[g].sum()),
                          "events": int(d.loc[d[g] == 1, "event"].sum())} for g in cols])


def fig_forest(a1: pd.DataFrame, a3: pd.DataFrame, gh: pd.DataFrame) -> plt.Figure:
    gcol = dict(GROUPS)
    items = [(r["analysis"], r, BLUE) for _, r in a1.iterrows()]
    items += [(f"{r['model']}: {'TH' if r['var'] == 'th_z' else 'calcium'}", r, BLUE if r["var"] == "th_z" else ORANGE)
              for _, r in a3.iterrows() if r["var"] != "th_x_calc"]
    items += [(f"2×2 group vs neither, adj: {r['group']} ({r['events']}/{r['n']})", r, gcol[r["group"]])
              for _, r in gh.iterrows()]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(items) + 1.2), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(items))[::-1]
    for yi, (lab, r, col) in zip(y, items):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r["hr"], yi, "o", color=col, ms=6, mec=SURFACE, mew=1.5)
        ax.text(3.3, yi, f"{r['hr']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]  p={r['p']:.1e}", fontsize=7, va="center", color=INK2)
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in items], fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.2)
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Hazard ratio for lung-cancer incidence (log scale; Cox, strata = split)", fontsize=8, color=INK2)
    ax.set_title("Lung-cancer HRs: thymic health (blue), cardiac calcium (orange)\nfull cohort n=6,041, 1,003 events",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def fig_cindex(c: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(c))[::-1]
    for yi, (_, r) in zip(y, c.iterrows()):
        ax.plot([r["c_lo"], r["c_hi"]], [yi, yi], color=BLUE, lw=2)
        ax.plot(r["c"], yi, "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
        txt = f"C={r['c']:.3f}" + ("" if r["model"] == "clinical" else f"   ΔC={r['dc_vs_clinical']:+.3f} [{r['dc_lo']:+.3f}, {r['dc_hi']:+.3f}]")
        ax.text(0.742, yi, txt, fontsize=7, va="center", color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(c["model"], fontsize=8, color=INK)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0.55, 0.74)
    ax.set_xlabel("Harrell's C on held-out test (95% bootstrap CI)", fontsize=8, color=INK2)
    ax.set_title(f"Held-out test: n={int(c['test_n'].iloc[0])}, {int(c['test_events'].iloc[0])} events; Cox fit on train+val",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, att = load_frame()
    print(att)

    a1 = a1_replication(df)
    a1.to_csv(OUT / "a1_th_replication.csv", index=False)
    print(a1.round(4).to_string(), "\nSchoenfeld p (TH):", a1.attrs["schoenfeld_p_th"])

    a2 = a2_correlations(df)
    a2.to_csv(OUT / "a2_partial_spearman.csv", index=False)
    print(a2.round(4).to_string())

    a3, summ = a3_joint(df)
    a3.to_csv(OUT / "a3_joint_cox.csv", index=False)
    print(a3.round(4).to_string(), "\n", {k: round(v, 4) for k, v in summ.items()})

    c, coefs, pred = a4_test(df)
    c.to_csv(OUT / "a4_test_cindex.csv", index=False)
    coefs.to_csv(OUT / "a4_trainval_coefs.csv", index=False)
    pred.to_csv(OUT / "a4_test_predictions.csv", index=False)
    print(c.round(4).to_string())

    q75 = float(df["log_calc"].quantile(0.75))
    km_rows, fig = km_2x2(df, q75)
    km_rows.to_csv(OUT / "a4_km_2x2.csv", index=False)
    fig.savefig(OUT / "fig_km_2x2.png", dpi=160)
    gh = group_hrs(df, q75)
    gh.to_csv(OUT / "a4_group_hrs.csv", index=False)
    print(km_rows.round(4).to_string(), "\n", gh.round(4).to_string())

    fig_forest(a1, a3, gh).savefig(OUT / "fig_forest.png", dpi=160)
    fig_cindex(c).savefig(OUT / "fig_test_cindex.png", dpi=160)

    json.dump({**att, **summ, "schoenfeld_p_th": a1.attrs["schoenfeld_p_th"], "calc_q75_log": q75},
              open(OUT / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
