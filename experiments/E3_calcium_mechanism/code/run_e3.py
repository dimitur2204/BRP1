#!/usr/bin/env python3
"""E3 -- calcium mechanism + TH scanner robustness (protocol.md, H7-H11).

C1 histology-specific cause-specific Cox (adeno vs squamous+small-cell), delta z-test
C2 risk horizon: <=1 yr truncation, 1-yr landmark, T0-screen vs later cancers
C3 dose-response: zero / non-zero tertiles
C4 robustness: age strata, age spline, smoking strata + interaction, scanner batch
C5 TH (and calcium) scanner leakage / partial R2 / batch-adjusted HR

Usage: env/.venv/bin/python experiments/E3_calcium_mechanism/code/run_e3.py
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
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from aging_data import (BLUE, CLIN, GRAY, INK, INK2, ORANGE, SURFACE, fit_cox, hr_row,  # noqa: E402
                        load_frame, style)

OUT = EXP / "results"
PRSN = PROJECT / "data" / "external" / "nlst_780" / "nlst_780" / "nlst_780_prsn_idc_20210527.csv"
ADENO = {8140, 8250, 8251, 8252, 8253, 8254, 8255, 8260, 8323, 8480, 8481, 8490, 8550}
SQUAM = set(range(8070, 8079)) | {8083, 8084}
SMALL = set(range(8041, 8046))


def histology(code) -> str:
    if pd.isna(code):
        return "none"
    c = int(code)
    return "adeno" if c in ADENO else "squamous" if c in SQUAM else "small_cell" if c in SMALL else "other"


def load() -> pd.DataFrame:
    df, _ = load_frame()
    p = pd.read_csv(PRSN, usecols=["pid", "de_type", "cancyr"])
    df = df.merge(p, on="pid", how="left", validate="1:1")
    df["histo"] = np.where(df["event"] == 1, df["de_type"].map(histology), "none")
    df["batch"] = df["manufacturer"].str.split().str[0] + "_" + df["kernel_group"]
    return df


def cause_specific(df: pd.DataFrame, mask_event: pd.Series) -> pd.DataFrame:
    return df.assign(event=((df["event"] == 1) & mask_event).astype(int))


def delta_test(a: dict, b: dict) -> dict:
    d = b["coef"] - a["coef"]
    se = np.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return {"delta_logHR": d, "hr_ratio": np.exp(d), "ratio_lo": np.exp(d - 1.96 * se),
            "ratio_hi": np.exp(d + 1.96 * se), "p": 2 * norm.sf(abs(d / se))}


def c1_histology(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {"adeno": df["histo"] == "adeno",
              "SQ/SC": df["histo"].isin(["squamous", "small_cell"]),
              "squamous": df["histo"] == "squamous", "small_cell": df["histo"] == "small_cell",
              "other": df["histo"] == "other"}
    specs = {"calcium": (CLIN + ["calc_z"], "calc_z"), "current smoking": (CLIN + ["calc_z"], "cigsmok"),
             "age": (CLIN + ["calc_z"], "age"), "emphysema": (CLIN + ["emph0"], "emph0"),
             "TH": (CLIN + ["th_z"], "th_z")}
    rows, fits = [], {}
    for g, m in groups.items():
        d = cause_specific(df, m)
        for name, (cols, var) in specs.items():
            key = (g, tuple(cols))
            if key not in fits:
                fits[key] = fit_cox(d, cols)
            rows.append({"histology": g, "marker": name, "events": int(d["event"].sum()), **hr_row(fits[key], var)})
    res = pd.DataFrame(rows)
    deltas = []
    for name in specs:
        a = res[(res["histology"] == "adeno") & (res["marker"] == name)].iloc[0]
        b = res[(res["histology"] == "SQ/SC") & (res["marker"] == name)].iloc[0]
        deltas.append({"marker": name, "hr_adeno": a["hr"], "hr_sqsc": b["hr"], **delta_test(a, b)})
    return res, pd.DataFrame(deltas)


def c2_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d1 = df.assign(event=((df["event"] == 1) & (df["time_yr"] <= 1)).astype(int), time_yr=df["time_yr"].clip(upper=1))
    rows.append({"window": "0-1 yr (truncated)", "n": len(d1), "events": int(d1["event"].sum()),
                 **hr_row(fit_cox(d1, CLIN + ["calc_z"]), "calc_z")})
    lm = df[df["time_yr"] > 1].assign(time_yr=lambda x: x["time_yr"] - 1)
    rows.append({"window": ">1 yr (landmark at 1 yr)", "n": len(lm), "events": int(lm["event"].sum()),
                 **hr_row(fit_cox(lm, CLIN + ["calc_z"]), "calc_z")})
    lm3 = df[df["time_yr"] > 3].assign(time_yr=lambda x: x["time_yr"] - 3)
    rows.append({"window": ">3 yr (landmark at 3 yr, exploratory)", "n": len(lm3), "events": int(lm3["event"].sum()),
                 **hr_row(fit_cox(lm3, CLIN + ["calc_z"]), "calc_z")})
    t0 = cause_specific(df, df["cancyr"] == 0)
    later = cause_specific(df, df["cancyr"] >= 1)
    a, b = hr_row(fit_cox(t0, CLIN + ["calc_z"]), "calc_z"), hr_row(fit_cox(later, CLIN + ["calc_z"]), "calc_z")
    rows.append({"window": "T0-screen cancers (cause-specific)", "n": len(df), "events": int(t0["event"].sum()), **a})
    rows.append({"window": "cancyr>=1 cancers (cause-specific)", "n": len(df), "events": int(later["event"].sum()), **b})
    res = pd.DataFrame(rows)
    res.attrs["t0_vs_later"] = delta_test(a, b)
    return res


def c3_dose(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    nz = df["card_calc_mass_proxy"] > 0
    cat = pd.Series(0, index=df.index)
    cat[nz] = pd.qcut(df.loc[nz, "card_calc_mass_proxy"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    d = df.assign(ca_cat=cat)
    for k in (1, 2, 3):
        d[f"ca_T{k}"] = (d["ca_cat"] == k).astype(int)
    cph = fit_cox(d, CLIN + ["ca_T1", "ca_T2", "ca_T3"])
    rows = [{"category": "zero calcium (ref)", "n": int((cat == 0).sum()), "events": int(d.loc[cat == 0, "event"].sum()),
             "range_mass_proxy": "0", "hr": 1.0, "lo": 1.0, "hi": 1.0, "p": np.nan}]
    for k in (1, 2, 3):
        s = d.loc[cat == k, "card_calc_mass_proxy"]
        r = hr_row(cph, f"ca_T{k}")
        rows.append({"category": f"non-zero tertile T{k}", "n": int((cat == k).sum()),
                     "events": int(d.loc[cat == k, "event"].sum()),
                     "range_mass_proxy": f"{s.min():.1f}-{s.max():.1f}", **{x: r[x] for x in ("hr", "lo", "hi", "p")}})
    trend = hr_row(fit_cox(d, CLIN + ["ca_cat"]), "ca_cat")
    return pd.DataFrame(rows), {"trend_hr_per_category": trend["hr"], "trend_p": trend["p"]}


def rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell), K knots -> K-2 non-linear columns."""
    k = knots
    K = len(k)
    pos = lambda u: np.clip(u, 0, None) ** 3  # noqa: E731
    cols = []
    for j in range(K - 2):
        cols.append((pos(x - k[j]) - pos(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                     + pos(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / (k[K - 1] - k[0]) ** 2)
    return np.column_stack(cols)


def c4_robust(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    add = lambda name, cph, d: rows.append({"spec": name, "n": len(d), "events": int(d["event"].sum()),  # noqa: E731
                                            **hr_row(cph, "calc_z")})
    add("reference: strata split, adj age/sex/smoking", fit_cox(df, CLIN + ["calc_z"]), df)
    ds = df.assign(agebin=(df["age"] // 5).astype(int))
    add("(i) strata split x sex x age5 (+age, smoking)",
        fit_cox(ds, ["age", "cigsmok", "calc_z"], strata=["split", "male", "agebin"]), ds)
    knots = np.percentile(df["age"], [5, 35, 65, 95])
    S = rcs(df["age"].to_numpy(float), knots)
    dsp = df.assign(**{f"age_s{j}": S[:, j] for j in range(S.shape[1])})
    add("(ii) age restricted cubic spline (4 knots)",
        fit_cox(dsp, ["age"] + [f"age_s{j}" for j in range(S.shape[1])] + ["male", "cigsmok", "calc_z"]), dsp)
    for s, lab in ((0, "former"), (1, "current")):
        d = df[df["cigsmok"] == s]
        add(f"(iii) {lab} smokers only (+age, sex)", fit_cox(d, ["age", "male", "calc_z"]), d)
    inter = fit_cox(df.assign(ca_x_smk=df["calc_z"] * df["cigsmok"]), CLIN + ["calc_z", "ca_x_smk"])
    rows.append({"spec": "(iii) calcium x current-smoking interaction", "n": len(df), "events": int(df["event"].sum()),
                 **hr_row(inter, "ca_x_smk")})
    B = pd.get_dummies(df["batch"], prefix="b", drop_first=True, dtype=int)
    B = B.loc[:, B.sum() >= 20]
    db = pd.concat([df, B], axis=1).assign(noise_z=lambda x: (x["card_core_noise_sd_hu"] - x["card_core_noise_sd_hu"].mean())
                                           / x["card_core_noise_sd_hu"].std())
    add("(iv) + scanner batch dummies + core noise SD", fit_cox(db, CLIN + list(B.columns) + ["noise_z", "calc_z"]), db)
    return pd.DataFrame(rows)


def c5_scanner(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    sharp = (df["kernel_group"] == "sharp").astype(int)
    man = df["manufacturer"].str.split().str[0]
    for var, lab in (("th_pct", "TH"), ("log_calc", "calcium (raw log)"), ("fat_z", "pericardial fat")):
        auc = roc_auc_score(sharp, df[var])
        row = {"marker": lab, "auc_sharp_vs_soft": auc, "leak_auc_sym": max(auc, 1 - auc)}
        for m in sorted(man.unique()):
            a = roc_auc_score((man == m).astype(int), df[var])
            row[f"leak_auc_{m}"] = max(a, 1 - a)
        # partial R2 of scanner terms beyond clinical
        Xc = np.column_stack([np.ones(len(df)), df[CLIN].to_numpy(float)])
        Xs = np.column_stack([Xc, pd.get_dummies(df["batch"], drop_first=True, dtype=float).to_numpy()])
        y = df[var].to_numpy(float)
        sse = lambda X: ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()  # noqa: E731
        row["scanner_partial_r2"] = (sse(Xc) - sse(Xs)) / sse(Xc)
        rows.append(row)
    means = df.groupby("batch").agg(n=("pid", "size"), th_mean=("th_pct", "mean"), th_sd=("th_pct", "std"),
                                    calc_mean=("log_calc", "mean")).reset_index()
    B = pd.get_dummies(df["batch"], prefix="b", drop_first=True, dtype=int)
    B = B.loc[:, B.sum() >= 20]
    db = pd.concat([df, B], axis=1)
    base = hr_row(fit_cox(df, CLIN + ["th_z"]), "th_z")
    adj = hr_row(fit_cox(db, CLIN + list(B.columns) + ["th_z"]), "th_z")
    summ = {"th_hr_base": base["hr"], "th_hr_batch_adj": adj["hr"], "th_hr_batch_adj_lo": adj["lo"],
            "th_hr_batch_adj_hi": adj["hi"], "th_hr_change": adj["hr"] - base["hr"]}
    res = pd.DataFrame(rows)
    res.attrs["means"] = means
    return res, summ


def fig_histology(hist: pd.DataFrame, deltas: pd.DataFrame) -> plt.Figure:
    markers = ["current smoking", "emphysema", "calcium", "TH"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax)
    y0 = np.arange(len(markers))[::-1] * 1.0
    for yi, mk in zip(y0, markers):
        for off, g, col in ((0.17, "adeno", BLUE), (-0.17, "SQ/SC", ORANGE)):
            r = hist[(hist["histology"] == g) & (hist["marker"] == mk)].iloc[0]
            ax.plot([r["lo"], r["hi"]], [yi + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(r["hr"], yi + off, "o", color=col, ms=6, mec=SURFACE, mew=1.5)
        dl = deltas[deltas["marker"] == mk].iloc[0]
        ax.text(3.05, yi, f"ratio SQ/SC ÷ adeno {dl['hr_ratio']:.2f} [{dl['ratio_lo']:.2f}, {dl['ratio_hi']:.2f}], "
                          f"p={dl['p']:.2g}", fontsize=7, va="center", color=INK2)
    ev_a = int(hist[(hist["histology"] == "adeno")]["events"].iloc[0])
    ev_b = int(hist[(hist["histology"] == "SQ/SC")]["events"].iloc[0])
    ax.plot([], [], "o-", color=BLUE, label=f"adenocarcinoma ({ev_a} events)")
    ax.plot([], [], "o-", color=ORANGE, label=f"squamous + small cell ({ev_b} events)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y0)
    ax.set_yticklabels(["current smoking\n(method control)", "emphysema (read)", "cardiac calcium /SD", "thymic health /SD"],
                       fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 3.0)
    ax.set_xticks([0.75, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0.75", "1", "1.5", "2", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Cause-specific HR (adj age/sex/smoking; strata split)", fontsize=8, color=INK2)
    ax.set_title("Does calcium behave like a smoking-dose marker? Histology-specific hazards", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def fig_dose(dose: pd.DataFrame, trend: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(dose))
    cols = [GRAY] + [ORANGE] * 3
    for xi, (_, r), c in zip(x, dose.iterrows(), cols):
        ax.plot([xi, xi], [r["lo"], r["hi"]], color=c, lw=2)
        ax.plot(xi, r["hr"], "o", color=c, ms=7, mec=SURFACE, mew=1.5)
        ax.text(xi, r["hi"] * 1.03, f"{r['hr']:.2f}\n{r['events']}/{r['n']}", ha="center", fontsize=7, color=INK2)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "low", "mid", "high"], fontsize=8, color=INK)
    ax.set_xlabel("Cardiac calcium (zero, then tertiles of non-zero)", fontsize=8, color=INK2)
    ax.set_ylabel("HR vs no calcium (adj)", fontsize=8, color=INK2)
    ax.set_ylim(0.6, max(dose["hi"]) * 1.25)
    ax.set_title(f"Dose-response: trend HR/category {trend['trend_hr_per_category']:.2f}, p={trend['trend_p']:.1e}",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print("histology counts (events):", df.loc[df["event"] == 1, "histo"].value_counts().to_dict())

    hist, deltas = c1_histology(df)
    hist.to_csv(OUT / "c1_histology_hrs.csv", index=False)
    deltas.to_csv(OUT / "c1_histology_deltas.csv", index=False)
    print(hist.round(4).to_string(), "\n", deltas.round(4).to_string())
    fig_histology(hist, deltas).savefig(OUT / "fig_histology.png", dpi=160)

    hz = c2_horizon(df)
    hz.to_csv(OUT / "c2_horizon.csv", index=False)
    print(hz.round(4).to_string(), "\n", {k: round(v, 4) for k, v in hz.attrs["t0_vs_later"].items()})

    dose, trend = c3_dose(df)
    dose.to_csv(OUT / "c3_dose_response.csv", index=False)
    print(dose.round(4).to_string(), "\n", trend)
    fig_dose(dose, trend).savefig(OUT / "fig_dose_response.png", dpi=160)

    rob = c4_robust(df)
    rob.to_csv(OUT / "c4_robustness.csv", index=False)
    print(rob.round(4).to_string())

    sc, summ = c5_scanner(df)
    sc.to_csv(OUT / "c5_scanner_leakage.csv", index=False)
    sc.attrs["means"].to_csv(OUT / "c5_batch_means.csv", index=False)
    print(sc.round(4).to_string(), "\n", sc.attrs["means"].round(2).to_string(), "\n", summ)

    json.dump({"t0_vs_later": hz.attrs["t0_vs_later"], **trend, **summ}, open(OUT / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
