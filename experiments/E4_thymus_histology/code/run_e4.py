#!/usr/bin/env python3
"""E4 -- stress test of the exploratory thymic health x histology finding (protocol.md, H12).

D1 Lunn-McNeil stacked competing-risks interaction test (robust SE)
D2 dose-marker adjustment (emphysema + calcium + pericardial fat)
D3 consistency across splits / sex / smoking status
D4 BAC/lepidic vs other adeno;  D5 early vs advanced stage
D6 comparator markers (fat, heart volume, calcium)
D7 held-out test: SQ/SC- and adeno-specific C-index gain from TH

Usage: env/.venv/bin/python experiments/E4_thymus_histology/code/run_e4.py
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
from lifelines import CoxPHFitter

EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments" / "E3_calcium_mechanism" / "code"))
from aging_data import (BLUE, CLIN, INK, INK2, ORANGE, SURFACE, fit_cox, hr_row,  # noqa: E402
                        paired_bootstrap_c, style, zscore)
from run_e3 import cause_specific, delta_test, load  # noqa: E402

OUT = EXP / "results"
PRSN = PROJECT / "data" / "external" / "nlst_780" / "nlst_780" / "nlst_780_prsn_idc_20210527.csv"
BAC = set(range(8250, 8256))
EARLY, ADVANCED = {110, 120, 210, 220}, {310, 320, 400}


def frame() -> pd.DataFrame:
    df = load()
    st = pd.read_csv(PRSN, usecols=["pid", "de_stag"])
    df = df.merge(st, on="pid", how="left", validate="1:1")
    df["is_adeno"] = df["histo"] == "adeno"
    df["is_sqsc"] = df["histo"].isin(["squamous", "small_cell"])
    df["is_bac"] = df["is_adeno"] & df["de_type"].isin(BAC)
    df["is_adeno_other"] = df["is_adeno"] & ~df["de_type"].isin(BAC)
    df["is_early"] = (df["event"] == 1) & df["de_stag"].isin(EARLY)
    df["is_adv"] = (df["event"] == 1) & df["de_stag"].isin(ADVANCED)
    df["hv_z"] = zscore(df["card_heart_volume_ml"])
    return df


def csh(df: pd.DataFrame, mask: pd.Series, cols: list[str], var: str = "th_z") -> dict:
    d = cause_specific(df, mask)
    return {**hr_row(fit_cox(d, cols), var), "events": int(d["event"].sum()), "n": len(d)}


def d1_lunn_mcneil(df: pd.DataFrame) -> dict:
    parts = []
    for cause, mask in (("adeno", df["is_adeno"]), ("sqsc", df["is_sqsc"])):
        d = cause_specific(df, mask).assign(sqsc=int(cause == "sqsc"))
        parts.append(d)
    st = pd.concat(parts, ignore_index=True)
    st["stratum"] = st["sqsc"].astype(str) + "_" + st["split"]
    for c in CLIN:
        st[f"{c}_x_sqsc"] = st[c] * st["sqsc"]
    st["th_x_sqsc"] = st["th_z"] * st["sqsc"]
    cols = CLIN + [f"{c}_x_sqsc" for c in CLIN] + ["th_z", "th_x_sqsc"]
    cph = CoxPHFitter().fit(st[cols + ["time_yr", "event", "stratum", "pid"]], "time_yr", "event",
                            strata=["stratum"], cluster_col="pid", robust=True)
    r = hr_row(cph, "th_x_sqsc")
    base = hr_row(cph, "th_z")
    return {"interaction_hr": r["hr"], "interaction_lo": r["lo"], "interaction_hi": r["hi"], "interaction_p": r["p"],
            "th_hr_adeno": base["hr"], "th_hr_sqsc": float(np.exp(base["coef"] + r["coef"]))}


def d2_dose_adjust(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, extra in (("clinical", []), ("+emphysema", ["emph0"]), ("+emphysema+calcium", ["emph0", "calc_z"]),
                        ("+emphysema+calcium+fat", ["emph0", "calc_z", "fat_z"])):
        a = csh(df, df["is_adeno"], CLIN + extra + ["th_z"])
        b = csh(df, df["is_sqsc"], CLIN + extra + ["th_z"])
        rows.append({"adjustment": name, "hr_adeno": a["hr"], "hr_adeno_lo": a["lo"], "hr_adeno_hi": a["hi"],
                     "hr_sqsc": b["hr"], "hr_sqsc_lo": b["lo"], "hr_sqsc_hi": b["hi"], "p_sqsc": b["p"],
                     "coef_sqsc": b["coef"], **{f"ratio_{k}": v for k, v in delta_test(a, b).items()}})
    res = pd.DataFrame(rows)
    res["sqsc_logHR_attenuation"] = 1 - res["coef_sqsc"] / res["coef_sqsc"].iloc[0]
    return res


def d3_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {f"split={s}": df["split"] == s for s in ("train", "val", "test")}
    subsets.update({"women": df["male"] == 0, "men": df["male"] == 1,
                    "former smokers": df["cigsmok"] == 0, "current smokers": df["cigsmok"] == 1})
    for name, m in subsets.items():
        d = df[m]
        cols = [c for c in CLIN if d[c].nunique() > 1] + ["th_z"]
        strata = ["split"] if d["split"].nunique() > 1 else None
        fits = {}
        for g in ("is_adeno", "is_sqsc"):
            dd = cause_specific(d, d[g])
            fits[g] = {**hr_row(fit_cox(dd, cols, strata=strata), "th_z"), "events": int(dd["event"].sum())}
        a, b = fits["is_adeno"], fits["is_sqsc"]
        rows.append({"subset": name, "n": len(d), "ev_adeno": a["events"], "ev_sqsc": b["events"],
                     "hr_adeno": a["hr"], "hr_sqsc": b["hr"], **delta_test(a, b)})
    return pd.DataFrame(rows)


def d4_d5_d6(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in (("adeno: BAC/lepidic 8250-8255", df["is_bac"]), ("adeno: other", df["is_adeno_other"]),
                       ("stage I-II (all histologies)", df["is_early"]), ("stage III-IV (all histologies)", df["is_adv"])):
        rows.append({"analysis": name, "marker": "TH", **csh(df, mask, CLIN + ["th_z"])})
    for mk, var in (("pericardial fat", "fat_z"), ("heart volume", "hv_z"), ("calcium", "calc_z"), ("TH", "th_z")):
        a = csh(df, df["is_adeno"], CLIN + [var], var)
        b = csh(df, df["is_sqsc"], CLIN + [var], var)
        rows.append({"analysis": "comparator: adeno", "marker": mk, **a})
        rows.append({"analysis": "comparator: SQ/SC", "marker": mk, **b, **{f"ratio_{k}": v for k, v in delta_test(a, b).items()}})
    return pd.DataFrame(rows)


def d7_test(df: pd.DataFrame) -> pd.DataFrame:
    tv, te = df[df["split"] != "test"], df[df["split"] == "test"].reset_index(drop=True)
    rows = []
    for g, lab in (("is_sqsc", "SQ/SC"), ("is_adeno", "adeno")):
        tvc, tec = cause_specific(tv, tv[g]), cause_specific(te, te[g])
        risks = {}
        for m, cols in (("clinical", CLIN), ("clinical+TH", CLIN + ["th_z"])):
            cph = fit_cox(tvc, cols)
            risks[m] = tec[cols].to_numpy(float) @ cph.params_[cols].to_numpy()
        c = paired_bootstrap_c(tec, risks, "clinical", 1000, 0)
        c["cause"], c["test_events"] = lab, int(tec["event"].sum())
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def fig_stress(d2: pd.DataFrame, d3: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    style(ax)
    y = np.arange(len(d2))[::-1]
    for yi, (_, r) in zip(y, d2.iterrows()):
        for off, pre, col in ((0.15, "hr_adeno", BLUE), (-0.15, "hr_sqsc", ORANGE)):
            ax.plot([r[f"{pre}_lo"], r[f"{pre}_hi"]], [yi + off] * 2, color=col, lw=2)
            ax.plot(r[pre], yi + off, "o", color=col, ms=6, mec=SURFACE, mew=1.5)
    ax.plot([], [], "o-", color=BLUE, label="adenocarcinoma")
    ax.plot([], [], "o-", color=ORANGE, label="squamous + small cell")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d2["adjustment"], fontsize=8, color=INK)
    ax.set_xlabel("TH HR/SD (cause-specific)", fontsize=8, color=INK2)
    ax.set_title("D2: adding smoking-dose markers", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    style(ax)
    y = np.arange(len(d3))[::-1]
    for yi, (_, r) in zip(y, d3.iterrows()):
        ax.plot([r["ratio_lo"], r["ratio_hi"]], [yi, yi], color=INK2, lw=2)
        ax.plot(r["hr_ratio"], yi, "o", color=INK, ms=6, mec=SURFACE, mew=1.5)
        ax.text(2.05, yi, f"{r['hr_ratio']:.2f}  ({r['ev_adeno']}/{r['ev_sqsc']} ev)", fontsize=7, va="center", color=INK2)
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d3["subset"], fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 2.0)
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2])
    ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("TH HR ratio, SQ/SC ÷ adeno (<1 = TH protects more against SQ/SC)", fontsize=8, color=INK2)
    ax.set_title("D3: consistency across disjoint subsets", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = frame()
    d1 = d1_lunn_mcneil(df)
    print("D1", {k: round(v, 4) for k, v in d1.items()})
    d2 = d2_dose_adjust(df)
    d2.to_csv(OUT / "d2_dose_adjustment.csv", index=False)
    print(d2.round(4).to_string())
    d3 = d3_consistency(df)
    d3.to_csv(OUT / "d3_consistency.csv", index=False)
    print(d3.round(4).to_string())
    d456 = d4_d5_d6(df)
    d456.to_csv(OUT / "d4_d6_subtypes_stage_comparators.csv", index=False)
    print(d456.round(4).to_string())
    d7 = d7_test(df)
    d7.to_csv(OUT / "d7_test_cause_specific_c.csv", index=False)
    print(d7.round(4).to_string())
    fig_stress(d2, d3).savefig(OUT / "fig_stress_test.png", dpi=160)
    json.dump(d1, open(OUT / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
