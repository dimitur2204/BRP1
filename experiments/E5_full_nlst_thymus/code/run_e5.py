#!/usr/bin/env python3
"""E5 -- thymic health on the full TH-scored NLST CT arm (protocol.md, H13-H14).

F1 paper replication (unadjusted 6-yr category HRs) + adjusted variants
F2 TH x histology (cause-specific + Lunn-McNeil) without case-cohort sampling
F3 descriptives / overlap with the E1 cohort

Usage: env/.venv/bin/python experiments/E5_full_nlst_thymus/code/run_e5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments" / "E3_calcium_mechanism" / "code"))
from aging_data import CTAB_CSV, TH_CSV, fit_cox, hr_row, load_frame, zscore  # noqa: E402
from run_e3 import PRSN, delta_test, histology  # noqa: E402

OUT = EXP / "results"
MANIFEST = Path("/faststorage/project/aura_thymus/experiments/lcrisk_discovery/data/manifest.csv")
CLIN = ["age", "male", "cigsmok"]


def load_full() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST, usecols=["pid", "time", "event"]).drop_duplicates("pid")
    th = pd.read_csv(TH_CSV).rename(columns={"ID": "pid", "Thymic_health_continuous": "th_pct",
                                             "Thymic_health_categories": "th_cat"})
    p = pd.read_csv(PRSN, usecols=["pid", "age", "gender", "cigsmok", "de_type"])
    df = m.merge(th, on="pid", how="inner").merge(p, on="pid", how="left")
    df = df.dropna(subset=["age", "gender", "cigsmok", "th_pct", "time"]).reset_index(drop=True)
    df["male"] = (df["gender"] == 1).astype(int)
    df["time_yr"] = df["time"] / 365.25
    df["th_z"] = zscore(df["th_pct"])
    df["th_cat"] = df["th_cat"].astype(int)
    df["th_avg"] = (df["th_cat"] == 1).astype(int)
    df["th_high"] = (df["th_cat"] == 2).astype(int)
    df["histo"] = np.where(df["event"] == 1, df["de_type"].map(histology), "none")
    c = pd.read_csv(CTAB_CSV, usecols=["pid", "study_yr", "sct_ab_desc"])
    df["emph0"] = df["pid"].isin(c.loc[(c["study_yr"] == 0) & (c["sct_ab_desc"] == 59), "pid"]).astype(int)
    df["agebin"] = (df["age"] // 5).astype(int)
    return df


def censor6(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(event=((df["event"] == 1) & (df["time_yr"] <= 6)).astype(int), time_yr=df["time_yr"].clip(upper=6))


def cox(d, cols, strata=None):
    return fit_cox(d, cols, strata=strata)


def f1_replication(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, d in (("6 yr", censor6(df)), ("full", df)):
        ev = int(d["event"].sum())
        un = cox(d, ["th_avg", "th_high"])
        rows += [{"horizon": horizon, "model": "unadjusted (paper: high 0.64 [0.53,0.76], avg 0.78)", **hr_row(un, v), "events": ev}
                 for v in ("th_avg", "th_high")]
        for name, cols, strata in (("adj age/sex/smoking", CLIN, None), ("strata sex x age5 + smoking", ["cigsmok"], ["male", "agebin"]),
                                   ("adj age/sex/smoking + emphysema", CLIN + ["emph0"], None)):
            cat = cox(d, cols + ["th_avg", "th_high"], strata)
            cont = cox(d, cols + ["th_z"], strata)
            rows += [{"horizon": horizon, "model": name, **hr_row(cat, "th_high"), "events": ev},
                     {"horizon": horizon, "model": name, **hr_row(cont, "th_z"), "events": ev}]
        rows.append({"horizon": horizon, "model": "unadjusted", **hr_row(cox(d, ["th_z"]), "th_z"), "events": ev})
    return pd.DataFrame(rows)


def cs(df, mask):
    return df.assign(event=((df["event"] == 1) & mask).astype(int))


def lunn_mcneil(df: pd.DataFrame, extra: list[str]) -> dict:
    parts = [cs(df, df["histo"] == "adeno").assign(sqsc=0),
             cs(df, df["histo"].isin(["squamous", "small_cell"])).assign(sqsc=1)]
    st = pd.concat(parts, ignore_index=True)
    cols = CLIN + extra
    for c in cols:
        st[f"{c}_x"] = st[c] * st["sqsc"]
    st["th_x_sqsc"] = st["th_z"] * st["sqsc"]
    use = cols + [f"{c}_x" for c in cols] + ["th_z", "th_x_sqsc"]
    cph = CoxPHFitter().fit(st[use + ["time_yr", "event", "sqsc", "pid"]], "time_yr", "event",
                            strata=["sqsc"], cluster_col="pid", robust=True)
    r = hr_row(cph, "th_x_sqsc")
    return {"interaction_hr": r["hr"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]}


def f2_histology(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, lm = [], {}
    groups = {"adeno": df["histo"] == "adeno", "SQ/SC": df["histo"].isin(["squamous", "small_cell"]),
              "squamous": df["histo"] == "squamous", "small_cell": df["histo"] == "small_cell",
              "other": df["histo"] == "other"}
    for horizon, d0 in (("full", df), ("6 yr", censor6(df))):
        for adj, extra in (("clinical", []), ("clinical + emphysema", ["emph0"])):
            fits = {}
            for g in groups:
                gm = groups[g] & (d0["event"] == 1)
                d = cs(d0, gm)
                fits[g] = {**hr_row(cox(d, CLIN + extra + ["th_z"]), "th_z"), "events": int(d["event"].sum())}
                rows.append({"horizon": horizon, "adjustment": adj, "histology": g, **fits[g]})
            rows.append({"horizon": horizon, "adjustment": adj, "histology": "ratio SQ/SC ÷ adeno",
                         **{k: v for k, v in delta_test(fits["adeno"], fits["SQ/SC"]).items()}})
            lm[f"{horizon}|{adj}"] = lunn_mcneil(d0, extra)
    return pd.DataFrame(rows), lm


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_full()
    e1, _ = load_frame()
    in_e1 = df["pid"].isin(e1["pid"])
    desc = {"n": len(df), "events": int(df["event"].sum()), "events_6yr": int(censor6(df)["event"].sum()),
            "n_not_in_E1": int((~in_e1).sum()), "events_not_in_E1": int(df.loc[~in_e1, "event"].sum()),
            "histology_events": df.loc[df["event"] == 1, "histo"].value_counts().to_dict(),
            "th_mean_by_sex": df.groupby("male")["th_pct"].mean().round(2).to_dict(),
            "th_mean_by_smoking": df.groupby("cigsmok")["th_pct"].mean().round(2).to_dict(),
            "emph0_frac": float(df["emph0"].mean())}
    print(desc)
    f1 = f1_replication(df)
    f1.to_csv(OUT / "f1_paper_replication.csv", index=False)
    print(f1[["horizon", "model", "var", "hr", "lo", "hi", "p", "events"]].round(4).to_string())
    f2, lm = f2_histology(df)
    f2.to_csv(OUT / "f2_histology.csv", index=False)
    print(f2.round(4).to_string())
    print({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in lm.items()})
    json.dump({**desc, "lunn_mcneil": lm}, open(OUT / "summary.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
