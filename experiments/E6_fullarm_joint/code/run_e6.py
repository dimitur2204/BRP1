#!/usr/bin/env python3
"""E6 -- TH + cardiac calcium on the TotalSegmentator-covered CT arm (~16k), no case-cohort
sampling (protocol.md, H15-H17). Needs data/radiomics/heart_features_fullarm.csv from
`sbatch src/submit_r2_fullarm.sh` + `r2_extract_radiomics.py --merge --tag fullarm ...`.

Usage:
    env/.venv/bin/python experiments/E6_fullarm_joint/code/run_e6.py
    env/.venv/bin/python experiments/E6_fullarm_joint/code/run_e6.py --dry-run   # enriched cohort only (code check)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments" / "E3_calcium_mechanism" / "code"))
from aging_data import CTAB_CSV, TH_CSV, fit_cox, hr_row, paired_bootstrap_c, zscore  # noqa: E402
from run_e3 import PRSN, delta_test, histology  # noqa: E402

OUT = EXP / "results"
PIDS = PROJECT / "data" / "pid_lists"
RAD = PROJECT / "data" / "radiomics"
CLIN = ["age", "male", "cigsmok"]
FEATS = ["pid", "status", "card_calc_mass_proxy", "card_fat_volume_ml", "card_heart_volume_ml",
         "qc_metal_voxels", "qc_heart_touches_z_edge"]


def load(dry_run: bool) -> tuple[pd.DataFrame, dict]:
    parts = [("v1", PIDS / "heart_cohort_v1.csv", RAD / "heart_features_v1.csv")]
    if not dry_run:
        parts.append(("fullarm", PIDS / "heart_cohort_fullarm.csv", RAD / "heart_features_fullarm.csv"))
    frames = []
    for tag, coh, feat in parts:
        c = pd.read_csv(coh)
        f = pd.read_csv(feat, usecols=FEATS)
        frames.append(c.merge(f, on="pid", how="left", validate="1:1").assign(source=tag))
    df = pd.concat(frames, ignore_index=True)
    att = {"built": len(df), "built_events": int(df["event"].sum())}
    keep = ((df["status"] == "ok") & (df["card_heart_volume_ml"] >= 250) & (df["qc_metal_voxels"] < 5)
            & (df["qc_heart_touches_z_edge"] == 0) & (df["kernel_group"] != "other"))
    df = df[keep].copy()
    th = pd.read_csv(TH_CSV).rename(columns={"ID": "pid", "Thymic_health_continuous": "th_pct"})
    df = df.merge(th[["pid", "th_pct"]], on="pid", how="inner")
    p = pd.read_csv(PRSN, usecols=["pid", "de_type"])
    df = df.merge(p, on="pid", how="left")
    ct = pd.read_csv(CTAB_CSV, usecols=["pid", "study_yr", "sct_ab_desc"])
    df["emph0"] = df["pid"].isin(ct.loc[(ct["study_yr"] == 0) & (ct["sct_ab_desc"] == 59), "pid"]).astype(int)
    df["male"] = (df["sex"] == "M").astype(int)
    df["time_yr"] = df["time"] / 365.25
    df["histo"] = np.where(df["event"] == 1, df["de_type"].map(histology), "none")
    df["log_calc"] = np.log1p(df["card_calc_mass_proxy"])
    df["calc_z"], df["th_z"], df["fat_z"] = zscore(df["log_calc"]), zscore(df["th_pct"]), zscore(df["card_fat_volume_ml"])
    att.update({"analysis_n": len(df), "analysis_events": int(df["event"].sum()),
                "by_source": df.groupby("source")["event"].agg(["size", "sum"]).to_dict()})
    return df.reset_index(drop=True), att


def cox(d, cols):
    return fit_cox(d, cols, strata=None)


def cs(df, mask):
    return df.assign(event=((df["event"] == 1) & mask).astype(int))


def resid(df, col):
    X = np.column_stack([np.ones(len(df)), df[CLIN].to_numpy(float)])
    y = df[col].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = OUT / ("dry_run" if args.dry_run else "")
    out.mkdir(parents=True, exist_ok=True)
    df, att = load(args.dry_run)
    print(att)
    rows = []
    # G1
    rows.append({"analysis": "G1 calcium adj", **hr_row(cox(df, CLIN + ["calc_z"]), "calc_z")})
    nz = df["card_calc_mass_proxy"] > 0
    cat = pd.Series(0, index=df.index)
    cat[nz] = pd.qcut(df.loc[nz, "card_calc_mass_proxy"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    d = df.assign(**{f"T{k}": (cat == k).astype(int) for k in (1, 2, 3)})
    rows += [{"analysis": f"G1 calcium T{k} vs none", **hr_row(cox(d, CLIN + ["T1", "T2", "T3"]), f"T{k}")} for k in (1, 2, 3)]
    # G2
    single = {m: cox(df, CLIN + [m]) for m in ("th_z", "calc_z")}
    joint = cox(df, CLIN + ["emph0", "th_z", "calc_z"])
    for m in ("th_z", "calc_z"):
        rows.append({"analysis": f"G2 single {m}", **hr_row(single[m], m)})
        rows.append({"analysis": f"G2 joint (+emph+other) {m}", **hr_row(joint, m),
                     "attenuation": 1 - joint.params_[m] / single[m].params_[m]})
    rho = {f"{a}~{b}": spearmanr(resid(df, a), resid(df, b))[0]
           for a, b in (("th_pct", "log_calc"), ("th_pct", "card_fat_volume_ml"), ("log_calc", "card_fat_volume_ml"))}
    # G3
    adj = CLIN + ["emph0", "calc_z"]
    fits = {}
    for g, m in (("adeno", df["histo"] == "adeno"), ("SQ/SC", df["histo"].isin(["squamous", "small_cell"]))):
        dd = cs(df, m)
        fits[g] = {"th": hr_row(cox(dd, adj + ["th_z"]), "th_z"), "ca": hr_row(cox(dd, CLIN + ["calc_z"]), "calc_z"),
                   "events": int(dd["event"].sum())}
        rows.append({"analysis": f"G3 TH {g} (adj clin+emph+calcium)", **fits[g]["th"], "events": fits[g]["events"]})
        rows.append({"analysis": f"G3 calcium {g} (adj clin)", **fits[g]["ca"], "events": fits[g]["events"]})
    parts = [cs(df, df["histo"] == "adeno").assign(sq=0), cs(df, df["histo"].isin(["squamous", "small_cell"])).assign(sq=1)]
    st = pd.concat(parts, ignore_index=True)
    for c in adj:
        st[f"{c}_x"] = st[c] * st["sq"]
    st["th_x_sq"] = st["th_z"] * st["sq"]
    use = adj + [f"{c}_x" for c in adj] + ["th_z", "th_x_sq"]
    lm = CoxPHFitter().fit(st[use + ["time_yr", "event", "sq", "pid"]], "time_yr", "event", strata=["sq"],
                           cluster_col="pid", robust=True)
    lm_row = hr_row(lm, "th_x_sq")
    ca_ratio = delta_test(fits["adeno"]["ca"], fits["SQ/SC"]["ca"])
    res = pd.DataFrame(rows)
    res.to_csv(out / "g1_g3_hrs.csv", index=False)
    print(res.round(4).to_string())
    # G4
    tv, te = df[df["split"] != "test"], df[df["split"] == "test"].reset_index(drop=True)
    risks = {}
    for name, cols in (("clinical", CLIN), ("clinical+emph", CLIN + ["emph0"]), ("clinical+emph+calcium", CLIN + ["emph0", "calc_z"]),
                       ("clinical+emph+calcium+TH", CLIN + ["emph0", "calc_z", "th_z"])):
        cph = cox(tv, cols)
        risks[name] = te[cols].to_numpy(float) @ cph.params_[cols].to_numpy()
    c = paired_bootstrap_c(te, risks, "clinical+emph", 1000, 0)
    c["test_n"], c["test_events"] = len(te), int(te["event"].sum())
    c.to_csv(out / "g4_test_cindex.csv", index=False)
    print(c.round(4).to_string())
    summ = {**att, "partial_rho": rho, "lunn_mcneil_th_x_sqsc": lm_row, "calcium_histology_ratio": ca_ratio}
    print({k: v for k, v in summ.items() if k != "by_source"})
    json.dump(summ, open(out / "summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
