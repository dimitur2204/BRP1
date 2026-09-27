#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, not in protocol): which covariate absorbs the TH -> lung-cancer
association, and does adiposity (pericardial fat) negatively confound it?

Usage: env/.venv/bin/python experiments/E1_thymus_heart_joint/code/explore_th_attenuation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
from aging_data import CLIN, fit_cox, hr_row, load_frame, zscore  # noqa: E402


def main() -> None:
    df, _ = load_frame()
    df["hv_z"] = zscore(df["card_heart_volume_ml"])
    df["fathu_z"] = zscore(df["card_fat_mean_hu"])
    X = np.column_stack([np.ones(len(df)), df[CLIN].to_numpy(float)])
    b, *_ = np.linalg.lstsq(X, df["th_pct"].to_numpy(float), rcond=None)
    r2 = 1 - (df["th_pct"] - X @ b).var() / df["th_pct"].var()
    print("TH percentile ~ age, male, cigsmok:", dict(zip(["const"] + CLIN, b.round(2))), "R2", round(r2, 3))

    rows = []
    specs = [("TH only", ["th_z"], None), ("+age", ["age", "th_z"], None), ("+sex", ["male", "th_z"], None),
             ("+smoking", ["cigsmok", "th_z"], None), ("+age+sex+smoking", CLIN + ["th_z"], None),
             ("+clin+pericardial fat", CLIN + ["fat_z", "th_z"], None),
             ("+clin+fat+heart volume", CLIN + ["fat_z", "hv_z", "th_z"], None),
             ("+clin+fat HU", CLIN + ["fathu_z", "th_z"], None),
             ("+clin+fat+calcium", CLIN + ["fat_z", "calc_z", "th_z"], None),
             ("women only (+age+smoking)", ["age", "cigsmok", "th_z"], df["male"] == 0),
             ("men only (+age+smoking)", ["age", "cigsmok", "th_z"], df["male"] == 1),
             ("former smokers (+age+sex)", ["age", "male", "th_z"], df["cigsmok"] == 0),
             ("current smokers (+age+sex)", ["age", "male", "th_z"], df["cigsmok"] == 1)]
    for name, cols, mask in specs:
        d = df if mask is None else df[mask]
        rows.append({"spec": name, **hr_row(fit_cox(d, cols), "th_z"), "n": len(d), "events": int(d["event"].sum())})
    for v in ("fat_z", "calc_z"):
        rows.append({"spec": f"{v} (adj clin)", **hr_row(fit_cox(df, CLIN + [v]), v), "n": len(df),
                     "events": int(df["event"].sum())})
    unadj = fit_cox(df, ["th_avg", "th_high"])
    rows.append({"spec": "TH high vs low, unadjusted (paper: 0.64)", **hr_row(unadj, "th_high"), "n": len(df),
                 "events": int(df["event"].sum())})
    res = pd.DataFrame(rows)
    res.to_csv(EXP / "results" / "explore_th_attenuation.csv", index=False)
    print(res[["spec", "hr", "lo", "hi", "p", "n", "events"]].round(4).to_string())


if __name__ == "__main__":
    main()
