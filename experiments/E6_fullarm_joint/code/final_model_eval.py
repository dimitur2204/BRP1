#!/usr/bin/env python3
"""Final-model summary for the report (descriptive, uses the E6-locked final model):
clinical + emphysema + calcium + TH, Cox fit on train+val of the 16k cohort.
Outputs: HR table (full 16k cohort), test tertile KM counts + log-rank, time-dependent AUC.

Usage: env/.venv/bin/python experiments/E6_fullarm_joint/code/final_model_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(EXP.parents[1] / "src"))
sys.path.insert(0, str(EXP.parents[1] / "experiments" / "E3_calcium_mechanism" / "code"))
from run_e6 import CLIN, cox, load  # noqa: E402
from aging_data import hr_row  # noqa: E402

FINAL = CLIN + ["emph0", "calc_z", "th_z"]


def main() -> None:
    df, _ = load(False)
    full = cox(df, FINAL)
    hr = pd.DataFrame([hr_row(full, v) for v in FINAL])
    hr.to_csv(EXP / "results" / "final_model_hrs_16k.csv", index=False)
    print(hr.round(4).to_string())

    tv, te = df[df["split"] != "test"], df[df["split"] == "test"].reset_index(drop=True)
    rows = []
    s_tr = Surv.from_arrays(tv["event"].astype(bool), tv["time_yr"])
    s_te = Surv.from_arrays(te["event"].astype(bool), te["time_yr"])
    for name, cols in (("clinical", CLIN), ("clinical+emph", CLIN + ["emph0"]), ("final", FINAL)):
        cph = cox(tv, cols)
        risk = te[cols].to_numpy(float) @ cph.params_[cols].to_numpy()
        auc, _ = cumulative_dynamic_auc(s_tr, s_te, risk, [2.0, 4.0, 6.0])
        tert = pd.qcut(pd.Series(risk).rank(method="first"), 3, labels=["low", "mid", "high"])
        lr = multivariate_logrank_test(te["time_yr"], tert, te["event"])
        ev = te.groupby(tert, observed=True)["event"].agg(["size", "sum"])
        cum6 = {g: float(1 - KaplanMeierFitter().fit(te.loc[tert == g, "time_yr"], te.loc[tert == g, "event"]).predict(6.0))
                for g in ("low", "mid", "high")}
        rows.append({"model": name, "auc_2y": auc[0], "auc_4y": auc[1], "auc_6y": auc[2], "logrank_p": lr.p_value,
                     **{f"n_{g}": int(ev.loc[g, "size"]) for g in ("low", "mid", "high")},
                     **{f"events_{g}": int(ev.loc[g, "sum"]) for g in ("low", "mid", "high")},
                     **{f"cuminc6_{g}": cum6[g] for g in ("low", "mid", "high")}})
    res = pd.DataFrame(rows)
    res.to_csv(EXP / "results" / "final_model_test_eval.csv", index=False)
    print(res.round(4).T.to_string())


if __name__ == "__main__":
    main()
