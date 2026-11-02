#!/usr/bin/env python3
"""S14 -- labelled training scans from every screening round (yr0, yr1, yr2).

The S7 cache already holds yr1/yr2 of every train + val patient, preprocessed
exactly like yr0 (own TotalSegmentator lung mask, same grid/window/encoding)
and passed through the S3 image QC (data/ssl_scans.csv). This stage gives each
of those scans a survival label so S4 can train on them (--train-scans).

Per scan of round r (0, 1, 2):
  origin    the scan itself: time = end - scr_days{r}, end = candx_days if the
            patient has a lung cancer, else canc_free_days (as S1 for yr0).
            scr_days{r} missing (scan on disk, date not in prsn) -> imputed as
            scr_days0 + 365 r, flagged in days_imputed
  event     the patient's lung cancer (same diagnosis for every round). Cancers
            diagnosed soon after the scan are kept (user, 2026-11-02)
  drop      scans with time <= 0 (taken at/after the diagnosis or the end of
            follow-up)
  age       age at that scan = cohort age + (scr_days{r} - scr_days0) / 365.25
  lung_ml, lung_mean_hu   from that scan's own mask (S7 rounds_index.csv)
  sex, race, cigsmok      baseline values (cohort.csv)
yr0 rows are the cohort.csv rows unchanged (time, age, covariates).

Output data/rounds_cohort.csv: scan_id, pid, round, split, cache ("yr0" =
lungs_u8.npy, "rounds" = rounds_u8.npy), row, time, event, covariates,
days_imputed, n_scans (scans of that patient kept here). Train + val only;
test patients are never in it (asserted). S4 uses the train rows; val and test
stay yr0-only (cohort.csv), so results compare with the yr0-only runs.

Usage (login node, seconds):
    env/.venv/bin/python src/s14_rounds_cohort.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nlst import COHORT_CSV, PRSN_CSV, ROUNDS_COHORT_CSV, SSL_CACHE_INDEX, SSL_SCANS_CSV

COVS = ["age", "sex", "race", "cigsmok", "lung_ml", "lung_mean_hu"]


def main() -> None:
    coh = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    scans = pd.read_csv(SSL_SCANS_CSV, dtype={"pid": str})
    idx = pd.read_csv(SSL_CACHE_INDEX, dtype={"pid": str})
    prsn = pd.read_csv(PRSN_CSV, dtype={"pid": str},
                       usecols=["pid", "candx_days", "canc_free_days",
                                "scr_days0", "scr_days1", "scr_days2"])
    test = set(coh.loc[coh["split"] == "test", "pid"])
    assert not test & set(scans["pid"]), "test patients in ssl_scans.csv"

    yr0 = scans[scans["round"] == "yr0"].merge(
        coh[["pid", "time", "event", *COVS]], on="pid", validate="1:1")
    yr0["days_imputed"] = False

    later = (scans[scans["round"] != "yr0"]
             .merge(idx[["scan_id", "lung_ml", "lung_mean_hu"]], on="scan_id", validate="1:1")
             .merge(coh[["pid", "event", "age", "sex", "race", "cigsmok"]], on="pid", validate="m:1")
             .merge(prsn, on="pid", validate="m:1"))
    r = later["round"].str[-1].astype(int).to_numpy()
    d0 = later["scr_days0"].fillna(0).to_numpy(float)  # as S1
    days = later[["scr_days0", "scr_days1", "scr_days2"]].to_numpy(float)[np.arange(len(r)), r]
    later["days_imputed"] = np.isnan(days)
    days = np.where(np.isnan(days), d0 + 365.0 * r, days)
    end = np.where(later["event"] == 1, later["candx_days"], later["canc_free_days"])
    later["time"] = end - days
    later["age"] = later["age"] + (days - d0) / 365.25

    n0 = len(later)
    after = ~(later["time"] > 0)
    print(f"yr1/yr2 QC-passing scans: {n0} ({int(later['days_imputed'].sum())} scan dates imputed)")
    print(f"  dropped, time <= 0 (at/after diagnosis or end of follow-up): {int(after.sum())} "
          f"({int(later.loc[after, 'event'].sum())} of them events)")
    later = later[~after]

    cols = ["scan_id", "pid", "round", "split", "cache", "row", "time", "event", *COVS, "days_imputed"]
    out = pd.concat([yr0[cols], later[cols]], ignore_index=True)
    out["time"] = out["time"].astype(int)
    out["n_scans"] = out.groupby("pid")["pid"].transform("size")
    out = out.sort_values(["split", "pid", "round"]).reset_index(drop=True)
    assert not test & set(out["pid"])
    out.to_csv(ROUNDS_COHORT_CSV, index=False)

    summ = (out.groupby(["split", "round"])
            .agg(scans=("pid", "size"), events=("event", "sum"),
                 median_years=("time", lambda t: round(t.median() / 365.25, 2)),
                 events_within_1y=("time", lambda t: int(((t <= 365) & (out.loc[t.index, "event"] == 1)).sum()))))
    print(summ.to_string())
    pts = out.groupby(["split", "pid"])["n_scans"].first().groupby("split").value_counts().unstack(fill_value=0)
    print("\npatients by number of scans:\n" + pts.to_string())
    print(f"\nwrote {ROUNDS_COHORT_CSV}: {len(out)} scans of {out['pid'].nunique()} patients")


if __name__ == "__main__":
    main()
