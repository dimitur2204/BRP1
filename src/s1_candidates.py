#!/usr/bin/env python3
"""S1 -- candidate patients: every NLST participant with a usable baseline CT.

Starts from all NLST participants in the public IDC-780 `prsn` table and
keeps those whose baseline (T0) low-dose CT is on disk as a resampled volume
with a TotalSegmentator mask. Nothing is sampled or enriched: the candidate
list has the natural NLST cancer rate among scans that have masks.

Sources
  prsn (IDC-780)      outcome + clinical covariates, one row per participant
                        candx_days      days randomisation -> lung cancer diagnosis
                        canc_free_days  days randomisation -> last known cancer-free
                        scr_days0       days randomisation -> T0 screen
                        age, gender, race, cigsmok
  scan list           pid -> series_uid of the CT that was resampled to
                      derived/ct_1x1x2mm/{pid}_yr0.nii.gz. Used only for the
                      series id. Its time/event/split columns are NOT used.
  nlst.csv            IDC series metadata. SeriesDescription is comma-coded:
                      f0 study year, f2 manufacturer, f4 kernel,
                      f6 slice thickness [mm], f7 kVp, f8 mA
  file checks         CT and mask existence + CT z-extent, by pid name only
                      (never enumerates either derived/ tree)

Outcome (time-to-event, origin = the T0 scan)
  event = 1 if a lung cancer diagnosis exists (candx_days not empty)
  time  = (candx_days if event else canc_free_days) - scr_days0   [days]
  scr_days0 is missing for ~145 participants; they are given 0 (the T0 scan
  is a median 5 days after randomisation).

Exclusions, in order, each logged in data/candidates_attrition.csv:
  1. in the IDC scan list (has a baseline CT series)
  2. that series is from study year 0
  3. resampled CT on disk
  4. TotalSegmentator mask on disk
  5. follow-up > 0 days after the T0 scan
Image-based QC (lungs fully scanned, mask plausible) happens in s2, the
train/val/test split in s3.

Usage:
    env/.venv/bin/python src/s1_candidates.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

from nlst import CANDIDATES_CSV, CT_DIR, DATA, PRSN_CSV, SCAN_LIST, SEG_DIR, SERIES_CSV

ATTRITION_CSV = DATA / "candidates_attrition.csv"


def check_files(pid: str) -> dict:
    ct = CT_DIR / f"{pid}_yr0.nii.gz"
    seg = SEG_DIR / f"{pid}_yr0" / "seg.nii.gz"
    has_ct, has_seg = ct.exists(), seg.exists()
    n_z = nib.load(ct).shape[2] if has_ct else np.nan  # header-only read
    return {"pid": pid, "has_ct": int(has_ct), "has_seg": int(has_seg), "n_z": n_z}


def load_series_meta() -> pd.DataFrame:
    cols = ["SeriesInstanceUID", "Manufacturer", "ManufacturerModelName", "SeriesDescription"]
    df = pd.read_csv(SERIES_CSV, usecols=cols, dtype=str).drop_duplicates("SeriesInstanceUID")
    f = df["SeriesDescription"].str.split(",", expand=True)
    num = lambda s: pd.to_numeric(s, errors="coerce")  # 'na' -> NaN
    return pd.DataFrame({
        "series_uid": df["SeriesInstanceUID"],
        "manufacturer": df["Manufacturer"].str.strip(),
        "model": df["ManufacturerModelName"].str.strip(),
        "series_study_yr": num(f[0]),
        "kernel": f[4],
        # a few series carry a malformed thickness (e.g. 350.2); >10 mm is not real
        "slice_thickness_mm": num(f[6]).where(lambda x: x <= 10),
        "kvp": num(f[7]),
        "tube_current_ma": num(f[8]),
    })


def main() -> None:
    prsn = pd.read_csv(PRSN_CSV, dtype={"pid": str},
                       usecols=["pid", "age", "gender", "race", "cigsmok",
                                "candx_days", "canc_free_days", "scr_days0"])
    df = prsn.rename(columns={"gender": "sex"})
    df["sex"] = df["sex"].map({1: "M", 2: "F"})
    df["event"] = df["candx_days"].notna().astype(int)
    t0 = df["scr_days0"].fillna(0)
    df["time"] = np.where(df["event"] == 1, df["candx_days"], df["canc_free_days"]) - t0

    scans = pd.read_csv(SCAN_LIST, usecols=["pid", "series_uid"], dtype=str)
    assert scans["pid"].is_unique
    df = df.merge(scans, on="pid", how="left")
    df = df.merge(load_series_meta(), on="series_uid", how="left")

    in_list = df["series_uid"].notna()
    print(f"checking files for {in_list.sum()} pids ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        files = pd.DataFrame(list(ex.map(check_files, df.loc[in_list, "pid"])))
    df = df.merge(files, on="pid", how="left")

    steps = [
        ("NLST participants (IDC-780 prsn)", np.ones(len(df), bool)),
        ("baseline CT series in IDC scan list", in_list),
        ("series is study year 0", df["series_study_yr"] == 0),
        ("resampled CT on disk", df["has_ct"] == 1),
        ("TotalSegmentator mask on disk", df["has_seg"] == 1),
        ("follow-up > 0 days after T0", df["time"] > 0),
    ]
    keep = np.ones(len(df), bool)
    rows = []
    for name, cond in steps:
        keep &= np.asarray(cond, bool)
        rows.append({"step": name, "n": int(keep.sum()), "events": int(df.loc[keep, "event"].sum())})
    attrition = pd.DataFrame(rows)
    attrition["event_rate"] = (attrition["events"] / attrition["n"]).round(4)
    print(attrition.to_string(index=False))

    out = df[keep].copy()
    out["time"] = out["time"].astype(int)
    out["n_z"] = out["n_z"].astype(int)
    cols = ["pid", "series_uid", "time", "event", "age", "sex", "race", "cigsmok",
            "manufacturer", "model", "kernel", "slice_thickness_mm", "kvp",
            "tube_current_ma", "n_z"]
    CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out[cols].sort_values("pid").to_csv(CANDIDATES_CSV, index=False)
    attrition.to_csv(ATTRITION_CSV, index=False)
    yrs = out["time"] / 365.25
    print(f"\nfollow-up years: median {yrs.median():.2f}, max {yrs.max():.2f}; "
          f"events: median time to diagnosis {yrs[out['event'] == 1].median():.2f} y, "
          f"{(yrs[out['event'] == 1] <= 1).mean():.0%} within 1 y")
    print(f"wrote {CANDIDATES_CSV} ({len(out)} rows), {ATTRITION_CSV}")


if __name__ == "__main__":
    main()
