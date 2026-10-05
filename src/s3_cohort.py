#!/usr/bin/env python3
"""S3 -- the training cohort: image QC exclusions + a fresh train/val/test split.

Reads data/candidates.csv (s1) and data/cache/lungs_index.csv (s2).
Exclusions (continuing s1's attrition table):
  6. preprocessing succeeded
  7. lungs fully scanned: lung cranio-caudal extent >= MIN_LUNG_Z_MM
  8. plausible mask: lung volume >= MIN_LUNG_ML
  9. lungs fit the fixed grid: clipped_frac <= MAX_CLIPPED_FRAC
Split: patient-level, random, stratified by event, 70/15/15, fixed seed
(SPLIT_SEED). One scan per patient, so there is no leakage between splits.

Output data/cohort.csv: one row per patient with `row` (index into
data/cache/lungs_u8.npy), time, event, split, covariates and QC columns.
This file is the source of truth for training and evaluation.

Usage:
    env/.venv/bin/python src/s3_cohort.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nlst import (CACHE_INDEX, CANDIDATES_CSV, COHORT_CSV, DATA, MAX_CLIPPED_FRAC,
                  MIN_LUNG_ML, MIN_LUNG_Z_MM, SPLIT_FRACS, SPLIT_SEED)

ATTRITION_CSV = DATA / "cohort_attrition.csv"
SPLITS_CSV = DATA / "cohort_splits.csv"


def stratified_split(event: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.empty(len(event), dtype=object)
    names = list(SPLIT_FRACS)
    cum = np.cumsum([SPLIT_FRACS[s] for s in names])
    for e in (0, 1):
        idx = rng.permutation(np.flatnonzero(event == e))
        cuts = np.rint(cum * len(idx)).astype(int)
        for name, a, b in zip(names, np.r_[0, cuts[:-1]], cuts):
            split[idx[a:b]] = name
    return split


def main() -> None:
    cand = pd.read_csv(CANDIDATES_CSV, dtype={"pid": str})
    qc = pd.read_csv(CACHE_INDEX, dtype={"pid": str})
    assert qc["pid"].tolist() == cand["pid"].tolist(), "cache index != candidates -- rerun s2"
    df = cand.merge(qc.drop(columns=["centre_x", "centre_y", "centre_z"]), on="pid")

    prev = pd.read_csv(DATA / "candidates_attrition.csv")
    steps = [
        ("preprocessing ok", df["status"] == "ok"),
        (f"lung z-extent >= {MIN_LUNG_Z_MM:g} mm", df["lung_z_mm"] >= MIN_LUNG_Z_MM),
        (f"lung volume >= {MIN_LUNG_ML:g} mL", df["lung_ml"] >= MIN_LUNG_ML),
        (f"clipped lung <= {MAX_CLIPPED_FRAC:.0%}", df["clipped_frac"] <= MAX_CLIPPED_FRAC),
    ]
    keep = np.ones(len(df), bool)
    rows = []
    for name, cond in steps:
        keep &= np.asarray(cond.fillna(False), bool)
        rows.append({"step": name, "n": int(keep.sum()), "events": int(df.loc[keep, "event"].sum())})
    att = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
    att["event_rate"] = (att["events"] / att["n"]).round(4)

    out = df[keep].reset_index(drop=True)
    out["split"] = stratified_split(out["event"].to_numpy(), SPLIT_SEED)
    splits = (out.groupby("split").agg(n=("pid", "size"), events=("event", "sum"),
                                       event_rate=("event", "mean"), age=("age", "mean"),
                                       follow_up_days=("time", "median"))
              .reindex(list(SPLIT_FRACS)).round(4))
    print(att.to_string(index=False), "\n")
    print(splits.to_string())

    cols = ["pid", "row", "split", "time", "event", "age", "sex", "race", "cigsmok",
            "manufacturer", "model", "kernel", "slice_thickness_mm", "kvp", "tube_current_ma",
            "lung_ml", "lung_z_mm", "lung_touches_z_edge", "clipped_frac", "lung_mean_hu"]
    out[cols].to_csv(COHORT_CSV, index=False)
    att.to_csv(ATTRITION_CSV, index=False)
    splits.to_csv(SPLITS_CSV)
    print(f"\nwrote {COHORT_CSV} ({len(out)} patients, {int(out['event'].sum())} events), "
          f"{ATTRITION_CSV}, {SPLITS_CSV}")


if __name__ == "__main__":
    main()
