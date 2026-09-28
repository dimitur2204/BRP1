#!/usr/bin/env python3
"""C0 -- build the lungs/sternum 3D CNN cohort (enriched cohort, ~6.3k patients).

Ported from the heart-radiomics branch's r0_build_cohort.py: identical joins
and exclusions, so the two projects use the same patients and are directly
comparable.

How the enriched cohort was made (mentor, discovery_pipeline): ALL lung-cancer
cases in the 26,254-scan manifest plus a random sample of controls. The
control sampling rate differs by split (printed + saved below as the
"sampling" table): ~8% of train controls but ~50% of val/test controls, so the
event rate is ~33% in train but ~8% in val/test. Ranking metrics (C-index,
AUC) are unaffected by this in expectation; absolute risks are not.

Joins, per baseline scan:
  - outcome + split:  experiments/lcrisk_discovery/data/manifest.csv
                      (time [days], event, split -- reused as-is, not re-derived;
                      the manifest's `label` column is identical to `event`)
  - acquisition:      nlst.csv SeriesDescription, a comma-coded string:
                        f0 study year      f4 kernel (== manifest kernel, 100%)
                        f5 recon FOV [mm]  f6 slice thickness [mm]
                        f7 kVp (== manifest kvp, 99.5%)  f8 tube current [mA]
  - clinical (OPTIONAL): the IDC-780 prsn table (age, gender, race, cigsmok).
                      Merged only if --prsn exists; it is not needed to build
                      the cohort (on the heart-radiomics branch the "has
                      age/sex/smoking" step removed nobody), only for the
                      clinical-adjustment analyses in c4_evaluate.py.
  - file checks:      derived/ct_1x1x2mm/{pid}_yr0.nii.gz,
                      derived/totalseg_fullres/{pid}_yr0/seg.nii.gz, and the
                      CT's z-extent (header-only read). Specific pids only --
                      never enumerates either tree.
  - in_subset_v1:     1 if the pid was in the legacy 400-patient subset.

Exclusions (in order, each logged in the attrition table):
  1. SeriesDescription study year != 0 -- the manifest's "baseline" series is
     actually a T1/T2 scan for a handful of pids; not a baseline image.
  2. missing CT or TotalSegmentator mask.
  3. CT z-extent < MIN_Z_SLICES (partial-coverage scans).
  4. missing clinical covariates (only when --prsn is available).
The per-organ mask-size gate (organs.MIN_VOX) is applied in c1, which loads
the mask anyway.

Usage:
    env/.venv/bin/python src/c0_build_cohort.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path("/faststorage/project/aura_thymus")
CT_DIR = REPO / "derived" / "ct_1x1x2mm"
TOTALSEG_DIR = REPO / "derived" / "totalseg_fullres"
ENRICHED_PIDS = REPO / "discovery_pipeline" / "enriched_cohort_pids.txt"
MANIFEST_CSV = REPO / "experiments" / "lcrisk_discovery" / "data" / "manifest.csv"
NLST_CSV = REPO / "nlst.csv"

PROJECT = Path(__file__).resolve().parents[1]
PRSN_CSV = PROJECT / "data" / "external" / "nlst_780" / "nlst_780" / "nlst_780_prsn_idc_20210527.csv"
SUBSET_V1 = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
OUT_CSV = PROJECT / "data" / "pid_lists" / "cnn_cohort_v1.csv"
ATTRITION_CSV = PROJECT / "data" / "pid_lists" / "cnn_cohort_v1_attrition.csv"
SAMPLING_CSV = PROJECT / "data" / "pid_lists" / "cnn_cohort_v1_sampling.csv"

MIN_Z_SLICES = 80

# Reconstruction-kernel sharpness. Vendor naming differs, so this is a manual
# mapping; R3 checks it empirically (measured image noise), so treat it as a
# first guess. Philips "C" is the least certain call (Philips A/B are smooth,
# C/D progressively sharper) -- kept as its own column value via `kernel` so
# R3 can regroup without rebuilding the cohort.
SOFT_KERNELS = {"STANDARD", "B30f", "B31s", "B20f", "B", "A", "FC10", "FC02"}
SHARP_KERNELS = {"BONE", "LUNG", "B50f", "B80f", "C", "FC50", "FC51", "FC53", "FC82"}


def kernel_group(k: str) -> str:
    if k in SOFT_KERNELS:
        return "soft"
    if k in SHARP_KERNELS:
        return "sharp"
    return "other"


def check_files(pid: str) -> dict:
    ct_path = CT_DIR / f"{pid}_yr0.nii.gz"
    seg_path = TOTALSEG_DIR / f"{pid}_yr0" / "seg.nii.gz"
    has_ct, has_seg = ct_path.exists(), seg_path.exists()
    n_z = nib.load(ct_path).shape[2] if has_ct else None  # header-only read
    return {"pid": pid, "has_ct": int(has_ct), "has_seg": int(has_seg), "n_z": n_z}


def load_acquisition() -> pd.DataFrame:
    cols = ["SeriesInstanceUID", "Manufacturer", "ManufacturerModelName", "SeriesDescription"]
    df = pd.read_csv(NLST_CSV, usecols=cols, dtype=str).drop_duplicates("SeriesInstanceUID")
    f = df["SeriesDescription"].str.split(",", expand=True)
    num = lambda s: pd.to_numeric(s, errors="coerce")  # 'na' -> NaN
    return pd.DataFrame({
        "series_uid": df["SeriesInstanceUID"],
        "manufacturer": df["Manufacturer"].str.strip(),
        "model": df["ManufacturerModelName"].str.strip(),
        "series_study_yr": num(f[0]),
        "sd_kernel": f[4],
        "recon_fov_mm": num(f[5]),
        # 2 series have a malformed thickness (350.2) -- anything >10mm is not
        # a real chest-CT slice thickness, so treat it as missing
        "slice_thickness_mm": num(f[6]).where(lambda x: x <= 10),
        "sd_kvp": num(f[7]),
        "tube_current_ma": num(f[8]),
    })


def sampling_table(m: pd.DataFrame, enriched: set) -> pd.DataFrame:
    """How the enriched cohort samples the full manifest, per split."""
    rows = []
    for s in ("train", "val", "test"):
        ms = m[m["split"] == s]
        es = ms[ms["pid"].isin(enriched)]
        cases, ctrl = int(ms["event"].sum()), int((ms["event"] == 0).sum())
        e_cases, e_ctrl = int(es["event"].sum()), int((es["event"] == 0).sum())
        rows.append({"split": s, "manifest_cases": cases, "manifest_controls": ctrl,
                     "enriched_cases": e_cases, "enriched_controls": e_ctrl,
                     "case_sampling_rate": round(e_cases / cases, 3),
                     "control_sampling_rate": round(e_ctrl / ctrl, 3),
                     "manifest_event_rate": round(cases / len(ms), 3),
                     "enriched_event_rate": round(e_cases / len(es), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prsn", default=str(PRSN_CSV),
                        help="IDC-780 prsn CSV; clinical columns are left empty if missing")
    args = parser.parse_args()

    enriched = set(ENRICHED_PIDS.read_text().split())
    m = pd.read_csv(MANIFEST_CSV, dtype={"pid": str, "series_uid": str})
    assert (m["label"] == m["event"]).all(), "manifest label != event"
    sampling = sampling_table(m, enriched)
    print(sampling.to_string(index=False), "\n")
    m = m[m["pid"].isin(enriched)]
    df = m[["pid", "series_uid", "time", "event", "split", "kernel", "kvp", "image_count"]]
    df = df.merge(load_acquisition(), on="series_uid", how="left")

    assert (df["sd_kernel"] == df["kernel"]).all(), "SeriesDescription kernel != manifest kernel"
    df = df.drop(columns="sd_kernel")
    df["kernel_group"] = df["kernel"].map(kernel_group)

    prsn_path = Path(args.prsn)
    has_clinical = prsn_path.exists()
    if has_clinical:
        prsn = pd.read_csv(prsn_path, usecols=["pid", "age", "gender", "race", "cigsmok", "candx_days"],
                           dtype={"pid": str})
        # sanity: manifest events should be exactly the prsn pids with a diagnosis date
        chk = df.merge(prsn[["pid", "candx_days"]], on="pid", how="left")
        mismatch = ((chk["event"] == 1) != chk["candx_days"].notna()).sum()
        print(f"manifest event vs prsn candx_days disagreement: {mismatch} pids")
        prsn = prsn.drop(columns="candx_days")
        prsn["sex"] = prsn["gender"].map({1: "M", 2: "F"})
        df = df.merge(prsn.drop(columns="gender"), on="pid", how="left")
    else:
        print(f"NOTE: {prsn_path} not found -- clinical columns left empty, "
              "clinical-completeness exclusion skipped")
        for c in ("age", "sex", "race", "cigsmok"):
            df[c] = np.nan
    subset_v1 = set(pd.read_csv(SUBSET_V1, dtype={"pid": str})["pid"])
    df["in_subset_v1"] = df["pid"].isin(subset_v1).astype(int)

    with ThreadPoolExecutor(max_workers=16) as ex:
        files = pd.DataFrame(list(ex.map(check_files, df["pid"])))
    df = df.merge(files, on="pid", how="left")

    steps = [
        ("enriched ∩ manifest", np.ones(len(df), bool)),
        ("series is baseline (study yr 0)", df["series_study_yr"] == 0),
        ("has CT + TotalSegmentator mask", (df["has_ct"] == 1) & (df["has_seg"] == 1)),
        (f"CT z-extent >= {MIN_Z_SLICES} slices", df["n_z"].fillna(0) >= MIN_Z_SLICES),
    ]
    if has_clinical:
        steps.append(("has age/sex/smoking", df[["age", "sex", "cigsmok"]].notna().all(axis=1)))
    keep = np.ones(len(df), bool)
    rows = []
    for name, cond in steps:
        keep &= np.asarray(cond)
        k = df[keep]
        row = {"step": name, "n": len(k), "events": int(k["event"].sum())}
        for s in ("train", "val", "test"):
            ks = k[k["split"] == s]
            row[f"{s}_n"], row[f"{s}_events"] = len(ks), int(ks["event"].sum())
        rows.append(row)
    attrition = pd.DataFrame(rows)
    print(attrition.to_string(index=False))

    out = df[keep].copy()
    out["n_z"] = out["n_z"].astype(int)
    cols = ["pid", "series_uid", "time", "event", "split",
            "age", "sex", "race", "cigsmok",
            "manufacturer", "model", "kernel", "kernel_group", "kvp",
            "slice_thickness_mm", "recon_fov_mm", "tube_current_ma", "image_count", "n_z",
            "in_subset_v1"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out[cols].sort_values("pid").to_csv(OUT_CSV, index=False)
    attrition.to_csv(ATTRITION_CSV, index=False)
    sampling.to_csv(SAMPLING_CSV, index=False)
    print(f"\nkernel_group: {out['kernel_group'].value_counts().to_dict()}")
    print(f"wrote {OUT_CSV} ({len(out)} rows), {ATTRITION_CSV}, {SAMPLING_CSV}")


if __name__ == "__main__":
    main()
