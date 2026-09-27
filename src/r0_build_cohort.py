#!/usr/bin/env python3
"""R0 -- build the heart-radiomics cohort (enriched cohort, ~6.4k patients).

Joins, per baseline scan:
  - outcome + split:  experiments/lcrisk_discovery/data/manifest.csv
                      (time [days], event, split -- reused as-is, not re-derived)
  - clinical:         data/external/nlst_780/nlst_780/nlst_780_prsn_idc_20210527.csv
                      (public TCIA IDC-780 package; age, gender, race, cigsmok)
  - acquisition:      nlst.csv SeriesDescription, a comma-coded string decoded
                      and cross-checked against the manifest in R0:
                        f0 study year      f4 kernel (== manifest kernel, 100%)
                        f5 recon FOV [mm]  f6 slice thickness [mm]
                        f7 kVp (== manifest kvp, 99.5%)  f8 tube current [mA]
                      f1/f9/f10 are left undecoded (not needed).
  - file checks:      derived/ct_1x1x2mm/{pid}_yr0.nii.gz,
                      derived/totalseg_fullres/{pid}_yr0/seg.nii.gz, and the
                      CT's z-extent (header-only read). Specific pids only --
                      never enumerates either tree.

Exclusions (in order, each logged in the attrition table):
  1. SeriesDescription study year != 0 -- the manifest's "baseline" series is
     actually a T1/T2 scan for a handful of pids; not a baseline image.
  2. missing CT or TotalSegmentator mask.
  3. CT z-extent < MIN_Z_SLICES (partial-coverage scans; same gate as the CNN
     pipeline's build_subset.py).
  4. missing clinical covariates.
The heart-mask size gate (organs.MIN_VOX) is applied in R2, which loads the
mask anyway.

Usage:
    env/.venv/bin/python src/r0_build_cohort.py
"""
from __future__ import annotations

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
OUT_CSV = PROJECT / "data" / "pid_lists" / "heart_cohort_v1.csv"
ATTRITION_CSV = PROJECT / "data" / "pid_lists" / "heart_cohort_v1_attrition.csv"

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


def main() -> None:
    enriched = set(ENRICHED_PIDS.read_text().split())
    m = pd.read_csv(MANIFEST_CSV, dtype={"pid": str, "series_uid": str})
    m = m[m["pid"].isin(enriched)]
    df = m[["pid", "series_uid", "time", "event", "split", "kernel", "kvp", "image_count"]]
    df = df.merge(load_acquisition(), on="series_uid", how="left")

    assert (df["sd_kernel"] == df["kernel"]).all(), "SeriesDescription kernel != manifest kernel"
    df = df.drop(columns="sd_kernel")
    df["kernel_group"] = df["kernel"].map(kernel_group)

    prsn = pd.read_csv(PRSN_CSV, usecols=["pid", "age", "gender", "race", "cigsmok", "candx_days"],
                       dtype={"pid": str})
    # sanity: manifest events should be exactly the prsn pids with a diagnosis date
    chk = df.merge(prsn[["pid", "candx_days"]], on="pid", how="left")
    mismatch = ((chk["event"] == 1) != chk["candx_days"].notna()).sum()
    print(f"manifest event vs prsn candx_days disagreement: {mismatch} pids")
    prsn = prsn.drop(columns="candx_days")
    prsn["sex"] = prsn["gender"].map({1: "M", 2: "F"})
    df = df.merge(prsn.drop(columns="gender"), on="pid", how="left")

    with ThreadPoolExecutor(max_workers=16) as ex:
        files = pd.DataFrame(list(ex.map(check_files, df["pid"])))
    df = df.merge(files, on="pid", how="left")

    steps = [
        ("enriched ∩ manifest", np.ones(len(df), bool)),
        ("series is baseline (study yr 0)", df["series_study_yr"] == 0),
        ("has CT + TotalSegmentator mask", (df["has_ct"] == 1) & (df["has_seg"] == 1)),
        (f"CT z-extent >= {MIN_Z_SLICES} slices", df["n_z"].fillna(0) >= MIN_Z_SLICES),
        ("has age/sex/smoking", df[["age", "sex", "cigsmok"]].notna().all(axis=1)),
    ]
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
            "slice_thickness_mm", "recon_fov_mm", "tube_current_ma", "image_count", "n_z"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out[cols].sort_values("pid").to_csv(OUT_CSV, index=False)
    attrition.to_csv(ATTRITION_CSV, index=False)
    print(f"\nkernel_group: {out['kernel_group'].value_counts().to_dict()}")
    print(f"wrote {OUT_CSV} ({len(out)} rows) and {ATTRITION_CSV}")


if __name__ == "__main__":
    main()
