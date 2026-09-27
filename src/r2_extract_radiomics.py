#!/usr/bin/env python3
"""R2 -- per-patient radiomics extraction (pyradiomics) + hand-crafted cardiac
features, chunked for a Slurm array.

Per patient (baseline scan, RAS canonical, same loader as stage 2 / R1):
  - mask: TotalSegmentator heart (label 51, largest connected component,
    see cardiac.py) by default, plus qc_* scan-quality fields; `--organ` takes any
    organs.ORGAN_SET key so R6's lungs/sternum controls reuse this script.
  - pyradiomics 3.1.0, `original` image only (no wavelet/LoG -> ~107 features):
      shape       on the full mask
      first-order + texture (GLCM/GLRLM/GLSZM/GLDM/NGTDM) on the mask eroded
                  by 1 voxel in-plane (limits partial volume with lung/fat)
    resampled to 2x2x2 mm isotropic (B-spline image / NN mask, IBSI
    recommendation for 3D texture), fixed bin width 25 HU, raw HU (no
    normalization). Texture is extracted on the raw (unsmoothed) CT; kernel
    effects are handled in R3 (card_core_noise_sd_hu + harmonization).
  - heart only: cardiac.cardiac_features() (volume, calcium proxy, fat shell,
    core noise) -- exactly what R1 visualized.
Output: one CSV per chunk (`--out-dir`/chunk_XXX.csv), one row per pid with
`status` = ok | no_mask | error:<msg>; failures are kept as rows, never
silently dropped. `--merge` concatenates chunks into
data/radiomics/{organ}_features_v1.csv and prints a failure summary.

Usage:
    # one chunk (what each Slurm array task runs)
    env/.venv/bin/python src/r2_extract_radiomics.py --chunk 0 --n-chunks 50
    # quick local test on the first N pids of the cohort
    env/.venv/bin/python src/r2_extract_radiomics.py --limit 3 --out-dir /tmp/r2test
    # after all array tasks finish
    env/.venv/bin/python src/r2_extract_radiomics.py --merge
    # full-CT-arm calcium run (cardiac features only; see submit_r2_fullarm.sh)
    env/.venv/bin/python src/r2_extract_radiomics.py --cardiac-only --tag fullarm \
        --cohort data/pid_lists/heart_cohort_fullarm.csv --chunk 0 --n-chunks 40
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import binary_erosion

from cardiac import heart_crop, heart_regions, cardiac_features
from organs import ORGAN_SET, organ_mask, MARGIN_VOX
from stage2_visualize_masks import load_patient

PROJECT = Path(__file__).resolve().parents[1]
COHORT_CSV = PROJECT / "data" / "pid_lists" / "heart_cohort_v1.csv"
RADIOMICS_DIR = PROJECT / "data" / "radiomics"

BASE_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": [2.0, 2.0, 2.0],
    "interpolator": "sitkBSpline",
    "normalize": False,
    "correctMask": True,
    "preCrop": True,
    "label": 1,
}
TEXTURE_CLASSES = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
# in-plane (x, y) 3x3 cross; no z erosion (z voxels are 2 mm vs 1 mm in-plane)
ERODE_STRUCT = np.zeros((3, 3, 3), bool)
ERODE_STRUCT[1, :, 1] = ERODE_STRUCT[:, 1, 1] = True

logging.getLogger("radiomics").setLevel(logging.ERROR)


def make_extractor(classes: list[str]) -> featureextractor.RadiomicsFeatureExtractor:
    ext = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    ext.disableAllFeatures()
    ext.enableImageTypes(Original={})
    for c in classes:
        ext.enableFeatureClassByName(c)
    return ext


def to_sitk(arr: np.ndarray, vox) -> sitk.Image:
    """numpy (x, y, z) -> SimpleITK (which indexes arrays as z, y, x)."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    img.SetSpacing(tuple(float(v) for v in vox))
    return img


def run_extractor(ext, ct_img: sitk.Image, mask: np.ndarray, vox) -> dict:
    res = ext.execute(ct_img, to_sitk(mask.astype(np.uint8), vox))
    # keep feature values only (drop pyradiomics' diagnostics_* provenance keys)
    return {k: float(v) for k, v in res.items() if k.startswith("original_")}


def organ_crop(organ: str, ct, seg, axcodes, vox):
    """(ct_crop, seg_crop, mask_crop, qc) around the organ, or None."""
    if organ == "heart":
        crop = heart_crop(ct, seg, vox)
        if crop is None:
            return None
        ct_c, seg_c, _, qc = crop
        return ct_c, seg_c, seg_c == ORGAN_SET["heart"][0], qc
    mask = organ_mask(organ, ct, seg, axcodes, vox)
    if mask is None:
        return None
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - MARGIN_VOX, 0)
    hi = np.minimum(idx.max(0) + MARGIN_VOX + 1, np.array(ct.shape))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return ct[sl], seg[sl], mask[sl], {}


def extract_one(pid: str, organ: str, ext_shape, ext_tex, cardiac_only: bool = False) -> dict:
    ct, seg, axcodes, vox = load_patient(pid)
    crop = organ_crop(organ, ct, seg, axcodes, vox)
    if crop is None:
        return {"status": "no_mask"}
    ct_c, seg_c, mask, qc = crop
    if cardiac_only:  # hand-crafted cardiac features + qc only, no pyradiomics (full-CT-arm run)
        row = {"mask_voxels": int(mask.sum()), **qc}
        row.update(cardiac_features(ct_c, heart_regions(ct_c, seg_c, vox), vox))
        row["status"] = "ok"
        return row
    ct_img = to_sitk(ct_c.astype(np.float32), vox)

    row = {"mask_voxels": int(mask.sum()), **qc}
    row.update(run_extractor(ext_shape, ct_img, mask, vox))
    eroded = binary_erosion(mask, structure=ERODE_STRUCT)
    row["eroded_mask_voxels"] = int(eroded.sum())
    row.update(run_extractor(ext_tex, ct_img, eroded, vox))
    if organ == "heart":
        row.update(cardiac_features(ct_c, heart_regions(ct_c, seg_c, vox), vox))
    row["status"] = "ok"
    return row


def run_chunk(args) -> None:
    cohort = pd.read_csv(args.cohort, dtype={"pid": str})
    pids = cohort["pid"].tolist()
    if args.limit:
        pids = pids[: args.limit]
    else:
        pids = np.array_split(pids, args.n_chunks)[args.chunk].tolist()

    out_dir = Path(args.out_dir or RADIOMICS_DIR / f"{args.organ}_{args.tag}_chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"chunk_{args.chunk:03d}.csv"
    print(f"organ={args.organ} chunk {args.chunk}/{args.n_chunks}: {len(pids)} pids -> {out_csv}", flush=True)

    ext_shape, ext_tex = (None, None) if args.cardiac_only else (make_extractor(["shape"]), make_extractor(TEXTURE_CLASSES))
    rows = []
    for i, pid in enumerate(pids):
        t0 = time.time()
        try:
            row = extract_one(pid, args.organ, ext_shape, ext_tex, args.cardiac_only)
        except Exception as e:  # keep going; the failure is recorded, not dropped
            row = {"status": f"error:{type(e).__name__}:{e}"[:300]}
        row = {"pid": pid, **row, "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"[{i + 1}/{len(pids)}] {pid} {row['status']} {row['seconds']}s", flush=True)
        # rewrite after every patient so a timed-out task keeps its partial work
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"done: {sum(r['status'] == 'ok' for r in rows)}/{len(rows)} ok", flush=True)


def merge(args) -> None:
    chunk_dir = RADIOMICS_DIR / f"{args.organ}_{args.tag}_chunks"
    chunks = sorted(chunk_dir.glob("chunk_*.csv"))
    df = pd.concat([pd.read_csv(c, dtype={"pid": str}) for c in chunks], ignore_index=True)
    cohort = pd.read_csv(args.cohort, dtype={"pid": str})
    missing = sorted(set(cohort["pid"]) - set(df["pid"]))
    print(f"{len(chunks)} chunks, {len(df)} rows; cohort pids never processed: {len(missing)}")
    print(df["status"].str.split(":").str[0].value_counts().to_string())
    errs = df[df["status"].str.startswith("error")]
    if len(errs):
        print("errors (first 10):\n" + errs[["pid", "status"]].head(10).to_string(index=False))
    out = RADIOMICS_DIR / f"{args.organ}_features_{args.tag}.csv"
    df.sort_values("pid").to_csv(out, index=False)
    print(f"wrote {out}; seconds/patient median {df['seconds'].median():.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="heart", choices=list(ORGAN_SET))
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="local test: first N pids only")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="v1", help="names the chunk dir ({organ}_{tag}_chunks) and merged CSV")
    ap.add_argument("--cardiac-only", action="store_true",
                    help="heart only: card_* + qc_* features, skip pyradiomics (full-CT-arm calcium run)")
    args = ap.parse_args()
    if args.cardiac_only and args.organ != "heart":
        ap.error("--cardiac-only needs --organ heart")
    merge(args) if args.merge else run_chunk(args)


if __name__ == "__main__":
    main()
