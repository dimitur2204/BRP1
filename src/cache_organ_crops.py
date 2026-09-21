#!/usr/bin/env python3
"""Stage 2 (full cohort) -- cache mask-zeroed organ crops for every patient in
data/pid_lists/subset_v1.csv, for all 7 organs in organs.ORGAN_SET.

Timed at ~3s/patient (load + all 7 organs) on the login node -> ~20 min for
400 patients. Run via run_in_background / nohup, not interactively.

Output:
  data/organ_crops/{organ}/{pid}.npy       float16, native resolution,
                                            mask-zeroed to air (-1000 HU)
  data/organ_crops/manifest.csv            per (pid, organ): found/shape/nvox
                                            -- Stage 3 reads this to know which
                                            (pid, organ) pairs actually exist
                                            (a mask can be missing/too small).

Usage:
    env/.venv/bin/python src/cache_organ_crops.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from organs import ORGAN_SET, organ_mask, mask_zeroed_crop

REPO = Path("/faststorage/project/aura_thymus")
CT_DIR = REPO / "derived" / "ct_1x1x2mm"
SEG_DIR = REPO / "derived" / "totalseg_fullres"

PROJECT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
OUT_DIR = PROJECT / "data" / "organ_crops"
MANIFEST_CSV = OUT_DIR / "manifest.csv"


def load_patient(pid: str, yr: int = 0):
    ct_img = nib.as_closest_canonical(nib.load(CT_DIR / f"{pid}_yr{yr}.nii.gz"))
    seg_img = nib.as_closest_canonical(nib.load(SEG_DIR / f"{pid}_yr{yr}" / "seg.nii.gz"))
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    seg = np.asarray(seg_img.dataobj, dtype=np.int16)
    axcodes = nib.orientations.aff2axcodes(ct_img.affine)
    vox = np.array(ct_img.header.get_zooms()[:3])
    if ct.shape != seg.shape:
        raise ValueError(f"{pid}: CT shape {ct.shape} != seg shape {seg.shape}")
    return ct, seg, axcodes, vox


def main() -> None:
    for name in ORGAN_SET:
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)

    pids = [row["pid"] for row in csv.DictReader(open(SUBSET_CSV))]
    print(f"caching crops for {len(pids)} patients x {len(ORGAN_SET)} organs", flush=True)

    manifest_rows = []
    t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            ct, seg, axcodes, vox = load_patient(pid)
        except Exception as exc:
            print(f"  ERROR loading {pid}: {exc}", flush=True)
            for name in ORGAN_SET:
                manifest_rows.append([pid, name, 0, "", 0])
            continue

        for name in ORGAN_SET:
            out_path = OUT_DIR / name / f"{pid}.npy"
            if out_path.exists():
                manifest_rows.append([pid, name, 1, "cached", -1])
                continue
            mask = organ_mask(name, ct, seg, axcodes, vox)
            if mask is None:
                manifest_rows.append([pid, name, 0, "", 0])
                continue
            crop, mask_crop, _ = mask_zeroed_crop(ct, mask)
            np.save(out_path, crop.astype(np.float16))
            manifest_rows.append([pid, name, 1, "x".join(map(str, crop.shape)), int(mask.sum())])

        if (i + 1) % 25 == 0 or (i + 1) == len(pids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pids) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"  [{i+1}/{len(pids)}] {elapsed:.0f}s elapsed, "
                  f"eta {eta:.0f}s", flush=True)

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "organ", "found", "shape", "nvox"])
        w.writerows(manifest_rows)
    print(f"wrote {MANIFEST_CSV}", flush=True)

    # summary: found-rate per organ
    from collections import Counter
    found = Counter()
    total = Counter()
    for pid, organ, f_, shape, nvox in manifest_rows:
        total[organ] += 1
        found[organ] += f_
    print("\nper-organ mask found-rate:", flush=True)
    for organ in ORGAN_SET:
        print(f"  {organ:22s} {found[organ]:4d}/{total[organ]:4d}", flush=True)


if __name__ == "__main__":
    main()
