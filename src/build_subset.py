#!/usr/bin/env python3
"""Stage 0 -- build the student_pipeline patient subset.

Samples ~150-400 NLST patients from the mentor's `enriched_cohort_pids.txt`
(itself already enriched for lung-cancer-positive cases relative to the full
~26k-patient cohort), stratified by event so the classifier/KM stages later
have enough positives to learn from (raw NLST incidence is ~4%).

Every candidate pid is checked for:
  - a baseline CT volume:      derived/ct_1x1x2mm/{pid}_yr0.nii.gz
  - a TotalSegmentator mask:   derived/totalseg_fullres/{pid}_yr0/seg.nii.gz
  - MIN_Z_SLICES z-coverage on the CT (added after Stage 1 visualization
    caught 3/400 patients with only 5-22 axial slices -- a few NLST series in
    this cohort are partial-coverage scans, not full-thorax exams, and are
    useless for 3D organ cropping. Median in this cohort is ~156 slices;
    MIN_Z_SLICES=80 rejects only genuinely broken/partial scans.)

Never enumerates the full derived/ct_1x1x2mm or derived/totalseg_fullres
trees (tens of thousands of entries each) -- only looks up specific pids.

Usage:
    env/.venv/bin/python src/build_subset.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import nibabel as nib

REPO = Path("/faststorage/project/aura_thymus")
CT_DIR = REPO / "derived" / "ct_1x1x2mm"
TOTALSEG_DIR = REPO / "derived" / "totalseg_fullres"
ENRICHED_PIDS = REPO / "discovery_pipeline" / "enriched_cohort_pids.txt"
MANIFEST_CSV = REPO / "experiments" / "lcrisk_discovery" / "data" / "manifest.csv"

PROJECT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"

SEED = 42
TARGET_POS = 120
TARGET_NEG = 280
MIN_Z_SLICES = 80


def check_files(pid: str) -> tuple[bool, bool, int | None]:
    """Return (has_ct, has_seg, n_z_slices) for one pid, baseline year only."""
    ct_path = CT_DIR / f"{pid}_yr0.nii.gz"
    seg_path = TOTALSEG_DIR / f"{pid}_yr0" / "seg.nii.gz"
    has_seg = seg_path.exists()
    if not ct_path.exists():
        return False, has_seg, None
    n_z = nib.load(ct_path).shape[2]  # header-only read, no voxel data loaded
    return True, has_seg, n_z


def is_usable(pid: str) -> tuple[bool, dict]:
    has_ct, has_seg, n_z = check_files(pid)
    ok = has_ct and has_seg and n_z is not None and n_z >= MIN_Z_SLICES
    return ok, {"has_ct": int(has_ct), "has_seg": int(has_seg), "n_z": n_z}


def main() -> None:
    random.seed(SEED)

    enriched = {line.strip() for line in ENRICHED_PIDS.read_text().splitlines() if line.strip()}

    pos, neg = [], []
    with open(MANIFEST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["pid"] not in enriched:
                continue
            (pos if row["event"] == "1" else neg if row["event"] == "0" else []).append(row)

    random.shuffle(pos)
    random.shuffle(neg)

    chosen = []
    rejected = []
    for pool, target, label in [(pos, TARGET_POS, "event=1"), (neg, TARGET_NEG, "event=0")]:
        n_kept = 0
        for row in pool:
            if n_kept >= target:
                break
            ok, info = is_usable(row["pid"])
            if ok:
                chosen.append({**row, **info})
                n_kept += 1
            else:
                rejected.append({**row, **info})
        if n_kept < target:
            print(f"WARNING: only found {n_kept}/{target} usable {label} patients "
                  f"in the enriched pool (pool size {len(pool)})")

    n_pos = sum(1 for c in chosen if c["event"] == "1")
    n_neg = sum(1 for c in chosen if c["event"] == "0")
    print(f"kept {len(chosen)} patients ({n_pos} positive, {n_neg} negative)")
    if rejected:
        print(f"rejected {len(rejected)} candidates (missing files or n_z < {MIN_Z_SLICES}):")
        for r in rejected:
            print(f"  {r['pid']} event={r['event']} has_ct={r['has_ct']} "
                  f"has_seg={r['has_seg']} n_z={r['n_z']}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "time", "event", "label", "split", "has_ct", "has_seg", "n_z"])
        for row in chosen:
            w.writerow([row["pid"], row["time"], row["event"], row["label"], row["split"],
                        row["has_ct"], row["has_seg"], row["n_z"]])
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
