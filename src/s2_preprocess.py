#!/usr/bin/env python3
"""S2 -- turn each candidate's baseline CT into one fixed-size lung volume.

Per patient (input definition in nlst.py):
  1. load the 1x1x2 mm CT and its TotalSegmentator mask, reoriented to RAS;
  2. lung mask = union of the 5 lobe labels;
  3. resample a fixed physical box (GRID voxels at SPACING_MM, 360x320x320 mm)
     centred on the lung bounding box: trilinear for HU, trilinear + 0.5
     threshold for the mask. No per-patient rescaling, so lung size and shape
     are preserved;
  4. dilate the mask by MASK_DILATE_VOX voxels (5 mm);
  5. encode as uint8: 0 outside the mask, 1-255 = lung-window HU inside.
QC columns written per patient (s3 applies the exclusions):
  lung_ml, lung_z_mm   lung volume and cranio-caudal extent (native mask)
  lung_touches_z_edge  lung mask reaches the first/last CT slice
  clipped_frac         share of native lung voxels outside the fixed box
  lung_mean_hu         mean HU inside the native lung mask
  status               "ok" or the error message (row is all zeros then)

Slurm array (CPU), then merge + QC on the login node:
    sbatch src/submit_s2_preprocess.sh
    env/.venv/bin/python src/s2_preprocess.py --merge   # -> data/cache/lungs_u8.npy + lungs_index.csv
    env/.venv/bin/python src/s2_preprocess.py --qc      # -> figs/s2_inputs.png, QC summary
Smoke test:
    env/.venv/bin/python src/s2_preprocess.py --pids 100002,100004 --out-dir /tmp/x
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from nlst import (AIR_HU, CACHE_DIR, CACHE_INDEX, CACHE_NPY, CANDIDATES_CSV, CT_DIR, FIGS,
                  GRID, LUNG_LABELS, MASK_DILATE_VOX, SEG_DIR, SPACING_MM, encode_hu)

CHUNK_DIR = CACHE_DIR / "chunks"


def preprocess(pid: str) -> tuple[np.ndarray, dict]:
    ct_img = nib.as_closest_canonical(nib.load(CT_DIR / f"{pid}_yr0.nii.gz"))
    seg_img = nib.as_closest_canonical(nib.load(SEG_DIR / f"{pid}_yr0" / "seg.nii.gz"))
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    seg = np.asarray(seg_img.dataobj)
    if ct.shape != seg.shape:
        raise ValueError(f"CT shape {ct.shape} != mask shape {seg.shape}")
    vox = np.array(ct_img.header.get_zooms()[:3], dtype=np.float64)

    lung = np.isin(seg, LUNG_LABELS)
    if not lung.any():
        raise ValueError("empty lung mask")
    idx = np.argwhere(lung)
    lo, hi = idx.min(0), idx.max(0)
    centre = (lo + hi) / 2.0                                   # native voxel coords

    # output voxel o (per axis) samples native coordinate centre + (o - (N-1)/2) * step
    grid = np.array(GRID)
    step = SPACING_MM / vox
    offset = centre - (grid - 1) / 2.0 * step
    hu = ndimage.affine_transform(ct, np.diag(step), offset=offset, output_shape=GRID,
                                  order=1, mode="constant", cval=AIR_HU)
    m = ndimage.affine_transform(lung.astype(np.float32), np.diag(step), offset=offset,
                                 output_shape=GRID, order=1, mode="constant", cval=0.0) >= 0.5
    m = ndimage.binary_dilation(m, iterations=MASK_DILATE_VOX)

    box_lo, box_hi = offset - 0.5 * step, offset + (grid - 0.5) * step
    outside = ((idx < box_lo) | (idx > box_hi)).any(1)
    qc = {
        "lung_ml": float(lung.sum() * vox.prod() / 1000.0),
        "lung_z_mm": float((hi[2] - lo[2] + 1) * vox[2]),
        "lung_touches_z_edge": int(lo[2] == 0 or hi[2] == ct.shape[2] - 1),
        "clipped_frac": float(outside.mean()),
        "lung_mean_hu": float(ct[lung].mean()),
        "centre_x": centre[0], "centre_y": centre[1], "centre_z": centre[2],
        "status": "ok",
    }
    return encode_hu(hu, m), qc


def run_pids(pids: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    arr = np.zeros((len(pids), *GRID), dtype=np.uint8)
    rows = []
    for i, pid in enumerate(pids):
        t = time.time()
        try:
            arr[i], qc = preprocess(pid)
        except Exception as e:  # recorded, excluded in s3
            qc = {"status": f"error: {type(e).__name__}: {e}"}
        rows.append({"pid": pid, **qc, "sec": round(time.time() - t, 2)})
        print(f"[{i + 1}/{len(pids)}] {pid} {qc['status']} {rows[-1]['sec']}s", flush=True)
    return arr, pd.DataFrame(rows)


def task_pids(task_id: int, n_tasks: int) -> list[str]:
    pids = pd.read_csv(CANDIDATES_CSV, dtype={"pid": str})["pid"].tolist()
    return np.array_split(np.array(pids), n_tasks)[task_id].tolist()


def merge(n_tasks: int) -> None:
    cand = pd.read_csv(CANDIDATES_CSV, dtype={"pid": str})
    parts = [CHUNK_DIR / f"{t:03d}" for t in range(n_tasks)]
    missing = [p.name for p in parts if not p.with_suffix(".csv").exists()]
    if missing:
        raise SystemExit(f"missing chunks {missing} -- resubmit those array tasks first")
    index = pd.concat([pd.read_csv(p.with_suffix(".csv"), dtype={"pid": str}) for p in parts],
                      ignore_index=True)
    if index["pid"].tolist() != cand["pid"].tolist():
        raise SystemExit("chunk pids != candidates.csv -- was s1 re-run after s2? redo s2")
    out = np.lib.format.open_memmap(CACHE_NPY, mode="w+", dtype=np.uint8,
                                    shape=(len(index), *GRID))
    row = 0
    for p in parts:  # stream, never hold the full ~37 GB in RAM
        a = np.load(p.with_suffix(".npy"))
        out[row:row + len(a)] = a
        row += len(a)
        print(f"merged {p.name} ({row}/{len(index)})", flush=True)
    out.flush()
    index.insert(0, "row", np.arange(len(index)))
    index.to_csv(CACHE_INDEX, index=False)
    ok = index["status"] == "ok"
    print(f"wrote {CACHE_NPY} {out.shape}, {CACHE_INDEX}: {ok.sum()} ok, {(~ok).sum()} errors")
    print("You can now delete the chunks:", CHUNK_DIR)


def qc_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = pd.read_csv(CACHE_INDEX, dtype={"pid": str})
    cand = pd.read_csv(CANDIDATES_CSV, dtype={"pid": str})[["pid", "event"]]
    idx = idx.merge(cand, on="pid")
    ok = idx[idx["status"] == "ok"]
    print(idx["status"].where(idx["status"] == "ok", "error").value_counts().to_string())
    print(ok[["lung_ml", "lung_z_mm", "clipped_frac", "lung_mean_hu", "sec"]]
          .describe(percentiles=[.01, .05, .5, .95, .99]).round(3).to_string())
    print("lung_touches_z_edge:", int(ok["lung_touches_z_edge"].sum()))

    arr = np.load(CACHE_NPY, mmap_mode="r")
    pick = pd.concat([ok[ok.event == 0].sample(4, random_state=0),
                      ok[ok.event == 1].sample(2, random_state=0)])
    fig, axes = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)))
    for r, (_, p) in enumerate(pick.iterrows()):
        v = np.asarray(arr[int(p["row"])])
        sl = [v[:, :, GRID[2] // 2].T, v[:, GRID[1] // 2, :].T, v[GRID[0] // 3, :, :].T]  # sagittal off-midline
        for c, (s, name) in enumerate(zip(sl, ["axial", "coronal", "sagittal"])):
            axes[r, c].imshow(s, cmap="gray", origin="lower", vmin=0, vmax=255)
            axes[r, c].set_title(f"{p['pid']} ev={p['event']} {name}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(f"S2 network inputs: {SPACING_MM} mm, grid {GRID}, 0 = outside lung mask")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "s2_inputs.png", dpi=110)
    print("wrote", FIGS / "s2_inputs.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--pids", help="comma-separated pids (smoke test)")
    ap.add_argument("--out-dir", default=str(CHUNK_DIR))
    a = ap.parse_args()

    if a.merge:
        return merge(a.n_tasks)
    if a.qc:
        return qc_figure()
    if a.pids:
        pids, name = a.pids.split(","), "smoke"
    elif a.task_id is not None:
        pids, name = task_pids(a.task_id, a.n_tasks), f"{a.task_id:03d}"
    else:
        ap.error("give --task-id, --pids, --merge or --qc")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{name}.csv").exists():
        print(f"{out / name}.csv exists -- skipping")
        return
    arr, df = run_pids(pids)
    np.save(out / f"{name}.npy", arr)
    df.to_csv(out / f"{name}.csv", index=False)  # written last = chunk complete
    print(f"wrote {out / name}.{{npy,csv}}: {len(df)} pids, "
          f"{(df['status'] == 'ok').sum()} ok, median {df['sec'].median():.1f} s/pid")


if __name__ == "__main__":
    main()
