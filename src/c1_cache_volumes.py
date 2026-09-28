#!/usr/bin/env python3
"""C1 -- cache the CNN input volumes for every cohort patient, lungs + sternum.

For each pid in data/pid_lists/cnn_cohort_v1.csv the CT + TotalSegmentator
seg are loaded ONCE (canonical RAS orientation), and for each organ:

  1. mask-zeroed bbox crop at native 1x1x2 mm (organs.mask_zeroed_crop, the
     same crop the legacy pipeline cached).
  2. ISO input (the new recipe): mask-aware area-averaged resampling to an
     isotropic grid, then centre pad/crop to a FIXED PHYSICAL field of view:
        lungs    4 mm -> 96 x 72 x 96   (384 x 288 x 384 mm)
        sternum  2 mm -> 48 x 64 x 128  ( 96 x 128 x 256 mm)
     Unlike the legacy 64^3 resize, one voxel is the same number of mm for
     every patient, so organ size and shape are preserved (a big lung stays
     big). Area averaging (instead of point-sampling) avoids aliasing the CT
     noise when going 1 mm -> 4 mm. Stored as int16 HU; voxels outside the
     mask (or padding) hold SENTINEL (-2048), from which the training code
     rebuilds a separate mask channel -- so "outside the lung" can no longer
     be confused with "very low-density (emphysematous) lung", which the
     legacy -1000 HU fill could not tell apart.
  3. LEGACY input: the exact legacy stage3_cnn3d.resize_and_window (crop
     rounded to float16 as the old cache did -> legacy window -> [0,1] ->
     trilinear 64^3). Lets the old checkpoints be scored on the big cohort and
     the old recipe be retrained at scale.
  4. interpretable covariates from the native-resolution mask, used later to
     ask WHAT the CNN learned: volume (mL); lungs: LAA-950 (% voxels < -950
     HU, the standard emphysema index), mean lung density, Perc15 (15th
     percentile HU); sternum: mean / median HU (bone density proxy).
  5. geometry (bbox, resample offset) so Grad-CAM can be reprojected into CT
     space later without recomputing anything.

Run as a Slurm CPU array (src/submit_c1_cache.sh): each task writes a chunk
file, then `--merge` concatenates the chunks into, per organ,
  data/cnn_cache/{organ}_iso.npy        int16   [N, X, Y, Z]
  data/cnn_cache/{organ}_legacy64.npy   float16 [N, 64, 64, 64]
  data/cnn_cache/{organ}_index.csv      row i of the arrays <-> pid + covariates
and `--qc` draws figs/c1_inputs/{organ}_inputs.png.

Usage:
    env/.venv/bin/python src/c1_cache_volumes.py --pids 100002,100004 --out-dir /tmp/x   # smoke test
    sbatch src/submit_c1_cache.sh                       # array; task id from SLURM_ARRAY_TASK_ID
    env/.venv/bin/python src/c1_cache_volumes.py --merge
    env/.venv/bin/python src/c1_cache_volumes.py --qc
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from organs import ORGAN_SET, LEGACY_WINDOW, organ_mask, mask_zeroed_crop, load_patient

PROJECT = Path(__file__).resolve().parents[1]
COHORT_CSV = PROJECT / "data" / "pid_lists" / "cnn_cohort_v1.csv"
CACHE_DIR = PROJECT / "data" / "cnn_cache"
QC_DIR = PROJECT / "figs" / "c1_inputs"

SENTINEL = -2048                     # "outside mask / padding" in the int16 ISO arrays
ISO_SPACING_MM = {"lungs": 4.0, "sternum": 2.0}
ISO_SHAPE = {"lungs": (96, 72, 96), "sternum": (48, 64, 128)}
LEGACY_SIZE = 64


def legacy_input(crop: np.ndarray, organ: str) -> np.ndarray:
    """Exact copy of the legacy stage3_cnn3d.resize_and_window (incl. the
    float16 round-trip of the old on-disk cache)."""
    crop = crop.astype(np.float16).astype(np.float32)
    lo, hi = LEGACY_WINDOW[organ]
    crop = np.clip(crop, lo, hi)
    crop = (crop - lo) / (hi - lo)
    t = torch.from_numpy(crop.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(LEGACY_SIZE,) * 3, mode="trilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float16)


def iso_input(crop: np.ndarray, mask_crop: np.ndarray, vox: np.ndarray, organ: str):
    """Mask-aware area resample to ISO_SPACING_MM, centre pad/crop to ISO_SHAPE.
    Returns (int16 volume with SENTINEL outside the mask, offset) where
    `offset[d]` is the resampled-grid index that lands on output index 0
    (negative = padding)."""
    sp = ISO_SPACING_MM[organ]
    res_shape = [max(1, int(round(n * v / sp))) for n, v in zip(crop.shape, vox)]
    m = torch.from_numpy(mask_crop.astype(np.float32))[None, None]
    hu = torch.from_numpy(np.where(mask_crop, crop, 0).astype(np.float32))[None, None]
    m_r = F.interpolate(m, size=res_shape, mode="area")[0, 0]
    hu_r = F.interpolate(hu, size=res_shape, mode="area")[0, 0]
    inside = m_r >= 0.5
    vol_r = torch.where(inside, hu_r / m_r.clamp_min(1e-6), torch.tensor(float(SENTINEL)))
    vol_r = vol_r.round().clamp(SENTINEL, 3071).numpy().astype(np.int16)

    target = ISO_SHAPE[organ]
    out = np.full(target, SENTINEL, dtype=np.int16)
    offset, src, dst = [], [], []
    for n_res, n_out in zip(vol_r.shape, target):
        o = n_res // 2 - n_out // 2               # resampled index at output 0
        s0, s1 = max(o, 0), min(o + n_out, n_res)
        offset.append(o)
        src.append(slice(s0, s1))
        dst.append(slice(s0 - o, s1 - o))
    out[tuple(dst)] = vol_r[tuple(src)]
    return out, offset, res_shape


def covariates(ct: np.ndarray, mask: np.ndarray, vox: np.ndarray, organ: str) -> dict:
    hu = ct[mask]
    c = {"volume_ml": float(mask.sum() * np.prod(vox) / 1000.0)}
    if organ == "lungs":
        c["laa950_pct"] = float((hu < -950).mean() * 100)
        c["mld_hu"] = float(hu.mean())
        c["perc15_hu"] = float(np.percentile(hu, 15))
    else:
        c["mean_hu"] = float(hu.mean())
        c["median_hu"] = float(np.median(hu))
    return c


def process_pids(pids: list[str], out_path: Path) -> None:
    iso = {o: [] for o in ORGAN_SET}
    legacy = {o: [] for o in ORGAN_SET}
    rows = []
    t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            ct, seg, vox, _ = load_patient(pid)
        except Exception as exc:  # logged, not silently dropped
            print(f"  ERROR loading {pid}: {exc}", flush=True)
            rows += [{"pid": pid, "organ": o, "found": 0, "error": str(exc)[:200]} for o in ORGAN_SET]
            continue
        for organ in ORGAN_SET:
            mask = organ_mask(organ, seg)
            if mask is None:
                rows.append({"pid": pid, "organ": organ, "found": 0, "error": "mask < MIN_VOX"})
                continue
            crop, mask_crop, (lo, hi) = mask_zeroed_crop(ct, mask)
            vol, offset, res_shape = iso_input(crop, mask_crop, vox, organ)
            iso[organ].append(vol)
            legacy[organ].append(legacy_input(crop, organ))
            row = {"pid": pid, "organ": organ, "found": 1, "error": "",
                   "vox_x": vox[0], "vox_y": vox[1], "vox_z": vox[2],
                   **{f"bbox_lo_{a}": int(v) for a, v in zip("xyz", lo)},
                   **{f"bbox_hi_{a}": int(v) for a, v in zip("xyz", hi)},
                   **{f"res_{a}": int(v) for a, v in zip("xyz", res_shape)},
                   **{f"iso_off_{a}": int(v) for a, v in zip("xyz", offset)},
                   "iso_clipped": int(any(o > 0 for o in offset)),
                   **covariates(ct, mask, vox, organ)}
            rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(pids):
            el = time.time() - t0
            print(f"  [{i+1}/{len(pids)}] {el:.0f}s, {el/(i+1):.1f}s/pid", flush=True)

    arrays = {}
    for organ in ORGAN_SET:
        arrays[f"{organ}_iso"] = (np.stack(iso[organ]) if iso[organ]
                                  else np.zeros((0, *ISO_SHAPE[organ]), np.int16))
        arrays[f"{organ}_legacy64"] = (np.stack(legacy[organ]) if legacy[organ]
                                       else np.zeros((0,) + (LEGACY_SIZE,) * 3, np.float16))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"wrote {out_path} (+ .csv)", flush=True)


def merge(cache_dir: Path) -> None:
    chunks = sorted((cache_dir / "chunks").glob("chunk_*.npz"))
    cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    print(f"merging {len(chunks)} chunks")
    for organ in ORGAN_SET:
        idx_parts, iso_parts, leg_parts = [], [], []
        for ch in chunks:
            meta = pd.read_csv(ch.with_suffix(".csv"), dtype={"pid": str})
            with np.load(ch) as z:
                iso_parts.append(z[f"{organ}_iso"])
                leg_parts.append(z[f"{organ}_legacy64"])
            idx_parts.append(meta[meta["organ"] == organ])
        idx = pd.concat(idx_parts, ignore_index=True)
        found = idx[idx["found"] == 1].reset_index(drop=True)
        iso = np.concatenate(iso_parts)
        leg = np.concatenate(leg_parts)
        assert len(found) == len(iso) == len(leg), (organ, len(found), len(iso), len(leg))
        found = found.merge(cohort[["pid", "time", "event", "split", "kernel_group",
                                    "manufacturer", "in_subset_v1"]], on="pid", how="left")
        np.save(cache_dir / f"{organ}_iso.npy", iso)
        np.save(cache_dir / f"{organ}_legacy64.npy", leg)
        found.to_csv(cache_dir / f"{organ}_index.csv", index=False)
        missing = set(cohort["pid"]) - set(found["pid"])
        idx[idx["found"] == 0].to_csv(cache_dir / f"{organ}_missing.csv", index=False)
        print(f"  {organ}: {len(found)} cached, {len(missing)} missing "
              f"({int(cohort[cohort['pid'].isin(missing)]['event'].sum())} events), "
              f"iso {iso.shape} {iso.nbytes/1e9:.1f} GB, "
              f"FOV-clipped {int(found['iso_clipped'].sum())}")
        print("   ", found.groupby("split")["event"].agg(n="size", events="sum").to_dict("index"))


def qc(cache_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from organs import WINDOW

    out_dir.mkdir(parents=True, exist_ok=True)
    for organ in ORGAN_SET:
        idx = pd.read_csv(cache_dir / f"{organ}_index.csv", dtype={"pid": str})
        iso = np.load(cache_dir / f"{organ}_iso.npy", mmap_mode="r")
        leg = np.load(cache_dir / f"{organ}_legacy64.npy", mmap_mode="r")
        # 2 soft-kernel + 2 sharp-kernel patients, one event and one not each
        picks = []
        for kg in ("soft", "sharp"):
            for ev in (1, 0):
                sub = idx[(idx["kernel_group"] == kg) & (idx["event"] == ev)]
                if len(sub):
                    picks.append(sub.index[len(sub) // 2])
        lo, hi = WINDOW[organ]
        leg_lo, leg_hi = LEGACY_WINDOW[organ]
        fig, axes = plt.subplots(len(picks), 5, figsize=(19, 3.8 * len(picks)))
        for r, i in enumerate(picks):
            v = np.asarray(iso[i]).astype(np.float32)
            inside = v != SENTINEL
            disp = np.where(inside, v, lo)
            zc = v.shape[2] // 2
            # 2nd view: coronal for lungs; sagittal for the (tilted, thin) sternum
            view = "coronal" if organ == "lungs" else "sagittal"
            cut = (lambda a, c: a[:, c, :]) if organ == "lungs" else (lambda a, c: a[c, :, :])
            c2 = v.shape[1] // 2 if organ == "lungs" else v.shape[0] // 2
            row = idx.loc[i]
            ax = axes[r]
            ax[0].imshow(np.rot90(disp[:, :, zc]), cmap="gray", vmin=lo, vmax=hi)
            ax[0].contour(np.rot90(inside[:, :, zc]), levels=[0.5], colors="red", linewidths=0.6)
            ax[0].set_title(f"pid {row.pid} ev={row.event} {row.kernel_group}\n"
                            f"NEW axial {v.shape}, {ISO_SPACING_MM[organ]:.0f} mm iso", fontsize=8)
            ax[1].imshow(np.rot90(cut(disp, c2)), cmap="gray", vmin=lo, vmax=hi)
            ax[1].contour(np.rot90(cut(inside, c2)), levels=[0.5], colors="red", linewidths=0.6)
            ax[1].set_title(f"NEW {view} (true aspect)", fontsize=8)
            lv = np.asarray(leg[i]).astype(np.float32)
            ax[2].imshow(np.rot90(lv[:, :, 32]), cmap="gray", vmin=0, vmax=1)
            ax[2].set_title(f"LEGACY axial 64^3, window [{leg_lo:.0f},{leg_hi:.0f}]", fontsize=8)
            ax[3].imshow(np.rot90(cut(lv, 32)), cmap="gray", vmin=0, vmax=1)
            ax[3].set_title(f"LEGACY {view} (squashed to a cube)", fontsize=8)
            ax[4].hist(v[inside], bins=120, color="gray")
            for x, c, lab in ((lo, "tab:green", "new window"), (hi, "tab:green", None),
                              (leg_lo, "tab:red", "legacy window"), (leg_hi, "tab:red", None)):
                ax[4].axvline(x, color=c, ls="--", lw=1, label=lab)
            ax[4].legend(fontsize=7)
            sat = (v[inside] > leg_hi).mean() * 100
            ax[4].set_title(f"in-mask HU; {sat:.0f}% of voxels > legacy upper bound", fontsize=8)
            for a in ax[:4]:
                a.axis("off")
        fig.suptitle(f"C1 input QC -- {organ}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = out_dir / f"{organ}_inputs.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--pids", default="", help="comma-separated pids (smoke test) instead of a cohort chunk")
    p.add_argument("--out-dir", default=str(CACHE_DIR))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--qc", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))

    if args.merge:
        merge(out_dir)
        return
    if args.qc:
        qc(out_dir, QC_DIR)
        return
    if args.pids:
        pids = args.pids.split(",")
        tag = "smoke"
    else:
        cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
        pids = np.array_split(cohort["pid"].to_numpy(), args.n_tasks)[args.task_id].tolist()
        tag = f"{args.task_id:03d}"
    print(f"task {args.task_id}/{args.n_tasks}: {len(pids)} pids", flush=True)
    process_pids(pids, out_dir / "chunks" / f"chunk_{tag}.npz")


if __name__ == "__main__":
    main()
