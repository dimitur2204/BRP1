#!/usr/bin/env python3
"""Stage 2 -- per-organ masking + visualization for a handful of subset patients.

For each sample patient and each of the 7 organs in organs.ORGAN_SET, shows:
  left column  -- context: the whole axial slice at the mask's centroid, with
                  the organ's mask contour overlaid (is the mask in the right
                  place, at the right scale?)
  right column -- crop: the mask-zeroed bounding-box crop at native
                  resolution (exactly what Stage 3's CNN will read, modulo
                  the final resize -- that's a modeling choice made in Stage 3)

Reuses the mentor's HU display windows (code/render_saliency_overlays.py):
soft-tissue WL40/WW400 for everything except "lungs", which uses the lung
window WL-600/WW1500 (a soft-tissue window would show nothing but noise
inside aerated lung).

Does NOT resize crops or touch torch -- that's deferred to Stage 3, which
decides the CNN's input size. This stage only proves the masks are correct.

Usage:
    env/.venv/bin/python src/stage2_visualize_masks.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from organs import ORGAN_SET, organ_mask, mask_zeroed_crop, window_for

REPO = Path("/faststorage/project/aura_thymus")
CT_DIR = REPO / "derived" / "ct_1x1x2mm"
SEG_DIR = REPO / "derived" / "totalseg_fullres"

PROJECT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
OUT_DIR = PROJECT / "figs" / "stage2_masks"


# same 5 sample patients as Stage 1
SAMPLE_PIDS = ["132386", "133201", "214994", "208196", "110999"]


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


def plot_patient(pid: str, event: int | None) -> Path:
    ct, seg, axcodes, vox = load_patient(pid)

    organ_names = list(ORGAN_SET.keys())
    fig, axes = plt.subplots(len(organ_names), 2, figsize=(8, 3.4 * len(organ_names)))
    title = f"pid {pid}" + (f"  event={event}" if event is not None else "")
    fig.suptitle(title, fontsize=13, y=0.995)

    for row, name in enumerate(organ_names):
        lo, hi = window_for(name)
        ax_ctx, ax_crop = axes[row, 0], axes[row, 1]

        mask = organ_mask(name, ct, seg, axcodes, vox)
        if mask is None:
            for ax in (ax_ctx, ax_crop):
                ax.text(0.5, 0.5, "mask not found\n(< MIN_VOX or missing)",
                        ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            ax_ctx.set_ylabel(name, fontsize=10)
            continue

        # context: whole axial slice at the mask's centroid z
        z = int(round(np.argwhere(mask).mean(axis=0)[2]))
        ax_ctx.imshow(np.rot90(ct[:, :, z]), cmap="gray", vmin=lo, vmax=hi)
        ax_ctx.contour(np.rot90(mask[:, :, z]), levels=[0.5], colors=["red"], linewidths=1.2)
        ax_ctx.set_title(f"{name}: context (axial z={z})", fontsize=9)
        ax_ctx.axis("off")

        # crop: mask-zeroed bbox at native resolution, mid axial slice of the crop
        crop, mask_crop, (bbox_lo, bbox_hi) = mask_zeroed_crop(ct, mask)
        cz = crop.shape[2] // 2
        ax_crop.imshow(np.rot90(crop[:, :, cz]), cmap="gray", vmin=lo, vmax=hi)
        sz_mm = (bbox_hi - bbox_lo) * vox
        ax_crop.set_title(
            f"{name}: crop {crop.shape} ({sz_mm[0]:.0f}x{sz_mm[1]:.0f}x{sz_mm[2]:.0f}mm)",
            fontsize=9)
        ax_crop.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    out = OUT_DIR / f"{pid}_yr0_masks.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events: dict[str, int] = {}
    with open(SUBSET_CSV, newline="") as f:
        for row in csv.DictReader(f):
            events[row["pid"]] = int(row["event"])

    for pid in SAMPLE_PIDS:
        out = plot_patient(pid, events.get(pid))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
