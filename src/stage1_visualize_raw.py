#!/usr/bin/env python3
"""Stage 1 -- visualize raw NLST CT volumes for a handful of subset patients.

Pure sanity check before any modeling: confirms orientation, voxel spacing,
and HU range look right for a few patients from data/pid_lists/subset_v1.csv,
before any masking or modeling touches the data (see Stage 2 for that).

HU display window reused from the mentor's code/render_saliency_overlays.py
(soft-tissue WL40/WW400 -> [-160, 240]) so figures stay visually comparable
to the rest of the project.

Usage:
    env/.venv/bin/python src/stage1_visualize_raw.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = Path("/faststorage/project/aura_thymus")
CT_DIR = REPO / "derived" / "ct_1x1x2mm"

PROJECT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
OUT_DIR = PROJECT / "figs" / "stage1_raw_ct"

# soft-tissue display window (HU): WL 40, WW 400 -> [-160, 240]
ST_LO, ST_HI = -160.0, 240.0

# 3 lung-cancer-positive + 2 negative patients from subset_v1.csv, picked by
# hand for a first look -- not a random sample, just "does the data look sane".
SAMPLE_PIDS = ["132386", "133201", "214994", "208196", "110999"]


def load_ct(pid: str, yr: int = 0):
    path = CT_DIR / f"{pid}_yr{yr}.nii.gz"
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # reorient to RAS+ so axis order is predictable
    data = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def plot_patient(pid: str, event: int | None) -> Path:
    data, spacing = load_ct(pid)
    x, y, z = (s // 2 for s in data.shape)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    title = f"pid {pid}  shape={data.shape}  spacing(mm)={tuple(round(s, 2) for s in spacing)}"
    if event is not None:
        title += f"  event={event}"
    fig.suptitle(title)

    axes[0].hist(data.ravel(), bins=100, range=(-1000, 1000))
    axes[0].set_title("HU histogram")
    axes[0].set_xlabel("Hounsfield units")
    axes[0].set_ylabel("voxel count")

    # RAS+ axes after as_closest_canonical: axis0=L->R, axis1=P->A, axis2=I->S
    panels = [
        ("sagittal (mid)", data[x, :, :]),
        ("coronal (mid)", data[:, y, :]),
        ("axial (mid)", data[:, :, z]),
    ]
    for ax, (label, sl) in zip(axes[1:], panels):
        ax.imshow(np.rot90(sl), cmap="gray", vmin=ST_LO, vmax=ST_HI)
        ax.set_title(label)
        ax.axis("off")

    fig.tight_layout()
    out = OUT_DIR / f"{pid}_yr0_overview.png"
    fig.savefig(out, dpi=130)
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
