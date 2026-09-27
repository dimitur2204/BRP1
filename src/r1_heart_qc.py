#!/usr/bin/env python3
"""R1 -- visual QC of the heart mask + the cardiac regions R2 will extract.

For a small sample spread over manufacturer x kernel group x event (plus the
thickest-slice scan in the cohort), one figure per patient:
  row 1  axial / coronal / sagittal through the heart centroid, soft-tissue
         window, heart contour (red), calcium-search zone (dashed yellow),
         calcium >= 130 HU (magenta fill), pericardial fat shell (green fill)
  row 2  axial slice with the most calcium (or centroid if none), zoomed;
         in-mask HU histogram; the cardiac feature values as text
Also writes figs/r1_heart_qc/r1_qc_summary.csv (one row per sampled patient).

What to look for: mask on the heart (not spilling into liver/lung), coronary
calcium landing inside the yellow zone, fat shell hugging the heart without
eating mediastinal vessels, and whether sharp kernels produce speckle
"calcium" (noise) that CALC_MIN_VOX fails to suppress.

Runs locally (10 patients, ~1 min). Usage:
    env/.venv/bin/python src/r1_heart_qc.py [--n-per-cell 1]
    env/.venv/bin/python src/r1_heart_qc.py --pids 123 456 --out-subdir outliers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from cardiac import heart_crop, heart_regions, cardiac_features, CALC_HU
from organs import ST_LO, ST_HI
from stage2_visualize_masks import load_patient

PROJECT = Path(__file__).resolve().parents[1]
COHORT_CSV = PROJECT / "data" / "pid_lists" / "heart_cohort_v1.csv"
OUT_DIR = PROJECT / "figs" / "r1_heart_qc"
SEED = 0

MAGENTA = ListedColormap(["none", "magenta"])
GREEN = ListedColormap(["none", "limegreen"])


def pick_sample(cohort: pd.DataFrame, n_per_cell: int) -> pd.DataFrame:
    """n per (manufacturer, kernel_group) cell, alternating event 1/0, plus the
    thickest-slice scan (worst case for texture/calcium)."""
    rng = np.random.default_rng(SEED)
    picks = []
    for i, (_, g) in enumerate(cohort[cohort.kernel_group != "other"]
                                .groupby(["manufacturer", "kernel_group"])):
        want_event = i % 2
        pool = g[g.event == want_event]
        pool = pool if len(pool) else g
        picks.append(pool.iloc[rng.choice(len(pool), size=min(n_per_cell, len(pool)), replace=False)])
    picks.append(cohort.nlargest(1, "slice_thickness_mm"))
    return pd.concat(picks).drop_duplicates("pid")


def overlay(ax, img2d, regions2d, aspect, title):
    ax.imshow(img2d, cmap="gray", vmin=ST_LO, vmax=ST_HI, aspect=aspect)
    ax.imshow(regions2d["fat_shell"], cmap=GREEN, alpha=0.45, aspect=aspect, interpolation="nearest")
    ax.imshow(regions2d["calc"], cmap=MAGENTA, alpha=0.9, aspect=aspect, interpolation="nearest")
    ax.contour(regions2d["heart"], levels=[0.5], colors="red", linewidths=0.9)
    ax.contour(regions2d["calc_zone"], levels=[0.5], colors="yellow", linewidths=0.6, linestyles="--")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_patient(row) -> dict:
    pid = row.pid
    ct, seg, axcodes, vox = load_patient(pid)   # RAS canonical, same as stage 2
    crop = heart_crop(ct, seg, vox)
    if crop is None:
        print(f"{pid}: heart mask missing or < MIN_VOX -- skipped")
        return {"pid": pid, "heart_ok": 0}
    ct_c, seg_c, (lo, hi), qc = crop
    reg = heart_regions(ct_c, seg_c, vox)
    feats = cardiac_features(ct_c, reg, vox)

    cx, cy, cz = np.round(np.argwhere(reg["heart"]).mean(0)).astype(int)
    z_aspect = vox[2] / vox[0]
    ax_sl = lambda a, z: np.rot90(a[:, :, z])
    cor_sl = lambda a, y: np.rot90(a[:, y, :])
    sag_sl = lambda a, x: np.rot90(a[x, :, :])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(f"pid {pid}  event={row.event}  {row.manufacturer} {row.model}  "
                 f"kernel={row.kernel} ({row.kernel_group})  slice={row.slice_thickness_mm}mm",
                 fontsize=11)
    overlay(axes[0, 0], ax_sl(ct_c, cz), {k: ax_sl(v, cz) for k, v in reg.items()}, 1, f"axial z={cz + lo[2]}")
    overlay(axes[0, 1], cor_sl(ct_c, cy), {k: cor_sl(v, cy) for k, v in reg.items()}, z_aspect, "coronal")
    overlay(axes[0, 2], sag_sl(ct_c, cx), {k: sag_sl(v, cx) for k, v in reg.items()}, z_aspect, "sagittal")

    zc = int(np.argmax(reg["calc"].sum((0, 1)))) if reg["calc"].any() else cz
    overlay(axes[1, 0], ax_sl(ct_c, zc), {k: ax_sl(v, zc) for k, v in reg.items()}, 1,
            f"axial, most calcium (z={zc + lo[2]})" if reg["calc"].any() else "axial (no calcium found)")

    ax = axes[1, 1]
    hu = ct_c[reg["heart"]]
    ax.hist(np.clip(hu, -300, 400), bins=140, color="gray")
    ax.axvline(CALC_HU, color="magenta", ls="--", lw=0.8, label=f"{CALC_HU:.0f} HU")
    ax.set_yscale("log")
    ax.set_xlabel("HU inside heart mask (clipped to [-300, 400])")
    ax.set_title(f"median {np.median(hu):.0f} HU, {(hu < -100).mean():.1%} of voxels < -100 HU", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0, 1, "\n".join(f"{k[5:]:<20s} {v:10.2f}" for k, v in feats.items()),
            va="top", family="monospace", fontsize=9, transform=ax.transAxes)
    ax.text(0, 0.02, "red: heart mask   yellow--: calcium zone (+3mm)\n"
                     "magenta: calcium >=130HU   green: fat shell (0-10mm)",
            fontsize=8, transform=ax.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / f"{pid}_heart_qc.png", dpi=110)
    plt.close(fig)
    return {"pid": pid, "heart_ok": 1, "event": row.event, "manufacturer": row.manufacturer,
            "kernel": row.kernel, "kernel_group": row.kernel_group,
            "slice_thickness_mm": row.slice_thickness_mm,
            "frac_below_-100HU": float((hu < -100).mean()), **qc, **feats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=1)
    ap.add_argument("--pids", nargs="+", default=None,
                    help="plot these pids instead of the default sample (e.g. R2 outliers)")
    ap.add_argument("--out-subdir", default="", help="subfolder of figs/r1_heart_qc/")
    args = ap.parse_args()
    global OUT_DIR
    OUT_DIR = OUT_DIR / args.out_subdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    sample = (cohort[cohort.pid.isin(args.pids)] if args.pids
              else pick_sample(cohort, args.n_per_cell))
    rows = []
    for row in sample.itertuples():
        rows.append(plot_patient(row))
        print(f"{row.pid}: done")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "r1_qc_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 2):
        print(out.drop(columns=["heart_ok"]))


if __name__ == "__main__":
    main()
