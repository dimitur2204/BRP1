#!/usr/bin/env python3
"""D1 -- DINOv2 views of the 2.5 mm lung volumes (dataset for ext/DINOv2-3D-Med).

Used from configs/dino/nlst_2p5mm.yaml as dino_data.LungViews; run.py imports
the repo as the `project` package first, so the repo's transforms are available.

Per scan (a row of data/ssl_scans.csv; yr0 from the S2 cache, yr1/yr2 from s7):
  1. read the 144x128x128 uint8 volume (2.5 mm, RAS, 0 = outside the dilated
     lung mask) and scale to float [0, 1];
  2. random 128^3 cube (shift along L-R only, the one axis longer than 128);
  3. 2 global + N local views with the repo's DINOv2Augmentation3D recipe
     (affine +-10 deg / shear 0.1, random resized crop, histogram shift,
     Gaussian smoothing). Crop side = scale x 128 voxels, resized to the view
     size, so the effective voxel size stays near the 2.5 mm used for
     feature extraction (D4):
       global: side 0.60-1.00 x 128 = 77-128 vox -> 96^3, 2.0-3.3 mm, 190-320 mm
       local:  side 0.15-0.30 x 128 = 19-38 vox  -> 24^3, 2.0-4.0 mm, 48-96 mm
     (the repo default 96^3 input with global scale 0.5-1.0 would upsample to
     1.25-2.5 mm);
  4. lung voxels fill only ~20% of the grid (the mediastinum between the lungs
     is 0), so a view with < min_lung of its voxels in the lung is redrawn,
     up to max_tries times (the last draw is kept).
Returns (views, False), the batch format the repo's training_step expects.

QC (login node, env/.venv-dino; figure + view statistics + CPU time per scan):
    env/.venv-dino/bin/python src/dino_data.py --qc
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlst import CACHE_NPY, COHORT_CSV, FIGS, PROJECT, SSL_CACHE_NPY, SSL_SCANS_CSV, SSL_SPLITS

REPO = PROJECT / "ext" / "DINOv2-3D-Med"
LUNG_THRESHOLD = 0.02     # float value above which a voxel counts as lung (0 = outside the mask)


def import_repo() -> None:
    """Make the repo importable as `project` (run.py does this itself)."""
    if "project" not in sys.modules:
        sys.path.insert(0, str(REPO))
        from utils.imports import import_module_from_path
        import_module_from_path("project", str(REPO))


class LungViews(Dataset):
    def __init__(self, scans_csv: str = str(SSL_SCANS_CSV), cube: int = 128,
                 global_size: int = 96, global_scale=(0.6, 1.0),
                 local_size: int = 24, local_scale=(0.15, 0.3), num_local_views: int = 4,
                 min_lung: float = 0.3, max_tries: int = 10, limit: int | None = None):
        import_repo()
        from project.transforms.dinov2_aug import DINOv2Augmentation3D

        cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
        if Path(scans_csv).exists():
            scans = pd.read_csv(scans_csv, dtype={"pid": str})
        else:  # before the s7 merge: yr0 scans of train + val only
            print(f"{scans_csv} not found -- using yr0 train + val scans only", flush=True)
            c = cohort[cohort["split"].isin(SSL_SPLITS)]
            scans = pd.DataFrame({"scan_id": c["pid"] + "_yr0", "pid": c["pid"], "round": "yr0",
                                  "split": c["split"], "cache": "yr0", "row": c["row"]})
        test = set(cohort.query("split == 'test'")["pid"])
        if test & set(scans["pid"]):
            raise SystemExit("test patients in the pretraining scan list")
        if limit:
            scans = scans.sample(limit, random_state=0)
        self.scans = scans.reset_index(drop=True)
        self.cube, self.min_lung, self.max_tries = cube, min_lung, max_tries
        aug = DINOv2Augmentation3D(global_view_scale=list(global_scale),
                                   global_view_size=global_size,
                                   local_view_scale=list(local_scale),
                                   local_view_size=local_size,
                                   num_local_views=num_local_views)
        self.global_aug, self.local_aug, self.n_local = aug.global_aug, aug.local_aug, num_local_views
        self._caches = None   # opened per worker process (memmaps don't survive fork well)
        self.redraws = 0      # QC counter (per process)

    def __len__(self) -> int:
        return len(self.scans)

    def volume(self, i: int) -> torch.Tensor:
        if self._caches is None:
            self._caches = {"yr0": np.load(CACHE_NPY, mmap_mode="r")}
            if (self.scans["cache"] == "rounds").any():
                self._caches["rounds"] = np.load(SSL_CACHE_NPY, mmap_mode="r")
        s = self.scans.iloc[i]
        v = np.asarray(self._caches[s["cache"]][int(s["row"])], dtype=np.float32) / 255.0
        x0 = np.random.randint(0, v.shape[0] - self.cube + 1)
        v = v[x0:x0 + self.cube, :self.cube, :self.cube]
        return torch.from_numpy(np.ascontiguousarray(v))[None]   # (1, 128, 128, 128)

    def _view(self, aug, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.max_tries):
            v = aug(x.clone())
            v = v.as_tensor() if hasattr(v, "as_tensor") else v
            if (v > LUNG_THRESHOLD).float().mean() >= self.min_lung:
                return v
            self.redraws += 1
        return v

    def __getitem__(self, i: int):
        x = self.volume(i)
        views = [self._view(self.global_aug, x) for _ in range(2)]
        views += [self._view(self.local_aug, x) for _ in range(self.n_local)]
        return views, False


def qc(n: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    np.random.seed(0)
    ds = LungViews(limit=n)
    t, fracs, n_local = time.time(), [], ds.n_local
    items = [ds[i] for i in range(len(ds))]
    sec = (time.time() - t) / len(ds)
    for views, _ in items:
        fracs.append([float((v > LUNG_THRESHOLD).float().mean()) for v in views])
    fracs = np.array(fracs)
    n_views = len(ds) * (2 + n_local)
    print(f"{len(ds)} scans ({ds.scans['round'].value_counts().to_dict()}), "
          f"{sec:.2f} s CPU per scan (1 process)")
    print(f"view shapes: global {tuple(items[0][0][0].shape)}, local {tuple(items[0][0][2].shape)}")
    print(f"redraws: {ds.redraws} over {n_views} views (min_lung {ds.min_lung})")
    for name, f in [("global", fracs[:, :2]), ("local", fracs[:, 2:])]:
        print(f"{name} lung fraction: median {np.median(f):.2f}, p5 {np.percentile(f, 5):.2f}, "
              f"< min_lung {np.mean(f < ds.min_lung):.1%}")

    rows = min(4, len(items))
    fig, axes = plt.subplots(rows, 2 + n_local + 1, figsize=(2.2 * (3 + n_local), 2.4 * rows))
    for r in range(rows):
        x = ds.volume(r)[0]
        axes[r, 0].imshow(x[:, :, 64].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{ds.scans['scan_id'][r]} 128³", fontsize=7)
        for c, v in enumerate(items[r][0]):
            v = v[0]
            axes[r, c + 1].imshow(v[:, :, v.shape[2] // 2].T, cmap="gray", origin="lower",
                                  vmin=0, vmax=1)
            axes[r, c + 1].set_title(f"{'global' if c < 2 else 'local'} {fracs[r, c]:.2f}",
                                     fontsize=7)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("DINOv2 views (axial mid-slice; title = lung fraction)", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(FIGS / "d1_views.png"))
    a = ap.parse_args()
    if not a.qc:
        ap.error("give --qc")
    qc(a.n, Path(a.out))


if __name__ == "__main__":
    main()
