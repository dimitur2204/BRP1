"""Shared paths and constants for the NLST lungs 3D CNN pipeline.

Every stage script (s1_ ... s5_) imports from here, so the input definition
(which voxels the network sees, at what resolution and intensity scaling)
is written down exactly once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── source data (read-only, shared project tree) ─────────────────────────────
ROOT = Path("/faststorage/project/aura_thymus")
CT_DIR = ROOT / "derived" / "ct_1x1x2mm"             # {pid}_yr0.nii.gz, int16 HU, 1x1x2 mm
SEG_DIR = ROOT / "derived" / "totalseg_fullres"      # {pid}_yr0/seg.nii.gz, same grid as CT
SCAN_LIST = ROOT / "experiments" / "lcrisk_discovery" / "data" / "manifest.csv"  # pid -> series of the CT on disk
SERIES_CSV = ROOT / "nlst.csv"                       # IDC series metadata (SeriesDescription codes)

# ── this project ─────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
PRSN_CSV = DATA / "external" / "nlst_780" / "nlst_780" / "nlst_780_prsn_idc_20210527.csv"
CANDIDATES_CSV = DATA / "candidates.csv"             # s1 output
CACHE_DIR = DATA / "cache"                           # s2 output
CACHE_NPY = CACHE_DIR / "lungs_u8.npy"               # (N, X, Y, Z) uint8
CACHE_INDEX = CACHE_DIR / "lungs_index.csv"          # row -> pid + QC
COHORT_CSV = DATA / "cohort.csv"                     # s3 output: the training cohort
RUNS_DIR = PROJECT / "runs"
FIGS = PROJECT / "figs"

# ── input definition ─────────────────────────────────────────────────────────
# TotalSegmentator "total" task labels 10-14 = the five lung lobes.
LUNG_LABELS = (10, 11, 12, 13, 14)

# Fixed physical grid: 2.5 mm isotropic, 144 x 128 x 128 voxels
# = 360 (L-R) x 320 (A-P) x 320 (S-I) mm, centred on the lung bounding box.
# Covers the largest lungs seen in a 30-patient sample (353 x 286 x 308 mm)
# without rescaling, so organ size and shape are preserved.
SPACING_MM = 2.5
GRID = (144, 128, 128)          # RAS order (x, y, z)

# Lung mask is dilated by this many output voxels (2 x 2.5 mm = 5 mm) so that
# juxtapleural nodules that TotalSegmentator leaves out of the lobes are kept.
MASK_DILATE_VOX = 2

# Intensity: lung window WL -600 / WW 1500 -> [-1350, 150] HU, stored as uint8.
#   0        = outside the (dilated) lung mask
#   1 .. 255 = HU inside the mask, linearly mapped from [-1350, 150]
# Outside-mask voxels therefore get a value no real lung voxel can have
# (air at -1000 HU maps to ~60), so the network can tell "not lung" from
# emphysema.
HU_LO, HU_HI = -1350.0, 150.0
AIR_HU = -1024.0                # fill value outside the scanned volume

# ── cohort / split ───────────────────────────────────────────────────────────
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 20261005
MIN_LUNG_Z_MM = 200.0           # lungs shorter than this are not fully scanned
MIN_LUNG_ML = 1500.0            # implausibly small lung mask -> segmentation failure
MAX_CLIPPED_FRAC = 0.02         # max share of lung-mask voxels outside the fixed grid


def encode_hu(hu: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """HU volume + bool mask -> uint8 network input (see the table above)."""
    v = np.clip((hu - HU_LO) / (HU_HI - HU_LO), 0.0, 1.0)
    out = (1.0 + np.rint(v * 254.0)).astype(np.uint8)
    out[~mask] = 0
    return out
