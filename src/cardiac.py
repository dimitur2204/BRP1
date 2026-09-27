"""Heart regions + hand-crafted cardiac features, shared by R1 (QC) and R2
(extraction) so what we look at is exactly what we extract.

All geometry works on a bbox crop around the heart (+ CROP_MARGIN_MM) for
speed, with physical-mm distances (distance_transform_edt with the voxel
sampling) because the CT grid is anisotropic (1x1x2 mm).

Regions (boolean arrays, crop space):
  heart        largest 3D connected component of TotalSegmentator label 51.
               Normal scans have one component (extra pieces < 0.1 mL), but R2
               outlier QC found e.g. pid 103665 with 34 + 20 mL detached
               "heart" blobs in the upper mediastinum (one at 314 HU). Dropped
               pieces are relabelled DROPPED_LABEL so they count as neither
               heart, unlabelled fat, nor calcium zone.
  calc_zone    heart dilated by CALC_DILATE_MM, restricted to voxels that are
               unlabelled (epicardial fat, where coronaries run -- TS has no
               coronary label) or cardiac labels. Excludes aorta, bones,
               esophagus etc., so their calcium can't leak in.
  calc         calc_zone voxels >= CALC_HU on the CT smoothed with a
               CALC_SMOOTH_MM Gaussian, kept only in 3D connected components
               >= CALC_MIN_VOX. The smoothing is essential: on sharp kernels
               (BONE, B50f, Philips C) raw noise SD is 45-55 HU, so 130 HU is
               ~1.6 SD above blood and R1 QC found 400-1,150 speckle "lesions"
               / 3-11 mL "calcium" per scan. With 0.7 mm smoothing, noise SD is
               9-24 HU on every kernel and volumes are plausible (0-0.8 mL).
               Applied uniformly to all scans (no per-kernel rules).
  core         heart voxels >= CORE_DEPTH_MM inside the surface. On
               non-contrast CT blood and myocardium have near-identical HU, so
               the HU SD here is mostly image noise -- a *measured* kernel/dose
               sharpness covariate for R3 harmonization.
  fat_shell    unlabelled voxels 0-FAT_SHELL_MM outside the heart mask with
               fat HU (FAT_HU_LO..FAT_HU_HI): pericardial/epicardial fat proxy.
               Lungs are TS-labelled, so "unlabelled" already excludes them.

The calcium measure is a *burden proxy*, not a clinical Agatston score: the
input is low-dose, non-gated, and resampled to 2 mm slices.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as cc_label

from organs import NAME_TO_LABEL, MIN_VOX

HEART = NAME_TO_LABEL["heart"]
# cardiac structures the calcium zone may overlap (besides heart + unlabelled)
CARDIAC_LABELS = [HEART, NAME_TO_LABEL["pulmonary_vein"], NAME_TO_LABEL["atrial_appendage_left"]]

CROP_MARGIN_MM = 15.0
CALC_DILATE_MM = 3.0
CALC_HU = 130.0
CALC_MIN_VOX = 3
CALC_SMOOTH_MM = 0.7
FAT_SHELL_MM = 10.0
CORE_DEPTH_MM = 5.0
FAT_HU_LO, FAT_HU_HI = -190.0, -30.0
DROPPED_LABEL = -1          # detached heart-label pieces (see `heart` above)
METAL_HU = 2000.0           # calcium stays well below this; wires/valves/leads exceed it


def heart_crop(ct: np.ndarray, seg: np.ndarray, vox) -> tuple | None:
    """Crop CT/seg to the heart (largest component) bbox + margin.

    Returns (ct_c, seg_c, (lo, hi), qc) or None if the heart mask is missing /
    below organs.MIN_VOX. seg_c is a copy with detached heart pieces set to
    DROPPED_LABEL. `qc` (prefix `qc_`) holds scan-quality fields that R3 uses
    for exclusions -- recorded here, thresholds decided later."""
    heart_all = seg == HEART
    if heart_all.sum() < MIN_VOX:
        return None
    cc, n = cc_label(heart_all)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    heart = cc == int(np.argmax(sizes))
    if heart.sum() < MIN_VOX:
        return None

    idx = np.argwhere(heart)
    m = np.ceil(CROP_MARGIN_MM / np.asarray(vox)).astype(int)
    lo = np.maximum(idx.min(0) - m, 0)
    hi = np.minimum(idx.max(0) + m + 1, np.array(seg.shape))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    seg_c = seg[sl].copy()
    seg_c[heart_all[sl] & ~heart[sl]] = DROPPED_LABEL

    vox_ml = float(np.prod(vox)) / 1000.0
    z = idx[:, 2]
    qc = {
        "qc_heart_n_components": int(n),
        "qc_heart_dropped_ml": float((heart_all.sum() - heart.sum()) * vox_ml),
        # heart cut off by the scan's z-extent -> incomplete heart
        "qc_heart_touches_z_edge": int(z.min() == 0 or z.max() == seg.shape[2] - 1),
        "qc_metal_voxels": int((ct[heart] >= METAL_HU).sum()),
    }
    return ct[sl], seg_c, (lo, hi), qc


def smooth_for_calc(ct_c: np.ndarray, vox) -> np.ndarray:
    return gaussian_filter(ct_c.astype(np.float32), sigma=CALC_SMOOTH_MM / np.asarray(vox, float))


def heart_regions(ct_c: np.ndarray, seg_c: np.ndarray, vox) -> dict:
    heart = seg_c == HEART
    dist_out = distance_transform_edt(~heart, sampling=vox)   # mm to heart surface
    dist_in = distance_transform_edt(heart, sampling=vox)
    unlabelled = seg_c == 0

    calc_zone = (dist_out <= CALC_DILATE_MM) & (unlabelled | np.isin(seg_c, CARDIAC_LABELS))
    calc = calc_zone & (smooth_for_calc(ct_c, vox) >= CALC_HU)
    cc, n = cc_label(calc)
    if n:
        sizes = np.bincount(cc.ravel())
        keep = np.flatnonzero(sizes >= CALC_MIN_VOX)
        calc = np.isin(cc, keep[keep > 0])

    fat_shell = ((dist_out > 0) & (dist_out <= FAT_SHELL_MM) & unlabelled
                 & (ct_c >= FAT_HU_LO) & (ct_c <= FAT_HU_HI))
    core = dist_in >= CORE_DEPTH_MM
    return {"heart": heart, "core": core, "calc_zone": calc_zone, "calc": calc, "fat_shell": fat_shell}


def cardiac_features(ct_c: np.ndarray, regions: dict, vox) -> dict:
    """Hand-crafted cardiac features (prefix `card_`). Volumes in mL."""
    vox_ml = float(np.prod(vox)) / 1000.0
    heart, calc, fat = regions["heart"], regions["calc"], regions["fat_shell"]
    hu_heart = ct_c[heart]
    hu_calc = smooth_for_calc(ct_c, vox)[calc]   # same image the calc mask was thresholded on
    n_lesions = cc_label(calc)[1] if calc.any() else 0
    return {
        "card_heart_volume_ml": heart.sum() * vox_ml,
        "card_heart_mean_hu": float(hu_heart.mean()),
        "card_heart_sd_hu": float(hu_heart.std()),   # partly anatomy, partly image noise
        "card_core_noise_sd_hu": float(ct_c[regions["core"]].std()) if regions["core"].any() else np.nan,
        "card_calc_volume_ml": calc.sum() * vox_ml,
        # density-weighted burden: sum of (HU - threshold) over calcium voxels, x voxel volume
        "card_calc_mass_proxy": float(np.clip(hu_calc - CALC_HU, 0, None).sum() * vox_ml),
        "card_calc_n_lesions": int(n_lesions),
        "card_calc_any": int(calc.any()),
        "card_fat_volume_ml": fat.sum() * vox_ml,
        "card_fat_mean_hu": float(ct_c[fat].mean()) if fat.any() else np.nan,
    }
