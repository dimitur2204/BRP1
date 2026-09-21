"""Shared organ definitions for the student pipeline.

Reuses the mentor's TotalSegmentator label map and sternum-anchored anterior-
mediastinum (thymus proxy) geometry from code/extract_organ_embeddings.py and
code/prevascular_roi.py verbatim/adapted -- NOT reimplemented from scratch --
so our organ masks stay consistent with the discovery_pipeline's conventions
and any Stage 6 benchmark against it is a fair comparison.

We only pull in the pure numpy/scipy geometry, not the torch/3DINO parts of
extract_organ_embeddings.py (those files import torch/dinov2 at module level,
which we don't need until Stage 3's CNN).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

# ── merged 117-class TotalSegmentator label map ──────────────────────────────
# copied verbatim from code/extract_organ_embeddings.py (_PARTS/LABEL_NAMES)
_PARTS = [
    (0,  ["spleen","kidney_right","kidney_left","gallbladder","liver","stomach",
          "pancreas","adrenal_gland_right","adrenal_gland_left",
          "lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right",
          "lung_middle_lobe_right","lung_lower_lobe_right","esophagus","trachea",
          "thyroid_gland","small_bowel","duodenum","colon","urinary_bladder",
          "prostate","kidney_cyst_left","kidney_cyst_right"]),
    (24, ["sacrum","vertebrae_S1","vertebrae_L5","vertebrae_L4","vertebrae_L3",
          "vertebrae_L2","vertebrae_L1","vertebrae_T12","vertebrae_T11",
          "vertebrae_T10","vertebrae_T9","vertebrae_T8","vertebrae_T7",
          "vertebrae_T6","vertebrae_T5","vertebrae_T4","vertebrae_T3",
          "vertebrae_T2","vertebrae_T1","vertebrae_C7","vertebrae_C6",
          "vertebrae_C5","vertebrae_C4","vertebrae_C3","vertebrae_C2",
          "vertebrae_C1"]),
    (50, ["heart","aorta","pulmonary_vein","brachiocephalic_trunk",
          "subclavian_artery_right","subclavian_artery_left",
          "common_carotid_artery_right","common_carotid_artery_left",
          "brachiocephalic_vein_left","brachiocephalic_vein_right",
          "atrial_appendage_left","superior_vena_cava","inferior_vena_cava",
          "portal_vein_and_splenic_vein","iliac_artery_left","iliac_artery_right",
          "iliac_vena_left","iliac_vena_right"]),
    (68, ["humerus_left","humerus_right","scapula_left","scapula_right",
          "clavicula_left","clavicula_right","femur_left","femur_right",
          "hip_left","hip_right","spinal_cord","gluteus_maximus_left",
          "gluteus_maximus_right","gluteus_medius_left","gluteus_medius_right",
          "gluteus_minimus_left","gluteus_minimus_right","autochthon_left",
          "autochthon_right","iliopsoas_left","iliopsoas_right","brain","skull"]),
    (91, [f"rib_left_{i}" for i in range(1, 13)] +
         [f"rib_right_{i}" for i in range(1, 13)] +
         ["sternum","costal_cartilages"]),
]
LABEL_NAMES = {off + i + 1: name
               for off, names in _PARTS for i, name in enumerate(names)}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}

# ── the 7 organs swept in this project ───────────────────────────────────────
# Mix of: expected-positive (lungs, anterior_mediastinum), plausible
# (heart, aorta), immune/abdominal (spleen, liver), and a bony NEGATIVE
# CONTROL (sternum -- no biological reason to carry lung-cancer signal, so if
# it ranks high in Stage 4 that's a red flag for scanner/positioning leakage
# rather than real biology).
LUNG_LOBE_NAMES = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
]
ORGAN_SET = {
    "anterior_mediastinum": "custom",              # thymus proxy -- see mediastinum_mask()
    "lungs": [NAME_TO_LABEL[n] for n in LUNG_LOBE_NAMES],  # union of 5 TS lobes
    "heart": [NAME_TO_LABEL["heart"]],
    "aorta": [NAME_TO_LABEL["aorta"]],
    "liver": [NAME_TO_LABEL["liver"]],
    "spleen": [NAME_TO_LABEL["spleen"]],
    "sternum": [NAME_TO_LABEL["sternum"]],          # bony negative control
}

MARGIN_VOX = 8      # same margin as extract_organ_embeddings.py
MIN_VOX = 500       # same minimum-size gate as extract_organ_embeddings.py
AIR_HU = -1000.0    # background value for mask-zeroed crops (real air HU)

# HU display/normalization windows, same convention as
# code/render_saliency_overlays.py (soft-tissue WL40/WW400, lung WL-600/WW1500)
ST_LO, ST_HI = -160.0, 240.0
LU_LO, LU_HI = -1350.0, 150.0


def window_for(organ: str):
    return (LU_LO, LU_HI) if organ == "lungs" else (ST_LO, ST_HI)


def _find_axis(axcodes, a, b):
    for i, c in enumerate(axcodes):
        if c in (a, b):
            return i
    raise ValueError(f"no {a}/{b} axis in axcodes {axcodes}")


def sternum_per_slice_roi(stern, axcodes, vox,
                          lateral_mm=20.0, posterior_mm=30.0,
                          skip_inferior_fraction=1 / 4,
                          skip_superior_fraction=1 / 6):
    """Per-axial-slice ROI anchored to the sternum (anterior mediastinum).
    Copied from code/extract_organ_embeddings.py -- identical geometry."""
    lr = _find_axis(axcodes, 'L', 'R')
    ap = _find_axis(axcodes, 'A', 'P')
    si = _find_axis(axcodes, 'S', 'I')

    lat_vox = int(np.round(lateral_mm / vox[lr]))
    post_vox = int(np.round(posterior_mm / vox[ap]))
    high_ap_is_post = axcodes[ap] == 'P'

    roi = np.zeros(stern.shape, dtype=bool)
    slice_axes = [i for i in range(3) if i != si]
    lr_col = slice_axes.index(lr)
    ap_col = slice_axes.index(ap)

    si_vals = np.argwhere(stern.any(axis=tuple(i for i in range(3) if i != si)))[:, 0]
    depth = si_vals.max() - si_vals.min()
    if axcodes[si] == 'S':
        si_keep = range(int(si_vals.min() + skip_inferior_fraction * depth),
                        int(si_vals.max() - skip_superior_fraction * depth) + 1)
    else:
        si_keep = range(int(si_vals.min() + skip_superior_fraction * depth),
                        int(si_vals.max() - skip_inferior_fraction * depth) + 1)

    for z in si_keep:
        idx = [slice(None), slice(None), slice(None)]
        idx[si] = z
        stern_sl = stern[tuple(idx)]
        if not stern_sl.any():
            continue
        pts = np.argwhere(stern_sl)
        lr_lo = max(0, pts[:, lr_col].min() - lat_vox)
        lr_hi = min(stern.shape[lr] - 1, pts[:, lr_col].max() + lat_vox)
        if high_ap_is_post:
            ap_lo = pts[:, ap_col].min()
            ap_hi = min(stern.shape[ap] - 1, pts[:, ap_col].max() + post_vox)
        else:
            ap_hi = pts[:, ap_col].max()
            ap_lo = max(0, pts[:, ap_col].min() - post_vox)
        roi_sl = roi[tuple(idx)]
        fill = [slice(None), slice(None)]
        fill[lr_col] = slice(lr_lo, lr_hi + 1)
        fill[ap_col] = slice(ap_lo, ap_hi + 1)
        roi_sl[tuple(fill)] = True

    return roi


def mediastinum_mask(ct: np.ndarray, seg: np.ndarray, axcodes, vox) -> np.ndarray | None:
    """Anterior mediastinum (thymus proxy): sternum-anchored ROI, minus all TS
    structures, soft-tissue voxels only, largest connected component. Adapted
    from code/extract_organ_embeddings.py's compute_custom_masks (label 203)."""
    stern = seg == NAME_TO_LABEL["sternum"]
    if not stern.any():
        return None
    roi = sternum_per_slice_roi(stern, axcodes, vox)

    ap = _find_axis(axcodes, 'A', 'P')
    si = _find_axis(axcodes, 'S', 'I')
    high_ap_is_post = axcodes[ap] == 'P'
    ap_col_2d = ap if ap < si else ap - 1
    post_of_stern = np.zeros(seg.shape, dtype=bool)
    for z in range(seg.shape[si]):
        sl_idx = [slice(None)] * 3
        sl_idx[si] = z
        stern_sl = stern[tuple(sl_idx)]
        if not stern_sl.any():
            continue
        ap_coords = np.argwhere(stern_sl)[:, ap_col_2d]
        post_edge = int(ap_coords.max() if high_ap_is_post else ap_coords.min())
        post_sl = post_of_stern[tuple(sl_idx)]
        post_range = [slice(None), slice(None)]
        if high_ap_is_post:
            post_range[ap_col_2d] = slice(post_edge + 1, None)
        else:
            post_range[ap_col_2d] = slice(0, post_edge)
        post_sl[tuple(post_range)] = True

    ts_any = seg > 0
    soft_tissue = (ct >= -200) & (ct <= 300)
    mediastinum = roi & post_of_stern & ~ts_any & soft_tissue
    cc, n_cc = cc_label(mediastinum)
    if n_cc == 0:
        return None
    largest = np.argmax(np.bincount(cc.ravel())[1:]) + 1
    mediastinum = cc == largest
    return mediastinum if mediastinum.sum() >= MIN_VOX else None


def organ_mask(name: str, ct: np.ndarray, seg: np.ndarray, axcodes, vox) -> np.ndarray | None:
    """Boolean mask (native CT shape) for one of the 7 organs in ORGAN_SET."""
    spec = ORGAN_SET[name]
    if spec == "custom":
        return mediastinum_mask(ct, seg, axcodes, vox)
    mask = np.isin(seg, spec)
    return mask if mask.sum() >= MIN_VOX else None


def mask_zeroed_crop(ct: np.ndarray, mask: np.ndarray, margin_vox: int = MARGIN_VOX):
    """Bbox-crop CT to the mask + margin, set outside-mask voxels to real air
    HU (-1000). Native resolution, RAW HU (no [-1,1] normalization -- that's a
    modeling-time transform, deferred to Stage 3). Returns (crop, mask_crop,
    (lo, hi) bbox in the full-volume index space)."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, np.array(ct.shape))
    ct_crop = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    mask_crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    ct_crop[~mask_crop] = AIR_HU
    return ct_crop, mask_crop, (lo, hi)
