"""Shared organ definitions for the lungs/sternum 3D CNN.

Reuses the mentor's TotalSegmentator label map (code/extract_organ_embeddings.py)
verbatim, so our masks stay consistent with the discovery_pipeline's
conventions. Only two organs are modelled:

  lungs    union of the 5 TotalSegmentator lobe labels -- POSITIVE control
           (lung cancer grows here; emphysema/density are known risk factors)
  sternum  TotalSegmentator sternum label -- NEGATIVE control (no biological
           route to lung cancer; any "signal" is a red flag for age/sex/scanner
           confounding)
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path("/faststorage/project/aura_thymus")
CT_DIR = REPO / "derived" / "ct_1x1x2mm"
SEG_DIR = REPO / "derived" / "totalseg_fullres"

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

LUNG_LOBE_NAMES = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
]
ORGAN_SET = {
    "lungs": [NAME_TO_LABEL[n] for n in LUNG_LOBE_NAMES],  # union of 5 TS lobes
    "sternum": [NAME_TO_LABEL["sternum"]],                  # bony negative control
}

MARGIN_VOX = 8      # same margin as extract_organ_embeddings.py
MIN_VOX = 500       # same minimum-size gate as extract_organ_embeddings.py
AIR_HU = -1000.0    # background value for mask-zeroed crops (real air HU)

# HU windows (lo, hi) used to rescale to [0, 1] before the CNN.
#   lungs:   lung window WL-600/WW1500 (code/render_saliency_overlays.py)
#   sternum: bone window WL400/WW1800. The legacy pipeline used the
#            soft-tissue window [-160, 240] here, which saturates all bone
#            (>240 HU) -- LEGACY_WINDOW keeps that only to reproduce the old
#            model exactly.
WINDOW = {"lungs": (-1350.0, 150.0), "sternum": (-500.0, 1300.0)}
LEGACY_WINDOW = {"lungs": (-1350.0, 150.0), "sternum": (-160.0, 240.0)}


def window_for(organ: str):
    return WINDOW[organ]


def load_patient(pid: str, yr: int = 0):
    """Canonical (RAS) CT + TotalSegmentator seg for one scan, plus the CT image
    (for its affine) and voxel size in mm."""
    ct_img = nib.as_closest_canonical(nib.load(CT_DIR / f"{pid}_yr{yr}.nii.gz"))
    seg_img = nib.as_closest_canonical(nib.load(SEG_DIR / f"{pid}_yr{yr}" / "seg.nii.gz"))
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    seg = np.asarray(seg_img.dataobj, dtype=np.int16)
    vox = np.array(ct_img.header.get_zooms()[:3], dtype=np.float64)
    if ct.shape != seg.shape:
        raise ValueError(f"{pid}: CT shape {ct.shape} != seg shape {seg.shape}")
    return ct, seg, vox, ct_img


def organ_mask(name: str, seg: np.ndarray) -> np.ndarray | None:
    """Boolean mask (native CT shape) for one organ in ORGAN_SET."""
    mask = np.isin(seg, ORGAN_SET[name])
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
