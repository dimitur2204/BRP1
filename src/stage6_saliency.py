#!/usr/bin/env python3
"""Stage 6 -- Grad-CAM saliency maps for the Stage 3v2 3D CNN.

Answers "which voxels drove this prediction?" for the two organs that
matter most: `lungs` (the validated positive control -- Stage 5's KM curve
already showed real signal there, logrank p=0.037) and
`anterior_mediastinum` (the actual scientific target, the thymus proxy).

Method: standard Grad-CAM (Selvaraju et al. 2017) hooked on the last conv
layer of Simple3DCNN (features[12], the 64->128 Conv3d right before
AdaptiveAvgPool3d) -- no new model needs training, this only inspects the
checkpoints already saved by stage3_cnn3d.py.

The cached crop's bounding-box offset into the full CT is NOT stored
anywhere (mask_zeroed_crop's (lo, hi) return value is discarded at its only
call site, in cache_organ_crops.py) -- so it's recomputed here from the
original CT+seg for each selected patient, and checked against the cached
crop's own shape before trusting the reprojection.

Outputs, per selected patient:
  data/saliency/{organ}/{pid}_gradcam.nii.gz   full-CT-space, aligned with
                                                the original scan's affine
  figs/stage6_saliency/{organ}/{pid}_montage.png   CT + Grad-CAM overlay,
                                                    axial slice at peak
                                                    CAM energy
  figs/stage6_saliency/{organ}_overview.png    one montage of all selected
                                                patients (high-risk vs
                                                low-risk rows)
  data/stage6_saliency_manifest.csv            organ,pid,group,event,risk,
                                                paths -- so results are
                                                traceable, not just images

Runs on CPU or GPU (trivial compute either way -- 4 conv layers, 20
patients); submitted via sbatch anyway because it has to re-read the
*original* full-resolution CT+seg files from the shared filesystem for each
selected patient (see submit_stage6_saliency.sh).

Usage:
    env/.venv/bin/python src/stage6_saliency.py --organ default
    env/.venv/bin/python src/stage6_saliency.py --organ lungs --top-k 5
"""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cache_organ_crops import CT_DIR, load_patient
from organs import ORGAN_SET, _find_axis, mask_zeroed_crop, organ_mask, window_for
from stage3_cnn3d import Simple3DCNN, resize_and_window
from stage4_ranking import hanley_mcneil_se
from stage5_km_curve import risk_scores_for_test

PROJECT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT / "models"
SALIENCY_DIR = PROJECT / "data" / "saliency"
FIGS_DIR = PROJECT / "figs" / "stage6_saliency"
MANIFEST_CSV = PROJECT / "data" / "stage6_saliency_manifest.csv"
RESULTS3D_CSV = PROJECT / "data" / "stage3_3d_results.csv"

DEFAULT_ORGANS = ["lungs", "anterior_mediastinum"]
TOP_K_DEFAULT = 5


def low_confidence(organ: str) -> bool:
    """True if this organ's Stage 4 test-AUC 95% CI (Hanley-McNeil) spans
    0.5 -- i.e. the classifier isn't demonstrably better than chance, so its
    Grad-CAM should be captioned as such rather than presented at face value."""
    if not RESULTS3D_CSV.exists():
        return True
    rows = list(csv.DictReader(open(RESULTS3D_CSV)))
    row = next((r for r in rows if r["organ"] == organ), None)
    if row is None or not row.get("test_auc"):
        return True
    cm = ast.literal_eval(row["test_confusion"])
    n0 = cm[0][0] + cm[0][1]
    n1 = cm[1][0] + cm[1][1]
    auc = float(row["test_auc"])
    se = hanley_mcneil_se(auc, n1, n0)
    lo, hi = auc - 1.96 * se, auc + 1.96 * se
    return lo <= 0.5 <= hi


class GradCAM3D:
    """Hooks Simple3DCNN's last conv layer (features[12]) to compute
    Grad-CAM: CAM = ReLU(sum_k alpha_k * A_k), alpha_k = global-average-pooled
    gradient of the logit w.r.t. channel k's activation."""

    def __init__(self, model: Simple3DCNN):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer = model.features[12]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: [1,1,64,64,64] model input. Returns (cam [8,8,8] in [0,1], risk)."""
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        logit.sum().backward()
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)   # [1,128,1,1,1]
        cam = F.relu((weights * self.activations).sum(dim=1))        # [1,8,8,8]
        cam = cam[0].cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        risk = torch.sigmoid(logit).item()
        return cam, risk


def axial_slice(vol: np.ndarray, si: int, z: int) -> np.ndarray:
    """2D display slice, oriented anterior-up (same convention as the
    mentor's render_saliency_overlays.py). Assumes RAS-canonical order, i.e.
    si == 2 -- true for every array here since all loads go through
    nib.as_closest_canonical()."""
    idx = [slice(None)] * 3
    idx[si] = z
    return np.flipud(vol[tuple(idx)].T)


def render_panel(ax, ct_2d, cam_2d, lo_w, hi_w, cam_vmax, title):
    ax.imshow(ct_2d, cmap="gray", vmin=lo_w, vmax=hi_w)
    alpha = np.clip(cam_2d / (cam_vmax + 1e-8), 0, 1)
    ax.imshow(cam_2d, cmap="hot", vmin=0, vmax=cam_vmax, alpha=alpha)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def process_patient(organ: str, pid: str, event: int, risk_stage5: float,
                     group: str, crops_dir: Path, model: Simple3DCNN,
                     cam_fn: GradCAM3D, device: torch.device, low_conf: bool) -> dict:
    crop_native = np.load(crops_dir / organ / f"{pid}.npy").astype(np.float32)

    x = resize_and_window(crop_native, organ, device)[None, None].to(device)
    cam8, risk = cam_fn(x)

    cam_t = torch.from_numpy(cam8)[None, None].float()
    cam_native = F.interpolate(cam_t, size=crop_native.shape, mode="trilinear",
                                align_corners=False)[0, 0].numpy()

    ct, seg, axcodes, vox = load_patient(pid)
    mask = organ_mask(organ, ct, seg, axcodes, vox)
    _, _, (lo, hi) = mask_zeroed_crop(ct, mask)
    assert tuple(hi - lo) == crop_native.shape, (
        f"{organ}/{pid}: recomputed bbox {tuple(hi - lo)} != cached crop shape "
        f"{crop_native.shape} -- mask/crop logic has drifted since caching")

    full = np.zeros(ct.shape, dtype=np.float32)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = cam_native

    affine = nib.as_closest_canonical(nib.load(CT_DIR / f"{pid}_yr0.nii.gz")).affine
    nii_dir = SALIENCY_DIR / organ
    nii_dir.mkdir(parents=True, exist_ok=True)
    nii_path = nii_dir / f"{pid}_gradcam.nii.gz"
    nib.save(nib.Nifti1Image(full, affine=affine), nii_path)

    si = _find_axis(axcodes, "S", "I")
    energy = cam_native.sum(axis=tuple(a for a in range(3) if a != si))
    global_z = int(lo[si] + np.argmax(energy))

    lo_w, hi_w = window_for(organ)
    ct_2d = np.clip(axial_slice(ct, si, global_z), lo_w, hi_w)
    cam_2d = axial_slice(full, si, global_z)
    cam_vmax = max(float(np.percentile(full[full > 0], 99.5)), 1e-6) if full.max() > 0 else 1.0

    caption = f"{organ} | pid {pid} | {group} | true event={event} | risk={risk:.2f} | z={global_z}"
    if low_conf:
        caption += "\n(classifier CI spans chance -- interpret with caution)"

    png_dir = FIGS_DIR / organ
    png_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ct_2d, cmap="gray", vmin=lo_w, vmax=hi_w)
    axes[0].set_title("CT"); axes[0].axis("off")
    render_panel(axes[1], ct_2d, cam_2d, lo_w, hi_w, cam_vmax, "Grad-CAM overlay")
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    png_path = png_dir / f"{pid}_montage.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    return {
        "organ": organ, "pid": pid, "group": group, "event": event,
        "risk_stage5": round(risk_stage5, 4), "risk_gradcam": round(risk, 4),
        "low_confidence": low_conf,
        "nifti_path": str(nii_path.relative_to(PROJECT)),
        "png_path": str(png_path.relative_to(PROJECT)),
        # kept for the overview montage, not written to the manifest CSV
        "_ct_2d": ct_2d, "_cam_2d": cam_2d, "_cam_vmax": cam_vmax,
    }


def run_organ(organ: str, crops_dir: Path, device: torch.device, top_k: int) -> list[dict]:
    low_conf = low_confidence(organ)
    print(f"[{organ}] low_confidence={low_conf}", flush=True)

    df = risk_scores_for_test(organ, crops_dir, device)
    df = df.sort_values("risk", ascending=False)
    picks = pd.concat([
        df.head(top_k).assign(group="high_risk"),
        df.tail(top_k).assign(group="low_risk"),
    ], ignore_index=True)

    model = Simple3DCNN().to(device)
    state = torch.load(MODELS_DIR / f"{organ}_cnn3d.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    cam_fn = GradCAM3D(model)

    results = []
    for _, row in picks.iterrows():
        r = process_patient(organ, row["pid"], int(row["event"]), float(row["risk"]),
                             row["group"], crops_dir, model, cam_fn, device, low_conf)
        results.append(r)
        print(f"[{organ}] {row['group']:9s} pid={row['pid']} "
              f"risk={r['risk_gradcam']:.2f} event={r['event']}  wrote {r['png_path']}",
              flush=True)

    # per-organ overview: high-risk row on top, low-risk row on bottom
    fig, axes = plt.subplots(2, top_k, figsize=(3 * top_k, 8))
    for col, r in enumerate(results[:top_k]):
        render_panel(axes[0, col], r["_ct_2d"], r["_cam_2d"], *window_for(organ),
                     r["_cam_vmax"], f"pid {r['pid']}\nevent={r['event']} risk={r['risk_gradcam']:.2f}")
    for col, r in enumerate(results[top_k:]):
        render_panel(axes[1, col], r["_ct_2d"], r["_cam_2d"], *window_for(organ),
                     r["_cam_vmax"], f"pid {r['pid']}\nevent={r['event']} risk={r['risk_gradcam']:.2f}")
    title = f"Stage 6 Grad-CAM overview: {organ}  (top: high-risk, bottom: low-risk)"
    if low_conf:
        title += "\nclassifier CI spans chance -- interpret with caution"
    fig.suptitle(title, y=0.995)
    # h_pad=3.0: without it, row 2's per-panel titles overlap row 1's images
    # (found when reviewing the first real run's anterior_mediastinum_overview.png)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
    fig.savefig(FIGS_DIR / f"{organ}_overview.png", dpi=130)
    plt.close(fig)
    print(f"[{organ}] wrote {FIGS_DIR / f'{organ}_overview.png'}", flush=True)

    for r in results:
        del r["_ct_2d"], r["_cam_2d"], r["_cam_vmax"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ", default="default",
                         choices=["default", "all"] + list(ORGAN_SET),
                         help="'default' = lungs + anterior_mediastinum only")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT,
                         help="highest- and lowest-risk test patients to render, per organ")
    parser.add_argument("--crops-dir", default=str(PROJECT / "data" / "organ_crops"),
                         help="override to read from a local $TMPDIR staging copy")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}", flush=True)

    if args.organ == "all":
        organs = list(ORGAN_SET)
    elif args.organ == "default":
        organs = DEFAULT_ORGANS
    else:
        organs = [args.organ]

    crops_dir = Path(args.crops_dir)
    all_results = []
    for organ in organs:
        all_results.extend(run_organ(organ, crops_dir, device, args.top_k))

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"wrote {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
