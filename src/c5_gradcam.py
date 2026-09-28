#!/usr/bin/env python3
"""C5 -- Grad-CAM for the headline lungs / sternum CNNs (final set, "main", seed 1).

Grad-CAM (Selvaraju et al. 2017) on the last conv layer (CNN3D.features[12]):
  alpha_k = spatial mean of d(eta)/d(A_k),  CAM = ReLU(sum_k alpha_k A_k)
i.e. "which regions pushed THIS patient's risk score up". It is coarse by
construction (the last conv map is 12x9x12 for lungs, 6x8x16 for sternum)
and shows what the model's score correlates with, not where a tumour is.

Reprojection into CT space uses the geometry c1 stored per patient (no
recomputation): CAM -> input grid (trilinear) -> undo the centre pad/crop
(iso_off_*) into the resampled grid (res_*) -> resample to the native bbox
(bbox_lo/hi) -> paste into a zero volume with the canonical CT's affine.

For the 5 highest- and 5 lowest-scored TEST patients per organ writes
  data/saliency/c5_{organ}/{pid}_gradcam.nii.gz   (opens aligned with the CT)
  figs/c5_gradcam/{organ}/{pid}.png               CT + overlay, axial + sagittal
  figs/c5_gradcam/{organ}_overview.png            high-risk row vs low-risk row
  data/c5_gradcam_manifest.csv

Usage:
    env/.venv/bin/python src/c5_gradcam.py [--organ lungs] [--top-k 5] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cnn3d import CNN3D, load_organ, to_input, CACHE_DIR, RUNS_DIR, PROJECT, SENTINEL
from organs import WINDOW, load_patient

SAL_DIR = PROJECT / "data" / "saliency"
FIG_DIR = PROJECT / "figs" / "c5_gradcam"
MANIFEST = PROJECT / "data" / "c5_gradcam_manifest.csv"


class GradCAM3D:
    def __init__(self, model: CNN3D):
        self.model = model
        layer = model.features[12]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def __call__(self, x: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        eta = self.model(x)
        eta.sum().backward()
        w = self.grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1))[0]
        return (cam / (cam.max() + 1e-8)).cpu().numpy(), float(eta.item())


def cam_on_input(cam: np.ndarray, in_shape) -> np.ndarray:
    t = torch.from_numpy(cam)[None, None].float()
    return F.interpolate(t, size=tuple(in_shape), mode="trilinear", align_corners=False)[0, 0].numpy()


def to_ct_space(t: np.ndarray, row: pd.Series, in_shape, ct_shape) -> np.ndarray:
    """t: CAM already on the model-input grid (cam_on_input)."""
    res = np.zeros([int(row[f"res_{a}"]) for a in "xyz"], np.float32)
    src, dst = [], []
    for d, a in enumerate("xyz"):
        o = int(row[f"iso_off_{a}"])
        s0, s1 = max(o, 0), min(o + in_shape[d], res.shape[d])
        dst.append(slice(s0, s1))
        src.append(slice(s0 - o, s1 - o))
    res[tuple(dst)] = t[tuple(src)]
    lo = np.array([int(row[f"bbox_lo_{a}"]) for a in "xyz"])
    hi = np.array([int(row[f"bbox_hi_{a}"]) for a in "xyz"])
    nat = F.interpolate(torch.from_numpy(res)[None, None], size=tuple(hi - lo), mode="trilinear",
                        align_corners=False)[0, 0].numpy()
    full = np.zeros(ct_shape, np.float32)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = nat
    return full


def panel(ax, ct2, cam2, organ, title):
    lo, hi = WINDOW[organ]
    ax.imshow(ct2, cmap="gray", vmin=lo, vmax=hi)
    vmax = max(float(cam2.max()), 1e-6)
    ax.imshow(cam2, cmap="hot", vmin=0, vmax=vmax, alpha=np.clip(cam2 / vmax, 0, 1) * 0.8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def views(ct, full, row):
    """Axial slice at peak CAM energy + sagittal slice through the organ's bbox
    centre, both cropped to the organ bbox (+20 voxels) so the thin sternum is visible."""
    lo = np.array([int(row[f"bbox_lo_{a}"]) for a in "xyz"])
    hi = np.array([int(row[f"bbox_hi_{a}"]) for a in "xyz"])
    pad = 20
    x0, x1 = max(lo[0] - pad, 0), min(hi[0] + pad, ct.shape[0])
    y0, y1 = max(lo[1] - pad, 0), min(hi[1] + pad, ct.shape[1])
    z = int(np.argmax(full.sum(axis=(0, 1))))
    xc = int(np.argmax(full.sum(axis=(1, 2)))) if full.max() > 0 else (lo[0] + hi[0]) // 2
    ax_ct = np.rot90(ct[x0:x1, y0:y1, z])
    ax_cam = np.rot90(full[x0:x1, y0:y1, z])
    sg_ct = np.rot90(ct[xc, y0:y1, lo[2]:hi[2]])
    sg_cam = np.rot90(full[xc, y0:y1, lo[2]:hi[2]])
    return (ax_ct, ax_cam, z), (sg_ct, sg_cam, xc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="all", choices=["all", "lungs", "sternum"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    organs = ["lungs", "sternum"] if args.organ == "all" else [args.organ]
    c4 = PROJECT / "data" / "c4_test_metrics.csv"
    metrics = pd.read_csv(c4) if c4.exists() else None

    manifest = []
    for organ in organs:
        run = next(Path(args.runs_dir, "final").glob(f"final__{organ}__main__*_s1"))
        cfg = json.loads((run / "config.json").read_text())
        model = CNN3D(in_ch=2, width=cfg["width"], dropout=cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(run / "model.pt", map_location=device))
        model.eval()
        cam_fn = GradCAM3D(model)

        idx, arr = load_organ(organ, "iso", Path(args.cache_dir))
        pred = pd.read_csv(run / "pred.csv", dtype={"pid": str})
        te = pred[pred.split == "test"].sort_values("eta", ascending=False)
        picks = pd.concat([te.head(args.top_k).assign(group="high"), te.tail(args.top_k).assign(group="low")])
        caveat = ""
        if metrics is not None:
            m = metrics[(metrics.organ == organ) & (metrics.group == "new_main")]
            if len(m) and m.c_lo.iloc[0] <= 0.5:
                caveat = " -- model test C CI spans 0.5: interpret with caution"

        out_nii = SAL_DIR / f"c5_{organ}"
        out_fig = FIG_DIR / organ
        out_nii.mkdir(parents=True, exist_ok=True)
        out_fig.mkdir(parents=True, exist_ok=True)
        over = []
        for _, p in picks.iterrows():
            i = int(np.where(idx.pid == p.pid)[0][0])
            row = idx.iloc[i]
            x = to_input(torch.from_numpy(np.array(arr[i:i + 1])).to(device), organ, "iso")
            cam, eta = cam_fn(x)
            ct, _, _, ct_img = load_patient(p.pid)
            cam_in = cam_on_input(cam, arr.shape[1:])
            # share of CAM mass inside the organ mask: ~1 = the model looks inside the
            # organ; low = it keys on the organ's outline/size or on padding
            inside = np.asarray(arr[i]) != SENTINEL
            frac_in = float(cam_in[inside].sum() / max(cam_in.sum(), 1e-8))
            full = to_ct_space(cam_in, row, arr.shape[1:], ct.shape)
            nii = out_nii / f"{p.pid}_gradcam.nii.gz"
            nib.save(nib.Nifti1Image(full, ct_img.affine), nii)
            (a_ct, a_cam, z), (s_ct, s_cam, xc) = views(ct, full, row)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
            lo, hi = WINDOW[organ]
            axes[0].imshow(a_ct, cmap="gray", vmin=lo, vmax=hi)
            axes[0].set_title("CT (axial)", fontsize=8)
            axes[0].axis("off")
            panel(axes[1], a_ct, a_cam, organ, f"Grad-CAM axial z={z}")
            panel(axes[2], s_ct, s_cam, organ, f"Grad-CAM sagittal x={xc}")
            fig.suptitle(f"{organ} | pid {p.pid} | {p.group}-risk | event={int(p.event)} | "
                         f"eta={eta:.2f} | CAM in mask {frac_in:.0%}{caveat}", fontsize=9)
            fig.tight_layout()
            png = out_fig / f"{p.pid}.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            over.append((p, a_ct, a_cam, s_ct, s_cam, eta))
            manifest.append({"organ": organ, "pid": p.pid, "group": p.group, "event": int(p.event),
                             "time_days": p.time, "eta": round(eta, 4), "cam_frac_in_mask": round(frac_in, 3),
                             "nifti": os.path.relpath(nii, PROJECT), "png": os.path.relpath(png, PROJECT)})
            print(f"[{organ}] {p.group} {p.pid} ev={int(p.event)} eta={eta:.2f} "
                  f"CAM-in-mask={frac_in:.2f} -> {png.name}", flush=True)

        k = args.top_k
        fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 7.5), squeeze=False)
        for j, (p, a_ct, a_cam, s_ct, s_cam, eta) in enumerate(over):
            r, c = divmod(j, k)
            ct2, cam2 = (a_ct, a_cam) if organ == "lungs" else (s_ct, s_cam)
            panel(axes[r, c], ct2, cam2, organ, f"{p.pid} ev={int(p.event)}\neta={eta:.2f}")
        fig.suptitle(f"C5 Grad-CAM {organ} ({'axial' if organ == 'lungs' else 'sagittal'}); "
                     f"top row: highest-risk test pts, bottom: lowest{caveat}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
        fig.savefig(FIG_DIR / f"{organ}_overview.png", dpi=110)
        plt.close(fig)
    pd.DataFrame(manifest).to_csv(MANIFEST, index=False)
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
