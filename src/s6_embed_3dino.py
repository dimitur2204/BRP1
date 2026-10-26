#!/usr/bin/env python3
"""S6 -- frozen 3DINO-ViT features for every cohort patient (no training).

3DINO-ViT (Xu et al., npj Digit. Med. 2025) is a ViT-Large pretrained with a
3D DINOv2 objective on ~100k unlabeled CT/MRI volumes. Here it is used as a
frozen feature extractor on the same S2 lung volumes the CNN sees, so the two
can be compared on the same split (S7 fits the Cox probe).

Input, per patient (cache row from data/cohort.csv):
  uint8 lung volume 144x128x128 at 2.5 mm (nlst.py). Outside-mask voxels (0)
  are set to the uint8 value of air, so the volume looks like lungs on an air
  background, not on an artificial black. Then 3DINO's own normalisation:
  the 0.05th/99.95th percentiles map to -1/1, clipped. All axes are
  multiples of 16, so the token grid is 9x8x8 = 576 patches (the position
  encoding is interpolated by 3DINO).
Saved features (last 4 transformer layers, layer-normed, as in DINOv2 probing):
  cls        (N, 1024)  [CLS] token of the last layer
  patch      (N, 1024)  mean of last-layer patch tokens inside the lung mask
                        (patches with >= 50% lung voxels)
  last4      (N, 4096)  [CLS] tokens of the last 4 layers, concatenated
  tokens     (N, 576, 1024) float16 last-layer patch tokens, only with
                        --save-tokens (~19 GB), for later regional analysis

Needs a clone of https://github.com/AICONSlab/3DINO and the weights from
https://huggingface.co/AICONSlab/3DINO-ViT (CC BY-NC-ND 4.0: non-commercial
use only). See docs/pipeline.md section 8.

Usage:
    sbatch src/submit_s6_embed.sh                       # -> data/embeddings/3dino.npz
    env/.venv/bin/python src/s6_embed_3dino.py --limit 4 --batch-size 2 --device cpu \
        --out <scratch>/s6_smoke.npz                     # smoke test (CPU, login node)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from nlst import CACHE_NPY, COHORT_CSV, DATA, GRID, HU_HI, HU_LO

EMB_DIR = DATA / "embeddings"
PATCH = 16
AIR_U8 = float(1 + round((-1000.0 - HU_LO) / (HU_HI - HU_LO) * 254))  # -1000 HU -> 60


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino-repo", default=str(Path(__file__).resolve().parents[1] / "external" / "3DINO"))
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "models" / "3dino_vit_weights.pth"))
    ap.add_argument("--config", default="train/vit3d_highres")
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--cache", default=str(CACHE_NPY))
    ap.add_argument("--out", default=str(EMB_DIR / "3dino.npz"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N cohort rows only (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-tokens", action="store_true")
    return ap.parse_args()


def load_model(repo: str, config: str, weights: str, device: torch.device):
    """Teacher ViT with strictly matched weights. (3DINO's own build_model_for_eval
    loads with strict=False, falls back to random weights if the file is missing,
    and always moves to cuda.)"""
    sys.path.insert(0, repo)
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    model, _ = build_model_from_cfg(load_and_merge_config_3d(config), only_teacher=True)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = sd.get("teacher", sd)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("dino_head") and not k.startswith("ibot_head")}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def to_dino_input(u8: torch.Tensor) -> torch.Tensor:
    """uint8 cache batch (B, X, Y, Z) -> (B, 1, X, Y, Z) in [-1, 1], 3DINO's per-volume scaling."""
    x = u8.float()
    x = torch.where(u8 == 0, torch.full_like(x, AIR_U8), x)
    # one volume at a time: torch.quantile refuses inputs over 16 M elements
    q = torch.stack([torch.quantile(v.flatten(), torch.tensor([0.0005, 0.9995], device=v.device))
                     for v in x])
    lo, hi = q[:, 0].view(-1, 1, 1, 1), q[:, 1].view(-1, 1, 1, 1)
    x = (x - lo) / (hi - lo).clamp_min(1e-6)
    return torch.clip(x * 2 - 1, -1, 1).unsqueeze(1)


@torch.no_grad()
def embed(model, u8: torch.Tensor):
    x = to_dino_input(u8)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if x.is_cuda else torch.autocast("cpu", enabled=False)
    with amp:
        layers = model.get_intermediate_layers(x, n=4, reshape=True, return_class_token=True)
    maps = [m.float() for m, _ in layers]                   # 4 x (B, C, 9, 8, 8)
    cls = [c.float() for _, c in layers]                    # 4 x (B, C)
    lung = F.avg_pool3d((u8 > 0).float().unsqueeze(1), PATCH) >= 0.5   # (B, 1, 9, 8, 8)
    last = maps[-1]
    w = lung.float()
    patch = (last * w).sum((2, 3, 4)) / w.sum((2, 3, 4)).clamp_min(1.0)
    tokens = last.flatten(2).transpose(1, 2)                # (B, 576, C), X-major like the grid
    return cls[-1], patch, torch.cat(cls, dim=1), tokens


def main() -> None:
    a = parse_args()
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    if a.limit:
        cohort = cohort.head(a.limit)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)   # before --save-tokens opens its file
    cache = np.load(a.cache, mmap_mode="r")
    assert cache.shape[1:] == GRID and all(g % PATCH == 0 for g in GRID), cache.shape
    device = torch.device(a.device)
    model = load_model(a.dino_repo, a.config, a.weights, device)
    print(f"3DINO loaded on {device}; {len(cohort)} patients, batch {a.batch_size}", flush=True)

    n, rows = len(cohort), cohort["row"].to_numpy()
    out = {k: [] for k in ("cls", "patch", "last4")}
    tokens = None
    if a.save_tokens:
        tok_path = Path(a.out).with_suffix(".tokens.npy")
        n_tok = int(np.prod([g // PATCH for g in GRID]))
        tokens = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.float16, shape=(n, n_tok, 1024))
    t0 = time.time()
    for i in range(0, n, a.batch_size):
        idx = rows[i:i + a.batch_size]
        u8 = torch.from_numpy(np.stack([cache[r] for r in idx])).to(device)
        cls, patch, last4, tok = embed(model, u8)
        out["cls"].append(cls.cpu().numpy())
        out["patch"].append(patch.cpu().numpy())
        out["last4"].append(last4.cpu().numpy())
        if tokens is not None:
            tokens[i:i + len(idx)] = tok.cpu().numpy().astype(np.float16)
        if (i // a.batch_size) % 200 == 0:
            print(f"{i + len(idx)}/{n}  {time.time() - t0:.0f} s", flush=True)

    feats = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    assert all(np.isfinite(v).all() for v in feats.values()), "non-finite features"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, pid=cohort["pid"].to_numpy(), row=rows, **feats)
    if tokens is not None:
        tokens.flush()
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v.shape}" for k, v in feats.items())
          + f"; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
