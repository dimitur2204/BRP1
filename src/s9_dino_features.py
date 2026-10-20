#!/usr/bin/env python3
"""S9 -- DINOv2 features for every cohort patient (D4).

For each requested checkpoint of a pretraining run (runs/dino/{run}/ckpt/),
the TEACHER backbone (the EMA copy, the standard DINO choice) reads the whole
yr0 lung volume of every cohort patient (all splits; no labels are used here)
in one forward pass: 144x128x128 at 2.5 mm, uint8 / 255, no crop or
augmentation -> 18x16x16 = 4,608 patch tokens + CLS (the position embedding
is resampled from the 12^3 training grid, RoPE is built for the new grid).
Per patient: [CLS token, mean of the patch tokens] -> 2 x embed_dim floats.

"random" instead of a checkpoint = the same backbone with random weights
(seed 0): the control a pretrained checkpoint must beat in the probe (D5).

The backbone is rebuilt from the run's launch.json (same config files and
overrides as pretraining), so architecture and weights always match. All
checkpoints share one pass over the data (the volumes are read once).

Outputs, data/features/{run}/:
  {ckpt}.npy    float32 (n_cohort, 2 x embed_dim), rows in data/cohort.csv order
  {ckpt}.json   run, checkpoint, epoch, dims, n, seconds
  pids.csv      pid per row (check against cohort.csv)

Usage:
    sbatch src/submit_s9_dino_features.sh --run trial_2h/big --ckpts random "dino_epoch=009" ...
    env/.venv-dino/bin/python src/s9_dino_features.py --run trial_2h/small --ckpts random --limit 8 \\
        --device cpu --out-dir <scratch>      # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "ext" / "DINOv2-3D-Med"
sys.path[:0] = [str(REPO), str(PROJECT / "src")]
from nlst import CACHE_NPY, COHORT_CSV, DATA  # noqa: E402

RUNS = PROJECT / "runs" / "dino"
TEACHER = "model.teacher_backbone."


def build_backbone(run: str):
    from monai.bundle import ConfigParser
    launch = json.loads((RUNS / run / "launch.json").read_text())
    parser = ConfigParser()
    parser.read_config(launch["config_files"])
    parser.update(launch["overrides"])      # before parse(), as in s8
    parser.parse()
    return parser.get_parsed_content("lightning_module#backbone")


def load_model(run: str, ckpt: str, device):
    import torch
    torch.manual_seed(0)
    model = build_backbone(run)
    meta = {"run": run, "ckpt": ckpt, "epoch": None}
    if ckpt != "random":
        c = torch.load(RUNS / run / "ckpt" / f"{ckpt}.ckpt", map_location="cpu", weights_only=False)
        sd = {k[len(TEACHER):]: v for k, v in c["state_dict"].items() if k.startswith(TEACHER)}
        if not sd:
            raise SystemExit(f"no {TEACHER}* weights in {ckpt}")
        model.load_state_dict(sd, strict=True)
        meta["epoch"], meta["global_step"] = c.get("epoch"), c.get("global_step")
    return model.to(device).eval(), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pretraining run under runs/dino/")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help='checkpoint names without .ckpt (e.g. "dino_epoch=039", last) or random')
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="first N cohort patients only (smoke test)")
    ap.add_argument("--out-dir", help="default data/features/{run}")
    a = ap.parse_args()

    import torch
    from utils.imports import import_module_from_path
    import_module_from_path("project", str(REPO))

    device = torch.device(a.device)
    out = Path(a.out_dir) if a.out_dir else DATA / "features" / a.run
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in a.ckpts if not (out / f"{c}.npy").exists()]
    for c in set(a.ckpts) - set(todo):
        print(f"{out / c}.npy exists -- skipping", flush=True)
    if not todo:
        return

    cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    if a.limit:
        cohort = cohort.head(a.limit)
    rows = cohort["row"].to_numpy()
    cache = np.load(CACHE_NPY, mmap_mode="r")
    models = {c: load_model(a.run, c, device) for c in todo}
    feats = {c: None for c in todo}
    print(f"{a.run}: {len(todo)} checkpoints x {len(rows)} patients, device {device}", flush=True)

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else \
        torch.autocast("cpu", enabled=False)
    t0 = time.time()
    with torch.no_grad(), amp:
        for start in range(0, len(rows), a.batch_size):
            idx = rows[start:start + a.batch_size]
            x = torch.from_numpy(np.stack([cache[i] for i in idx]).astype(np.float32) / 255.0)
            x = x[:, None].to(device, non_blocking=True)             # (B, 1, 144, 128, 128)
            for c, (model, _) in models.items():
                tok = model(x).float()                               # (B, 1 + 4608, D)
                f = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], dim=1).cpu().numpy()
                if feats[c] is None:
                    feats[c] = np.zeros((len(rows), f.shape[1]), dtype=np.float32)
                feats[c][start:start + len(idx)] = f
            if (start // a.batch_size) % 250 == 0:
                done = start + len(idx)
                print(f"  {done}/{len(rows)} patients, {time.time() - t0:.0f} s", flush=True)

    sec = time.time() - t0
    cohort[["pid"]].to_csv(out / "pids.csv", index=False)
    for c, (_, meta) in models.items():
        f = feats[c]
        if not np.isfinite(f).all():
            raise SystemExit(f"{c}: non-finite features")
        np.save(out / f"{c}.npy", f)
        sd = f.std(0)
        meta.update({"n": len(rows), "dim": f.shape[1], "seconds_all_ckpts": round(sec),
                     "feature_sd_median": float(np.median(sd)),
                     "dims_sd_below_1e-6": int((sd < 1e-6).sum())})
        (out / f"{c}.json").write_text(json.dumps(meta, indent=1))
        print(f"wrote {out / c}.npy {f.shape}: median per-feature SD {meta['feature_sd_median']:.4g}, "
              f"{meta['dims_sd_below_1e-6']} constant dims", flush=True)


if __name__ == "__main__":
    main()
