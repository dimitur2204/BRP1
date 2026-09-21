#!/usr/bin/env python3
"""Stage 3v2 -- real 3D CNN classifier, trained per organ, on GPU.

Supersedes src/stage3_cnn.py's single-2D-slice baseline, which came out at
chance for 5/7 organs -- including `lungs`, the positive control -- most
likely because reducing a whole 3D organ to one 2D cross-section throws away
most of the signal. This version feeds the CNN the WHOLE 3D organ crop
(resized to a fixed cube) instead of one slice.

Meant to run as a Slurm GPU job (see submit_stage3_cnn3d.sh), not on the
login node -- a 64^3 volume x ~180 training patients x several organs is too
slow on CPU for interactive iteration. `--device` defaults to auto-detect so
it still runs (slowly) on CPU for local smoke-testing.

Usage (on a GPU node, via sbatch):
    python src/stage3_cnn3d.py --organ all --crops-dir $TMPDIR/organ_crops
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

from organs import ORGAN_SET, AIR_HU, window_for

PROJECT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
FIGS_DIR = PROJECT / "figs" / "stage3_cnn3d"
MODELS_DIR = PROJECT / "models"
RESULTS_CSV = PROJECT / "data" / "stage3_3d_results.csv"

IMG_SIZE = 64      # fixed cube side length the CNN trains on
SEED = 0
EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-3


def load_subset_rows():
    with open(SUBSET_CSV, newline="") as f:
        return list(csv.DictReader(f))


def resize_and_window(crop: np.ndarray, organ: str, device: torch.device) -> torch.Tensor:
    """HU-window + rescale to [0,1], resize to IMG_SIZE^3 via trilinear interp."""
    lo, hi = window_for(organ)
    crop = np.clip(crop, lo, hi)
    crop = (crop - lo) / (hi - lo)
    t = torch.from_numpy(crop.astype(np.float32))[None, None].to(device)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE, IMG_SIZE), mode="trilinear",
                      align_corners=False)
    return t[0, 0].cpu()


class OrganVolumeDataset(torch.utils.data.Dataset):
    """Precomputes every patient's resized volume ONCE at construction (same
    lesson as stage3_cnn.py's bugfix: never redo this work per epoch)."""

    def __init__(self, organ: str, rows: list[dict], crops_dir: Path, device: torch.device):
        self.volumes = []
        self.events = []
        for row in rows:
            path = crops_dir / organ / f"{row['pid']}.npy"
            if not path.exists():
                continue
            crop = np.load(path).astype(np.float32)
            self.volumes.append(resize_and_window(crop, organ, device))
            self.events.append(int(row["event"]))

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        return self.volumes[idx][None], torch.tensor(self.events[idx], dtype=torch.float32)


class Simple3DCNN(nn.Module):
    """4 conv3d blocks + global pool + FC -- still a "simple, familiar" CNN,
    just extended to 3D instead of collapsing to one 2D slice."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),   # 64->32
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),  # 32->16
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),  # 16->8
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x).squeeze(-1)


def run_organ(organ: str, crops_dir: Path, device: torch.device) -> dict:
    torch.manual_seed(SEED)
    rows = load_subset_rows()
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}

    t0 = time.time()
    ds = {s: OrganVolumeDataset(organ, by_split[s], crops_dir, device) for s in by_split}
    print(f"[{organ}] loaded+resized in {time.time()-t0:.0f}s", flush=True)

    n_pos = sum(ds["train"].events)
    n_neg = len(ds["train"]) - n_pos
    print(f"[{organ}] n = train {len(ds['train'])} ({n_pos}+/{n_neg}-), "
          f"val {len(ds['val'])}, test {len(ds['test'])}", flush=True)
    if n_pos == 0 or len(ds["val"]) == 0:
        print(f"[{organ}] SKIP: not enough data", flush=True)
        return {"organ": organ, "val_auc": None, "test_auc": None}

    loaders = {
        "train": torch.utils.data.DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True),
        "val": torch.utils.data.DataLoader(ds["val"], batch_size=16),
        "test": torch.utils.data.DataLoader(ds["test"], batch_size=16),
    }

    model = Simple3DCNN().to(device)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    def epoch_pass(split, train: bool):
        model.train(train)
        losses, ys, ps = [], [], []
        for x, y in loaders[split]:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            ys.extend(y.cpu().tolist())
            ps.extend(torch.sigmoid(logits).detach().cpu().tolist())
        auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
        return float(np.mean(losses)), auc, ys, ps

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(EPOCHS):
        tr_loss, tr_auc, _, _ = epoch_pass("train", train=True)
        va_loss, va_auc, _, _ = epoch_pass("val", train=False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(va_auc)
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"[{organ}] trained {EPOCHS} epochs in {time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    _, val_auc, _, _ = epoch_pass("val", train=False)
    _, test_auc, test_y, test_p = epoch_pass("test", train=False)
    test_pred = [1 if p >= 0.5 else 0 for p in test_p]
    cm = confusion_matrix(test_y, test_pred).tolist()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS_DIR / f"{organ}_cnn3d.pt")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{organ}: loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_auc"], label="train")
    axes[1].plot(history["val_auc"], label="val")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{organ}: AUC (best val={best_val_auc:.3f}, test={test_auc:.3f})")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{organ}_training_curves.png", dpi=120)
    plt.close(fig)

    print(f"[{organ}] best val AUC={best_val_auc:.3f}  test AUC={test_auc:.3f}  "
          f"test confusion={cm}", flush=True)

    return {
        "organ": organ,
        "n_train": len(ds["train"]), "n_pos_train": n_pos,
        "n_val": len(ds["val"]), "n_test": len(ds["test"]),
        "val_auc": round(best_val_auc, 4), "test_auc": round(test_auc, 4),
        "test_confusion": json.dumps(cm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ", default="all", choices=["all"] + list(ORGAN_SET))
    parser.add_argument("--crops-dir", default=str(PROJECT / "data" / "organ_crops"),
                        help="override to read from a local $TMPDIR staging copy")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}  (cuda available: {torch.cuda.is_available()})", flush=True)
    if device.type == "cpu":
        print("WARNING: running on CPU -- this is a GPU-sized job, expect it to be slow. "
              "If this is unexpected inside a GPU sbatch job, check the CUDA torch install.",
              flush=True)

    crops_dir = Path(args.crops_dir)
    organs = list(ORGAN_SET) if args.organ == "all" else [args.organ]
    results = [run_organ(o, crops_dir, device) for o in organs]

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(results)
    print(f"appended results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
