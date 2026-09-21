#!/usr/bin/env python3
"""Stage 3 -- simple 2D-slice CNN classifier, trained per organ.

Deliberately the simplest model that could work, as a warm-up before any 3D
CNN or the mentor's frozen-embedding approach: for each organ, take the axial
slice with the most in-mask coverage from the cached crop (data/organ_crops,
built by src/cache_organ_crops.py), resize to a fixed 64x64, and train a small
2D CNN (3 conv blocks + FC) to predict the binary lung-cancer event.

CPU-friendly by design (per project decision): 2D not 3D, small crops, few
epochs -- trains in well under a minute per organ on the login node's CPU.

Train/val/test split is reused as-is from data/pid_lists/subset_v1.csv (which
carries it over from the mentor's manifest.csv), not re-shuffled.

Usage:
    env/.venv/bin/python src/stage3_cnn.py --organ lungs
    env/.venv/bin/python src/stage3_cnn.py --organ all   # loops all 7, Stage 4 input
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score, confusion_matrix

from organs import ORGAN_SET, AIR_HU, window_for

PROJECT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
CROPS_DIR = PROJECT / "data" / "organ_crops"
FIGS_DIR = PROJECT / "figs" / "stage3_cnn"
RESULTS_CSV = PROJECT / "data" / "stage3_results.csv"

IMG_SIZE = 64
SEED = 0
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3


def load_subset_rows():
    with open(SUBSET_CSV, newline="") as f:
        return list(csv.DictReader(f))


def best_slice_2d(crop: np.ndarray, organ: str) -> np.ndarray:
    """Pick the axial slice with the most in-mask (non-air) coverage, window
    and rescale it to [0, 1] for the CNN, resize to IMG_SIZE x IMG_SIZE."""
    coverage = (crop > AIR_HU + 1.0).sum(axis=(0, 1))
    z = int(np.argmax(coverage))
    sl = crop[:, :, z]
    lo, hi = window_for(organ)
    sl = np.clip(sl, lo, hi)
    sl = (sl - lo) / (hi - lo)  # -> [0, 1]
    zoom_factors = (IMG_SIZE / sl.shape[0], IMG_SIZE / sl.shape[1])
    return zoom(sl, zoom_factors, order=1).astype(np.float32)


class OrganSliceDataset(torch.utils.data.Dataset):
    """Precomputes each patient's best-coverage 2D slice ONCE at construction
    (not per __getitem__ call) -- crops (esp. lungs, ~17MB/patient at native
    resolution) are far too large to reload from disk + resize every epoch;
    the final 64x64 float image is tiny, so caching it in memory is cheap."""

    def __init__(self, organ: str, rows: list[dict]):
        self.images = []
        self.events = []
        for row in rows:
            path = CROPS_DIR / organ / f"{row['pid']}.npy"
            if not path.exists():
                continue
            crop = np.load(path).astype(np.float32)
            self.images.append(best_slice_2d(crop, organ))
            self.events.append(int(row["event"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])[None]
        return img, torch.tensor(self.events[idx], dtype=torch.float32)


class SimpleCNN2D(nn.Module):
    """3 conv blocks + global pool + FC -- deliberately small/familiar."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x).squeeze(-1)


def run_organ(organ: str) -> dict:
    torch.manual_seed(SEED)
    rows = load_subset_rows()
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}

    ds = {s: OrganSliceDataset(organ, by_split[s]) for s in by_split}
    n_pos = sum(ds["train"].events)
    n_neg = len(ds["train"]) - n_pos
    print(f"[{organ}] n = train {len(ds['train'])} ({n_pos}+/{n_neg}-), "
          f"val {len(ds['val'])}, test {len(ds['test'])}", flush=True)
    if n_pos == 0 or len(ds["val"]) == 0:
        print(f"[{organ}] SKIP: not enough data (missing masks?)", flush=True)
        return {"organ": organ, "val_auc": None, "test_auc": None}

    loaders = {
        "train": torch.utils.data.DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True),
        "val": torch.utils.data.DataLoader(ds["val"], batch_size=64),
        "test": torch.utils.data.DataLoader(ds["test"], batch_size=64),
    }

    model = SimpleCNN2D()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    def epoch_pass(split, train: bool):
        model.train(train)
        losses, ys, ps = [], [], []
        for x, y in loaders[split]:
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            ys.extend(y.tolist())
            ps.extend(torch.sigmoid(logits).detach().tolist())
        auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
        return float(np.mean(losses)), auc, ys, ps

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc, best_state = -1.0, None
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

    model.load_state_dict(best_state)
    _, val_auc, val_y, val_p = epoch_pass("val", train=False)
    _, test_auc, test_y, test_p = epoch_pass("test", train=False)
    test_pred = [1 if p >= 0.5 else 0 for p in test_p]
    cm = confusion_matrix(test_y, test_pred).tolist()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    org_dir = FIGS_DIR / organ
    org_dir.mkdir(exist_ok=True)

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
    fig.savefig(org_dir / "training_curves.png", dpi=120)
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
    args = parser.parse_args()

    organs = list(ORGAN_SET) if args.organ == "all" else [args.organ]
    results = [run_organ(o) for o in organs]

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
