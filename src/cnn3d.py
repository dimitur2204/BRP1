"""Shared 3D CNN pieces: model, Cox loss, cached-input loading, C-index.

Used by c2_train.py (training), c4_evaluate.py and c5_gradcam.py.

The model outputs ONE number per patient, eta = f(image). Nothing about time
goes into the network. With the Cox loss, eta is a log relative hazard:
patient A with eta_A = eta_B + 1 is modelled as having e^1 = 2.7x the
instantaneous lung-cancer hazard of B, at every time point (proportional
hazards). Time only enters the LOSS, through who is still "at risk" when each
cancer happens -- see cox_ph_loss.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from organs import WINDOW

PROJECT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT / "data" / "cnn_cache"
RUNS_DIR = PROJECT / "runs"
SENTINEL = -2048  # must match c1_cache_volumes.SENTINEL


class CNN3D(nn.Module):
    """4 conv3d blocks (conv -> BatchNorm -> ReLU [-> MaxPool]) + global
    average pool + dropout + linear -> 1 scalar.

    Layer layout (`features` indices, `fc`) is identical to the legacy
    stage3_cnn3d.Simple3DCNN, so CNN3D(in_ch=1, width=16) loads the old
    checkpoints unchanged; features[12] is the last conv (Grad-CAM target).
    Global average pooling makes the network accept any input size, so the
    same class serves the legacy 64^3 cube and the new per-organ grids.
    """

    def __init__(self, in_ch: int = 2, width: int = 16, dropout: float = 0.0):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, w, 3, padding=1), nn.BatchNorm3d(w), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(w, 2 * w, 3, padding=1), nn.BatchNorm3d(2 * w), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm3d(4 * w), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(4 * w, 8 * w, 3, padding=1), nn.BatchNorm3d(8 * w), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8 * w, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(self.dropout(x)).squeeze(-1)


def cox_ph_loss(eta: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), averaged over events.

    For every patient i who got cancer at time t_i, the Cox model asks: of
    everyone still cancer-free and under follow-up at t_i (the risk set
    R_i = {j : t_j >= t_i}), how likely was it to be *i* who got it?
        P_i = exp(eta_i) / sum_{j in R_i} exp(eta_j)
    The loss is -mean_i log P_i over events. Censored patients never
    contribute a term of their own, but they sit in the denominator of every
    event that happened while they were still followed -- which is exactly
    how censoring is handled correctly (BCE instead treats a patient
    censored after 2 years as a definite non-case).

    Computed on a minibatch (the risk set is batch-local, a standard
    approximation; batch 32 at a 33% train event rate gives ~10 events per
    batch). Sorting by time descending makes each risk-set sum a cumulative
    logsumexp; tied times all get the risk set of the whole tie group.
    """
    eta, time, event = eta.float(), time.float(), event.float()
    order = torch.argsort(time, descending=True)
    eta, time, event = eta[order], time[order], event[order]
    log_risk = torch.logcumsumexp(eta, dim=0)
    _, inv, counts = torch.unique_consecutive(time, return_inverse=True, return_counts=True)
    last = torch.cumsum(counts, 0) - 1               # last index of each tie group
    log_risk = log_risk[last[inv]]
    n_events = event.sum()
    if n_events == 0:
        return eta.sum() * 0.0                        # no events in batch: no signal
    return -((eta - log_risk) * event).sum() / n_events


def harrell_c(time, eta, event) -> float:
    """Harrell's C: fraction of comparable pairs (the earlier time is an event)
    in which the earlier-cancer patient has the higher eta. 0.5 = chance."""
    from lifelines.utils import concordance_index
    return float(concordance_index(np.asarray(time), -np.asarray(eta), np.asarray(event)))


def load_organ(organ: str, recipe: str, cache_dir: Path = CACHE_DIR, mmap: bool = True):
    """(index DataFrame, array) for one organ; recipe 'iso' or 'legacy64'.
    Row i of the array belongs to index row i."""
    idx = pd.read_csv(cache_dir / f"{organ}_index.csv", dtype={"pid": str})
    arr = np.load(cache_dir / f"{organ}_{recipe}.npy", mmap_mode="r" if mmap else None)
    assert len(idx) == len(arr), (organ, recipe, len(idx), len(arr))
    return idx, arr


def to_input(batch: torch.Tensor, organ: str, recipe: str,
             hu_jitter: torch.Tensor | None = None) -> torch.Tensor:
    """Cached batch -> network input [B, C, X, Y, Z] float.

    legacy64: already windowed to [0, 1]; 1 channel.
    iso:      int16 HU with SENTINEL outside the mask ->
              ch0 = HU windowed to [0, 1] (organs.WINDOW), 0 outside the mask
              ch1 = mask (1 inside the organ, 0 outside / padding)
    `hu_jitter` ([B, 2] = offset HU, scale) is the training-time intensity
    augmentation, applied to in-mask HU before windowing.
    """
    if recipe == "legacy64":
        return batch.float()[:, None]
    x = batch.float()
    inside = x != SENTINEL
    if hu_jitter is not None:
        x = x * hu_jitter[:, 1, None, None, None] + hu_jitter[:, 0, None, None, None]
    lo, hi = WINDOW[organ]
    ch0 = ((x - lo) / (hi - lo)).clamp(0, 1) * inside
    return torch.stack([ch0, inside.float()], dim=1)
