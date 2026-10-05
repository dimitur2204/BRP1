"""Network, loss, metric and augmentation for the lungs 3D CNN.

Kept deliberately small and plain so it is easy to read, debug and extend:

  LungCNN    a VGG-style 3D CNN. A stride-2 stem, then 4 stages of
             [conv 3^3 -> conv 3^3 stride 2], each followed by GroupNorm +
             ReLU. Widths 16-32-64-128-256, global average pool, dropout,
             one linear output = the log-risk score eta. GroupNorm (not
             BatchNorm) means train and eval mode compute the same function,
             whatever the batch size.
  cox_loss   negative Cox partial log-likelihood (Breslow ties) over the
             patients in a batch.
  harrell_c  Harrell's concordance index (higher eta = earlier event).
  augment    random rigid-ish transform on the GPU: rotation about the
             cranio-caudal axis, isotropic scale, translation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )


class LungCNN(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers = [conv_block(1, widths[0], stride=2)]        # 144x128x128 -> 72x64x64
        for cin, cout in zip(widths[:-1], widths[1:]):        # -> 36, 18, 9, 5 (x)
            layers += [conv_block(cin, cout), conv_block(cout, cout, stride=2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(widths[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, 1, X, Y, Z) in [0, 1]
        return self.head(self.features(x)).squeeze(1)        # (B,) log-risk eta


def to_input(u8: torch.Tensor) -> torch.Tensor:
    """uint8 cache batch (B, X, Y, Z) -> float network input (B, 1, X, Y, Z) in [0, 1]."""
    return (u8.float() / 255.0).unsqueeze(1)


def cox_loss(eta: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Mean negative Cox partial log-likelihood per event, Breslow ties.

    Risk set of patient i = everyone with time >= time_i. With patients sorted
    by descending time, that is the prefix up to the last patient tied with i,
    so log sum_{j in R_i} exp(eta_j) = logcumsumexp(eta)[last tied index].
    Returns 0 (with a gradient path) if the batch has no events.
    """
    order = torch.argsort(time, descending=True)
    eta, time, event = eta[order].float(), time[order], event[order].float()
    lcse = torch.logcumsumexp(eta, dim=0)
    n_ge = torch.searchsorted(-time.contiguous(), -time.contiguous(), right=True)  # #{j: t_j >= t_i}
    log_risk = lcse[n_ge - 1]
    n_events = event.sum()
    if n_events == 0:
        return eta.sum() * 0.0
    return -((eta - log_risk) * event).sum() / n_events


def harrell_c(eta, time, event) -> float:
    """Harrell's C: share of comparable pairs (earlier time has an event) in
    which the earlier patient has the higher eta; eta ties count 1/2."""
    from sksurv.exceptions import NoComparablePairException
    from sksurv.metrics import concordance_index_censored
    eta, time, event = (np.asarray(v, dtype=float) for v in (eta, time, event))
    if event.sum() == 0:
        return float("nan")
    try:
        return float(concordance_index_censored(event.astype(bool), time, eta)[0])
    except NoComparablePairException:  # e.g. no events in a (bootstrap) sample
        return float("nan")


def augment(x: torch.Tensor, max_rot_deg: float = 10.0, max_scale: float = 0.10,
            max_shift_vox: float = 6.0) -> torch.Tensor:
    """Random transform per sample of a (B, 1, X, Y, Z) batch. Out-of-volume
    voxels become 0 (= "outside the lung mask"). Built in voxel units and then
    converted to grid_sample's normalised coordinates, so rotations are not
    sheared by the non-cubic grid."""
    b = x.shape[0]
    dev = x.device
    size = torch.tensor(x.shape[2:], dtype=torch.float32, device=dev)  # (X, Y, Z)
    # grid_sample's (x, y, z) components index (W, H, D) = our (Z, Y, X)
    half = (size.flip(0) - 1) / 2.0                                    # (Z, Y, X) half sizes
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    scale = 1.0 + (torch.rand(b, device=dev) * 2 - 1) * max_scale
    shift = (torch.rand(b, 3, device=dev) * 2 - 1) * max_shift_vox
    c, s = torch.cos(ang), torch.sin(ang)
    A = torch.zeros(b, 3, 3, device=dev)
    A[:, 0, 0] = 1.0                       # Z (cranio-caudal) axis unchanged
    A[:, 1, 1], A[:, 1, 2] = c, -s         # rotation in the axial (Y, X) plane
    A[:, 2, 1], A[:, 2, 2] = s, c
    A = A * scale[:, None, None]
    S, Sinv = torch.diag(half), torch.diag(1.0 / half)
    theta = torch.cat([Sinv @ A @ S, (shift / half)[:, :, None]], dim=2)  # (B, 3, 4)
    grid = F.affine_grid(theta, list(x.shape), align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
