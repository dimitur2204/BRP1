"""D2 -- the repo's collapse monitors, logging to the Lightning CSV logger instead of W&B.

The three callbacks in ext/DINOv2-3D-Med/callbacks only log to a WandbLogger
(no W&B account here), so with a CSVLogger they would compute and drop their
numbers. These subclasses keep all the computation and only change where the
numbers go: trainer.logger.log_metrics -> {log_dir}/metrics.csv. The Gram
matrix heatmap is saved as {log_dir}/gram/step_{N}.png, and its off-diagonal
mean / std are logged as metrics (mean -> 1 with std -> 0 = collapse).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from project.callbacks import (EffectiveRankCallback, GramMatrixCallback,  # noqa: E402
                               TeacherEntropyCallback)


def _log(trainer, metrics: dict) -> None:
    if trainer.global_rank == 0 and trainer.logger is not None:
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float, np.floating, np.integer))
                 or (torch.is_tensor(v) and v.numel() == 1)}
        if clean:
            trainer.logger.log_metrics(clean, step=trainer.global_step)


class CSVEffectiveRank(EffectiveRankCallback):
    def _log_metrics(self, trainer, metrics: dict) -> None:
        _log(trainer, metrics)


class CSVTeacherEntropy(TeacherEntropyCallback):
    def _log_metrics(self, trainer, metrics: dict) -> None:
        _log(trainer, metrics)


class CSVGramMatrix(GramMatrixCallback):
    def _report_debug_diagnostics(self, trainer, metrics: dict) -> None:
        _log(trainer, metrics)

    def _report_extraction_issue(self, trainer, message: str) -> None:
        if trainer.global_rank == 0:
            print(f"[gram matrix] {message}", flush=True)

    def _log_gram_matrix(self, trainer, gram_matrix: torch.Tensor) -> None:
        if trainer.global_rank != 0:
            return
        g = gram_matrix.float().cpu().numpy()
        off = g[~np.eye(len(g), dtype=bool)] if len(g) > 1 else np.zeros(1)
        _log(trainer, {"gram_offdiag_mean": off.mean(), "gram_offdiag_std": off.std()})
        out = Path(trainer.logger.log_dir) / "gram"
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(g, cmap="viridis", vmin=max(-1.0, float(off.min()) - 0.05), vmax=1.0)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"step {trainer.global_step}: off-diag {off.mean():.3f} ± {off.std():.3f}",
                     fontsize=8)
        fig.savefig(out / f"step_{trainer.global_step:06d}.png", dpi=80)
        plt.close(fig)


class Throughput(pl.Callback):
    """D2: seconds per dataloader batch, the share spent waiting for data, peak GPU memory.

    data_wait_frac near 1 = the GPU waits for the CPU-side 3D augmentation
    (GenomeDK cancels GPU jobs averaging < 75% utilisation after 2 h).
    """

    def __init__(self, log_every_n_steps: int = 50):
        self.every = log_every_n_steps
        self.t_end = self.t_start = None
        self.wait = self.total = 0.0
        self.n = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self.t_end is not None:
            self.wait += now - self.t_end
        self.t_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self.t_end is not None:
            self.total += now - self.t_end
        self.t_end = now
        self.n += 1
        if self.total > 0 and self.n % self.every == 0:
            m = {"sec_per_batch": self.total / self.every, "data_wait_frac": self.wait / self.total}
            if torch.cuda.is_available():
                m["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            _log(trainer, m)
            print(f"[throughput] step {trainer.global_step} " +
                  " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
            self.wait = self.total = 0.0
