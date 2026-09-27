#!/usr/bin/env python3
"""Headline figure: cumulative incidence by thymic-health category, adenocarcinoma vs
squamous + small cell (Aalen-Johansen; other lung cancers = competing event), full NLST CT arm.

Usage: env/.venv/bin/python experiments/E5_full_nlst_thymus/code/fig_e5.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter

EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(PROJECT / "experiments" / "E3_calcium_mechanism" / "code"))
from aging_data import INK, INK2, SURFACE, style  # noqa: E402
from run_e5 import load_full  # noqa: E402

# ordinal single-hue ramp (dataviz sequential blue): low -> high thymic health
CAT_COLORS = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
CAT_LABELS = {0: "low TH (≤25th pct)", 1: "average", 2: "high TH (>75th pct)"}


def main() -> None:
    warnings.filterwarnings("ignore")
    df = load_full()
    f2 = pd.read_csv(EXP / "results" / "f2_histology.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE, sharey=True)
    for ax, (lab, mask, key) in zip(axes, [
            ("Adenocarcinoma", df["histo"] == "adeno", "adeno"),
            ("Squamous + small cell", df["histo"].isin(["squamous", "small_cell"]), "SQ/SC")]):
        style(ax)
        code = np.where(df["event"] == 0, 0, np.where(mask, 1, 2))
        counts = []
        for c in (0, 1, 2):
            s = (df["th_cat"] == c).to_numpy()
            aj = AalenJohansenFitter(calculate_variance=False, seed=0).fit(df.loc[s, "time_yr"].to_numpy(), code[s], event_of_interest=1)
            ci = aj.cumulative_density_
            ax.step(ci.index, ci.iloc[:, 0], where="post", color=CAT_COLORS[c], lw=2, label=CAT_LABELS[c])
            counts.append(f"{CAT_LABELS[c].split(' (')[0]}: {int(((code == 1) & s).sum())}/{int(s.sum())}")
        ax.text(0.03, 0.97, "events / n\n" + "\n".join(counts), transform=ax.transAxes, fontsize=7,
                color=INK2, va="top", ha="left")
        r = f2[(f2["horizon"] == "full") & (f2["adjustment"] == "clinical") & (f2["histology"] == key)].iloc[0]
        ax.set_title(f"{lab}\nadj HR per SD of TH: {r['hr']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, 8.5)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xlabel("Years since randomization", fontsize=8, color=INK2)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("Cumulative incidence (Aalen–Johansen)", fontsize=8, color=INK2)
    fig.suptitle("Thymic health and lung cancer by histology: full NLST CT arm, n = 25,031 "
                 "(adj = age, sex, smoking; interaction p = 0.019)", fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(EXP / "results" / "fig_th_histology_cif.png", dpi=160)


if __name__ == "__main__":
    main()
