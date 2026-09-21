#!/usr/bin/env python3
"""Stage 5 -- Kaplan-Meier survival curves from the Stage 3v2 3D CNN's risk
scores, one organ at a time.

Follows the mentor's own KM convention (discovery_pipeline/src/km_analysis.py):
  continuous risk score on the HELD-OUT test set
    -> tertile split via pd.qcut on a tie-safe rank (not a raw qcut, which
       breaks on duplicate risk scores)
    -> lifelines.KaplanMeierFitter per tertile
    -> multivariate_logrank_test for a trend p-value.

Test-only (not val+test) to match the mentor's convention exactly: val was
already used for checkpoint selection during training, so it's not a clean
held-out set anymore. This does mean small groups (test n=127, 18 events ->
~6 events per tertile) -- that's reported honestly, not hidden.

Runs on CPU (inference-only, ~127 patients, no training) -- no GPU/sbatch
needed for this stage.

Usage:
    env/.venv/bin/python src/stage5_km_curve.py --organ lungs
    env/.venv/bin/python src/stage5_km_curve.py --organ all
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from organs import ORGAN_SET
from stage3_cnn3d import Simple3DCNN, OrganVolumeDataset

PROJECT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT / "data" / "pid_lists" / "subset_v1.csv"
MODELS_DIR = PROJECT / "models"
FIGS_DIR = PROJECT / "figs" / "stage5_km_curve"
RESULTS_CSV = PROJECT / "data" / "stage5_results.csv"

GROUP_LABELS = ["Low risk", "Mid risk", "High risk"]
GROUP_COLORS = {"Low risk": "tab:blue", "Mid risk": "tab:orange", "High risk": "tab:red"}


def load_subset_rows():
    with open(SUBSET_CSV, newline="") as f:
        return list(csv.DictReader(f))


def risk_scores_for_test(organ: str, crops_dir: Path, device: torch.device) -> pd.DataFrame:
    rows = load_subset_rows()
    test_rows = [r for r in rows if r["split"] == "test"]

    model = Simple3DCNN().to(device)
    state = torch.load(MODELS_DIR / f"{organ}_cnn3d.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = OrganVolumeDataset(organ, test_rows, crops_dir, device)
    # OrganVolumeDataset silently drops rows with no cached crop -- recover
    # which pids actually made it in, in the same order, to attach time/event.
    kept_pids = [r["pid"] for r in test_rows
                 if (crops_dir / organ / f"{r['pid']}.npy").exists()]
    assert len(kept_pids) == len(ds)

    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    risks = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            risks.extend(torch.sigmoid(model(x)).cpu().tolist())

    by_pid = {r["pid"]: r for r in test_rows}
    return pd.DataFrame({
        "pid": kept_pids,
        "risk": risks,
        "time_years": [float(by_pid[p]["time"]) / 365.25 for p in kept_pids],
        "event": [int(by_pid[p]["event"]) for p in kept_pids],
    })


def run_organ(organ: str, crops_dir: Path, device: torch.device) -> dict:
    df = risk_scores_for_test(organ, crops_dir, device)
    n_events = int(df["event"].sum())
    print(f"[{organ}] test n={len(df)}, events={n_events}", flush=True)

    # tie-safe tertile split, same as the mentor's km_analysis.py
    df["group"] = pd.qcut(df["risk"].rank(method="first"), 3, labels=GROUP_LABELS)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in GROUP_LABELS:
        sub = df[df["group"] == label]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time_years"], sub["event"],
                label=f"{label} (n={len(sub)}, events={int(sub['event'].sum())})")
        kmf.plot_survival_function(ax=ax, color=GROUP_COLORS[label])

    result = multivariate_logrank_test(df["time_years"], df["group"], df["event"])
    p_value = result.p_value

    ax.set_title(f"Stage 5 KM curve: {organ}\n"
                 f"3D CNN risk score, test set (n={len(df)}, {n_events} events), "
                 f"tertile split\nlogrank trend p={p_value:.3f}")
    ax.set_xlabel("years")
    ax.set_ylabel("survival probability (event-free)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / f"{organ}_km.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[{organ}] logrank trend p={p_value:.4f}  wrote {out}", flush=True)

    return {"organ": organ, "n_test": len(df), "n_events": n_events,
            "logrank_p": round(p_value, 4)}


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
    print(f"device: {device}", flush=True)

    crops_dir = Path(args.crops_dir)
    organs = list(ORGAN_SET) if args.organ == "all" else [args.organ]
    results = [run_organ(o, crops_dir, device) for o in organs]

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
