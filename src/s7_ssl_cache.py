#!/usr/bin/env python3
"""S7 -- the pretraining scan list for self-supervised learning (DINOv2, D1).

Pretraining uses no labels, so it can use every screening round (yr0, yr1, yr2)
of the train + val patients. Test patients are never included (asserted).

  --list     data/cache/rounds_todo.csv: train + val cohort pids x {yr1, yr2}
             whose CT and TotalSegmentator mask exist (looked up by name)
  --task-id  one Slurm array task: preprocess its share of rounds_todo.csv with
             exactly the S2 input definition (s2_preprocess.preprocess) into
             data/cache/rounds_chunks/{task}.{npy,csv}
  --merge    stream the chunks into data/cache/rounds_u8.npy + rounds_index.csv,
             then write data/ssl_scans.csv: yr0 rows of cohort.csv (train + val,
             already QC'd by S3) + yr1/yr2 rows passing the same S3 image QC
             (status ok, lung z-extent, lung volume, clipped fraction)

ssl_scans.csv columns: scan_id ({pid}_{round}), pid, round, split, cache
("yr0" = S2 lungs_u8.npy, "rounds" = rounds_u8.npy), row (in that cache).

    env/.venv/bin/python src/s7_ssl_cache.py --list       # login node, ~5 min
    sbatch src/submit_s7_ssl_cache.sh                      # 40 CPU tasks
    env/.venv/bin/python src/s7_ssl_cache.py --merge
Smoke test:
    env/.venv/bin/python src/s7_ssl_cache.py --list --pids 100002,100004 --out-dir /tmp/x
    env/.venv/bin/python src/s7_ssl_cache.py --task-id 0 --n-tasks 1 --out-dir /tmp/x
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nlst import (CACHE_DIR, COHORT_CSV, CT_DIR, GRID, MAX_CLIPPED_FRAC, MIN_LUNG_ML,
                  MIN_LUNG_Z_MM, SEG_DIR, SSL_CACHE_INDEX, SSL_CACHE_NPY, SSL_ROUNDS,
                  SSL_SCANS_CSV, SSL_SPLITS)
from s2_preprocess import run_pids_rounds

CHUNK_DIR = CACHE_DIR / "rounds_chunks"
TODO_CSV = CACHE_DIR / "rounds_todo.csv"


def ssl_cohort() -> pd.DataFrame:
    df = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    return df[df["split"].isin(SSL_SPLITS)]


def assert_no_test(scans: pd.DataFrame) -> None:
    test = set(pd.read_csv(COHORT_CSV, dtype={"pid": str}).query("split == 'test'")["pid"])
    leak = test & set(scans["pid"])
    if leak:
        raise SystemExit(f"{len(leak)} test patients in the pretraining list, e.g. {sorted(leak)[:3]}")


def make_todo(pids: list[str] | None, out: Path) -> None:
    coh = ssl_cohort()
    if pids:
        coh = coh[coh["pid"].isin(pids)]
    rows = []
    for pid, split in zip(coh["pid"], coh["split"]):
        for rnd in SSL_ROUNDS:
            if ((CT_DIR / f"{pid}_{rnd}.nii.gz").exists()
                    and (SEG_DIR / f"{pid}_{rnd}" / "seg.nii.gz").exists()):
                rows.append({"scan_id": f"{pid}_{rnd}", "pid": pid, "round": rnd, "split": split})
    todo = pd.DataFrame(rows, columns=["scan_id", "pid", "round", "split"])
    assert_no_test(todo)
    out.parent.mkdir(parents=True, exist_ok=True)
    todo.to_csv(out, index=False)
    print(f"wrote {out}: {len(todo)} scans of {coh['pid'].nunique()} pids")
    print(todo.groupby(["split", "round"]).size().to_string())


def merge(n_tasks: int) -> None:
    todo = pd.read_csv(TODO_CSV, dtype={"pid": str})
    parts = [CHUNK_DIR / f"{t:03d}" for t in range(n_tasks)]
    missing = [p.name for p in parts if not p.with_suffix(".csv").exists()]
    if missing:
        raise SystemExit(f"missing chunks {missing} -- resubmit those array tasks first")
    index = pd.concat([pd.read_csv(p.with_suffix(".csv"), dtype={"pid": str}) for p in parts],
                      ignore_index=True)
    if index["scan_id"].tolist() != todo["scan_id"].tolist():
        raise SystemExit("chunk scans != rounds_todo.csv -- was --list re-run? redo the array")
    out = np.lib.format.open_memmap(SSL_CACHE_NPY, mode="w+", dtype=np.uint8,
                                    shape=(len(index), *GRID))
    row = 0
    for p in parts:  # stream, never hold the full cache in RAM
        a = np.load(p.with_suffix(".npy"))
        out[row:row + len(a)] = a
        row += len(a)
        print(f"merged {p.name} ({row}/{len(index)})", flush=True)
    out.flush()
    index.insert(0, "row", np.arange(len(index)))
    index.to_csv(SSL_CACHE_INDEX, index=False)
    print(f"wrote {SSL_CACHE_NPY} {out.shape}, {SSL_CACHE_INDEX}")

    # same image QC as S3
    checks = [("preprocessing ok", index["status"] == "ok"),
              (f"lung z-extent >= {MIN_LUNG_Z_MM:g} mm", index["lung_z_mm"] >= MIN_LUNG_Z_MM),
              (f"lung volume >= {MIN_LUNG_ML:g} mL", index["lung_ml"] >= MIN_LUNG_ML),
              (f"clipped lung <= {MAX_CLIPPED_FRAC:.0%}", index["clipped_frac"] <= MAX_CLIPPED_FRAC)]
    keep = pd.Series(True, index=index.index)
    print(f"yr1/yr2 scans: {len(index)}")
    for name, ok in checks:
        keep &= ok.fillna(False)
        print(f"  {name}: {int(keep.sum())} left")

    yr0 = ssl_cohort()
    scans = pd.concat([
        pd.DataFrame({"scan_id": yr0["pid"] + "_yr0", "pid": yr0["pid"], "round": "yr0",
                      "split": yr0["split"], "cache": "yr0", "row": yr0["row"]}),
        index.loc[keep, ["scan_id", "pid", "round", "split"]].assign(
            cache="rounds", row=index.loc[keep, "row"]),
    ], ignore_index=True)
    assert_no_test(scans)
    scans.to_csv(SSL_SCANS_CSV, index=False)
    print(f"wrote {SSL_SCANS_CSV}: {len(scans)} scans of {scans['pid'].nunique()} pids")
    print(scans.groupby(["split", "round"]).size().to_string())
    print("You can now delete the chunks:", CHUNK_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pids", help="comma-separated pids (smoke test, with --list)")
    ap.add_argument("--out-dir", help="smoke test: write todo/chunks here instead of data/cache")
    a = ap.parse_args()

    out = Path(a.out_dir) if a.out_dir else None
    todo_csv = out / "rounds_todo.csv" if out else TODO_CSV
    if a.list:
        return make_todo(a.pids.split(",") if a.pids else None, todo_csv)
    if a.merge:
        return merge(a.n_tasks)
    if a.task_id is None:
        ap.error("give --list, --task-id or --merge")

    chunk_dir = out or CHUNK_DIR
    name = f"{a.task_id:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if (chunk_dir / f"{name}.csv").exists():
        print(f"{chunk_dir / name}.csv exists -- skipping")
        return
    todo = pd.read_csv(todo_csv, dtype={"pid": str})
    part = np.array_split(np.arange(len(todo)), a.n_tasks)[a.task_id]
    todo = todo.iloc[part]
    arr, df = run_pids_rounds(todo["pid"].tolist(), todo["round"].tolist())
    df.insert(0, "scan_id", todo["scan_id"].to_numpy())
    df.insert(2, "round", todo["round"].to_numpy())
    df.insert(3, "split", todo["split"].to_numpy())
    np.save(chunk_dir / f"{name}.npy", arr)
    df.to_csv(chunk_dir / f"{name}.csv", index=False)  # written last = chunk complete
    print(f"wrote {chunk_dir / name}.{{npy,csv}}: {len(df)} scans, "
          f"{(df['status'] == 'ok').sum()} ok, median {df['sec'].median():.1f} s/scan")


if __name__ == "__main__":
    main()
