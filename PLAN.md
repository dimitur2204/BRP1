# Lungs 3D CNN on NLST: baseline from scratch

Branch `cnn3d-lungs-cox` (from `cnn3d-lungs-sternum` 7382553), 2026-10-05. All previous code, cohorts,
caches and results were deleted (they are in git history up to `7382553`).
Design and rationale: `docs/pipeline.md`.

## Goal

A working, simple 3D CNN that reads the lungs on the NLST baseline CT and
predicts time to lung cancer (Cox loss). It is a base to build on, so there is
no hyperparameter grid.

## Decisions (user, 2026-10-05)

- Input: lungs only, from TotalSegmentator masks. This limits the cohort to
  patients with masks (all cancers, ~63% of controls).
- Outcome: time-to-event, Cox partial likelihood.
- New cohort and a fresh random split. Nothing from the old pipeline is reused.
- `learn/` (teaching workspace, git-excluded) is kept.

## Stages

- **S1 candidates** ✅ `src/s1_candidates.py` → 16,543 pts / 1,047 events.
- **S2 preprocess** ✅ code, smoke-tested. ✅ array run (job 1681218): 16,515 ok, 28 empty lung masks.
  ⬜ **user: `s2_preprocess.py --merge`, then `--qc`.**
- **S3 cohort** ✅ code, smoke-tested. ⬜ run after the S2 merge.
- **S4 train** ✅ code, CPU smoke-tested. Cox loss verified. ⬜ **user:
  `sbatch src/submit_s4_train.sh`** after S3.
- **S5 evaluate** ✅ code, smoke-tested. ⬜ run after S4.

## Open items

- First GPU run: check the epoch time and GPU memory in `logs/s4_*.out`.
  The 12 h limit and 96 GB of RAM are estimates.
