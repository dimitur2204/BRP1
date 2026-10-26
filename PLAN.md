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

- **S6 embed** (D1, frozen 3DINO-ViT features) ✅ CPU smoke on the login node (2026-10-26).
  ✅ run (job 6295033, gpu-short L40S, 8 min 50 s, masked input, `--save-tokens`) →
  `data/embeddings/3dino.npz` + `3dino.tokens.npy` (19 GB), verified.
- **S7 probe** (elastic-net Cox on frozen features → S5) ✅ val (2026-10-26): cls 0.726, patch 0.719,
  **cls_patch 0.729** (chosen), last4 0.723; clinical 0.621; `cosine` CNN 0.688.
  ✅ S5 test: 3DINO 0.647 [0.609, 0.688] (= CNN 0.648), clinical 0.653,
  3DINO + clinical 0.675 [0.638, 0.713] (= clinical + CNN 0.675). Frozen general SSL
  matches the supervised CNN; neither clearly beats clinical. Val Spearman 3DINO η vs CNN η 0.66.
  ⬜ paired ΔC CI vs clinical (S5 extension). ⬜ user decision: next step (unmasked input / D2 / D4).

## Next direction (2026-10-26, proposal: `docs/directions.md`)

Self-supervised 3D encoders on NLST: D1 frozen 3DINO probe (S6/S7, started),
D2 `ctab` positive-control probes, D3 in-domain SSL on NLST, D4 lobe-level
risk from patch tokens. D2–D4 wait on D1's result and user sign-off.

## Open items

- First GPU run: check the epoch time and GPU memory in `logs/s4_*.out`.
  The 12 h limit and 96 GB of RAM are estimates.
- Fill in the S4 baseline test metrics (needed as the D1 comparison).
- Verify on the cluster: `prsn` cancer-location/histology/stage columns and `ctab` codes + nodule fields (needed for D2/D4).
