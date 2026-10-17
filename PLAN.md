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
- **S2 preprocess** ✅ array run (job 1681218): 16,515 ok, 28 empty lung masks. ✅ merged.
- **S3 cohort** ✅ 16,345 pts / 1,034 events; train 11,442/724, val 2,451/155, test 2,452/155.
- **S4/S5 run 1 `baseline`** ✅ (job 1683009): constant lr 3e-4, early stop at epoch 17,
  best epoch 5. Test C 0.614 [0.570, 0.660], below age alone (0.630). Underfits.
- **S4/S5 run 2 `cosine`** ✅ (job 1688465): peak lr 1e-3, warm-up 3 epochs, cosine decay,
  80 epochs. Best epoch 57. Test C 0.648 [0.604, 0.689], above age alone (0.630).
- **S5 sex/smoking adjustment** ✅ run on `cosine` (2026-10-06): HR/SD 1.64 [1.41, 1.91] →
  1.59 [1.37, 1.86] adjusted. ✅ `--adjust sex,cigsmok,age,lung_ml,lung_mean_hu` (2026-10-17): HR/SD
  1.64 → 1.37 [1.15, 1.62], p 0.0004. CNN adds risk info beyond lung size/density/age (age ρ 0.23 with η). ⬜ Time-dependent AUCs have no CI yet (bootstrap them like C; not built).
- **S4 run 3: seeds + training variants** (code 2026-10-17): new `s4_train.py` flags `--min-lr`,
  `--ema`, `--select last`, `--events-per-batch` (all default to the old behaviour).
  ✅ 40-epoch screen on `gpu-short` (2 epb4 runs cut at epoch 37 by the time limit; compared
  at epoch 37): ema 0.674 ± 0.006 vs epb4 0.695 ± 0.037 val C. Leaning epb4, not proven; EMA ≈ no
  effect. ⬜ **user decision:** final recipe + seeds. ⬜ Train it at 80 epochs (train split), S5.
  That is the baseline SSL must beat.
- **S4 final runs** `epb4_s0–2` (80 epochs, epb4 + EMA + last): s0 ✅ **overfits** (full-train C
  0.86 vs val 0.66; val loss worse than constant by epoch 80). s1/s2 ✅ confirm it (full-train
  0.97/0.90 vs val 0.70/0.67).
  → Recommended: keep `cosine` as the baseline; ⬜ optional `cosine_s1/s2` for seed spread.
- **S1b mask-selection check** ✅ (2026-10-17): kept vs dropped controls balanced on every
  clinical/scanner variable (|SMD| < 0.05, CV AUC 0.50–0.51); only processing order (old split,
  pid) differs. IPW weights 0.95–1.09 → not needed. Masked controls ≈ a random subset.
- **S6 saliency** ✅ on `cosine` (2026-10-17): saliency sits in the posterior lung bases in
  everyone (RLL 2.3×, LLL 1.6× their volume share; upper lobes ~0.1×), events ≈ controls; no
  cancer-lobe localisation (p 0.24 / 0.47, hit 0.24 vs 0.21 chance). A lung size/inflation score,
  not a nodule detector. ✅ Added discrimination (S5, 2026-10-17): test C clinical 0.654 → clinical + η 0.675,
  ΔC +0.021 [−0.017, +0.058] (not significant, 155 events); η alone 0.648. SSL target: ΔC CI > 0.
- **Supervisor meeting** (prep 2026-10-16): results + limitations + questions in PROGRESS.

## Open items

- Discussed, not built: select the epoch by val loss; a reference Cox model of
  lung volume + age.
