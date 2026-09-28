# 3D CNN on lungs + sternum → lung-cancer risk: retrain at scale, fix, explain

Branch `cnn3d-lungs-sternum` (from `main` b423d60). The heart-radiomics work
lives on `heart-radiomics` / `thymus-heart-aging`; the original 7-organ CNN
pipeline is in git history (`eb3b89b`).

## Question

The legacy 3D CNN (400-patient subset, 18 test cancers) reported lungs test
AUC 0.687 [0.54, 0.83] and sternum 0.607. Retrain it on the full enriched
cohort (6,306 pts / 1,037 cancers; test 2,067 / 157), on **lungs (positive
control) and sternum (negative control) only**. Find out how big the real
difference between them is and *why* it exists. Explain every design choice
(`docs/cnn3d_explained.md` plus a published web page).

## Decisions (user, 2026-09-28)

- Cohort: the enriched 6.3k (same pids as `heart-radiomics:heart_cohort_v1.csv`).
  The mentor's split is reused.
- Objective: Cox partial likelihood as primary, BCE as an ablation.
- Output: repo markdown plus a web page.
- Cleanup: 2D CNN, other 5 organs and the multi-organ ranking are removed. Old
  lungs/sternum numbers and figures are kept in `data/legacy_400/` and
  `figs/legacy_400/`.

## Legacy recipe defects being fixed

1. Every crop is squashed to a 64³ cube, so volume and shape are lost.
2. The sternum used the soft-tissue window [−160, 240], saturating most of
   the bone. The negative control was blinded.
3. Outside-mask voxels were set to −1000 HU, the same as emphysematous lung,
   with no mask channel.
4. BCE ignores time and censoring.
5. Selection used the best val AUC over 60 epochs on 18 val cancers: no early
   stopping, one seed, no augmentation.
6. Results were read off a 0.5 threshold, which is meaningless under
   `pos_weight` and the prevalence shift.
7. Hanley–McNeil CIs were used instead of the bootstrap.
8. The crop bbox was never stored.

## Stages

- **C0 — cohort** ✅ `src/c0_build_cohort.py` → `data/pid_lists/cnn_cohort_v1{,_attrition,_sampling}.csv`.
  6,306 / 1,037 events, pid-identical to heart_cohort_v1. Clinical columns
  are complete (the user downloaded the TCIA prsn package; the "has
  age/sex/smoking" step dropped nobody).
- **C1 — input cache** ✅ code, smoke-tested (legacy path matches the old
  cache to within 2.4e-4). ⬜ **user: `sbatch src/submit_c1_cache.sh`**, then
  `--merge` and `--qc`.
- **C2 — training** ✅ code, CPU smoke-tested (grid, legacy, score_old, final
  sets). Cox loss verified. ⬜ **user: `sbatch --array=0-11 src/submit_c2_train.sh grid`**.
- **C3 — selection** ✅ code. ⬜ run after grid → `data/c3_chosen_configs.json`.
  ⬜ **user: `sbatch --array=0-15 src/submit_c2_train.sh final`**.
- **C4 — evaluation** ✅ code. ⬜ run after final.
- **C5 — Grad-CAM** ✅ code. ⬜ **user: `sbatch src/submit_c5_gradcam.sh`** after C4.
- **Doc + web page** 🔄 `docs/cnn3d_explained.md`: sections 0–8 written;
  section 9 (results) is pending C4. The web page will be published once the
  results are in.

## Open items

- None besides the pending Slurm runs.
