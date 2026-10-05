# PROGRESS — lungs / sternum 3D CNN at scale

Running log. Plan/status: `PLAN.md`. Explanations: `docs/cnn3d_explained.md`.

## 2026-09-28 — branch, cleanup, C0–C5 code, smoke tests

**Branch** `cnn3d-lungs-sternum` from `main`.
- Removed: 2D CNN (`stage3_cnn.py`, figs, results), `stage4_ranking.py`, the
  5 non-target organs' figures, anterior-mediastinum saliency, and the
  superseded legacy scripts (`build_subset`, `cache_organ_crops`,
  `stage3_cnn3d`, `stage5_km_curve`, `stage6_saliency` + submits). They are
  all in git history.
- Kept: `data/legacy_400/` (lungs/sternum rows of the old results),
  `figs/legacy_400/`, `subset_v1.csv`.
- `organs.py` is reduced to lungs + sternum. It adds a bone window for sternum
  and a `LEGACY_WINDOW` for exact reproduction. `stage2_visualize_masks.py`
  is narrowed and its figures regenerated.

**Facts found while planning**
- Enriched cohort = all cases + controls sampled at **8.4% (train) vs ~50%
  (val/test)**, giving event rates of 33% vs 7.7%. See
  `cnn_cohort_v1_sampling.csv`.
- The manifest's `label` equals `event` exactly.
- 26% of cancers are diagnosed within 1 year of the baseline scan (median
  828 d). Lead-time analysis is essential for the lungs model.
- 399/400 legacy subset pids are in the cohort. The old test set (127/18) is
  fully inside the new test set (2,067/157).

**C0** (`src/c0_build_cohort.py`, a port of the radiomics R0): 6,306 pts /
1,037 events; train 2,191/725, val 2,048/155, test 2,067/157. pid/time/event/split
are identical to `heart-radiomics:heart_cohort_v1.csv`. The prsn clinical
table is optional. The TCIA download was **denied by the session's permission
classifier (PII)**, so clinical columns are empty for now. The cohort is
unaffected (that exclusion step removed nobody on the radiomics branch).

**C1** (`src/c1_cache_volumes.py`) smoke test on 3 + 40 pids:
- 2.9 s/pid, so 20 array tasks take ~16 min.
- The legacy path reproduces the old `resize_and_window` of the old cached
  crops to max |diff| 2.4e-4 (float16).
- **Sternum saturation confirmed:** 56–57% of in-mask sternum voxels are
  > 240 HU, the legacy upper bound, with median sternum HU 233–253. The old
  sternum "negative control" saw mostly clipped bone.
- In the legacy lungs input, outside-mask background has the same value
  (0.23) as aerated lung (visible in the QC figure).
- The new iso grid keeps true aspect. 1/40 lungs and 1/40 sternum exceeded
  the FOV slightly (`iso_clipped`); the count over the full cohort is
  reported at merge.

**C2/C3/C4/C5**
- `cox_ph_loss` equals a brute-force Breslow partial likelihood (diff 4e-8).
  Minimizing it reproduces sksurv's Breslow β (0.5756 vs 0.5756).
- CPU smoke runs (24 pids/split, 2 epochs) pass for new-recipe grid, legacy
  scale/subset, score_old, and final (main/bce/lc/null). Each writes
  config/history/pred/curves/model/result.
- The grid is 34 runs in 12 array tasks; final is 46 runs in 16 tasks.
- `c4_evaluate.py` smoke test (scratch runs, `--out-dir`) runs end to end:
  metrics table, paired ΔC, lead-time/kernel strata, score correlations,
  adjusted HRs, reference models, KM, learning curve, headline figure.
  - Fixed: C-index/AUC now return NaN on strata with no comparable pairs
    (e.g. sharp kernel with no events). Previously this crashed.
  - The runtime is dominated by imports and bootstraps (≈10 min on the login
    node at n-boot 20; plan for longer at 1,000).
- `c5_gradcam.py` smoke test: the NIfTI shape and affine equal the canonical
  CT's. Added `cam_frac_in_mask` per patient (share of CAM mass inside the
  organ mask). It is 0–2% for the 2-epoch smoke model, whose CAM sits in the
  padding, so this column is the check that the real model looks *inside*
  the organ rather than at its outline/size.

**Clinical data (same day):** the user downloaded the TCIA IDC-780 package
into `data/external/` and re-ran C0. Age/sex/race/cigsmok are 100% complete,
"has age/sex/smoking" dropped 0 pts, and manifest event vs prsn `candx_days`
disagree for 0 pids. C4's clinical adjustment is now active. The user ran C4
too early (no C2 runs yet) and got a KeyError; C4 now exits with a clear
message when no runs exist, and doc §8 no longer suggests running it before
training.

**Cleanup (user-approved):** deleted the gitignored legacy data for the 5
dropped organs: `data/organ_crops/{anterior_mediastinum,aorta,heart,liver,spleen}`
(~5.2 GB), their `models/*_cnn3d.pt`, and `data/saliency/anterior_mediastinum`.
The lungs/sternum legacy crops and checkpoints are kept (C4 re-scores the old
checkpoints).

**C1 run + merge (2026-09-29):** all 20 array tasks finished (1.3–5.9 s/pid).
The first C2 grid submission (job 1025992) failed in every task with
`FileNotFoundError: data/cnn_cache/lungs_index.csv` because `--merge` had not
been run yet. `--merge` now streams chunks into memmapped `.npy` files, so it
never holds the ~12 GB/organ arrays in RAM on the login node (4 min).
`load_organ` now names the missing merge step.
- lungs: 6,302 cached, 4 missing (0 events), FOV-clipped 11. Split n/events:
  train 2,189/725, val 2,046/155, test 2,067/157.
- sternum: 6,296 cached, 10 missing (1 event), FOV-clipped 77. Split:
  train 2,188/724, val 2,042/155, test 2,066/157.
- Merged arrays equal the chunk data (tail spot check). QC figures are in
  `figs/c1_inputs/` and look correct. 36–57% of in-mask sternum voxels
  exceed the legacy 240 HU bound; the new [-500, 1300] window covers them.

**C2 grid + C3 (2026-09-29):** grid job 1053159 finished all 34 runs.
Validation Harrell C by grid config (seed 0):
- lungs: 0.630–0.684. Best is lr 3e-4, wd 1e-2, dropout 0. lr 1e-4 is
  consistently lower (0.65–0.67). lr 1e-3 with dropout 0.3 stopped early
  (0.630).
- sternum: 0.541–0.556, a flat surface. Best is lr 1e-3, wd 1e-2, dropout
  0, which ties with 5 other configs within 0.005.
- Legacy recipe at scale (64³, BCE, 60 epochs, best val AUC), val C over 3
  seeds: lungs 0.640–0.665, sternum 0.548–0.586.
- Legacy recipe on the 400-pt subset: lungs 0.633, sternum 0.558.

→ `data/c3_chosen_configs.json`, `data/c3_val_grid.csv`, `figs/c3_val_grid.png`.

**C2 final (job 1073738, 16 tasks, ≤ 21 min each, 2026-09-29):** all 46 runs
have result/pred/model. pred.csv holds all splits (6,302 / 6,296 rows) with
finite η. The only stderr is the harmless non-writable-numpy warning.

Validation C, mean [min–max] over seeds:

| organ | main (5) | bce (5) | noaug (3) | lc subset_v1 (3) | lc 25% | lc 50% | null |
|---|---|---|---|---|---|---|---|
| lungs | 0.646 [0.624–0.664] | 0.647 | 0.620 | 0.623 | 0.628 | 0.618 | 0.541 |
| sternum | 0.557 [0.551–0.565] | 0.555 | 0.545 | 0.538 | 0.553 | 0.540 | 0.530 |

- **Winner's curse is visible.** The lungs grid winner had val C 0.684 at
  seed 0. The identical config (verified: configs differ only in seed)
  averages 0.646 over 5 new seeds. Val C swings ±0.03 per epoch, so the
  best of 12 configs × ~50 epochs is optimistic. Patience 12 also stops
  some seeds early: best epoch 13–36 vs 47 for grid seed 0. Test numbers
  from C4 are the honest ones.
- **The null runs' val C (0.53–0.54) is above 0.5 for the same reason:**
  early stopping picks the luckiest epoch on val. Their test C is the real
  null check.

**C4 (2026-09-29, n-boot 1000, ~20 min):** full write-up is in doc §9.
- Reproduction check passes: old checkpoints on the old 127/18 test set give
  binary AUC 0.688 (lungs) and 0.607 (sternum), the published values.
- The same checkpoints on the full test set (2,067/157) give C 0.615 and
  0.502. The old sternum result was noise.
- Test C:
  - lungs: legacy_scale 0.680, new_main 0.688 [0.652, 0.724], new_bce 0.689,
    noaug 0.656, null 0.520;
  - sternum: new_main 0.538, null 0.502.
- Paired ΔC:
  - data size (legacy_scale − old_ckpt) +0.065, p = 0.004;
  - new recipe − legacy_scale +0.008, p = 0.60 (n.s.);
  - Cox − BCE ≈ 0 for lungs;
  - aug − noaug +0.032;
  - lungs − sternum 0.151 [0.100, 0.203].
- Adjustment:
  - lungs HR/SD 1.87 → 1.68 after age/sex/smoking + kernel; ΔC clinical+CNN
    vs clinical +0.059 [0.025, 0.093];
  - sternum HR/SD 1.13 → 1.01; ΔC 0.000.
- The sternum score is ρ −0.76 with bone HU and +0.21 with age. Its
  unadjusted KM log-rank p = 0.035 is an age confound, not biology.
- Lead time: lungs 1-y landmark C 0.682 (vs 0.653 legacy_scale), so the
  model predicts future cancers, not just prevalent ones.
- Train C: new 0.69 vs legacy_scale 0.79. The new recipe overfits less.
- Learning curve still rising at 2,189 pts.
- Pitfall: wait loops must use `kill -0 <pid>`. `pgrep -f <script>` matches
  the loop's own command line and never ends.

**Figure review (2026-10-02).** An external review of the c* figures was checked against the code
and run artifacts. All points held:
- Cox curves: train loss used batch-local risk sets (32 pts), val the whole split. Zero-score
  Cox loss is 3.24 (batch 32) vs 7.52 (val), 7.45 (whole train), which explains the "3.2 vs
  7.5" gap entirely. Both curves were only ~0.1 below their own constant-score baseline.
- BCE curves: train weighted by pos_weight, val unweighted. The val spikes (lungs legacy_scale
  up to 45 at epoch 43, val C ~0.62) can't be diagnosed from what was saved: only the
  selected-epoch model and preds exist.
- Train C was the online, train-mode-BN, augmented number. The plots showed one seed without
  saying so.
- c3_val_grid.png: right-column y-labels covered the left column's cells.
- KM: no CIs/at-risk tables, independent y-axes, no enriched-cohort caveat; sternum tertiles
  not ordered (mid 61 events > high 58).

Fixes:
- `c2_train.py` logs optimisation loss (+ constant-score baseline) separately from an eval-mode,
  unaugmented, same-definition evaluation loss on train and val (Cox whole-split risk sets /
  unweighted BCE, each with its no-information baseline), eval-mode train C, and diagnostics
  (val logit quantiles, BCE pos/neg contributions, top-1% share, BN running stats). New
  `curves` run set (8 runs, 4 tasks): the plotted seeds re-trained full-length, plus
  BN-batch-stat val pass and per-epoch checkpoints, no test preds, excluded by C4.
- New `c2_plot_curves.py` → `figs/c2_training_curves/` + `data/c2_curves_spikes.csv`. The old
  `figs/c4_training_curves/` was removed (misleading; in git history). C4 no longer copies curves.
- `c3_select.py`: constrained layout, contrast-aware cell text. Re-run: the CSV/JSON are
  byte-identical, only the PNG changed.
- `c4_evaluate.py --figs-only` redraws figures without the bootstraps (~2 min). KM: 95% CI
  bands, at-risk/events tables, one y-axis across all KM panels of both organs, enriched-cohort
  footnote. Learning curve: seed-SD footnote, legacy SD bars where present.
- Pre-existing bug: with a NaN selection score on every epoch (smoke val subset has no events)
  `best_state` stayed None and the run crashed. Epoch 0 is now the fallback.
- Doc §6 has a new "Training curves" subsection; §6 KM row, §9.2 learning curve, §9.4 sternum KM
  updated.

**Curves runs (job 1361681, 2026-10-03):** all 8 runs finished (60/80 epochs each, no errors).
- Corrected curves: the lungs new recipe at its selected epoch has train C (eval) 0.720 vs val
  0.657. Legacy_scale has 0.792 vs 0.671 and reaches train C 0.90 by epoch 22. The null runs
  memorise permuted labels (train C 0.66, val ~0.5). Sternum barely learns (val Cox excess
  −0.02 nats per event vs −0.17 for lungs).
- Re-run nondeterminism: lungs main s1 selected epoch 48 (val C 0.657) vs epoch 5 (0.624) in
  the original run.
- **BCE spikes are stale BatchNorm running stats.**
  - Lungs legacy_scale epoch 39: val median logit +31.4 (min +17.8). All of the loss comes from
    negatives, the top-1% share is 1.5%, and val C is 0.631.
  - The train eval-mode mean logit tracks the val median at r = 1.000, so this is not a val
    shift. BN batch-stats val BCE excess is 1.18 vs 28.7.
  - Decisive test (scratch `bn_recal.py`, CPU): re-estimating the running stats over all train
    patients on the epoch-39 checkpoint moves the val median +31.4 → −2.3, BCE 29.1 → 1.22,
    C 0.631 → 0.619. Epoch 40 (stored offset −14.7, low BCE because negatives dominate) recalibrates to −2.5, the same level.
  - Mechanism: EMA momentum 0.1 over batch-8 batches under fast-changing Adam weights. BN4
    error becomes a shared logit shift. The new recipe shows the same effect, smaller.
- Rankings and C4 headline unaffected: an additive offset is invisible to Cox and C, and C4
  val-standardises per seed. No architecture change needed. "Precise BN" would be the fix if
  calibration is ever needed.
- Old `c4_training_curves` remain deleted. Doc §6 now has the findings table and the spike
  analysis.

**Next:** user: `sbatch src/submit_c5_gradcam.sh` → doc §9.6 → web page.
