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

**Next:** user submits C1 → merge/QC → C2 grid → C3 → C2 final → C4 → C5.
Then fill doc section 9 and publish the web page.
