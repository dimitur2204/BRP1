# Heart radiomics → lung-cancer risk (DeepSurv + baselines → KM)

## Question

Do quantitative **radiomic features of the heart** on the NLST baseline
low-dose CT carry information about **future lung-cancer incidence**, beyond
basic clinical covariates? Deliberately exploratory and novel: a null result
is an acceptable, reportable outcome. It is not a failure.

The previous multi-organ CNN pipeline (Stages 0–6, commit `eb3b89b`) found
the heart **at chance** for this outcome (3D CNN test AUC CI spanning 0.5,
n=127 test). This project asks the same question with a completely different
representation: roughly 100 hand-defined, interpretable features instead of
learned image filters, on about 16× more patients.

## Feasibility assessment (why this is worth doing, and what can go wrong)

**Plausible routes by which heart features could relate to lung cancer:**
1. **Smoking burden proxy.** Coronary calcification tracks cumulative smoking
   exposure. We have smoking *status* but **not pack-years**, so heart features
   may pick up the unmeasured intensity of smoking. That would be real signal,
   but it is a proxy, not cardiac biology. It must be interpreted that way.
2. **Emphysema / hyperinflation.** Hyperinflated lungs compress the heart
   and make it more vertical, so heart volume and shape change. Emphysema is
   an established lung-cancer risk factor. Heart *shape* features could
   therefore carry lung signal indirectly.
3. **Age / systemic inflammation.** Heart density, calcium and pericardial fat
   all change with age. This is why the clinical baseline is mandatory.

**Threats:**
- **Scanner/kernel confounding.** Radiomic texture features are highly
  sensitive to reconstruction kernel. The manifest mixes STANDARD, B30f, C,
  FC51, B50f, BONE and others. Stage R3 must probe and harmonize this, or
  "signal" may just be which site scanned which patient.
- **Non-contrast, low-dose CT.** Heart chambers are not separable and texture
  is noisy. Shape, first-order statistics, calcium and fat are expected to be
  more robust than higher-order texture.
- **Case-enriched cohort.** The enriched 6,373 cohort contains all 1,061
  events but is not a random sample, and its splits are unbalanced (train
  742/2,226 events = 33%; val 159/2,067 and test 160/2,080 ≈ 8%). Rankings
  (C-index, KM separation, HRs) remain valid. **Absolute risk / calibration
  / Brier scores are biased** and must be labelled as such.
- **Multiple testing.** About 110 features are screened univariately, so
  Benjamini–Hochberg FDR is applied and reported.
- **Neural nets vs linear models on tabular data.** DeepSurv frequently fails
  to beat Cox-LASSO at this sample size. The comparison itself is a result,
  so it is reported either way.

**Verdict:** feasible and methodologically sound as an exploratory study.
Compute is cheap (CPU-only extraction). Test-set power is far better than the
CNN work (about 160 test events vs 18, so roughly 50 per KM tertile vs 6). The
main scientific risk is confounding (kernel, age, smoking), which the plan
addresses explicitly rather than hoping away.

## Decisions (confirmed with user, 2026-09-26)

- **Outcome:** lung-cancer incidence from `experiments/lcrisk_discovery/data/manifest.csv`
  (`time` = days, `event`), same as the CNN pipeline. No mortality outcome
  (not in our IDC-780 package).
- **Cohort:** `discovery_pipeline/enriched_cohort_pids.txt` ∩ manifest =
  **6,373 patients, 1,061 events**. The mentor's `split` column is reused as-is.
- **Models:** DeepSurv (MLP + Cox partial-likelihood loss, plain torch) as the
  neural network, benchmarked against clinical-only Cox, Cox elastic-net and a
  Random Survival Forest.
- Old CNN-pipeline PLAN/PROGRESS were committed (`eb3b89b`), then removed.
  `src/stage*` CNN code stays in the repo untouched. `src/organs.py` is reused.

## Blockers / open items

- **Clinical covariates are not readable by us.** `meta/nlst_780/nlst_780_prsn_idc_20210527.csv`
  (age, sex, race, cigsmok) is `rw-rw----` owned by `mer:mer`, and we are
  in group `aura_thymus`, not `mer`. The mentor needs to grant access (e.g.
  `chgrp aura_thymus` + `chmod g+r`, or put a copy somewhere readable).
  **Without it, the clinical baseline and confound adjustment are impossible,**
  and results can't separate heart signal from age. Stages R0–R2 can proceed
  meanwhile.
- The cardiac sanity label (`ctab` code 60, "Significant cardiovascular
  abnormality") sits in `nlst_780_ctab_idc_20210527.csv`, which has the same
  permission problem.
- Checked alternatives (2026-09-26), and none carries clinical covariates:
  `nlst.csv` has `PatientAge`/`PatientSex` 100% empty. The second copy at
  `discovery_pipeline/nlst_root/meta/nlst_780/` is also owner-only.
  `derived/` contains only images, masks and embeddings.

## Staged pipeline (status: nothing built yet)

**R0 — Cohort build.** ⬜ Enriched pids ∩ manifest. For each pid, check that
`derived/ct_1x1x2mm/{pid}_yr0.nii.gz` and `derived/totalseg_fullres/{pid}_yr0/seg.nii.gz`
exist (lookup by name only, never enumerate). Join kernel/kVp from the
manifest, and manufacturer/model plus acquisition parameters parsed from
`nlst.csv` `SeriesDescription` (kernel, kVp, mA, slice thickness), for R3
harmonization; join clinical data once
accessible. Output `data/pid_lists/heart_cohort_v1.csv` plus an attrition
table (how many dropped, how many events lost, per split).

**R1 — Visual QC.** ⬜ Heart mask (TotalSegmentator label 51) overlaid on CT
for about 10 patients spanning different kernels/manufacturers. Plot the
in-mask HU histogram. Look specifically at the lung/fat border (partial volume)
and whether coronary calcium falls inside or just outside the mask.
Output: `figs/r1_heart_qc/`.

**R2 — Radiomics extraction (Slurm array, CPU `short`).** ⬜
`src/r2_extract_radiomics.py` + `src/submit_r2_radiomics.sh`.
- pyradiomics 3.1.0 (has a wheel for our py3.9 venv). Image
  `ct_1x1x2mm`, mask = seg==51. Resample to 2×2×2 mm isotropic (B-spline
  image, nearest-neighbour mask; IBSI-recommended for 3D texture). Fixed
  bin width 25 HU, mask eroded by 1 voxel to limit lung/fat partial volume.
- Feature classes: `original` only (shape 14, first-order 18, GLCM 24,
  GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5 ≈ 107). No wavelet/LoG filters: those
  would add about 1,000 features, which is too many for 1,061 events.
- **Hand-crafted cardiac features** (the most likely carriers of signal):
  heart volume; **calcium burden proxy** (volume and density-weighted
  sum of voxels ≥130 HU within the heart mask dilated 3 mm, excluding
  bone/aorta labels; explicitly *not* a clinical Agatston score on 2 mm
  resampled low-dose CT); **pericardial fat volume and mean HU** (−190…−30 HU
  in a 0–10 mm shell around the heart, excluding lung).
- Per-patient output: one row. Merged to `data/radiomics/heart_features_v1.csv`.
  Failures are logged, not silently dropped.

**R3 — Feature QC, harmonization, univariate association.** ⬜
- Drop features that are constant, missing or near-zero-variance. Cluster
  features with |Spearman| > 0.95 and keep one representative each.
- **Kernel-leakage probe:** how well do the features predict kernel group
  (soft/standard vs sharp)? A high AUC means ComBat harmonization by kernel
  group (fit on training folds only) or restriction to the robust subset
  (shape + first-order + cardiac).
- **Univariate Cox per feature** (HR per SD, 95% CI), unadjusted and adjusted
  for age/sex/smoking, with BH-FDR. Volcano/forest plot. This is the most
  interpretable answer to "is there *any* relation."

**R4 — Models (train+val via 5-fold CV for tuning; test held out once).** ⬜
1. Clinical-only Cox (age, sex, smoking status): the bar to beat.
2. Cox elastic-net on radiomics; the same with clinical covariates added.
3. Random Survival Forest (scikit-survival).
4. **DeepSurv**: MLP (e.g. 2 × 32–64 hidden units, dropout, weight decay),
   Cox partial-likelihood loss, early stopping on validation C-index, and
   standardization fit on the training fold. Run on radiomics alone and on
   radiomics + clinical.
5. Null control: DeepSurv on permuted survival labels, which should give a
   C-index of about 0.5.

**R5 — Evaluation on held-out test (2,080 patients, 160 events).** ⬜
- Harrell's and Uno's C-index with bootstrap 95% CI. **ΔC vs the clinical
  model** (paired bootstrap): the key incremental-value number.
- Time-dependent AUC at 2/4/6 years. HR per SD of each model's risk score
  (unadjusted and clinical-adjusted).
- **Kaplan–Meier** risk-score tertiles plus a multivariate log-rank trend
  test (same convention as `discovery_pipeline/src/km_analysis.py`), with
  number-at-risk tables.
- Brier / calibration are reported **only with the enrichment caveat**.
- Interpretability: permutation importance (and SHAP for DeepSurv). KM curves
  for the top 1–2 individual features (e.g. calcium proxy tertiles).
- Robustness: results stratified by kernel group / manufacturer.

**R6 — Controls on the same pipeline (recommended, cheap).** ⬜ Rerun R2–R5
with `lungs` (positive control: lung-cancer signal is expected) and `sternum`
(negative control). If lungs radiomics clear the clinical baseline and
sternum doesn't, the pipeline is validated and a heart null result means
something.

**R7 (optional).** ⬜ Cardiac sanity check: do heart features predict the
radiologist's "significant cardiovascular abnormality" flag (ctab code 60)?
This confirms the features encode cardiac information even if lung-cancer
signal is null. It depends on ctab access.

## Environment additions

`env/.venv/bin/pip install pyradiomics==3.1.0 SimpleITK scikit-survival neuroCombat shap`
(the venv is actually **Python 3.9.25**, not 3.11 as CLAUDE.md states).

## Reporting norms (carried over)

Weak/null results are reported plainly. Any metric whose CI spans chance is
flagged "interpret with caution". Event counts per split/tertile are stated
wherever risk scores or KM curves appear. Multiple-comparisons (FDR) and
enrichment caveats are always attached.
