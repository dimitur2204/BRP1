# PROGRESS — heart radiomics → lung-cancer risk

Running log of what was built, found and decided. The plan and status live in `PLAN.md`.
The previous CNN pipeline's log is in git history (commit `eb3b89b`).

## 2026-09-26 — Pivot + feasibility check

**Pivot:** from per-organ 3D CNNs to heart radiomics + DeepSurv, with
lung-cancer incidence as the outcome (user decision: exploratory, novel, a null
result is acceptable).

**Facts established while checking feasibility:**
- Manifest (`experiments/lcrisk_discovery/data/manifest.csv`): 26,254
  baseline scans, 1,061 lung-cancer events. `time` is in days (median 2,424,
  max 2,983, so about 8 years of follow-up).
- Enriched cohort ∩ manifest: 6,373 pids with **all 1,061 events**. Split
  event counts: train 742/2,226, val 159/2,067, test 160/2,080. The heavy
  imbalance between train and val/test is by construction (mentor's
  enrichment), so calibration on this cohort is biased.
- Kernels are heterogeneous: STANDARD 13,738, B30f 7,219, C 2,184, FC51
  1,118, B50f 822, FC10 419, B 320, BONE 319 (full manifest). This is a major
  radiomics confounder and has to be handled in R3.
- `ct_1x1x2mm` and `totalseg_fullres` seg share the same grid (checked for
  pid 132386: 320×320×133, 1×1×2 mm). Heart = TotalSegmentator label 51
  (`organs.NAME_TO_LABEL`).
- The venv is **Python 3.9.25** (CLAUDE.md says 3.11, which is out of date).
  pyradiomics 3.1.0 has a cp39 manylinux wheel, and scikit-survival 0.23.1
  is available.
- **IDC-780 clinical data has no mortality, cause of death, pack-years,
  BMI or comorbidities.** Per the data dictionary, `prsn` only has age,
  gender, race, cigsmok, screening results and lung-cancer fields. `ctab` has
  code 60 "Significant cardiovascular abnormality" (a possible cardiac sanity
  label).
- **Blocker:** `meta/nlst_780/*.csv` are `rw-rw---- mer:mer`, so they are
  unreadable for us (group `aura_thymus`). Mentor access is needed for
  clinical covariates. The world-readable `.zip` copies were deliberately
  not used, since the restriction on the CSVs looks intentional. Confirm
  with the mentor.

**Housekeeping:** committed Stage 6 + final CNN PLAN/PROGRESS (`eb3b89b`).
Added `data/saliency/` (254 MB of regenerable NIfTI) to `.gitignore`. Then
removed the old PLAN.md/PROGRESS.md.

**Follow-up search for clinical data (user asked):**
- `nlst.csv` (203,087 series / 26,254 patients) is IDC DICOM series metadata.
  `PatientAge`, `PatientSex`, `SeriesDate`, `AnalysisResult` are 100% empty
  (de-identified). Useful part: `SeriesDescription` is a comma-coded
  acquisition string, e.g. `1,OPA,GE,LSULT,STANDARD,380,1.2,120,60,44.4,1.4`
  (study yr, ?, vendor, model, kernel, ?, ?, kVp, mA, ?, ?). The exact field
  meanings still need to be confirmed in R0 before use.
- `derived/` holds only images/masks/embeddings/saliency, no tables.
- `discovery_pipeline/nlst_root/meta/nlst_780/nlst_780_prsn_idc_20210527.csv`
  is another copy of prsn, also `rw-r----- mer:mer`, so it is unreadable too.
- `discovery_pipeline/NLST_thymic_health_scores_published.csv` (25,030 pids:
  continuous + categorical thymic health) is readable. It isn't needed now,
  but could be a later covariate/comparison.
- Conclusion: the only missing inputs are age, sex, cigsmok (prsn) and the
  ctab code-60 flag. Both need the mentor to grant access.
- CLAUDE.md updated for the radiomics project (Python 3.9, new deps, access
  rule, new source-of-truth cohort file).

**Clinical data unblocked (same day):** the IDC-780 package is public on
TCIA (collection page → `package-nlst-780.2021-05-28.zip`, CC BY 4.0).
Downloaded to `data/external/nlst_780/` (gitignored; regeneration command is
in `.gitignore`). `cmp` confirms prsn.zip is byte-identical to the mentor's
restricted copy. prsn: 53,452 rows, 39 cols, all 6,373 enriched pids present.
ctab code 60: 4,564 rows / 2,714 pids across study years 0–2. Full NLST
(death, pkyr, comorbidities) would still need a CDAS request.



## 2026-09-26 — R0 cohort, R1 heart QC, R2 extraction ready

**R0 (`src/r0_build_cohort.py`, runs locally in ~20 s)** → `data/pid_lists/heart_cohort_v1.csv`
+ `heart_cohort_v1_attrition.csv`.

| step | n | events | test n / events |
|---|---|---|---|
| enriched ∩ manifest | 6,373 | 1,061 | 2,080 / 160 |
| series is baseline (study yr 0) | 6,359 | 1,047 | 2,079 / 159 |
| has CT + TS mask | 6,359 | 1,047 | 2,079 / 159 |
| CT z ≥ 80 slices | 6,306 | 1,037 | 2,067 / 157 |
| has age/sex/smoking | 6,306 | 1,037 | 2,067 / 157 |

- `nlst.csv` `SeriesDescription` decoded and validated: f4 = kernel (matches
  manifest 100%), f7 = kVp (99.5%), f5 = recon FOV mm, f6 = slice thickness
  mm (thickness × image count ≈ 34 cm of chest), f8 = tube current mA. 2
  malformed thicknesses (350.2) were set to NaN.
- **All 14 non-baseline series (study yr 1/2 labelled as baseline in the
  manifest) were event patients.** The manifest apparently fell back to a
  later scan when the T0 scan was missing. Excluded; worth telling the mentor.
- Manifest `event` agrees exactly with prsn `candx_days` (0 disagreements).
- Cohort profile (event vs non-event): age 63.6 vs 61.3, current smokers 59%
  vs 48%, male 60% vs 61%, **sharp kernel 16% vs 17%** (kernel is not
  associated with outcome, so less risk of confounding via the outcome itself).
- Kernel groups (manual mapping in R0): soft 5,253, sharp 1,051, other 2
  (EXPERIMENTAL7).

**R1 (`src/r1_heart_qc.py`, `src/cardiac.py`)** → `figs/r1_heart_qc/` (9 patients: one per
manufacturer × kernel group, alternating event, plus the only 5 mm-slice scan).
- TotalSegmentator heart masks look correct on all 9: no spill into liver or
  lung, and in-mask HU median ~40 with <1.2% of voxels below −100 HU.
- The pericardial fat shell hugs the heart as intended (67–326 mL, mean
  −79 to −92 HU).
- **Calcium definition initially failed on sharp kernels:** core noise SD is
  45–55 HU on BONE/B50f/Philips C vs 12–28 on soft kernels, so 130 HU
  thresholding produced 386–1,156 speckle "lesions" (3–11 mL). Compared raw /
  Gaussian 0.7 mm / 1.0 mm / adaptive threshold (mean + 4 SD). **Chose 0.7 mm
  Gaussian smoothing before thresholding, applied uniformly.** Noise SD then
  becomes 9–24 HU on every kernel, and calcium is 0–0.8 mL with real lesions
  kept (e.g. 108121: posterior, probably mitral-annulus calcification; 200397:
  aortic root / coronary). 1.0 mm suppressed real small lesions on soft
  kernels.
- Kernel labels are an imperfect sharpness proxy: Toshiba FC51 ("sharp") has
  core noise 23 HU, like the soft kernels. So the measured `card_core_noise_sd_hu`
  is kept as the harmonization covariate.
- Known limitation: calcium that TotalSegmentator labels as aorta (aortic
  root/ostia) is excluded from the calcium zone by design.
- Heart texture SD roughly doubles on sharp kernels (heart_sd 50–58 vs 20–33
  HU), so raw texture features will be kernel-dominated. This is the main
  R3 harmonization job.

**R2 (`src/r2_extract_radiomics.py`, `src/submit_r2_radiomics.sh`)**
- Env: pyradiomics 3.1.0's C extension won't import on NumPy 2 → pinned
  `numpy==1.26.4` in `env/.venv`. torch 2.8, lifelines, pandas, sklearn and
  nibabel were re-verified.
- Local test on 3 pids: 122 columns (shape 14, first-order 18, GLCM 24, GLRLM
  16, GLSZM 16, GLDM 14, NGTDM 5, card_ 10), no NaNs, ~2.6 s/patient.
  pyradiomics MeshVolume matches card_heart_volume_ml (607 vs 608 mL).
- Full run: 20-task array on `short`, ~315 pids/task, expected ~15–20 min.
  **Pending: user runs `sbatch src/submit_r2_radiomics.sh`, then
  `env/.venv/bin/python src/r2_extract_radiomics.py --merge`.**


## 2026-09-26 — R2 run 1 done → outlier QC → mask fix, R2 re-run needed

**Run 1:** 20/20 tasks OK, 6,295/6,306 extracted, 11 `no_mask` (1 event),
median 2.4 s/patient. No NaNs and no constant features across 117 features.
Medians are plausible (heart 499 mL, fat 191 mL, calcium present in 87%), but
177 patients hit at least one outlier criterion (volume <250 or >1,100 mL,
mean HU <20, core noise >60 or <5 HU, calcium >5 mL). Rendered 9 of them
(`figs/r1_heart_qc/outliers/`, via the new `r1_heart_qc.py --pids`):
- **Detached "heart" blobs** (103665): TS labelled 34 + 20 mL of upper
  mediastinum as heart (one piece at 314 HU), which inflated calcium to 39 mL.
  In normal scans the heart label is one component (extra pieces <0.1 mL). →
  **Heart = largest connected component**, and dropped pieces are relabelled so
  they aren't counted as fat or calcium zone. 103665 afterwards: 321 mL, 3.2
  mL calcium, which on visual check is real, dense mitral-annular calcification.
- **Garbage / non-covering scans** (130827, 116761, 134498): striped,
  heart volume 3–23 mL, meaningless features. → exclude on volume in R3.
- **Metal** (215394: wires/device streaks, 333 voxels ≥2000 HU in the heart;
  every other checked scan has 0). Metal ruins both calcium (18 mL) and
  texture. → flag `qc_metal_voxels`, exclude in R3.
- **Real cardiomegaly** (119777, 1,107 mL): the mask is good, keep.
- **Very noisy sharp-kernel scans** (100456 GE LUNG, noise 80 HU): not
  broken, just the kernel. Kept, and handled via the noise covariate +
  harmonization.

**Code change:** `cardiac.heart_crop` now returns the largest component plus
`qc_*` fields (`qc_heart_n_components`, `qc_heart_dropped_ml`,
`qc_heart_touches_z_edge`, `qc_metal_voxels`). R2 writes them per patient.
Exclusion thresholds are applied in R3, not at extraction, so they can be
revised without re-extracting. Checked on 5 outliers.

**Run 2** (job 777378): 20/20 tasks done, merged 6,295 ok / 11 no_mask. A
duplicate submission (job 779102) was cancelled after 1–4 pids/task. Its
partial chunks were overwritten by 777378 (checked via timestamps and per-task
row counts), so the merged file is entirely run 2.

## 2026-09-26 — R3: exclusions, harmonization, univariate association

`src/r3_feature_qc.py` (local, ~1 min). Env: installed scikit-survival 0.23.1
(downgraded scikit-learn 1.6.1 → 1.5.2, which it requires) and neuroCombat.
FDR uses scipy's `false_discovery_control` (statsmodels isn't installed).

**Exclusions** (`r3_attrition.csv`): 6,306 → 6,205 patients, 1,037 → 1,018 events.
The steps: no mask −11, heart <250 mL −63, metal −23, heart cut off at scan
edge −4. **Final: train 2,147/710, val 2,018/154, test 2,040/154.**

**Harmonization works** (`r3_scanner_leakage.png`). Yeo-Johnson, then ComBat
with batch = manufacturer × kernel group (8 batches, train n 20–1,138),
preserving age/sex/smoking, fit on train and applied to val/test. Leakage AUC
on val+test: kernel 0.89 → 0.54, manufacturer 0.92 → 0.52. Texture was the
worst family (0.85/0.88) and shape was barely affected (0.53/0.56). Biology
survives: heart volume → sex AUC 0.86 before and after.

**Redundancy:** 115 → 57 features (|Spearman| > 0.95, cardiac/shape features
kept preferentially).

**Univariate Cox** (discovery = train+val, n=4,165, 864 events, stratified by
split; HR per SD):
- Clinical reference: age HR 1.08/yr, current smoker 1.66, male 0.96 (ns).
- FDR<0.05: unadjusted 39/57, **adjusted for age/sex/smoking 23/57**,
  +noise/kernel 8/57.
- **Test replication** (23 features: adj-FDR hits ∪ top 10; Bonferroni
  threshold 0.05/23 = 0.0022):
  - **`card_calc_mass_proxy` (density-weighted cardiac calcium): discovery HR
    1.12 [1.04, 1.20], test HR 1.32 [1.13, 1.55], p=0.0006. It replicates, and
    it also survives the noise/kernel adjustment (adj_tech FDR 0.007).**
  - `card_calc_volume_ml`: test HR 1.28 [1.09, 1.49], p=0.0022, right at the
    Bonferroni line (same signal, highly correlated).
  - **None of the ~20 texture/first-order/shape discovery hits replicate**
    (all test CIs cross 1). Most also weaken or vanish after adjusting for
    measured noise. This is consistent with residual acquisition/body-habitus
    effects, not robust biology.
- **Interpretation (don't overstate):** heart calcium burden is associated
  with lung-cancer incidence *beyond age, sex and smoking status*. The most
  plausible explanation is that calcium tracks cumulative smoking exposure
  (pack-years, which we don't have; see `docs/future_data_request_cdas.md`)
  rather than a cardiac mechanism. This is an association, not yet
  incremental prediction (R4/R5 will show ΔC-index).
- Nuance: the measured noise covariate partly reflects body size (larger
  patients produce noisier low-dose scans), and BMI is itself inversely
  linked to lung cancer. So the `adj_tech` model may over-adjust. It is
  reported as a sensitivity analysis, not the primary model.


## 2026-09-26 — R4 models + R5 held-out test evaluation

**R4 (`src/r4_models.py`, ~13 min on CPU on the login node).** Hyperparameters
were tuned by val C with each model fit on train. The chosen config was then
refit on train+val (Cox stratified by split, DeepSurv loss per split, RSF with
a split indicator that is constant on test), and the test split was predicted
once. Outputs: `data/r4_test_predictions.csv`, `r4_val_tuning.csv`,
`r4_chosen_configs.json`.
- Val C: clinical 0.613, +calcium 0.618, Cox-EN heart 0.598 / all 0.630 (pen
  0.03, l1 0.5, 12 nonzero), RSF 0.635 (leaf 15), DeepSurv heart 0.588 / all
  0.638 (hidden 16, dropout 0.5, wd 1e-2, 351 epochs).
- DeepSurv heart-only picked **8 epochs**, meaning val C peaked almost
  immediately. That's a sign there is little stable heart-only signal to learn.

**R5 (`src/r5_evaluate.py`).** Test set: n=2,040, 154 events (~51/tertile).
1,000 paired bootstrap resamples. Uno's C (τ=7 yr) was within ±0.003 of
Harrell's for every model.

| model | test C [95% CI] | ΔC vs clinical [95% CI] | HR/SD adj. clinical |
|---|---|---|---|
| clinical (age, sex, smoking) | 0.647 [0.604, 0.691] | — | — |
| clinical + calcium (pre-specified) | 0.658 [0.616, 0.700] | +0.011 [−0.002, +0.023], p=0.08 | calcium score 2.99* |
| calcium alone (raw feature) | 0.605 [0.559, 0.650] | −0.042 | 1.33 [p=0.0006] |
| Cox-EN heart only | 0.600 [0.553, 0.644] | −0.048 | 1.17 (p=0.046) |
| **Cox-EN heart + clinical** | **0.662 [0.619, 0.702]** | **+0.014 [+0.001, +0.027], p=0.03** | 1.82* |
| RSF heart + clinical | 0.621 [0.577, 0.666] | −0.026 | 1.16 (p=0.08) |
| DeepSurv heart only | 0.571 [0.523, 0.621] | −0.076 | 1.14 (p=0.14) |
| DeepSurv heart + clinical | 0.641 [0.598, 0.682] | −0.006 | 1.29 (p=0.08) |
| DeepSurv null (shuffled labels) | 0.497 [0.452, 0.546] | −0.151 | 0.98 |

\* For models that already contain the clinical covariates, the
"clinical-adjusted HR" is collinear by construction and not interpretable.

Time-dependent AUC at 2/4/6 yr: clinical 0.64/0.65/0.66, Cox-EN all
0.66/0.66/0.68, clinical+calcium 0.65/0.65/0.67.

KM tertiles (`r5_km_tertiles.png`): every panel separates (log-rank trend p ≤
0.002), including raw calcium alone (p=2e-5: high-tertile 75 vs low-tertile
32 events). Clinical + heart models give the cleanest low tertile: Cox-EN all
24 events low vs 88 high, compared with clinical's 30 vs 85.

**Read (plain):**
- Heart features carry real but small lung-cancer information. Heart-only
  models reach C ≈ 0.57–0.61, which clears chance (null control 0.50) but sits
  well below the clinical model. None of the heart-only CIs span 0.5.
- **The incremental value over clinical is small:** best ΔC +0.014 (Cox-EN,
  CI just excludes 0), and pre-specified calcium +0.011 (CI touches 0). Both
  are driven by the calcium signal found in R3. With 154 test events,
  differences of this size are at the edge of detectability.
- **Neural nets did not help.** DeepSurv ≈ clinical (−0.006) and RSF is worse
  (−0.026, overfitting on 710 train events). The linear elastic-net Cox won,
  as expected for ~60 tabular features and ~860 events. The DeepSurv null at
  0.497 confirms the training/evaluation pipeline doesn't leak.
- The likely mechanism remains calcium as a proxy for cumulative smoking.
  Pack-years (CDAS request) would test this directly.
- Caveats: the enriched cohort (no calibration/Brier reported), a single
  test split, 7 models compared (the multiple-comparisons caveat applies to
  the Cox-EN p=0.03), and calcium was chosen on train+val, not test.

**Next options:** R6 controls (lungs positive, sternum negative) on the same
pipeline; a sensitivity analysis without ComBat; commit.
