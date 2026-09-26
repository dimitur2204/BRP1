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

**Next:** R0 cohort build + R1 visual QC (both local, light). Then write the
R2 extraction script and sbatch array for the user to submit.
