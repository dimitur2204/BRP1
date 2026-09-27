# E1 — Thymic health × cardiac calcium → lung-cancer incidence

**Status:** protocol locked before any TH-outcome analysis was run (2026-09-27).
Hypotheses H1–H4 (`research-state.yaml`). Everything below is CONFIRMATORY.
Anything else is labelled EXPLORATORY in `analysis.md`.

## Data (fixed)

- Cohort: `data/radiomics/heart_features_v1_r3.csv` (R3-QC'd, 6,205 pids;
  train 2,147/710 ev, val 2,018/154, test 2,040/154) ⨝ published NLST thymic
  health (`/faststorage/project/aura_thymus/discovery_pipeline/NLST_thymic_health_scores_published.csv`,
  `ID`, `Thymic_health_continuous` ∈ [0,100] percentile,
  `Thymic_health_categories` 0/1/2 = low ≤25th / average / high >75th).
  Complete case on TH (expected ~6,041). Missing-TH patients are reported
  (n, events), not imputed.
- Outcome: lung-cancer incidence, `time_yr`, `event` (manifest ≡ prsn candx_days).
- Clinical covariates (fixed set): `age`, `male`, `cigsmok` (current = 1).
- **Calcium (pre-specified):** `log1p(card_calc_mass_proxy)` from the *raw*
  R2 table `heart_features_v1.csv` (not ComBat'd, so it is interpretable),
  z-scored on the analysis cohort. Sensitivity: the ComBat-harmonized r3 value.
- Pericardial fat: raw `card_fat_volume_ml`, z-scored (used for H2 only).
- TH: continuous z-scored (HR per SD), plus categories (reference = low).

## Analyses

**A1 (H1) — TH replication, full cohort.** TH is an external, frozen score,
so no fitting on our data is involved, and all splits are used for power.
Cox PH, strata = split (enrichment sampling differs by split), covariates:
age, male, cigsmok + TH. Report the HR per SD and category HRs (avg vs low,
high vs low) with 95% CI. Also the unadjusted model.
*Prediction:* HR per SD < 1. High vs low HR ∈ [0.5, 0.85], p < 0.05.
*Sensitivity:* (i) administrative censoring at 6 yr (paper's LC-incidence
horizon); (ii) strata = split × sex × 5-yr age bin (paper's PH handling);
(iii) Schoenfeld PH test for TH.

**A2 (H2) — cross-organ correlation.** Residualize TH, log-calcium and
pericardial fat on age + male + cigsmok (OLS), then Spearman ρ between the
residuals, with bootstrap CIs (1,000). Also report raw Spearman.
*Prediction:* ρ(TH, calcium) ∈ [−0.15, −0.02]. ρ(TH, fat) < 0 with |ρ| > 0.1.

**A3 (H3) — independence, full cohort.** Cox (strata = split) with clinical +
TH + calcium, compared with clinical + TH and clinical + calcium. Report each
marker's log-HR attenuation = 1 − β_joint/β_single. Test the TH × calcium
interaction (EXPLORATORY-free: pre-specified as secondary).
*Prediction:* both p < 0.05 in the joint model, and each attenuation < 15%.

**A4 (H4) — incremental prediction, held-out test.** Fit Cox on train+val
(strata = split), predict the linear predictor on test. Models: clinical;
clinical + TH; clinical + calcium; clinical + TH + calcium. Harrell's C;
paired bootstrap (1,000 resamples of the test set, seed 0) for ΔC vs clinical
and vs clinical + calcium, with 95% CI and two-sided bootstrap p.
*Prediction:* ΔC(clinical+TH+Ca vs clinical) > 0 with CI excluding 0.
ΔC(vs clinical+Ca) > 0 as a point estimate.
Plus KM on test and on the full cohort by the 2 × 2 grouping
TH low (category 0) vs not × calcium top quartile vs not, with a log-rank test.

## Reporting rules

Event counts next to every HR/C. CIs that span the null are flagged. There is
no Brier/calibration (enriched cohort). Seven heart models were already
compared in R5, so the ΔC p-values are nominal.
