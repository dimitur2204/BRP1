# E6 — Thymic health + cardiac calcium without case-cohort sampling (TotalSegmentator-covered CT arm, ~16k)

**Status:** protocol locked 2026-09-27, **before the data exist**. It waits on
the user running `sbatch src/submit_r2_fullarm.sh` and the merge step.
Hypotheses H15–H17.

## Why

Calcium is currently measured only on the enriched case-cohort sample (all
cancers + ~5k controls). E5 showed that this sample dilutes TH (0.95 vs 0.91
per SD). The TotalSegmentator-covered CT arm is 16,004 TH-scored participants
with 1,021 of the 1,030 cancers. It matches the uncovered 9,027 on age, sex,
smoking and TH (mean 49.8 vs 50.3), and gives the same TH HR (0.909 vs
0.908 full arm), so it is representative.

## Data (fixed)

- Features: `data/radiomics/heart_features_v1.csv` (enriched cohort) ∪
  `data/radiomics/heart_features_fullarm.csv` (cardiac-only run). Identical
  `cardiac.py` code; the pyradiomics columns are ignored.
- Cohort covariates: `heart_cohort_v1.csv` ∪ `heart_cohort_fullarm.csv`. The
  outcome comes from those (manifest), TH from the published CSV, emphysema
  from ctab yr0 (as E2), histology from prsn (as E3).
- R3 exclusions applied identically: status ok, heart ≥ 250 mL, metal voxels
  < 5, heart not touching the z-edge, kernel group ≠ other.
- Calcium = log1p(card_calc_mass_proxy), z-scored on the analysis cohort (as
  E1). No split strata (no sampling). Cox adjusted for age + male + cigsmok
  unless stated.

## Analyses and predictions

**G1 (H15) — calcium HR without sampling.** HR/SD in [1.08, 1.25], p < 0.001.
Dose-response T3 vs none > 1.2.

**G2 (H16) — joint model.** Clinical + TH + calcium + emphysema. Both TH and
calcium p < 0.05. Each log-HR attenuates < 15% vs its single-marker model.
Partial ρ(TH, calcium) in [−0.15, −0.05]; ρ(TH, pericardial fat) < −0.2.

**G3 (H17) — TH × histology, calcium-adjusted.** Lunn–McNeil interaction
(adeno vs SQ/SC) with clinical + emphysema + calcium: HR < 1, p < 0.05.
SQ/SC TH HR ≤ 0.90. Calcium's own histology ratio is reported (E3 got 1.08).

**G4 — held-out prediction (manifest split; fit train+val, evaluate test).**
Harrell C for clinical; + emphysema; + emphysema + calcium; + emphysema +
calcium + TH; paired bootstrap ΔC (1,000). *Prediction:* calcium ΔC over
clinical + emphysema > 0. Also the SQ/SC-specific ΔC for TH.

Test events will be few, since nearly all cancers are in the enriched cohort's
splits. The counts are reported, and ΔC is descriptive when there are fewer
than 100 test events.
