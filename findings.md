# Research Findings

## Research Question

Do CT-derived markers of immune aging (thymic health) and vascular aging
(cardiac calcium, pericardial fat) carry independent, additive information
about lung-cancer incidence on NLST baseline low-dose CT, or do they proxy
the same unmeasured smoking burden?

## Current Understanding

*Starting point (from heart-radiomics R0–R5, before this branch):* a cardiac
calcium burden proxy is associated with lung-cancer incidence beyond age, sex
and smoking status (test HR/SD 1.32), but adds only +0.011 C-index to a
clinical Cox model. Heart texture features do not replicate. Neural survival
models (DeepSurv, RSF) do not beat linear Cox at this sample size.

## Key Results

_(none yet on this branch)_

## Patterns and Insights

## Lessons and Constraints

- Carried over from R0–R5: radiomic texture is kernel-dominated (leakage AUC
  0.89 before ComBat). Calcium needs 0.7 mm smoothing on sharp kernels.
  Enriched cohort means no calibration/Brier, rankings only. Stratify Cox by
  split (enrichment differs by split).
- Never enumerate `derived/totalseg_fullres/` (~55k folders).
- `ctab` rows exist only for reported abnormalities. For a scanned participant
  with no yr0 row, treat the flag as absent. Never adjust for nodule codes
  (51/52) because they sit on the detection pathway.

## Open Questions

- Does calcium proxy pack-years? (Pack-years need a CDAS request. Emphysema is
  the interim proxy.)

## Optimization Trajectory

Baseline clinical test C = 0.647. Best so far = 0.662 (Cox-EN heart + clinical).
