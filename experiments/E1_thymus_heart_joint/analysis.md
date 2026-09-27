# E1 — analysis (2026-09-27)

Cohort: 6,041 patients / 1,003 events (164 R3-QC'd pids lack a published
TH score, and only 15 of them are events, so TH-missingness is enriched for
controls). Test: 1,983 / 149. Code: `code/run_e1.py`. Outputs: `results/`.

## CONFIRMATORY results vs locked predictions

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H1 | TH high vs low adj HR ∈ [0.5, 0.85], p < 0.05 | **adj HR 0.90 [0.75, 1.08], p=0.26**; per SD 0.95 [0.89, 1.02], p=0.13 | **refuted** (direction as predicted, magnitude and significance not) |
| H2 | ρ(TH, calcium) ∈ [−0.15, −0.02]; ρ(TH, fat) < 0, \|ρ\| > 0.1 | partial ρ **−0.100 [−0.125, −0.075]**; **−0.323 [−0.348, −0.299]** | **supported** |
| H3 | both p<0.05 jointly, attenuation < 15% each | calcium HR 1.14 → 1.14 (atten 3%); TH 0.95 → 0.97 (atten 38%, n.s.) | **refuted** for TH; calcium is independent of TH |
| H4 | ΔC(clin+TH+Ca vs clin) CI > 0 | +0.008 [−0.004, +0.020]; TH adds −0.002 over clin+Ca | **refuted** |

Sensitivities for H1 all agree: censoring at 6 yr gives 0.95 per SD; strata
split × sex × 5-yr age give 0.94 [0.88, 1.00], p=0.06. Schoenfeld p=0.22 (no
PH violation). TH × calcium interaction p=0.51. The ComBat-harmonized calcium
gives the same result (HR 1.14).

## Why TH loses its association (EXPLORATORY, `code/explore_th_attenuation.py`)

- **Age absorbs it.** Unadjusted TH HR/SD is 0.895 (p=0.0006) and high vs low
  is 0.77 [0.65, 0.92]. That is the same direction as the paper's unadjusted
  0.64, and weaker in our enriched subset. Adding **age alone** moves it to
  0.966 (p=0.30). Sex and smoking alone do nothing (0.885, 0.884). TH falls
  about 1.1 percentile points per year of age and is 15 points lower in men
  (R² = 0.11 for age + sex + smoking).
- This matches the paper, whose fully adjusted LC-incidence effect was only
  type III P=0.036 in 25k patients (literature/survey.md §1). Our 1,003 events
  cannot resolve an adjusted HR/SD of about 0.95.
- **Adiposity is a weak negative confounder.** TH is strongly anti-correlated
  with pericardial fat (ρ −0.32). Fat itself is slightly protective for lung
  cancer (HR 0.95, n.s., in line with the known inverse BMI–lung-cancer link).
  Adjusting for fat strengthens TH only slightly (0.950 → 0.934, p=0.05).
- There is no subgroup where TH is clearly significant (men 0.92, current
  smokers 0.94, both n.s.).

## Descriptive 2 × 2 (low TH × top-quartile calcium)

The KM curves separate on both the full cohort and test (log-rank 5e-9 and
6e-6). Adjusted HR vs neither: low TH only 1.04 (n.s.), high Ca only 1.25
[1.05, 1.48], **both 1.48 [1.21, 1.80]**. The "both" group looks more than
additive on the log scale (0.04 + 0.22 < 0.39), but the continuous interaction
is null (p=0.51). Treat this as an exploratory pattern that may come from the
quartile cut-points, not as evidence of synergy.

## What this rules out / suggests

- It rules out TH as an independent lung-cancer predictor at our sample size
  once age is modelled. TH's lung-cancer signal in NLST is mostly an age
  signal.
- Calcium's association does not run through thymic health (3% attenuation).
- TH is biologically coherent with the heart. Lower TH goes with more calcium
  (ρ −0.10) and a lot more pericardial fat (ρ −0.32), and with denser, less
  negative fat HU (ρ +0.26). The inverse thymus–CAC link (Walther 2026,
  n=206, no smoking adjustment) replicates at n=6,041 with smoking adjustment.
  The thymus–pericardial-fat link is, as far as the survey found, new.
