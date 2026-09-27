# E3 — What is the calcium signal? Histology specificity, risk horizon, dose-response, robustness; TH scanner robustness

**Status:** protocol locked before analysis (2026-09-27). Hypotheses H7–H11.
Before locking, only the histology/timing *counts* among events were
inspected (no marker-outcome associations): adeno ≈463, squamous ≈228,
small-cell ≈125, other/NOS ≈172. 272 events came from the T0 screen.

## Data (fixed)

Same frame and marker definitions as E1/E2 (`src/aging_data.load_frame`,
6,041 / 1,003 events). All Cox models use strata = split and adjust for age,
male and cigsmok unless stated otherwise. Histology comes from prsn `de_type`
(ICD-O-3):
- **adeno**: 8140, 8250–8255, 8260, 8323, 8480, 8481, 8490, 8550
- **squamous**: 8070–8078, 8083, 8084
- **small cell**: 8041–8045
- **other**: everything else, including NSCLC NOS 8046, large cell, carcinoid
  and NOS

## Analyses

**C1 (H7) — histology specificity: an indirect pack-years test.** Squamous
and small-cell carcinoma have much steeper smoking dose–response than
adenocarcinoma. Cause-specific Cox models (other histologies censored at
diagnosis) for (a) adeno and (b) squamous + small cell ("SQ/SC"). Compare the
log-HR per SD of calcium between (a) and (b): Δ = β_SQSC − β_adeno, SE =
√(se_a² + se_b²), two-sided z-test.
- *Method positive control:* current-smoking HR must show Δ > 0 (a steeper
  association with SQ/SC). If it doesn't, the test is uninformative.
- *Proxy prediction:* calcium Δ > 0, HR ratio ≥ 1.10, p < 0.05.
- *Independent-biology prediction:* calcium Δ ≈ 0 (ratio in [0.9, 1.1]).
- Also reported: emphysema and TH Δ (descriptive).

**C2 (H8) — risk horizon.** (i) Follow-up truncated at 1 yr (events ≤1 yr;
this mostly captures T0-prevalent cancers). (ii) Landmark at 1 yr: patients
event-free at 1 yr, time reset. *Long-term-marker prediction:* the calcium HR
in window (ii) is ≥ 1.10 with a CI excluding 1. Also C2b: events with cancyr
= 0 vs ≥ 1 (cause-specific, same Δ-test as C1).

**C3 (H9) — dose-response.** Calcium categories: zero (mass proxy = 0, ~13%)
= reference, then tertiles of the non-zero values (T1–T3). *Prediction:* HRs
monotone non-decreasing T1 ≤ T2 ≤ T3, T3 vs zero HR > 1.3, and a trend test
(category index as a numeric) p < 0.01.

**C4 (H10) — robustness of calcium.** (i) Strata split × sex × 5-yr age bin,
plus linear age and cigsmok. (ii) Age as a restricted cubic spline (4 knots
at the 5/35/65/95th percentiles) + male + cigsmok. (iii) Within former and
within current smokers separately, plus the calcium × cigsmok interaction p.
(iv) Adjusted for the scanner batch (manufacturer × kernel group dummies) and
core noise SD. *Prediction:* calcium HR/SD ≥ 1.10 with p < 0.01 in (i), (ii)
and (iv). HR > 1 in both smoking strata.

**C5 (H11) — TH scanner robustness (the TH authors did no harmonization).**
(i) AUC of TH percentile for sharp vs soft kernel (the leakage probe used in
R3). (ii) OLS TH ~ age + male + cigsmok + manufacturer + kernel group:
partial R² of the scanner terms, and group means. (iii) TH → LC Cox HR with
scanner batch dummies added. *Prediction:* the leakage AUC stays below 0.60,
the scanner partial R² is below 2%, and the TH HR changes by less than 0.02.
The same leakage probe is run on raw log-calcium for comparison.
