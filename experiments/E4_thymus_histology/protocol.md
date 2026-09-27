# E4 — Stress test: is thymic health's protection specific to squamous / small-cell lung cancer?

**Status:** protocol locked 2026-09-27, **after** the exploratory E3 finding
(TH HR/SD: adeno 1.06, SQ/SC 0.86, ratio 0.82, p=0.009, full cohort). E4 is
therefore not independent confirmation. It is a pre-specified *stress test*
of an exploratory finding: each check below could have broken it, and each
prediction is written down before running. Hypothesis H12.

**H12:** The TH × histology heterogeneity is robust (not a split, subgroup or
dose-proxy artifact) and is consistent with immune surveillance of
smoking-mutagenized tumours rather than with TH acting as an inverse
pack-years proxy.

## Data (fixed)

As in E3 (`experiments/E3_calcium_mechanism/code/run_e3.py::load`). Histology
groups are the same. BAC/lepidic subtypes = ICD-O-3 8250–8255 (inside
"adeno"). Stage from prsn `de_stag`: early = 110/120/210/220 (I–II),
advanced = 310/320/400 (III–IV), and everything else unknown.

## Analyses and predictions

**D1 — formal competing-risks heterogeneity test (Lunn–McNeil).** Stack one
copy per cause (adeno, SQ/SC). Cox with strata = cause × split, covariates =
clinical × cause (all interacted) + TH + TH × [cause = SQ/SC]. Robust
(cluster = pid) SEs. *Prediction:* the interaction HR is < 1, p < 0.05, and it
agrees with the E3 z-test (0.82).

**D2 — dose-marker adjustment (the discriminating test).** SQ/SC and adeno
cause-specific TH HRs, adjusted for clinical + emphysema + calcium +
pericardial fat.
- *Immune-surveillance prediction:* the SQ/SC TH HR stays ≤ 0.90 with p <
  0.05, and the ratio stays ≤ 0.87.
- *Pack-years-proxy prediction:* the SQ/SC log-HR attenuates by more than 50%.

**D3 — consistency.** The ratio (SQ/SC ÷ adeno) computed separately in the
train, val and test splits (disjoint patients), in men and women, and in
current and former smokers. *Prediction:* the ratio is < 1 in ≥ 5 of these 7
subsets, and in all 3 splits. The splits are small, so no significance is
required.

**D4 — adeno subtypes.** TH HR for BAC/lepidic (8250–8255) vs other adeno.
*Prediction (overdiagnosis/indolence):* BAC HR ≥ other-adeno HR. This is
descriptive: BAC has ~100 events.

**D5 — stage.** TH HR for early vs advanced cancers (all histologies,
cause-specific). *Prediction (surveillance → aggressive disease, paper's
mortality > incidence):* advanced HR < early HR. Descriptive, with the Δ z-test
reported.

**D6 — comparators (negative-control logic).** Histology ratios for
pericardial fat, heart volume (age/sex/body-size correlated like TH) and
calcium (already 1.08). *Prediction:* none of them shows a protective ratio
≤ 0.87 with p < 0.05. If fat did, TH could be riding on adiposity.

**D7 — held-out test, SQ/SC-specific prediction.** Cause-specific Cox for
SQ/SC fit on train+val: clinical vs clinical + TH. Harrell C for SQ/SC on
test (other cancers censored), paired bootstrap ΔC (1,000). *Prediction:* ΔC >
0 as a point estimate (test SQ/SC events are few, ~50, so the CI will be
wide). Adeno ΔC is reported for contrast.

**Multiplicity note.** In E3 the TH Δ was one of 5 marker deltas (Bonferroni
α = 0.01; observed p = 0.009). E4 adds 7 analyses. Only D1/D2 are treated as
decisive; D3–D7 are supporting or descriptive.
