# Research Findings

## Research Question

Do CT-derived markers of immune aging (thymic health, Bernatz et al. Nature
2026) and vascular aging (cardiac calcium, pericardial fat) carry independent,
additive information about lung-cancer incidence on NLST baseline low-dose CT,
or do they proxy the same unmeasured smoking burden?

## Current Understanding (after 5 experiments, 3 outer loops)

Two aging axes can be read off the same screening CT. They are only weakly
linked, and they relate to lung cancer in different ways.

1. **Vascular axis: cardiac calcium is a robust, independent, general
   lung-cancer risk marker.**
   - HR/SD 1.14 [1.08, 1.22], adjusted for age, sex and smoking (case-cohort,
     6,041 pts / 1,003 events). The effect is threshold-like: only the top
     third of non-zero calcium carries risk (HR 1.36 vs none; trend p=0.001).
   - It is stable over time (HR 1.15 in year 0–1 and 1.14 after year 1),
     unchanged by flexible age modelling or scanner batch, and unaffected by
     thymic health (3% attenuation) or radiologist-read emphysema (1.5%).
     Calcium and emphysema are uncorrelated (OR 0.99): they are orthogonal
     smoking phenotypes, vascular vs parenchymal.
   - It is only weakly histology-specific (SQ/SC ÷ adeno 1.08 vs 1.74 for
     current smoking), so it is not simply a stand-in for smoking dose.
   - It adds a small, reproducible **+0.01 test C-index** on top of every
     baseline (clinical 0.657 → 0.667; clinical + emphysema 0.678 → 0.688).
2. **Immune axis: the published thymic-health score protects against
   non-adenocarcinoma lung cancer only.**
   - On the full NLST CT arm (25,031 / 1,030 events) we reproduce the paper's
     unadjusted HR (0.649 vs 0.64).
   - About half of that effect is age. The adjusted effect is real: 0.91/SD
     (p=0.004), high vs low 0.81.
   - It is **entirely non-adenocarcinoma**: squamous 0.80, small cell 0.89,
     other 0.86, **adenocarcinoma 1.00 [0.91, 1.10]**. The Lunn–McNeil
     interaction is 0.835 (p=0.019).
   - Emphysema adjustment doesn't change it. In the case-cohort subset where
     calcium is available, dose markers attenuate the SQ/SC effect by only 25%.
   - The pattern has the same direction in 7/7 disjoint subsets. It isn't
     shared by body-size comparators, and it isn't driven by indolent BAC
     adenocarcinoma. It is stronger for advanced (0.91) than early (0.98)
     cancers.
   - Candidate mechanism: thymic output (naïve T-cell repertoire) constrains
     the most mutagenized, neoantigen-rich tumours (squamous/small cell, the
     smoking-signature cancers). This would also explain why the paper's LC
     *mortality* HR is stronger than its incidence HR. A pack-years confounding
     explanation can't be excluded without CDAS data. It would predict larger
     attenuation by dose markers than the 25% observed.
3. **The axes are connected through adiposity and vascular aging.**
   - Lower thymic health goes with more cardiac calcium (partial ρ −0.10),
     much more pericardial fat (ρ −0.32), denser fat HU (ρ +0.26),
     radiologist CV abnormality (OR 0.85) and emphysema (OR 0.89).
   - This replicates at n = 6k, with smoking adjustment, the only prior
     thymus–CAC study (Walther 2026, n=206). The thymus–pericardial-fat link
     appears to be new.
   - The two axes **share about a quarter of the SQ/SC signal**. Otherwise
     they are complementary: calcium is general, thymic health is histology
     specific.

## Key Results

| Result | Numbers | Where |
|---|---|---|
| TH, paper replication (unadj 6-yr high vs low) | 0.649 [0.54, 0.78] (paper 0.64 [0.53, 0.76]) | E5 F1 |
| TH adj age/sex/smoking, full CT arm | 0.908/SD [0.85, 0.97]; high vs low 0.81 [0.67, 0.97] | E5 F1 |
| TH by histology (full) | adeno 1.00, SQ/SC 0.83 [0.75, 0.93], squamous 0.80; LM p=0.019 | E5 F2 |
| TH ↔ calcium / pericardial fat (partial ρ) | −0.10 / −0.32 | E1 A2 |
| Calcium adj HR/SD (with TH / with emphysema) | 1.14 / 1.14 | E1 A3, E2 B2 |
| Calcium dose-response (T3 vs none) | 1.36 [1.09, 1.70], trend p=0.001 | E3 C3 |
| Calcium histology ratio vs smoking control | 1.08 vs 1.74 | E3 C1 |
| Test C: clin / +Ca / +emph / +emph+Ca | 0.657 / 0.667 / 0.678 / **0.688** | E1, E2 |
| Test SQ/SC-specific C: clin → +TH | 0.720 → 0.733 (+0.013, p=0.056) | E4 D7 |

## Patterns and Insights

- **Pooling histologies hides immune-axis signal.** A marker that protects
  against one histology and is null for another looks weak in pooled
  analyses. Lung-cancer biomarker studies should report histology-specific
  HRs.
- **Simple physical CT quantities survive; high-order features don't.**
  Calcium mass replicates everywhere, texture didn't (R3). A deep TH score
  works, but only for a biologically specific endpoint.
- **Age is the dominant confounder of "aging" markers.** It halves TH's
  log-HR. Always report age-adjusted estimates.
- **Additivity comes from orthogonal compartments:** vascular calcium vs
  parenchymal emphysema vs immune (thymus, histology-specific).

## Lessons and Constraints

- **Case-cohort (enriched) sample ≠ full cohort.** The same events with ~5k
  instead of ~24k controls, plus split strata, diluted TH (0.95 vs 0.91/SD).
  Estimate HRs on the full cohort when the marker is available there; treat
  enriched-sample HRs as approximate. Calcium currently exists only on the
  enriched sample. A full-CT-arm extraction is a big sbatch (user runs it).
- Carried over from R0–R5: radiomic texture is kernel-dominated. Calcium needs
  0.7 mm smoothing on sharp kernels. There is no calibration/Brier on the
  enriched cohort. Stratify Cox by split there.
- Never enumerate `derived/totalseg_fullres/` (~55k folders).
- `ctab`: no yr0 row means the flag is absent. Never adjust for nodule codes.
  ctab-60 is a poor CAC reference (calcium AUC only 0.62).
- TH is a uniform within-NLST percentile with no identifiable QC-failure mass.
  It is scanner-robust (kernel leakage AUC 0.52, partial R² 1.8%).
- statsmodels is not in the venv. `aging_data.logit` (IRLS) gives the ORs.
- Multiplicity: the TH × histology lead came from 5 exploratory marker
  deltas (p=0.009 vs Bonferroni 0.01). E4/E5 stress tests use the same events,
  so this is robust but not independently replicated.

## Open Questions

- Pack-years (CDAS) would decide immune surveillance vs residual dose
  confounding for TH × SQ/SC, and quantify how much of calcium is dose.
- Calcium on the full CT arm (19k more scans; the big sbatch) would remove the
  case-cohort approximation for calcium and allow a full-cohort TH + calcium
  joint model.
- External replication of TH × histology (e.g., FHS or other cohorts with TH
  scores) is needed. TRACERx (in the paper) is NSCLC-only and could compare
  adeno vs squamous immune markers.
- Does TH predict squamous *stage* or aggressiveness among cases?

## Optimization Trajectory

Test Harrell C (Cox fit on train+val; TH-complete test n=1,983, 149 events):
clinical 0.657 → +TH 0.658 → +calcium 0.667 → +emphysema 0.678 →
**+emphysema+calcium 0.688**. TH doesn't help pooled prediction but helps
SQ/SC-specific prediction (+0.013).
