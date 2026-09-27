# E3 — analysis (2026-09-27)

Cohort as E1/E2 (6,041 / 1,003 events). Event histology: adeno 463, squamous
228, small cell 125, other 175, missing 12. Code: `code/run_e3.py` (40 s).

## CONFIRMATORY results vs locked predictions

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H7 (method control) | current-smoking Δ > 0 | HR adeno 1.19, SQ/SC 2.08, **ratio 1.74 [1.31, 2.31], p=0.0002** | the test is informative |
| H7 | proxy: calcium ratio ≥ 1.10, p < 0.05; independent: ratio in [0.9, 1.1] | adeno 1.09 [0.99, 1.19], SQ/SC 1.18 [1.06, 1.31]; **ratio 1.08 [0.94, 1.25], p=0.25** | **inconclusive, leaning "independent"**. The point estimate is inside the independent band, but the CI can't exclude a modest proxy effect. Calcium is ~7× less histology-specific than smoking on the log scale (0.08 vs 0.55) |
| H8 | landmark >1 yr HR ≥ 1.10, CI > 1 | 0–1 yr 1.15 [1.02, 1.30]; **>1 yr 1.14 [1.06, 1.23]**; >3 yr 1.17 [1.05, 1.29]; T0-screen vs later ratio 1.00 | **supported**: a stable long-term marker, not prevalent-cancer driven |
| H9 | monotone, T3 vs 0 > 1.3, trend p < 0.01 | zero (ref), T1 1.07, T2 1.08, **T3 1.36 [1.09, 1.70]**, trend p=0.0013 | **supported**, threshold-like: the risk sits in the top third of non-zero calcium (mass proxy > 11) |
| H10 | HR ≥ 1.10, p<0.01 under age strata / spline / scanner; >1 in both smoking strata | 1.144 / 1.145 / **1.171**; former 1.08 [0.98, 1.19], current 1.19 [1.10, 1.30]; interaction p=0.17 | **supported** (the former-smoker CI touches 1; the stronger effect in current smokers is compatible with ongoing exposure) |
| H11 | TH leakage AUC < 0.60, scanner partial R² < 2%, ΔHR < 0.02 | kernel AUC 0.52, max manufacturer 0.57; **partial R² 1.8%**; ΔHR −0.003 | **supported** (the R² is borderline, driven by GE-soft vs Siemens mean TH 52 vs 44; calcium's own scanner R² is 3.7%) |

## EXPLORATORY: thymic health is histology-specific

The protocol listed TH's histology Δ as *descriptive*, so this is exploratory:

| | adeno (463) | SQ/SC (353) | squamous (228) | small cell (125) | other (175) |
|---|---|---|---|---|---|
| TH HR/SD (adj) | **1.06** [0.96, 1.16] | **0.86** [0.77, 0.97] | 0.84 [0.72, 0.97] | 0.92 [0.76, 1.11] | 0.88 [0.75, 1.03] |

Ratio SQ/SC ÷ adeno = **0.82 [0.70, 0.95], p=0.009**. The null pooled TH
effect (E1) may be two opposing histology-specific effects cancelling out.

Two competing explanations, which E4 must separate:
1. **Immune surveillance.** Squamous and small-cell tumours are the most
   smoking-mutagenized (highest TMB, most neoantigens). A larger naïve T-cell
   repertoire (high thymic output) would plausibly constrain them more than
   adenocarcinoma. Screening-detected adenocarcinomas also include indolent,
   overdiagnosed lepidic/BAC tumours, where immunity should matter least. This
   would also explain why the paper's lung-cancer *mortality* HR (0.52) was
   stronger than its incidence HR (0.64): SQ/SC are the more lethal types.
2. **Inverse pack-years proxy.** TH falls with smoking duration and intensity
   (paper, FHS), and SQ/SC are dose-driven. So high TH could mean "fewer
   pack-years". Prediction: this TH–SQ/SC effect should attenuate when other
   dose markers (emphysema, calcium) are added, and TH should look like
   current smoking/emphysema. One check against it: dose markers (smoking,
   emphysema, calcium) all raise SQ/SC risk *and* adeno risk (all HR > 1
   for adeno). TH is ≥ 1 for adeno, so a pure inverse-dose proxy would
   need to be protective for adeno too, just less so.

## What this rules out / suggests

- Calcium is a stable, dose-responsive, scanner-robust, age-robust
  long-term risk marker. It has only weak histology specificity, so it is
  unlikely to be a pure stand-in for smoking dose. This fits the MESA
  pack-years result (17% attenuation).
- TH may have a real but **histology-restricted** association that pooled
  analyses dilute. This is the most novel lead so far, and it needs a
  stress test (E4) before anyone believes it.
