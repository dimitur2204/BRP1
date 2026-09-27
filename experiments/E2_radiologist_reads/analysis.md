# E2 — analysis (2026-09-27)

Cohort as E1 (6,041 / 1,003 events). yr0 radiologist flags: emphysema 1,940
(32%), significant CV abnormality 382 (6.3%). Code: `code/run_e2.py`
(about 8 min; the attenuation bootstrap has 500 resamples × 4 Cox fits).

## CONFIRMATORY results vs locked predictions

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H6a | calcium AUC for ctab-60 ≥ 0.75 | **AUC 0.62 [0.59, 0.65]**; adj OR/SD 1.40 [1.27, 1.54]; monotone decile gradient 2.8% → 11.4% | **refuted** on the AUC threshold. There is a clear, monotone association, but ctab-60 is a heterogeneous flag (aneurysm, cardiomegaly, effusion, valve, CAC) that NLST readers didn't systematically use for CAC |
| H6b | TH OR/SD for ctab-60 < 1 | **0.85 [0.76, 0.95], p=0.003** | **supported** (matches the paper's CV-mortality finding) |
| H5 | calcium attenuation by emphysema: ≥25% (proxy) vs <10% (independent) | **1.5% [−6%, +10%]** | **"independent" supported**: calcium is not a stand-in for the emphysema-type smoking damage |
| — | emphysema HR (sanity) > 1 | 1.67 [1.48, 1.90] | ok |

Other descriptive associations (adjusted for age, sex and smoking):
- Calcium vs emphysema: OR 0.99 (null).
- TH vs emphysema: OR 0.89 (p=1e-4).
- Pericardial fat: emphysema OR 0.79 (emphysematous patients are leaner) and
  CV-abnormality OR 1.31.

## Held-out test (B3, secondary)

| model | test C | ΔC vs clinical | ΔC vs clinical + emph |
|---|---|---|---|
| clinical | 0.657 | — | — |
| clinical + emphysema | 0.678 | +0.021 [−0.002, +0.046] | — |
| **clinical + emphysema + calcium** | **0.688** | **+0.031 [+0.005, +0.060]** | **+0.010 [+0.000, +0.020], p=0.05** |
| clinical + emphysema + TH + calcium | 0.685 | +0.028 | +0.007 (n.s.) |

(n=1,983, 149 events.)

## What this rules out / suggests

- Calcium and emphysema are **orthogonal** smoking-related CT phenotypes
  (vascular vs parenchymal): they are uncorrelated and their HRs don't
  attenuate each other. Each adds to lung-cancer prediction. This fits the
  literature: CAC tracks pack-years dose-dependently and persists after
  quitting (Yao 2025), while emphysema reflects susceptibility.
- Calcium's +0.01 C is **stable across three adjustment sets** (R5 +0.011,
  E1 +0.010, E2 +0.010 over clinical + emphysema). This is a small but
  consistent incremental signal.
- It is still open whether calcium proxies *pack-years* specifically. Emphysema
  is only a partial dose marker. The next indirect test is histology
  specificity (E3).
