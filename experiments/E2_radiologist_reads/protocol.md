# E2 — Radiologist baseline reads: calcium validity and the smoking-burden proxy test

**Status:** protocol locked before analysis (2026-09-27). Hypotheses H5, H6.

## Data (fixed)

- Same analysis cohort and marker definitions as E1 (`../E1_thymus_heart_joint/protocol.md`).
- Radiologist reads: `data/external/nlst_780/nlst_780/nlst_780_ctab_idc_20210527.csv`,
  **`study_yr == 0` only** (same screen as our CT). Per pid: `emph0` = any row
  with `sct_ab_desc == 59` (emphysema); `cv0` = any row with `sct_ab_desc == 60`
  (significant cardiovascular abnormality). A cohort pid with no yr0 row → 0.
  (Every cohort pid had a yr0 CT, so "no row" means nothing was reported.)
- Nodule codes (51/52/62) are **not** used as covariates, because they sit on the
  detection pathway of the outcome.

## Analyses

**B1 (H6) — calcium validity.** AUC of log-calcium alone for `cv0`, with a
bootstrap CI, on the full cohort. Logistic regression cv0 ~ calcium + age +
male + cigsmok (OR per SD).
*Prediction:* AUC ≥ 0.75, OR per SD > 1.
TH: logistic cv0 ~ TH + age + male + cigsmok.
*Prediction:* OR per SD < 1 (paper: high TH → lower CV mortality).
Also emphysema ~ TH and ~ calcium (same adjustment): descriptive, feeds H5.

**B2 (H5) — smoking-burden proxy test.** Full cohort, Cox strata = split.
Compare the calcium log-HR (clinical + calcium) with and without `emph0`
added, and the same for TH. Attenuation = 1 − β_with/β_without, with a
bootstrap CI (500 resamples).
*Competing predictions:* "proxy" → calcium attenuation ≥ 25%;
"independent" → < 10%. TH attenuation is expected to be smaller than
calcium's (TH reflects immune aging and is less dose-specific).
Report the HR of emph0 itself (sanity check: expected > 1, since emphysema
is an established lung-cancer risk factor).

**B3 — held-out test (secondary).** Add emph0 to the E1-A4 test-set models:
clinical + emph; clinical + emph + TH + calcium. ΔC vs clinical + emph asks
whether the imaging markers still add anything once the radiologist's
emphysema read is known.
