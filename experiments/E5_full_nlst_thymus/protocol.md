# E5 — Full NLST CT arm (~25k): paper replication and design-robustness of the TH × histology finding

**Status:** protocol locked 2026-09-27, before running. Hypotheses H13, H14.

## Why

E1–E4 use the mentor's enriched **case-cohort** sample (all cases + a random
subcohort, with different sampling fractions per split). Unweighted Cox on
such samples can distort HRs, and a TH result from it can't be compared
directly with the paper's full-NLST numbers. TH is published for 25,031 NLST
participants, and the outcome and covariates come from the manifest and
prsn. None of this needs our imaging features, so the whole TH-scored CT arm
can be analysed with no sampling. **The events are almost the same as in
E1–E4**, so this is a design-robustness check and a check that our data
pipeline matches the paper, *not* independent replication. There is no
calcium here (it isn't extracted outside the 6,306 cohort).

## Data (fixed)

- Manifest (`/faststorage/project/aura_thymus/experiments/lcrisk_discovery/data/manifest.csv`):
  `pid`, `time` (days → years), `event`. One row per pid.
- ⨝ TH (published, as in E1) ⨝ prsn (`age`, `gender` → male, `cigsmok`,
  `de_type` → histology as in E3) ⨝ ctab yr0 emphysema (as in E2).
- Complete case on TH and covariates. No split strata (no sampling).

## Analyses

**F1 (H13) — paper replication.** Unadjusted Cox with TH categories (ref =
low), follow-up censored at 6 yr (the paper's LC-incidence horizon).
*Prediction:* high vs low HR inside the paper's CI [0.53, 0.76], and average
vs low inside roughly [0.67, 0.90].
Then adjusted: (i) age + male + cigsmok; (ii) strata sex × 5-yr age +
cigsmok (the paper's PH approach); (iii) + emphysema. Report TH per SD and
high vs low, at full follow-up and at 6 yr.

**F2 (H14) — TH × histology in the full cohort.** Cause-specific Cox (adeno;
SQ/SC; squamous; small cell; other), adjusted for age + male + cigsmok, and
the Lunn–McNeil interaction (strata = cause, robust SE).
*Prediction:* interaction HR < 1, p < 0.05. SQ/SC TH HR/SD < 0.92. Adeno TH HR
CI includes 1. The same with + emphysema.
Sensitivity: 6-yr censoring.

**F3 — descriptive.** Full-cohort n, events per histology, how many
participants and events are *not* in the E1 cohort (showing how much the
event sets overlap), and the TH mean by sex / smoking.
