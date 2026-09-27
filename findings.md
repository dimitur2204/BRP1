# Research Findings

## Research Question

Do CT-derived markers of immune aging (thymic health) and vascular aging
(cardiac calcium, pericardial fat) carry independent, additive information
about lung-cancer incidence on NLST baseline low-dose CT, or do they proxy
the same unmeasured smoking burden?

## Current Understanding

On the same 6,041 NLST baseline LDCTs (1,003 incident lung cancers), the two
aging axes behave very differently.

1. **Cardiac calcium is a robust, independent lung-cancer marker.** HR/SD 1.14
   [1.08, 1.22], adjusted for age, sex and smoking status. The HR does not move
   when thymic health (3% attenuation) or the radiologist's emphysema read
   (1.5% [−6%, +10%]) is added, and it is identical after ComBat harmonization.
   On held-out test it adds a small but consistent **+0.01 C-index** on top of
   every baseline tried (clinical; clinical + emphysema), reaching test C 0.688
   with clinical + emphysema + calcium.
2. **Thymic health's lung-cancer association is mostly an age effect.**
   Unadjusted HR/SD 0.90 (high vs low 0.77, same direction as the paper's
   unadjusted 0.64). Adjusting for age alone takes it to 0.97 (n.s.). This is
   consistent with the paper's weak fully adjusted result (P=0.036 in 25k).
   TH adds no C-index.
3. **But TH is biologically connected to the heart.** Lower TH goes with more
   cardiac calcium (partial ρ −0.10) and much more pericardial fat (partial ρ
   −0.32), with denser fat HU, with the radiologist's "significant CV
   abnormality" flag (OR/SD 0.85) and with emphysema (OR 0.89). All of these
   are adjusted for age, sex and smoking. Thymic fatty involution tracks
   visceral (pericardial) adiposity and vascular aging. That fits TH predicting
   CV death in the paper, but it doesn't translate into lung-cancer risk beyond
   age.
4. **Calcium and emphysema are orthogonal smoking phenotypes** (vascular vs
   parenchymal, OR 0.99 for their association), and each adds to lung-cancer
   risk. So calcium is not a proxy for emphysema-type damage. Whether it
   proxies *pack-years* remains open (no pack-years in IDC-780).

## Key Results

| Result | Numbers | Where |
|---|---|---|
| TH → LC, adj age/sex/smoking | HR/SD 0.95 [0.89, 1.02]; high vs low 0.90 [0.75, 1.08] | E1 A1 |
| TH → LC, unadjusted | HR/SD 0.90 [0.84, 0.95]; high vs low 0.77 [0.65, 0.92] | E1 explore |
| TH ↔ calcium / pericardial fat (partial ρ) | −0.10 [−0.13, −0.08] / −0.32 [−0.35, −0.30] | E1 A2 |
| Calcium → LC, joint with TH | HR/SD 1.14 [1.07, 1.21] | E1 A3 |
| Low TH + top-quartile calcium vs neither | adj HR 1.48 [1.21, 1.80] | E1 2×2 |
| Calcium attenuation by emphysema | 1.5% [−6%, +10%] | E2 B2 |
| Test C: clin / +Ca / +emph / +emph+Ca | 0.657 / 0.667 / 0.678 / **0.688** | E1 A4, E2 B3 |
| Calcium AUC for radiologist CV flag | 0.62 [0.59, 0.65]; OR/SD 1.40 | E2 B1 |

## Patterns and Insights

- **Simple physical CT quantities survive and learned or high-order features
  don't.** Calcium mass replicates everywhere. Texture didn't (R3). A deep TH
  score is mostly age at this endpoint.
- **Age is the dominant confounder for "aging" markers.** Any aging-axis score
  must be evaluated *age-adjusted*. Unadjusted HRs mainly measure age.
- **Additivity comes from orthogonal tissue compartments** (vascular calcium
  vs lung parenchyma), not from two aging scores that share age and adiposity.

## Lessons and Constraints

- Carried over from R0–R5: radiomic texture is kernel-dominated (leakage AUC
  0.89 before ComBat). Calcium needs 0.7 mm smoothing on sharp kernels.
  Enriched cohort means no calibration/Brier, rankings only. Stratify Cox by
  split (enrichment differs by split).
- Never enumerate `derived/totalseg_fullres/` (~55k folders).
- `ctab` rows exist only for reported abnormalities. For a scanned participant
  with no yr0 row, treat the flag as absent. Never adjust for nodule codes
  (51/52/62) because they sit on the detection pathway.
- Published TH is a within-NLST percentile (QC failures are dumped into
  "low" and can't be identified). With 1,003 events the MDE for an adjusted
  HR/SD is about 0.90, so a null here is "not large", not "zero".
- ctab-60 is a poor CAC reference standard (heterogeneous flag). If a real CAC
  reference is needed, use the DeepCAC2 NLST release (Nürnberg 2026) once it
  is public.
- statsmodels is not in the venv. `aging_data.logit` (IRLS) gives the ORs.

## Open Questions

- Does calcium proxy pack-years? Indirect test: histology specificity
  (squamous/small-cell are more dose-driven than adenocarcinoma) → E3.
- Is the calcium association front-loaded in the first year (prevalent,
  undiagnosed cancer) or stable (long-term risk)? → E3.
- Is TH itself kernel/scanner-dependent (never harmonized by its authors)? → E3.
- The 2×2 "both" group looks super-additive: real, or a cut-point artifact?

## Optimization Trajectory

Test Harrell C (Cox fit on train+val, TH-complete test n=1,983, 149 events):
clinical 0.657 → +TH 0.658 → +calcium 0.667 → +emphysema 0.678 →
**+emphysema+calcium 0.688**. TH never helps. Calcium gives +0.01 on every base.
