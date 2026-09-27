# E6 — analysis (2026-09-27)

Data: the user ran `sbatch src/submit_r2_fullarm.sh` (job 832790; 40/40 tasks
completed, about 6 min each, median 1.2 s/patient). Merged: 9,861 ok, 11 no_mask.
Feature distributions match the enriched cohort: heart volume median 498 vs
499 mL, calcium present in 87.2% vs 87.0%, calcium mass median 3.1 vs 3.3.
After the R3 exclusions and TH join, the analysis cohort is
**15,784 participants / 1,003 cancers**, of which 9,743 are added full-arm
participants. All of the added participants are non-cases.
Code: `code/run_e6.py` (75 s).

**Design correction (vs the protocol's "representative").** TotalSegmentator
was run on essentially all cancer cases plus about 62% of the other
CT-arm participants. The uncovered 9,027 have only 9 events. So this cohort is
still case-enriched, but far less than the original ~5k-control sample.
Uncovered and covered participants match on age, sex, smoking and TH, and the
TH HR is identical (0.909 vs 0.908 on all 25k). So the milder enrichment
doesn't distort HRs. Absolute risks are still not interpretable.

## Results vs locked predictions

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H15 | calcium HR/SD ∈ [1.08, 1.25], p<0.001; T3 vs none > 1.2 | **1.138 [1.071, 1.209], p=3e-5**; T1 1.05, T2 1.09, **T3 1.33 [1.06, 1.66]** | **supported**: the case-cohort estimate (1.14) holds |
| H16 | joint (clin + emph + TH + Ca): both p<0.05, each attenuation <15%; ρ(TH, Ca) ∈ [−0.15, −0.05]; ρ(TH, fat) < −0.2 | TH **0.934 [0.874, 0.997], p=0.042** (attenuation 29%); calcium **1.126 [1.059, 1.197], p=1e-4** (attenuation 8%); ρ −0.094 / −0.332 | **partial**: both are independent, calcium is unaffected, and TH's *pooled* HR is attenuated more than predicted |
| H17 | LM interaction < 1, p<0.05; SQ/SC TH (adj clin + emph + Ca) ≤ 0.90 | interaction **0.838 [0.721, 0.974], p=0.021**; SQ/SC **0.859 [0.766, 0.963], p=0.009**; adeno 1.02 | **supported** |
| G4 | calcium ΔC over clinical + emph > 0 | test n=2,622, 149 events: clin 0.660 → +emph 0.680 → **+Ca 0.689 (+0.009 [+0.000, +0.019], p=0.04)** → **+Ca+TH 0.692 (+0.012 [+0.002, +0.023], p=0.016)** | **supported**. Best model so far, and TH now adds on top |

## SQ/SC attenuation by dose markers (EXPLORATORY; E4 found 25% in the case-cohort)

| adjustment | TH HR/SD for SQ/SC | attenuation |
|---|---|---|
| clinical | 0.831 [0.742, 0.932] | — |
| + emphysema | 0.839 | 5% |
| + emphysema + calcium | 0.859 [0.766, 0.963], p=0.009 | 18% |
| + emphysema + calcium + fat | 0.849 [0.754, 0.956], p=0.007 | 11% |

With 3× more controls, the shared thymus–calcium component shrinks to 11–18%.
After every available dose marker, the SQ/SC effect stays significant. The
pack-years-proxy account predicted more than 50% attenuation, and it now looks
less likely. It still can't be excluded without pack-years.

## What changes in the story

- Calcium: nothing changes. It is the same general marker (1.14/SD,
  threshold-like, histology ratio 1.09) in the less-sampled cohort.
- Thymic health: the non-adenocarcinoma specificity is now shown *with
  calcium and emphysema adjustment* in the less-sampled cohort (p=0.009).
  The E4 borderline result (p=0.08) was a precision problem of the ~5k-control
  sample.
- Joint prediction: clinical + emphysema + calcium + TH is the best model
  (test C 0.692). Both CT aging markers contribute on top of the radiologist's
  emphysema read.
