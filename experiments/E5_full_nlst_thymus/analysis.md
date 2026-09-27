# E5 — analysis (2026-09-27)

Full TH-scored NLST CT arm: **25,031 participants, 1,030 lung cancers** (974
within 6 yr). Only 27 of these events are outside the E1 cohort, so the event
set is essentially the same. What changes is the comparison group: all
~24k non-cases instead of the ~5k case-cohort subcohort, and no sampling
weights are needed. Code: `code/run_e5.py` (2 min).

## Results vs locked predictions

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H13 | unadjusted 6-yr high vs low HR within the paper's CI [0.53, 0.76]; average ≈ 0.78 | **high 0.649 [0.542, 0.777]**; average **0.791** [0.684, 0.914] (paper: 0.64 [0.53, 0.76] and 0.78) | **supported**: our join of published TH + manifest outcome reproduces the paper almost exactly |
| H14 | LM interaction < 1, p<0.05; SQ/SC HR/SD < 0.92; adeno CI includes 1 | interaction **0.835 [0.719, 0.970], p=0.019**; SQ/SC **0.834 [0.745, 0.933]**; squamous 0.80; small cell 0.89; other 0.86; **adeno 0.998 [0.907, 1.098]** | **supported**. Unchanged with + emphysema (0.835, p=0.018) and at 6 yr (0.842, p=0.029) |

## Adjusted TH association (full cohort)

| model | HR/SD | high vs low |
|---|---|---|
| unadjusted | 0.847 [0.796, 0.901] | 0.666 |
| + age only | 0.919 [0.862, 0.979] | — |
| + age, sex, smoking | **0.908 [0.850, 0.969], p=0.004** | **0.81 [0.67, 0.97]** |
| strata sex × age5 + smoking | 0.902 [0.845, 0.963] | 0.79 |
| + emphysema | 0.914 [0.857, 0.976] | 0.83 |

## Correction to E1

E1 concluded that TH's lung-cancer association is "mostly age" (adjusted
0.95, n.s.). In the full cohort **about half** of the log-HR is age
(0.847 → 0.919), and **the adjusted association is real** (0.91/SD,
p=0.004). The E1 estimate came from the case-cohort subsample (same events,
~5k controls, split strata). Restricting the full-cohort data to the E1 pids
gives 0.934 [0.874, 0.997]. So E1 was diluted and underpowered, not
contradictory. Lesson: **TH (and probably any marker) should be estimated on
the full cohort where possible. The enriched sample loses precision and may
shift HRs.**

## The main new result

**The published thymic-health score's lung-cancer association is entirely
non-adenocarcinoma.** Per SD: squamous 0.80, small cell 0.89, other/NOS
0.86, adenocarcinoma 1.00. The paper's headline HR 0.64 is therefore an average
over a strong effect in squamous/small-cell/other tumours and no effect
in adenocarcinoma, which is ~46% of screen-era cancers.
