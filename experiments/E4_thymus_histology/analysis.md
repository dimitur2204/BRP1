# E4 — analysis (2026-09-27)

Stress test of the exploratory E3 finding. Cohort as E1–E3 (6,041 / 1,003
events; adeno 463, SQ/SC 353). Code: `code/run_e4.py` (1 min).
**Caveat up front:** all E4 checks use the same events that produced the
finding, so this is robustness, not independent replication.

## Results vs locked predictions

| Check | Prediction | Result | Verdict |
|---|---|---|---|
| D1 Lunn–McNeil (robust SE) | interaction HR < 1, p<0.05 | **0.82 [0.70, 0.95], p=0.010** | ✓ |
| D2 + emphysema + calcium + fat | immune: SQ/SC HR ≤ 0.90, p<0.05 and ratio ≤ 0.87; proxy: >50% attenuation | SQ/SC **0.90 [0.80, 1.01], p=0.08**; **ratio 0.85 [0.72, 0.99], p=0.036**; attenuation **25%** (almost all from calcium: emphysema 10% → +calcium 27%) | **partial**: neither a clean immune nor a clean proxy result. The heterogeneity survives; a quarter of the SQ/SC effect is shared with vascular calcium |
| D3 consistency | ratio < 1 in ≥5/7 subsets and all 3 splits | **7/7**: train 0.88, val 0.59, test 0.82, women 0.81, men 0.83, former 0.74, current 0.86 | ✓ |
| D4 adeno subtypes | BAC ≥ other adeno | BAC 1.07 [0.88, 1.30], other adeno 1.05 [0.94, 1.18] | ✓ nominally, but they're **equal**, so the adeno null is *not* explained by indolent/overdiagnosed BAC |
| D5 stage | advanced HR < early | III–IV **0.91** [0.83, 1.01] vs I–II 0.98 | ✓ direction (descriptive); consistent with the paper's stronger mortality HR |
| D6 comparators | none protective ratio ≤ 0.87 | fat 1.11, heart volume 1.08, calcium 1.08 | ✓ not a generic body-size or age artifact |
| D7 test SQ/SC ΔC | > 0 | SQ/SC C 0.720 → **0.733, ΔC +0.013 [−0.000, +0.026], p=0.056** (58 events); adeno ΔC **−0.016** (63 events) | ✓ point estimate |

**H12 verdict: mostly supported.** The heterogeneity is formally significant,
has the same direction in every disjoint subset, is not shared by comparator
markers, and shows up as a histology-specific C-index gain on held-out data.
But it is partly (25%) shared with calcium, and after full adjustment the
SQ/SC effect alone is borderline (p=0.08).

## Side findings (exploratory)

- **Body size and adenocarcinoma.** Pericardial fat (HR 0.89) and heart volume
  (HR 0.86) are *protective for adeno only*. This matches the known inverse
  BMI–lung-cancer association, and here it looks adenocarcinoma-specific.
- **Calcium absorbs part of TH's SQ/SC signal.** These are the two aging axes
  (immune ↔ vascular, ρ −0.10) overlapping on the most smoking-driven
  cancers. The shared component is plausibly cumulative exposure. The
  residual TH component is what an immune-surveillance account would predict.

## Interpretation for the story

Pooling histologies hides a thymic-health association with the most
mutagenized, immunogenic lung cancers (squamous + small cell), and it is
absent for adenocarcinoma. That is consistent with thymic output (naïve T-cell
repertoire) mattering most where neoantigen load is highest, and with the
Bernatz et al. observation that TH relates more strongly to lung-cancer
*death* than to incidence. We can't separate this from residual pack-years
confounding without CDAS data (the most important caveat), but the proxy
account would predict larger attenuation by the dose markers than the 25%
observed.
