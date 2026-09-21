# `subset_v1.csv` — student pipeline patient subset

Built by intersecting `discovery_pipeline/enriched_cohort_pids.txt` (6,373
pids, itself already enriched for lung-cancer-positive cases relative to the
full ~26k-patient cohort) with `experiments/lcrisk_discovery/data/manifest.csv`
(baseline-scan-level `pid,time,event,label,split`), then stratifying:
**all available positives capped at 120**, plus **280 randomly sampled
negatives** (seed=42), for **400 patients total (30% event rate)** — enough
events for both a first CNN classifier and a legible KM curve, far above
NLST's raw ~3.8% incidence.

Every row was individually verified to have both:
- `derived/ct_1x1x2mm/{pid}_yr0.nii.gz` (baseline CT, resampled 1x1x2mm)
- `derived/totalseg_fullres/{pid}_yr0/seg.nii.gz` (full-res TotalSegmentator
  117-label mask)

(`has_ct`/`has_seg` columns are both 1 for every row in this file — checked at
build time, not re-verified downstream.)

## Columns

| column | meaning |
|---|---|
| `pid` | NLST PatientID (join key for everything: `derived/*/{pid}_yr0*`, `nlst_780` clinical CSVs via `discovery_pipeline/src/data.py`-style joins) |
| `time` | days to event or censoring (from `manifest.csv`, already derived by the mentor's pipeline from `candx_days`/`canc_free_days` — do not re-derive) |
| `event` | 1 = lung cancer diagnosed, 0 = censored |
| `label` | original NLST screening result label carried over from `manifest.csv` |
| `split` | train/val/test split as already assigned in `manifest.csv` — reused as-is, not re-shuffled, so results stay comparable to the mentor's splits |
| `has_ct`, `has_seg` | file-existence checks at build time (both always 1 here) |

## Regenerating / extending this subset

Only ever look up files for specific pids from this list — never `find`/`ls -R`
over `derived/ct_1x1x2mm` or `derived/totalseg_fullres` (each has tens of
thousands of per-patient entries and is multi-terabyte). If Stage 3/4/5 later
need more events or a larger pool, extend by sampling further from
`discovery_pipeline/enriched_cohort_pids.txt` ∩ `manifest.csv` the same way,
and append rather than resample from scratch, so `subset_v1.csv` pids remain a
stable subset of any later `subset_v2.csv`.
