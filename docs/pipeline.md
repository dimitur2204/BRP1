# Lungs 3D CNN on NLST: design and rationale

A baseline 3D CNN that reads the lungs on a participant's baseline (T0)
low-dose CT and outputs a lung-cancer log-risk score η, trained with a Cox
loss. It is meant to be simple and correct, a base for later work. There is
no hyperparameter search. Every choice below is a default that can be
changed with one flag or one constant (`src/nlst.py`).

The project was restarted from scratch on 2026-10-05. The previous pipeline
(enriched 6.3k cohort, mentor split, lungs/sternum comparison) is only in git
history (branch `cnn3d-lungs-sternum` up to commit `7382553`). Nothing from
it is reused: no cohort, no split, no cache, no code.

## Pipeline

| stage | script | runs where | output |
|---|---|---|---|
| S1 candidates | `src/s1_candidates.py` | login node, ~1 min | `data/candidates.csv`, `data/candidates_attrition.csv` |
| S2 preprocess | `src/s2_preprocess.py` via `submit_s2_preprocess.sh` (40 CPU tasks), then `--merge`, `--qc` | Slurm + login node | `data/cache/lungs_u8.npy` (~39 GB), `data/cache/lungs_index.csv`, `figs/s2_inputs.png` |
| S3 cohort | `src/s3_cohort.py` | login node, seconds | `data/cohort.csv`, `data/cohort_attrition.csv`, `data/cohort_splits.csv` |
| S4 train | `src/s4_train.py` via `submit_s4_train.sh` (1 GPU) | Slurm | `runs/{name}/` |
| S5 evaluate | `src/s5_evaluate.py --run {name}` | login node | `runs/{name}/eval/` |
| S1b mask selection check | `src/s1b_mask_selection.py` | login node, minutes | `data/mask_selection_*.csv`, `data/mask_ipw.csv`, `figs/mask_selection.png` |
| S6 saliency | `src/s6_saliency.py` via `submit_s6_saliency.sh` (1 GPU, `gpu-short`) | Slurm | `runs/{name}/saliency_val/` |

Shared code: `src/nlst.py` (paths and the whole input definition) and
`src/model.py` (network, Cox loss, C-index, augmentation).

## 1. Cohort (S1 + S3)

**Population.** All NLST participants in the public IDC-780 `prsn` table
(53,452, both arms), restricted to those whose baseline CT is on disk with a
TotalSegmentator mask. Nothing is sampled or enriched.

| step | n | events |
|---|---|---|
| NLST participants (IDC-780 prsn) | 53,452 | 2,058 |
| baseline CT series in the IDC scan list | 26,254 | 1,061 |
| series is study year 0 | 25,868 | 1,047 |
| resampled CT on disk | 25,861 | 1,047 |
| TotalSegmentator mask on disk | 16,618 | 1,047 |
| follow-up > 0 days after T0 | 16,543 | 1,047 |
| preprocessing ok (S2; 28 empty lung masks) | 16,515 | 1,046 |
| lung z-extent ≥ 200 mm | 16,350 | 1,034 |
| lung volume ≥ 1.5 L | 16,348 | 1,034 |
| ≤ 2% of lung outside the grid | **16,345** | **1,034** |

Split: train 11,442 / 724 events, val 2,451 / 155, test 2,452 / 155.

- **The mask requirement is the main selection step.** Every cancer case has
  a mask, but only ~63% of controls do. The masked controls are not a known
  random sample: TotalSegmentator was run upstream on an enriched list plus
  later batches. The event rate is therefore 6.3% rather than NLST's ~4%.
  Ranking metrics (C, AUC) are unaffected in expectation. Absolute risks are
  not calibrated to NLST. Running TotalSegmentator on the remaining ~9.2k
  controls would remove this.
- **Checking it (S1b, `src/s1b_mask_selection.py`).** "Unaffected" holds only
  if the dropped controls are like the kept ones. Among controls with a
  year-0 CT on disk, S1b compares kept vs dropped on clinical (age, sex,
  race, smoking, T0 screen result), scanner (manufacturer, kernel, slice
  thickness, kVp, mA, diagnostic quality) and administrative (older
  experiment's split, pid) variables. Standardised mean differences
  (|SMD| < 0.1 = balanced), plus the cross-validated AUC of predicting "has a
  mask" from each variable set (0.5 = unrelated). It also writes
  inverse-probability weights (`data/mask_ipw.csv`) for a later weighted C/AUC
  that would estimate performance on all NLST controls with a CT. S5 does not
  use them yet. Known from the scan list: 60% of controls in the old
  experiment's train split have masks vs 69% in its val/test, so masks were
  generated in some order, not as one random draw. Whether that order relates
  to anything clinical is what S1b checks. Synthetic check: with mask
  availability built to depend on age and one kernel, S1b flags exactly those
  two (SMD 0.42, 0.44), and the weights recover all controls' mean age
  (64.72 vs 64.71).
  **Result (2026-10-17):** every clinical and scanner variable is balanced
  (|SMD| < 0.05; e.g. age 61.3 vs 61.4, positive T0 screen 26.1% vs 26.2%), and
  none of them predicts having a mask (CV AUC 0.50–0.51). Only the processing
  order differs (old split SMD −0.18, pid −0.12; AUC 0.57). The weights are
  0.95–1.09, so weighting would change nothing. On everything measured, the
  masked controls are a random subset. Unmeasured traits (pack-years,
  emphysema) can't be checked from the IDC table.
- **Image source.** `derived/ct_1x1x2mm/{pid}_yr0.nii.gz` (1×1×2 mm, int16 HU)
  and `derived/totalseg_fullres/{pid}_yr0/seg.nii.gz`. The IDC scan list
  (`experiments/lcrisk_discovery/data/manifest.csv`) is used only to know
  which series that CT came from, to look up acquisition metadata. Its own
  time/event/split columns are not used.

**Outcome (time-to-event, origin = the T0 scan).** From `prsn`:

- `event = 1` if `candx_days` (days to lung-cancer diagnosis) exists;
- `time = (candx_days if event else canc_free_days) − scr_days0`, in days.

The origin is the scan, not randomisation (the T0 scan is a median 5 days
later), because the image is what the model sees. `scr_days0` is missing for
145 participants, who get 0. Median follow-up is 6.6 years. Median time to
diagnosis is 2.2 years. **27% of cancers are diagnosed within a year of T0**,
most of them screen-detected on that same scan. Those cases are partly a
detection task, so S5 also reports C among patients still at risk at 1 year.

**Image QC (S3)**, from S2's per-patient measurements:

- preprocessing succeeded;
- lung cranio-caudal extent ≥ 200 mm (lungs fully scanned; a 30-patient
  sample had 158 mm and 48 mm truncated scans, with normal lungs at
  240–310 mm);
- lung volume ≥ 1.5 L (catches segmentation failures);
- ≤ 2% of lung voxels fall outside the fixed grid.

**Split.** Fresh, patient-level, random 70/15/15, stratified by event, seed
`20261005`. There is one scan per patient, so nothing leaks between splits.
All three splits have the same event rate. Unlike the old mentor split,
train and test come from the same distribution.

## 2. Network input (S2, constants in `src/nlst.py`)

| choice | value | why |
|---|---|---|
| region | union of TotalSegmentator lobes 10–14, dilated 5 mm | the organ where lung cancer arises. The dilation keeps juxtapleural nodules that the lobe masks can cut off |
| grid | 2.5 mm isotropic, 144×128×128 voxels (360×320×320 mm), centred on the lung bounding box | fixed **physical** size, so lung volume and shape are preserved (no per-patient rescaling). Covers the largest lungs in the sample (353×286×308 mm). The resampling is anisotropic-to-isotropic, from 1×1×2 mm |
| resampling | trilinear for HU. Mask: trilinear then ≥ 0.5 | simple and fast (~1.5 s/patient including file reads) |
| intensity | lung window [−1350, 150] HU → 1…255 (uint8). 0 = outside the mask | one byte per voxel (2.4 MB/patient). Outside-mask voxels get a value no lung voxel can have (−1000 HU air → ~60), so "not lung" is distinguishable from emphysema |
| storage | one memmapped `(N, 144, 128, 128)` uint8 array | sequential reads. The train split fits in RAM (~27 GB) |

Known limits, all deliberate for a first version:

- 2.5 mm voxels blur nodules under ~5 mm.
- A large central tumour that TotalSegmentator leaves out of the lobes would
  show up only as a hole in the mask. A spot check of one case's hole found
  hilar vessels, not a tumour.
- HU above 150 (calcification) saturates.

## 3. Model (`src/model.py`)

`LungCNN`, about 3.5 M parameters:

```
input 1×144×128×128
stem   conv3³ stride 2, 16 ch                    → 72×64×64
stage  [conv3³ → conv3³ stride 2] × 4, 32/64/128/256 ch → 36×32×32 … 5×4×4
every conv: no bias → GroupNorm(8) → ReLU
head   global average pool → dropout 0.3 → linear → η
```

- **VGG-style, no residuals.** It is easy to read and shape-trace, and deep
  enough (9 convs, receptive field covering the volume) for a baseline.
- **GroupNorm, not BatchNorm.** Train and eval mode compute the same function
  at any batch size. There are no running statistics, which in the previous
  project went stale and shifted every eval-mode output.
- **Single output η.** The Cox model needs a log-risk score, not a
  probability.

## 4. Training (`src/s4_train.py`)

| choice | value | why |
|---|---|---|
| loss | Cox negative partial log-likelihood, Breslow ties, per event, over each batch | uses every patient's follow-up and censoring. Verified against a brute-force Breslow sum (exact) and sksurv's fitted β (0.67041 vs 0.67041) |
| batch | 32 shuffled patients | ~2 events per batch at 6.3% prevalence. A batch with 0 events has no Cox gradient and is skipped (logged as `skipped_batches`, ~12%); another ~27% have exactly 1 event |
| events per batch (option) | `--events-per-batch N`: every batch = N events + (32 − N) non-events. Same number of batches per epoch; non-events are each used ~once per epoch, events ~2× (N = 4) | removes skipped and 1-event batches, so every step ranks several cases against controls (less noisy gradient). The partial likelihood is still a ranking objective; η's absolute level is not calibrated either way, and S5 refits HRs on test. Off by default |
| optimiser | AdamW, weight decay 1e-2 | standard default |
| learning rate | peak 1e-3. Linear warm-up over the first 3 epochs, then cosine decay to 0 by the last epoch, set every batch (`--schedule cosine`) | the first run (constant 3e-4) underfit: train loss barely moved and train C stayed below val C. A higher peak lets the model move further; warm-up keeps the first steps from blowing up; the decay lets it settle at the end. `--schedule constant` gives the old behaviour. `--min-lr` sets a floor for the cosine (default 0): in `cosine`, the last 20 epochs (lr < 1e-4) moved val C by < 0.005, so a floor is not expected to matter |
| length | 80 epochs, always run to the end (`--patience 0`) | early stopping on noisy val C ended the first run at epoch 17, when its val loss was still improving. Running the whole schedule also means the decay actually happens |
| precision | bf16 autocast on GPU | ~2× faster, no loss scaling needed |
| augmentation | per sample on GPU: rotation ±10° in the axial plane, scale ±10%, shift ±6 voxels | mild pose/size variation. No L-R flip, because the lungs are not symmetric (3 vs 2 lobes) |
| weight averaging (option) | `--ema 0.999`: an exponential moving average of the weights, `ema = 0.999·ema + 0.001·w` after every optimiser step (≈ the last 1,000 steps, ~3 epochs), with the decay ramped up over the first steps. Evaluation, selection and saving then use the EMA weights; `val_c_raw` logs the raw weights for comparison | with ~2 events per batch the weights jitter from step to step (val C ±0.04 between epochs); the average sits in the middle of that noise. Costs one extra model copy. Off by default |
| model selection | Harrell C on the whole val split after each epoch. Keep the best epoch (`--select val_c`, default) or the last epoch (`--select last`) | simple. With ~155 val events, val C is noisy, so the best epoch is slightly optimistic; the test split is the honest number. In `cosine` the end of training is flat (best 0.688 at ep 57, last 0.687), so `last` removes the selection step at no cost |
| logged per epoch | train loss; val Cox loss with whole-split risk sets and its η=0 baseline; train C (eval mode, fixed 2,000 patients); val C; SD of val η; learning rate; peak GPU memory (`gpu_gb`) | train and val C are measured the same way, so the gap is real overfitting. A val loss below its η=0 baseline means the score carries information. `gpu_gb` shows whether a larger batch fits |

**Test discipline.** `pred.csv` contains η for every patient, including test,
but only `s5_evaluate.py` reads the test rows. Nothing in training or
selection does.

## 5. Evaluation (`src/s5_evaluate.py`)

Per split (val for reference, test as the result):

- Harrell C with a 1,000× patient-bootstrap 95% CI;
- C among patients event-free and in follow-up at 1 year;
- time-dependent cumulative/dynamic AUC at 1, 2, 3 and 5 years (IPCW,
  censoring estimated on train);
- C of age alone, as a reference point;
- Kaplan–Meier curves of the test split in η tertiles, with cut points from
  val, and a log-rank p.

**Adjustment for sex and smoking** (`--adjust sex,cigsmok`, the default; any
`cohort.csv` columns can be listed, e.g. `--adjust sex,cigsmok,age`). Both
are known lung-cancer risk factors and could be partly what the image score
picks up (men have larger lungs; current smokers differ in lung density). On
the test split:

- `km_test_strata.png`: KM by η tertile inside each of the 4 sex × smoking
  groups (cigsmok 1 = current, 0 = former). If the tertiles still separate
  within every panel, sex and smoking cannot explain the separation. Each
  panel has only 23–56 events, so its CIs are wide.
- `km_test_adjusted.png`: unadjusted KM (left) next to covariate-adjusted
  curves (right). These come from a Cox model `tertile + sex + cigsmok`
  fitted on test, then standardised: each curve is the mean predicted
  survival if every test patient were in that tertile, keeping their own sex
  and smoking. The 95% bands come from 200 patient-bootstrap refits.
- `adjusted_hr.csv`: Cox hazard ratios without and with the adjustment, for η
  per SD (val SD) and for mid/high vs low tertile. An adjusted HR that stays
  well above 1 means the score carries information beyond sex and smoking.

Checked with a synthetic score built only from sex and smoking plus noise.
Its HR per SD fell from 1.13 to 0.91 after adjustment, while smoking kept HR
≈ 1.9. That is the expected collapse.

Result with `--adjust sex,cigsmok,age,lung_ml,lung_mean_hu` (`cosine`, test,
2026-10-17): HR per SD 1.64 → 1.37 [1.15, 1.62], p 0.0004 (sex + smoking alone:
1.59); high vs low tertile 3.59 → 2.33. About a third of the log-HR is shared
with lung size, density and age, and two thirds remain. The adjustment is
linear, and mean HU is only a crude emphysema measure. Use `--out` to write
a re-run to another folder than `runs/{run}/eval/`.

**Added value over clinical variables** (always run; `--added-only` runs only
this). Does η improve *discrimination*, not just association? Three Cox
models: clinical (`--clinical`, default age, sex, cigsmok, lung_ml,
lung_mean_hu, i.e. the lung size and density the CNN tracks), clinical + η per SD,
and η alone. They are fitted on **val** (`--fit-split`): train would overweight
η, since the CNN fitted those patients (`epb4_s1`: train C 0.97), and test stays
untouched until scoring. On test: C, C after 1 y and AUC 1/2/3/5 y per model,
plus Δ = (clinical + η) − clinical with a paired bootstrap (both models on the
same resampled patients, no refit). Output `added_value.csv`,
`added_value_fit.csv`. Synthetic checks: η carrying information outside the
clinical variables → ΔC +0.035 [0.020, 0.050]; η built from them only → ΔC
−0.003 [−0.006, 0.001].

Result (`cosine`, 2026-10-17): test C clinical 0.654 [0.612, 0.696], η alone
0.648, clinical + η 0.675 [0.636, 0.713]; **ΔC +0.021 [−0.017, +0.058]**
(p_boot 0.135), C after 1 y +0.028, AUCs −0.014 to +0.011. η adds about 0.02 C,
in line with the adjusted HR 1.37, but 155 test events cannot confirm it
(detectable ΔC ~0.04). This is the bar for SSL.

## 5b. Saliency: where does the model look? (`src/s6_saliency.py`)

Applied to a saved `model.pt`; nothing is retrained. Val split only (`--split
test` is refused).

| choice | value | why |
|---|---|---|
| main method | occlusion: a 40 mm cube (`--patch 16`) slid in 20 mm steps (`--stride 8`); saliency = mean over covering cubes of η − η(occluded) | model-agnostic and easy to explain ("hiding this region lowered the risk by X"). ~2k forward passes per patient, seconds on a GPU |
| occlusion fill | lung voxels in the cube → the patient's median lung value (`--fill median`) | filling with 0 ("outside lung") would shrink the lung, and η tracks lung volume (ρ 0.65 in `baseline`), so every cube would look important. Median fill asks "what if this region looked like the rest of this lung?" |
| second method | Grad-CAM at `model.features[4]` (64 ch, 18×16×16, ~2 cm cells) | a cross-check. The last layer is 5×4×4 (~7 cm cells), too coarse |
| patients | all val events + as many random val controls; figures of the top/bottom-η events and top-η controls (`--top-k 4`) | events for the lobe test, controls as its reference |
| lobe labels | TotalSegmentator lobes (labels 10–14) resampled exactly as in S2 (nearest neighbour, box centre from `lungs_index.csv`); the 5 mm dilation rim gets the nearest lobe | the cache only stores the merged lung. Juxtapleural nodules sit in the rim. An orientation check (left lobes at lower x, upper lobes higher z in RAS) is printed |
| cancer lobe | prsn `loclup`/`loclin` → LUL, `locllow` → LLL, `locrup` → RUL, `locrmid` → RML, `locrlow` → RLL; hilar/mediastinal/unknown-only events left out | 1,779/2,058 NLST cancers have exactly one lobe flagged |
| lobe test | enrichment = lobe's share of positive saliency / its share of lung volume (1 = no preference). Each event's cancer lobe vs the same lobe in val controls and in events whose cancer was elsewhere; Mann–Whitney p; also by diagnosis ≤ 1 y vs > 1 y | cancers are mostly upper-lobe, and a model that looks at the upper lobes in everyone would beat a naive 1-in-5 chance level without finding any tumour. Cancers diagnosed within a year are likely visible at T0, so the effect should be strongest there |

Caveat: a saliency map shows where η is sensitive, not what feature it uses
(size, density, a nodule). Checked on synthetic CT/mask files run through
the real S2 code: orientation checks pass, η per patient matches the batch
prediction (6e-8), figures render.

**Result on `cosine` (2026-10-17, 313 val patients, 20 min).** Orientation
checks pass (1.0 / 0.987), confirming labels 10–14 = LUL/LLL/RUL/RML/RLL. The
positive occlusion saliency sits in the **posterior lung bases** in nearly
every patient, cancers and controls alike: median enrichment RLL 2.34, LLL
1.55, RML 0.19, RUL 0.12, LUL 0.10. Grad-CAM agrees. The cancer lobe is not
singled out: 141 events, enrichment 0.148 vs 0.124 for the same lobe in
controls (p 0.24) and in other-lobe events (p 0.47), top-lobe hit rate 0.24 vs
0.21 chance. Cancers diagnosed within 1 year show no effect either. The CNN
reads lung extent and inflation at the bases (consistent with η ~ volume,
ρ 0.65), not tumours. Figures: `runs/cosine/saliency_val/`.

## 6. Running it

```bash
env/.venv/bin/python src/s1_candidates.py              # done 2026-10-05
sbatch src/submit_s2_preprocess.sh                     # 40 CPU tasks
env/.venv/bin/python src/s2_preprocess.py --merge
env/.venv/bin/python src/s2_preprocess.py --qc         # look at figs/s2_inputs.png
env/.venv/bin/python src/s3_cohort.py
sbatch src/submit_s4_train.sh --run-name cosine       # runs/cosine
env/.venv/bin/python src/s5_evaluate.py --run cosine   # incl. sex/smoking adjustment
```

Mask-selection check and saliency (2026-10-17):

```bash
env/.venv/bin/python src/s1b_mask_selection.py              # login node, minutes
sbatch src/submit_s6_saliency.sh --run cosine               # gpu-short, ~1 h
```

Seed/variant comparison (2026-10-17). A short 40-epoch screen (cosine compressed
to 40 epochs), 2 configs × seeds 0–2, that differ only in `--events-per-batch 4`.
Both use EMA + last epoch; EMA vs raw weights is read from `val_c_raw` in the
same runs. Compare configs by mean ± SD of val C over seeds; test stays untouched.
The winner is then trained with the full 80 epochs. ~50 min per job, so it fits
the `gpu-short` partition (8 L40S, 2 h max), which usually has free GPUs, unlike
`gpu`/`gpu-l40s` (7 L40S, often full). Command-line `-p`/`-t` override the script's
`#SBATCH` lines and must come before the script name:

```bash
for s in 0 1 2; do
  sbatch -p gpu-short -t 01:30:00 src/submit_s4_train.sh --run-name e40_ema_s$s --seed $s \
      --epochs 40 --ema 0.999 --select last
  sbatch -p gpu-short -t 01:30:00 src/submit_s4_train.sh --run-name e40_epb4_s$s --seed $s \
      --epochs 40 --ema 0.999 --select last --events-per-batch 4
done
```

Run 2026-10-17: the six jobs loaded their 27 GB at once and each load took 29 min
instead of ~7, so two runs hit the 1.5 h limit at epoch 37. Results in
`PROGRESS.md`. For several jobs, use `-t 02:00:00` and stagger the starts, e.g.
`--begin=now+10minutes`, `+20minutes`, … so the loads don't overlap.

GenomeDK cancels GPU jobs that average < 75% GPU utilisation after 2 h
(https://genome.au.dk/docs/computing-with-gpus/); check a running job with
`jobinfo <jobid>`. An 80-epoch run takes ~1.5 h (on `gpu`, ~87 min for `cosine`).

The outcome and clinical table (`prsn`) is the public TCIA IDC-780 package
(CC BY 4.0), in `data/external/` (gitignored). The project copy under
`meta/nlst_780/` is owner-only. Regenerate with:

```bash
mkdir -p data/external/nlst_780 && cd data/external/nlst_780 && curl -sSLO \
  https://www.cancerimagingarchive.net/wp-content/uploads/package-nlst-780.2021-05-28.zip \
  && unzip -q package-nlst-780.2021-05-28.zip && cd nlst_780 \
  && unzip -q nlst780.idc.delivery.052821.zip && for z in *.csv.zip; do unzip -q "$z"; done
```

Smoke test without Slurm (CPU, ~3 min): run `s2_preprocess.py --pids ...
--out-dir <scratch>`, build a small cohort CSV from it, then run
`s4_train.py --smoke --device cpu --cohort ... --cache ... --runs-dir ...`.

## 7. Obvious next steps (not done)

- Seeds 1–4 of the same recipe to measure run-to-run spread (3 seeds of the
  40-epoch screen in §6, 2026-10-17).
- Batch 64 (`--batch-size 64`) once `gpu_gb` shows it fits.
- TotalSegmentator for the remaining controls, which gives the full ~25.8k
  cohort at NLST prevalence.
- 1.5–2 mm resolution, or a nodule-scale crop, for small nodules.
- Select the epoch by val loss instead of val C (smoother: in `baseline` the best
  val loss was at epoch 14, the best val C at epoch 5).
- A non-image reference Cox model (lung volume + age, from `cohort.csv`): the
  `baseline` score correlates ρ 0.65 with lung volume, so the CNN should beat it.
- Residual blocks or pretrained weights, each tested against the current
  recipe one at a time.
- Clinical adjustment (age, sex, smoking are in `cohort.csv`).
