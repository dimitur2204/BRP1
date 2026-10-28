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
| D0 SSL setup | `src/setup_dino_env.sh` | login node, ~15 min | `ext/DINOv2-3D-Med/`, `env/.venv-dino/` |
| S7 SSL scans | `src/s7_ssl_cache.py --list`, `submit_s7_ssl_cache.sh` (40 CPU tasks), `--merge` | login node + Slurm | `data/cache/rounds_u8.npy`, `data/ssl_scans.csv` |
| D1 views | `src/dino_data.py` (`--qc`) | used by S8; QC on login node | `figs/d1_views.png` |
| S8 DINOv2 pretrain | `src/s8_dino_pretrain.py` via `submit_s8_dino_pretrain.sh` (1 GPU) | Slurm | `runs/dino/{name}/` |
| S9 DINOv2 features | `src/s9_dino_features.py` via `submit_s9_dino_features.sh` (1 GPU) | Slurm | `data/features/{run}/{ckpt}.npy` |
| S10 Cox probe | `src/s10_dino_probe.py` (`--summary`) | login node, minutes | `runs/dino_probe/{run}/{ckpt}_{use}/` (S4 format → S5) |
| S11 thymus score | `src/s11_thymus_score.py` | login node, ~1 min | `runs/thymus_score/` |
| S12 ViT fine-tune (D6) | `src/s12_dino_finetune.py` via `submit_s12_dino_finetune.sh` (1 GPU, ~4 h on an L40S) | Slurm | `runs/finetune/{name}/` (S4 format → S5) |

Cluster folder (since 2026-10-26): `/faststorage/project/aura_thymus/plamen_pipeline`. The large
shared items (`data/cache`, `data/external`, `env`, `ext`, the runs up to 2026-10-20) are symlinks
into `student_pipeline/`, which another pipeline now uses; new runs and logs live here.

Run folders (`runs/`, regrouped 2026-10-20; every script takes the path below `runs/` as the
run name, e.g. `s5_evaluate.py --run cosine_improved_trial/epb4_s0`):

| folder | contents |
|---|---|
| `runs/baseline`, `runs/cosine` | the first two CNN runs (S4/S5/S6); `cosine` is the reference |
| `runs/cosine_improved_trial/` | the CNN recipe trials of 2026-10-17 (`e40_ema_s*`, `e40_epb4_s*`, `epb4_s*`) |
| `runs/dino/trial_2h/{small,big}` | DINOv2 2 h trial pretraining (D2; were `d2_s` / `d2_m`) |
| `runs/dino/small_150ep/{default,stable}` | DINOv2 small model, authors' run length, default vs anti-collapse recipe |
| `runs/dino_probe/{dino run}/{ckpt}_{use}` | Cox probes (S10) on the features in `data/features/{dino run}/` |
| `runs/finetune/{dino_ep39,scratch}` | the DINO ViT fine-tuned with the Cox loss vs the same ViT from scratch (S12) |
| `runs/clinical_cox/{offset,joint}` | the CNN trained with the clinical covariates in the Cox loss (S4 `--clinical`, §11) |

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
  sbatch -p gpu-short -t 01:30:00 src/submit_s4_train.sh --run-name cosine_improved_trial/e40_ema_s$s --seed $s \
      --epochs 40 --ema 0.999 --select last
  sbatch -p gpu-short -t 01:30:00 src/submit_s4_train.sh --run-name cosine_improved_trial/e40_epb4_s$s --seed $s \
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
- Clinical adjustment (age, sex, smoking are in `cohort.csv`). Built as S4 `--clinical` (§11).

## 8. Self-supervised pretraining (DINOv2, branch `ssl-dino`)

Goal: learn lung features **without labels**, then predict time to cancer from
them (linear Cox probe, D5) and compare with `cosine` through the unchanged S5.
Plan and status: `PLAN.md` (D0–D6).

| choice | value | why |
|---|---|---|
| method / code | DINOv2 (DINO + iBOT + KoLeo losses) from [AIM-Harvard/DINOv2-3D-Med](https://github.com/AIM-Harvard/DINOv2-3D-Med) at `77bde4d`, cloned into `ext/` | 3D port with collapse monitors. No released weights, so we pretrain ourselves. Not CT-FM: 128,037 of its ~148k pretraining scans are NLST, so it has seen our test scans (arXiv 2501.09001, Suppl. S5) |
| environment | `env/.venv-dino` (Python 3.11, the repo's `uv.lock`, + pandas) | the repo needs ≥ 3.11; `env/.venv` (3.9) stays for S1–S6 |
| pretraining scans | yr0 + yr1 + yr2 of **train + val** patients that pass the S3 image QC (`data/ssl_scans.csv`, ~34.8k) | no labels are used, so later screening rounds are fair game and give ~2.5× the scans. Test patients are excluded (asserted in S7 and `dino_data.py`), so test scans stay unseen. Consequence: DINO sees more images than the CNN did (without labels); a yr0-only DINO run separates the two effects if needed |
| input | the S2 encoding and grid (2.5 mm, 144×128×128, uint8, 0 outside the lung) for every round | same view as `cosine`, so the comparison isolates self-supervised vs supervised learning. The ViT does not know the spacing; only consistency between pretraining and feature extraction matters |
| views | random 128³ cube; 2 global views (side 0.6–1.0 × 128 → 96³, 2.0–3.3 mm) and 4 local views (0.15–0.30 × 128 → 24³, 2.0–4.0 mm); repo augmentations (affine ±10°, shear 0.1, histogram shift, Gaussian smoothing) | effective voxel size stays near the 2.5 mm used for extraction. Lung fills only ~20% of the grid, so a view with < 30% lung voxels is redrawn (≤ 10 tries) |
| backbone | PRIMUS-M (EVA ViT, 864-dim, depth 16, patch 8³, RoPE + learned position embedding; 146 M parameters) | repo default; it accepts any input size, so the whole 144×128×128 lung (4,608 tokens) is one forward pass at extraction (D4): one embedding per patient, no crop pooling. A smaller ViT is compared in D2 |
| training | repo recipe (`train.yaml`: base lr 0.004 × √(batch/1024), layer decay 0.9, wd 0.04→0.2, 10 warm-up epochs, teacher EMA 0.992→1); 1 GPU, bf16 | the schedules depend on `max_epochs` × steps per epoch, so a run must keep them fixed when it resumes (S8 checks) |
| logging | Lightning CSVLogger, `runs/dino/{name}/metrics.csv`; the repo's collapse monitors (Gram matrix, teacher entropy, effective rank) re-routed from W&B by `src/dino_callbacks.py`; throughput (s/batch, share waiting for data, GPU memory) | no W&B account |

| features (S9) | teacher backbone, whole yr0 volume 144×128×128 at 2.5 mm, no augmentation → [CLS, mean of the 4,608 patch tokens] | the teacher (EMA) is the usual DINO feature extractor; the whole lung in one pass gives one vector per patient. `random` (same backbone, random weights) is the control |
| probe (S10) | z-score on train → elastic-net Cox (l1_ratio 0.5, 50 penalties) on train → penalty by val Harrell C → η for all patients in `pred.csv` | answers "is cancer risk in the frozen features?" with the backbone untouched. Same split, time and event as S4, and the S4 output format, so S5 evaluates it unchanged; S10 itself never computes a test metric. Conservative: on synthetic data, 126 noise columns cost ~0.03 C (PROGRESS 2026-10-19) |

Repo quirks found while wiring it up (2026-10-19):

- `scripts/run.py` applies command-line overrides **after** `parse()`, so values
  copied with `%` macros (`embed_dim: "%hidden_size"`, the callbacks'
  `log_every_n_steps`) ignore overrides. S8 updates before parsing.
- `scripts/run.py` cannot resume (`trainer.fit` without `ckpt_path`). S8 resumes
  from `ckpt/last.ckpt` and refuses if the launch arguments changed.
- Warm-up is fixed at 10 epochs of the run's own length, so a run needs
  `max_epochs` ≥ 11.
- The collapse callbacks only write to a WandbLogger (silently nothing otherwise).
- 1 of 25 repo unit tests fails (a test expects an error wrapper `run.py` lacks;
  cosmetic). `configs/predict.yaml` refers to modules that no longer exist.

Small-model comparison at the authors' run length (2026-10-20). The authors trained on 4 GPUs
with 5 scans × 6 accumulation each = 120 scans per optimiser step and 150 epochs of 250 steps per
GPU (750k samples). On one GPU: batch 8 × accumulation 15 = 120, epoch = 625 batches = 5,000
scans, 150 epochs. `stable` adds the authors' anti-collapse experiment (exp07: base lr 0.002,
centre momentum 0.9, gradient clip 1.0, projection 16,384). The winner's recipe then goes to the
big model.

```bash
COMMON="trainer#max_epochs=150 trainer#limit_train_batches=625 trainer#accumulate_grad_batches=15 \
  hidden_size=384 lightning_module#backbone#eva_depth=12 lightning_module#backbone#eva_numheads=4 \
  data_module#batch_size=8 data_module#num_workers=20"
sbatch -p gpu,gpu-h200 -t 1-00:00:00 -c 24 src/submit_s8_dino_pretrain.sh \
  --run-name small_150ep/default $COMMON
sbatch -p gpu,gpu-h200 -t 1-00:00:00 -c 24 src/submit_s8_dino_pretrain.sh \
  --run-name small_150ep/stable $COMMON lightning_module#base_lr=0.002 \
  lightning_module#center_momentum=0.9 lightning_module#gradient_clip_val=1.0 \
  lightning_module#projection_dim=16384
```

```bash
bash src/setup_dino_env.sh                                        # D0, login node
env/.venv/bin/python src/s7_ssl_cache.py --list                   # login node
sbatch src/submit_s7_ssl_cache.sh                                 # 40 CPU tasks
env/.venv/bin/python src/s7_ssl_cache.py --merge
env/.venv-dino/bin/python src/dino_data.py --qc                   # figs/d1_views.png
sbatch src/submit_s8_dino_pretrain.sh --run-name trial_2h/big trainer#max_epochs=100   # D2 (2 h, cut by the limit)
```

## 9. Thymus step 0: the published thymic-health score (`src/s11_thymus_score.py`)

Before building a thymus SSL model we measure how much a thymus model can tell us here.
The AIM-Harvard thymic-health score (Bernatz et al. 2026, *Nature*; per-participant CSV on
Zenodo 10.5281/zenodo.18306999, path `THYMUS_SCORE_CSV` in `src/nlst.py`) is a 0–100
percentile per NLST participant (higher = less fatty thymus) from their full system
(thymic-bed segmenter → SwAV ResNet-50 → classifier fine-tuned on expert involution grades).
It is roughly the ceiling for a thymus model, and later the label-free target for "did our
SSL learn thymus anatomy".

- **Test patients are never read.** Cohort and reference predictions are cut to train + val
  on load. Complete cases only (score present).
- **Discrimination:** Cox models fitted on train, Harrell C on val, 1,000× patient bootstrap,
  paired ΔC for clinical → clinical + thymus and clinical + η → clinical + η + thymus
  (clinical = S5's 5 variables, η = `runs/cosine`). Models containing η are fitted on val,
  because η was trained on the train patients (as S5 `--fit-split val`); both sides of that ΔC
  are fitted the same way.
- **HR per SD** (train SD): train + val for unadjusted / + age / + age, sex, smoking /
  + clinical; val only for + clinical + η, with a likelihood-ratio test of adding the score.
- **Histology:** cause-specific Cox (other cancers censored), adenocarcinoma vs the rest
  (prsn `de_type`, ICD-O-3 codes as in the thymus-heart-aging E3/E5 analyses).
- **All-cause death:** if prsn has `death_days` / `fup_days`, time from T0 as for the outcome.

```bash
env/.venv/bin/python src/s11_thymus_score.py      # login node, ~1 min → runs/thymus_score/
```

## 10. Fine-tuning the DINO ViT with the Cox loss (`src/s12_dino_finetune.py`, D6)

The S10 probe asks whether cancer risk can be read **linearly from frozen** DINO features (val C
0.658, below `cosine` 0.688). Fine-tuning asks the other question: does starting from the DINO
weights help when **every** weight learns from the cancer labels, as the CNN did in S4? Only 724
train events is little for a 22 M-parameter ViT from scratch (ViTs lack the CNN's locality
prior), so pretrained weights can help even when the linear probe is weak. The test is the same
ViT trained twice with the same script: from the DINO weights vs from random weights.

| choice | value | why |
|---|---|---|
| runs | `finetune/dino_ep39`: teacher of `small_150ep/stable` epoch 39; `finetune/scratch`: same backbone with random weights (seed 0, = S9's `random`) | epoch 39 is the checkpoint already probed (S10 was flat from epoch 9). The teacher, as in S9 |
| network | backbone rebuilt from the pretraining `launch.json` (S9's `load_model`, strict load); DINO/iBOT heads, student and the unused `up_projection` decoder are not trained | architecture and weights always match pretraining; drop-path / attention dropout 0.2 / 0.2 as pretrained |
| input / augment | whole 144×128×128 S2 volume (18×16×16 tokens, position embedding resampled from 12³), S4's `augment` | same view as `cosine`, S9 and S10 |
| head | [CLS, mean patch token] (768) → dropout 0.3 → Linear(768 → 1), zero-initialised, no bias | the S10 probe's features, so probe vs fine-tune is like for like. η = 0 at the start, so the first updates do not push a random head's noise into the pretrained blocks. A constant shift of η does not change the Cox loss |
| loss / batch | S4's `cox_loss` over 32 patients per batch; batches without events skipped (as S4) | the risk set must be in one forward pass, so no gradient accumulation. Block activations are recomputed in the backward pass (gradient checkpointing, default on; `--no-grad-checkpoint` to disable): without it batch 32 ran out of memory on an L40S (> 44 GB, job 6287322) |
| learning rates | head peak 1e-3 (as S4); backbone peak 1e-4 × layer decay 0.75: block *i* of 12 (0-based) × 0.75^(12−*i*), cls/position/patch embedding × 0.75^13 (3e-6 for the first block), final norm × 1. 3 warm-up epochs, cosine to 0, per batch | standard fine-tuning: the top blocks adapt, the early (local anatomy) filters barely move. **Scratch:** `--layer-decay 1.0 --lr-backbone 1e-3` (everything at 1e-3): layer decay only exists to protect pretrained weights and would freeze random early blocks |
| optimiser | AdamW, weight decay 0.05 (0 for biases, norms, layer-scale gammas, cls/position embedding), gradient-norm clip 1.0 | usual ViT fine-tuning settings; both runs identical |
| length / selection | 30 epochs, last epoch kept | ~12.5 min per epoch on an L40S (measured; ~7 h per run plus up to ~45 min cache load), so 80 epochs (S4) would be ~17 h. Pretrained fine-tunes usually settle within 20–30 epochs. A scratch curve still rising at epoch 30 is noted as a limit of the comparison |
| outputs | S4 format (`config.json`, `history.csv` + `lr_backbone`, `grad_norm`, `model.pt`, `pred.csv` for all splits, `curves.png`) | S5 evaluates it unchanged; test rows are only read by S5 |
| C index | `harrell_c_np` in S12 | `env/.venv-dino` has no scikit-survival; identical to sksurv on synthetic data with tied times/scores and on `cosine` val (0.68839) |

Repo bug: `Eva.set_grad_checkpointing()` calls the reentrant `torch.utils.checkpoint` with
`rope=` as a keyword, which it rejects; S12 wraps each block's forward with the non-reentrant
version instead (parameter names unchanged). `--smoke` crops each volume to a central 64³
because whole-lung attention on CPU exceeds the login node's ~16 GB; the full-size memory and
speed check is a short GPU job.

```bash
sbatch -p gpu-short -t 00:45:00 src/submit_s12_dino_finetune.sh --run-name finetune/timing \
  --ckpt "dino_epoch=039" --epochs 2                                   # memory + s/epoch
sbatch src/submit_s12_dino_finetune.sh --run-name finetune/dino_ep39 --ckpt "dino_epoch=039"
sbatch --begin=now+15minutes src/submit_s12_dino_finetune.sh --run-name finetune/scratch \
  --ckpt random --layer-decay 1.0 --lr-backbone 1e-3
env/.venv/bin/python src/s5_evaluate.py --run finetune/dino_ep39       # and finetune/scratch
```

**Results (2026-10-27).** S5, 155 events in each of val and test; ΔC = clinical + η − clinical
(clinical = age, sex, cigsmok, lung_ml, lung_mean_hu; Cox models fitted on val, bootstrap on test).

| | `cosine` CNN | `finetune/dino_ep39` | `finetune/scratch` |
|---|---|---|---|
| val C (last epoch) | 0.688 | 0.690 | 0.583 |
| test C (η alone) | 0.648 | 0.654 | 0.571 |
| test C after 1 y | 0.653 | 0.664 | 0.568 |
| test ΔC over clinical (0.654) | +0.021, p 0.135 | +0.027 [−0.009, +0.063], p 0.083 | −0.001, p 0.548 |
| test HR per SD η, adjusted for sex + smoking | — | 1.69 | 1.37 |

- **The DINO initialisation is what makes the ViT work.** From random weights the same recipe
  never fitted the training set (train C 0.53, η SD 0.03 against 0.75; gradient spikes up to
  ~200 at the peak lr). Its weak signal is lung size: with clinical in the model its η flips sign
  (HR 0.34) next to `lung_ml`. A lower lr might help a little, but 724 events are too few for a
  ViT without pretraining.
- **Fine-tuned DINO ≈ the CNN**, slightly ahead on every test number, nothing significant. Its
  AUC is flat from 1 to 5 years and C after 1 y ≥ overall, so it behaves like a general
  lung-risk score, not a detector of visible cancers. Much of η overlaps with age: age's HR per
  year falls from 1.059 to 1.029 when η is added (fitted on val).

## 11. Clinical covariates in the training loss (`s4_train.py --clinical`, branch `cnn-clinical-cox`)

So far the CNN was trained on the image alone and the clinical variables were added only
afterwards (S5). Its η therefore also learns risk that age, sex, smoking, lung size and density
already explain (`cosine`: with η in the clinical model, age's HR falls 1.060 → 1.029; scratch
ViT: η ≈ a lung-size proxy). The goal is a score from general lung features that adds to
clinical (ΔC over clinical, PROGRESS 2026-10-26). Here the Cox loss sees the clinical covariates
during training, so the CNN is rewarded only for risk they do not explain:

η_total = η_img(x) + zᵀβ, Cox loss on η_total (`model.ClinicalCox`).

| choice | value | why |
|---|---|---|
| covariates | `--clinical age,sex,cigsmok,lung_ml,lung_mean_hu` (= S5's clinical model); sex → male 0/1; z-scored with train mean/SD | the same reference S5 measures ΔC against. `lung_ml` and `lung_mean_hu` are image-derived, so the CNN is pushed away from lung size/density, which `cosine`'s η largely tracked. No missing values in `cohort.csv` |
| β start | Cox fit of the covariates alone on the train split (lifelines) | the CNN starts from the clinical model and its first updates go to the residual |
| `--clinical-mode offset` (default) | β fixed | the cleanest "image learns only the residual": η_img is exactly what is added to a fixed clinical model |
| `--clinical-mode joint` | β trained with the CNN (same lr schedule, no weight decay) | the usual joint model (DeepSurv with covariates). β can trade off against η_img, e.g. lung size may move from β into the image |
| everything else | the `cosine` recipe (S4 defaults: 80 epochs, peak lr 1e-3, warm-up 3, cosine, batch 32, best val C) | a one-change comparison with `cosine` |
| selection, logged C | val C of η_total; `history.csv` also logs `val_c_img` (image part alone), `val_c_clin` / `val_loss_clin` (clinical part alone) and `val_img_sd` | η_total is what is trained. `val_c_img` may be *below* `cosine` by design: the image no longer gets credit for age or lung size |
| `pred.csv` | `eta` = η_img (image part), `eta_clin`, `eta_total`; `clinical.json`: columns, mean/SD, β start/selected, clinical-only val C/loss | S5 runs unchanged on `eta` = the image part, so its ΔC (clinical + η − clinical, refit on val) is exactly the question; "η alone" in S5 is the residual score and is expected to be weak |
| input | the 2.5 mm `data/cache/lungs_u8.npy` (S2; the "big cut" used by every run so far). S4 stops if the cache volumes are not 144×128×128, so the 1.4 mm cache (`lungs_1p4mm_u8.npy`, another pipeline) cannot be used by mistake | same input as `cosine` |

Without `--clinical`, S4 is unchanged (same model, `model.pt` keys and `pred.csv`). `s6_saliency.py`
loads a plain `LungCNN` and does not read `ClinicalCox` checkpoints (keys `image.*`, `beta`).

```bash
sbatch src/submit_s4_train.sh --run-name clinical_cox/offset --clinical age,sex,cigsmok,lung_ml,lung_mean_hu
sbatch --begin=now+20minutes src/submit_s4_train.sh --run-name clinical_cox/joint \
  --clinical age,sex,cigsmok,lung_ml,lung_mean_hu --clinical-mode joint
env/.venv/bin/python src/s5_evaluate.py --run clinical_cox/offset     # and clinical_cox/joint
```

Success: test ΔC above `cosine`'s +0.021 [−0.017, +0.058], ideally with a CI > 0, and C after 1 y.

**Results (2026-10-28, test, 155 events; S5 Cox models fitted on val).** ΔC = clinical + η − clinical:

| run | val C total / image part | test η alone | test clinical + η | ΔC [95% CI], p_boot |
|---|---|---|---|---|
| `cosine` (reference) | 0.688 | 0.648 | 0.675 | +0.021 [−0.017, +0.058], 0.135 |
| `clinical_cox/offset` | 0.699 / 0.668 | 0.612 | 0.677 | +0.023 [−0.012, +0.055], 0.093 |
| `clinical_cox/joint` | 0.712 / 0.682 | 0.625 | 0.689 | +0.035 [−0.002, +0.075], 0.033 |

η's Spearman ρ with cigsmok / lung_ml / lung_mean_hu falls from 0.18 / 0.18 / −0.17 (`cosine`)
to ≤ 0.10 in absolute value; with age from 0.23 to 0.16–0.19.

Joint seeds (2026-10-28): ΔC s0 +0.035, s1 +0.026 [−0.012, +0.061], s2 +0.049 [+0.013, +0.090].
Ensemble of the three (mean of val-z-scored η, `runs/clinical_cox/joint_ens3`, S5 `--added-only`):
test clinical + η 0.697, **ΔC +0.043 [+0.007, +0.084]**, p_boot 0.014; ΔC after 1 y +0.034.
