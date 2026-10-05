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
| image QC (S3, below) | to be filled after S2 | |

- **The mask requirement is the main selection step.** Every cancer case has
  a mask, but only ~63% of controls do. The masked controls are not a known
  random sample: TotalSegmentator was run upstream on an enriched list plus
  later batches. The event rate is therefore 6.3% rather than NLST's ~4%.
  Ranking metrics (C, AUC) are unaffected in expectation. Absolute risks are
  not calibrated to NLST. Running TotalSegmentator on the remaining ~9.2k
  controls would remove this.
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
| batch | 32 shuffled patients | ~2 events per batch at 6.3% prevalence. A batch with 0 events has no Cox gradient and is skipped (logged as `skipped_batches`, ~12%) |
| optimiser | AdamW, lr 3e-4, weight decay 1e-2, constant lr | standard defaults. No schedule to tune |
| precision | bf16 autocast on GPU | ~2× faster, no loss scaling needed |
| augmentation | per sample on GPU: rotation ±10° in the axial plane, scale ±10%, shift ±6 voxels | mild pose/size variation. No L-R flip, because the lungs are not symmetric (3 vs 2 lobes) |
| model selection | Harrell C on the whole val split after each epoch. Keep the best epoch, stop after 12 epochs without improvement (max 60) | simple. With ~157 val events, val C is noisy, so the best epoch is slightly optimistic; the test split is the honest number |
| logged per epoch | train loss; val Cox loss with whole-split risk sets and its η=0 baseline; train C (eval mode, fixed 2,000 patients); val C; SD of val η | train and val C are measured the same way, so the gap is real overfitting. A val loss below its η=0 baseline means the score carries information |

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

## 6. Running it

```bash
env/.venv/bin/python src/s1_candidates.py              # done 2026-10-05
sbatch src/submit_s2_preprocess.sh                     # 40 CPU tasks
env/.venv/bin/python src/s2_preprocess.py --merge
env/.venv/bin/python src/s2_preprocess.py --qc         # look at figs/s2_inputs.png
env/.venv/bin/python src/s3_cohort.py
sbatch src/submit_s4_train.sh                          # runs/baseline
env/.venv/bin/python src/s5_evaluate.py --run baseline
```

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

- Seeds 1–4 of the same recipe to measure run-to-run spread.
- TotalSegmentator for the remaining controls, which gives the full ~25.8k
  cohort at NLST prevalence.
- 1.5–2 mm resolution, or a nodule-scale crop, for small nodules.
- Residual blocks, a cosine lr schedule, or pretrained weights, each tested
  against this baseline one at a time.
- Clinical adjustment (age, sex, smoking are in `cohort.csv`).
