# The lungs / sternum 3D CNN, explained

This document explains every part of the 3D CNN pipeline on branch
`cnn3d-lungs-sternum`:
- which NLST data goes in;
- how the cohort is built;
- what the network sees and outputs;
- how the Cox loss uses time;
- which hyperparameters are tuned and why;
- which metrics are reported and what they mean for *this* data;
- what "stratification" means here;
- whether age/sex/smoking adjustment is needed.

Section 9 holds the results and the explanation of *why* lungs and sternum differ.
It is filled in from `data/c4_*.csv` once the GPU runs have finished.

The code lives in `src/c0_…c5_*.py`, the shared pieces in `src/cnn3d.py` and
`src/organs.py`.

---

## 0. The question and why two organs

**Question:** does a CNN reading a baseline low-dose CT learn anything that
predicts *future* lung-cancer diagnosis?

- **Lungs = positive control.** This is where lung cancer grows. Emphysema,
  lung density and already-visible nodules are all known risk markers, so a
  working pipeline *must* find signal here.
- **Sternum = negative control.** It is a bone with no biological route to
  lung cancer. If the sternum model "predicts" cancer, something other than
  biology is leaking in: age, sex, scanner/kernel, or body size.

The *difference* between the two is the informative quantity. "Lungs
C = 0.65" means little on its own. "Lungs 0.65 while sternum is 0.52 and
drops to 0.50 after adjusting for age" is a finding.

---

## 1. Which NLST data is used

| Source | What it gives us | Used for |
|---|---|---|
| NLST CT images (IDC), resampled by the mentor to `derived/ct_1x1x2mm/{pid}_yr0.nii.gz` | Baseline (T0) low-dose chest CT, 1×1 mm in-plane, 2 mm slices, Hounsfield units (HU) | CNN input |
| `derived/totalseg_fullres/{pid}_yr0/seg.nii.gz` | TotalSegmentator 117-label segmentation on the same grid | Organ masks: **lungs** = union of the 5 lobe labels (10–14); **sternum** = label 116 |
| `experiments/lcrisk_discovery/data/manifest.csv` | Per baseline scan: `time` (days from the T0 scan to lung-cancer diagnosis, or to last cancer-free follow-up), `event` (1 = diagnosed), `split` (the mentor's train/val/test), `kernel`, `kvp` | Outcome and split. Reused as-is and not re-derived. Its `label` column is **identical** to `event` (checked in C0) |
| `nlst.csv` (IDC series metadata) | `SeriesDescription`, a comma-coded acquisition string: study year, vendor, kernel, FOV, slice thickness, kVp, mA | Scanner / kernel covariates, and dropping non-baseline series |
| IDC-780 `prsn` table (TCIA public package, in `data/external/`) | Age, gender, race, `cigsmok` (current vs former smoker) | Clinical adjustment (section 8). Complete for all 6,306 patients |

**Not available at all:** pack-years, smoking duration, death/cause of death,
BMI, comorbidities (these would need a CDAS data request). No never-smokers are
in NLST (everyone had ≥30 pack-years), so `cigsmok` is a weak smoking axis.

**Only the baseline (T0) scan is used.** The T1/T2 follow-up scans exist, but
using them would mix "predicting the future" with "watching a tumour grow".

**Outcome timing.** Median time to diagnosis among cases is 828 days, and
**26% of cases are diagnosed within 365 days of the baseline scan.** Many of
these were probably already visible on that scan, as the screen-detected
nodule that led to the diagnosis. This matters a lot for interpreting the
lungs model (section 9, lead time).

---

## 2. How the enriched 6.3k cohort is constructed

The full manifest has 26,254 baseline scans and 1,061 lung cancers (4.0%). The
mentor built a **case-enriched cohort** (`discovery_pipeline/enriched_cohort_pids.txt`)
to make per-organ embedding extraction cheaper: **all cases plus a random
sample of controls**. The sampling rate of controls differs by split
(`data/pid_lists/cnn_cohort_v1_sampling.csv`):

| split | manifest cases / controls | kept cases | kept controls | control sampling rate | event rate after enrichment |
|---|---|---|---|---|---|
| train | 742 / 17,635 | 742 (100%) | 1,484 | **8.4%** | **33.3%** |
| val | 159 / 3,778 | 159 (100%) | 1,908 | 50.5% | 7.7% |
| test | 160 / 3,780 | 160 (100%) | 1,920 | 50.8% | 7.7% |

So this is a **case-control (outcome-dependent) sample**, and the train split is
far more enriched than val/test. `src/c0_build_cohort.py` then applies the same
exclusions as the heart-radiomics project (`cnn_cohort_v1_attrition.csv`):

| step | patients | cancers | test patients / cancers |
|---|---|---|---|
| enriched ∩ manifest | 6,373 | 1,061 | 2,080 / 160 |
| series really is the baseline scan (study year 0)¹ | 6,359 | 1,047 | 2,079 / 159 |
| has CT + TotalSegmentator mask | 6,359 | 1,047 | 2,079 / 159 |
| CT covers ≥ 80 slices (drops partial scans) | 6,306 | 1,037 | 2,067 / 157 |
| has age / sex / smoking status | **6,306** | **1,037** | **2,067 / 157** |

The manifest's cancer flag agrees with the clinical table's diagnosis date
for every patient (0 disagreements).

¹ For 14 patients, all cancer cases, the manifest's "baseline" series was
actually a T1/T2 scan. They are excluded because that image isn't a baseline.

Per organ, C1 additionally requires the mask to have ≥ 500 voxels (see
`data/cnn_cache/{organ}_missing.csv`).

The **legacy model** used a 400-patient subset of this cohort (`subset_v1.csv`).
399 of those pids survive the exclusions above. The old test set (127 patients,
18 cancers) is contained in the new test set (2,067 / 157), so the old
checkpoint can be re-scored on 16× more test patients.

**What the enrichment does and doesn't break**
- **Ranking metrics survive.** C-index and AUC compare cases with controls.
  Sampling controls at random (within a split) does not change the expected
  case-vs-control ranking, so these metrics stay unbiased.
- **Absolute risk does not.** Any "probability of cancer" learned on a 33%
  event-rate training set is inflated about 8× relative to the real 4%, and
  the val/test rate (7.7%) is different again. So we never report calibrated
  absolute risks or Brier scores. The model output is used only as a
  *ranking* score (section 4).
- **Class balance in training** is actually helped: 725 training cancers
  instead of about 30 at a natural 4% rate among 2,191 patients.

---

## 3. What the network sees (preprocessing): legacy vs new

Both recipes start from the same **mask-zeroed crop**: a bounding box around
the organ mask plus 8 voxels, with every voxel outside the mask set to a
background value. Everything after that differs (`src/c1_cache_volumes.py`,
figure `figs/c1_inputs/{organ}_inputs.png`):

| | legacy (old model) | new recipe | why it matters |
|---|---|---|---|
| **Geometry** | whatever the crop size, trilinearly **squashed into a 64³ cube** | resampled to **isotropic** voxels (lungs 4 mm, sternum 2 mm), then centre-padded to a **fixed physical box** (lungs 384×288×384 mm → 96×72×96; sternum 96×128×256 mm → 48×64×128) | In the legacy cube every organ is the same size, so volume (e.g. hyperinflated lungs in emphysema) is erased, and the aspect ratio is distorted differently per patient. The new grid keeps "1 voxel = 4 mm" for everyone. |
| **Downsampling** | trilinear point sampling | mask-aware **area averaging** | Going from 1 mm to 4 mm by point sampling picks single noisy voxels (aliasing). Averaging keeps the mean density, which is what emphysema measures. |
| **Lung window** | [−1350, 150] HU | same | Covers air (−1000) to soft tissue. Emphysema lives at < −950 HU. |
| **Sternum window** | [−160, 240] HU (soft tissue) | **[−500, 1300] HU (bone)** | **Bug in the legacy model:** sternal bone is mostly 100–1000 HU. On the smoke-test patients, 56–57% of sternum voxels were above 240 HU and were clipped to the same white, so the old sternum model was nearly blind to bone density. |
| **Outside the mask** | set to −1000 HU (air), one channel | channel 0 = windowed HU (0 outside), **channel 1 = mask** | For lungs, −1000 HU is also the value of emphysematous lung. The legacy model could not tell "outside the lung" from "destroyed lung". The mask channel removes that ambiguity. |
| **Augmentation** | none | random ±3-voxel shift, HU offset (sd 10 HU) and scale (sd 2%) | Makes the model robust to small positioning / calibration differences, so it doesn't memorise training patients. |

---

## 4. The model, and how Cox uses time

### Architecture (`src/cnn3d.py: CNN3D`)
Four blocks of `Conv3d(3×3×3) → BatchNorm → ReLU`, with 2× max-pool after the
first three. Channels are 16 → 32 → 64 → 128. Then global average pooling,
dropout and one linear layer, giving **one number per patient, η ("eta")**. It
is deliberately the same small network as the legacy model, and the layer
layout is identical, so old checkpoints load unchanged. That way the
comparison isolates **data + preprocessing + objective**, not architecture.

The receptive field grows with depth: each unit of the last conv layer sees
about 38 input voxels across (≈ 15 cm for lungs at 4 mm). The network has
292k parameters. Global average pooling then averages the
final feature map over the whole organ. So the model is suited to **diffuse,
organ-wide** patterns (density, emphysema, texture, shape), and poorly suited
to a single 6 mm nodule, which is 1–2 voxels at 4 mm.

### Time is *not* an input
A common confusion: the CNN never receives the follow-up time. The input is
only the image. Time enters **only the loss function**, through *risk sets*.

The Cox proportional-hazards model says each patient's hazard (instantaneous
risk of diagnosis) is `h(t | image) = h₀(t) · exp(η)`. Here `h₀(t)` is a
shared, unknown baseline hazard over time, and `exp(η)` is the patient's
multiplier. The partial likelihood asks, **at each moment a cancer occurs:
among everyone still cancer-free and under follow-up at that moment, how
likely was it to be this patient?**

`P_i = exp(η_i) / Σ_{j still at risk at t_i} exp(η_j)`,  loss = −mean over cancers of log P_i.

`h₀(t)` cancels out of this ratio, which is why time never needs to be
modelled or fed to the network. It only decides *who is in each
denominator*.

**Worked example (4 patients):**

| patient | outcome |
|---|---|
| A | cancer at year 1 |
| B | cancer at year 3 |
| C | cancer-free, left follow-up at year 2 (censored) |
| D | cancer-free, followed to year 5 (censored) |

- At year 1 (A's diagnosis) everyone is at risk: term `η_A − log(e^{η_A}+e^{η_B}+e^{η_C}+e^{η_D})`.
- At year 3 (B's diagnosis) only B and D are still at risk. A already had
  cancer and C left at year 2: term `η_B − log(e^{η_B}+e^{η_D})`.
- C and D add no term of their own. But C sits in A's denominator: "C was
  still cancer-free at year 1, when A got it", so the model is pushed to rank
  A above C. After year 2 we know nothing about C, and C drops out.

Compare the **BCE classifier** (legacy): it labels C as 0, "never gets
cancer", even though C was only watched for 2 years. It also ranks A and B as
equally positive, although A was diagnosed 2 years earlier. Cox uses the
censoring and the ordering correctly. In a minibatch the risk set is
approximated by the patients in that batch. Batch 32 at the 33% training
event rate gives about 10 events per batch (`cox_ph_loss`, verified to match
a brute-force partial likelihood to 4·10⁻⁸ and sksurv's Breslow fit).

### What the prediction is
- **η is a log relative hazard**: unitless, and defined only up to an
  additive constant (adding 5 to every η leaves the loss unchanged). What
  matters is differences. If `η_A − η_B = 1`, A has `e¹ ≈ 2.7×` B's
  instantaneous risk, at every time point (the "proportional hazards"
  assumption).
- We therefore **standardise η on the validation set** (mean 0, SD 1) and
  report *HR per SD*. When several seeds are trained we average their
  standardised η (the ensemble).
- An absolute probability, e.g. "7% risk within 5 years", would need the
  Breslow baseline survival `S₀(t)`: `S(t|x) = S₀(t)^{exp(η)}`. `S₀` estimated
  on this enriched cohort is biased (section 2), so **we use η only for
  ranking**: C-index, KM tertiles, HRs.
- The BCE model's output is a logit of "cancer at any time during follow-up".
  With `pos_weight` (≈ 2 for train) and the train/test prevalence shift, its
  0.5 threshold is meaningless. That is why the legacy confusion matrices
  ("predicts all positive / all negative") were not informative and are no
  longer reported.

---

## 5. Hyperparameters: what each does, which are tuned, and why

| hyperparameter | value(s) | what it does | why this choice |
|---|---|---|---|
| **learning rate** (AdamW) | **grid: 1e-4, 3e-4, 1e-3** | Step size of each weight update | This is the most important knob. Too high and the loss is noisy or diverges; with BatchNorm, BCE can collapse to a constant output, as the legacy 2D model did. Too low and the model underfits within the epoch budget. The best value depends on the loss scale, so it is tuned. |
| **weight decay** (AdamW, decoupled L2) | **grid: 1e-4, 1e-2** | Shrinks weights toward 0 every step, a preference for simpler functions | The main defence against memorising 2,191 training patients. Tuned because the right strength is data-size dependent. |
| **dropout** before the final layer | **grid: 0, 0.3** | Randomly zeroes 30% of the 128 pooled features during training | Stops the final layer relying on a few features. Tuned because it helps small data and can hurt large data. |
| batch size | 32 (legacy 8) | Patients per gradient step | For Cox, the batch *is* the risk set, and too few events per batch makes the loss noisy. 32 gives about 10 events. Fixed: it is linked to the learning rate, so tuning both would double the grid. |
| max epochs / **early stopping** | 80 / patience 12 on **val Harrell C** | Stop when val C hasn't improved for 12 epochs, and keep the best epoch | Early stopping is itself a regulariser, so no epoch count is chosen by hand. The legacy model trained a fixed 60 epochs and took the best val AUC over only 18 val cancers, which is very noisy selection (see "winner's curse" below). |
| augmentation | on (ablation: off) | Random shifts and intensity jitter | Regularisation. Tested as an ablation rather than tuned, to show its effect. |
| width | 16 base channels | Capacity | Kept equal to the legacy model so the comparison isolates data, preprocessing and loss. Not tuned. |
| input spacing / FOV | lungs 4 mm, sternum 2 mm | Resolution vs memory | Fixed at cache time (re-caching costs about 20 CPU-hours). 4 mm keeps all 6.3k lung volumes in GPU memory (8 GB). |
| loss | Cox (ablation: BCE) | Training objective | Section 4. |

**How the tuning works**
- 12 configs per organ, seed 0, trained on train and scored on val (C2 `grid`).
- The best val C per organ is chosen (C3, `data/c3_chosen_configs.json`).
- The chosen config is retrained with **5 fresh seeds** (C2 `final`), and
  only those are evaluated on test (C4).

**Why val Harrell C as the selection metric:** it is the same quantity we
report, it uses the time information, and val has 155 cancers (vs 18 for the
legacy model). Loss is a worse selection metric because it depends on the
batch composition.

**Winner's curse:** the maximum of 12 noisy val estimates is optimistic, so
the chosen config's val C overstates its true performance. Retraining with new
seeds and reporting only the test set removes that bias.

**The test set is looked at once**, after all choices are frozen. Test
predictions are written by every run for convenience, but only C4 reads them,
and only for the pre-declared model groups.

---

## 6. Metrics, and how they relate to this data

| metric | definition | why we use it / caveat |
|---|---|---|
| **Harrell's C-index** (primary) | Over all *comparable* pairs (the patient with the earlier time had cancer), the fraction where that patient got the higher score. 0.5 = chance, 1 = perfect ranking | Uses time and censoring and needs no time horizon. Unbiased under our case-control sampling. It can be mildly influenced by the censoring pattern, hence Uno's C as a check. |
| **Uno's C** (τ = 6 years) | Harrell's C with inverse-probability-of-censoring weights, truncated at 6 years | Robust to how much censoring there is. It should agree with Harrell's; if not, censoring is distorting things. |
| **time-dependent AUC** at 1 / 3 / 5 years | Patients diagnosed by year t vs those still cancer-free after t | Answers "how well does it separate cancers *within t years*". **Year 1 is dominated by cancers that may already be visible**, so compare it with years 3 and 5 (section 9). |
| **binary AUC** (event yes/no, time ignored) | The legacy metric | Reported only to compare with the old 0.687. It treats a patient censored at 2 years as a definite non-case. |
| **HR per SD** | Cox hazard ratio for a 1-SD increase in the score, unadjusted and adjusted | A clinically readable effect size. The adjusted version is the key to section 8. |
| **ΔC** (paired bootstrap) | C(model A) − C(model B) on the same test patients | Tests whether one model is really better. Paired because the two models share patients, which makes it much more precise than comparing two separate CIs. |
| **KM curves + log-rank** | Test patients split into score tertiles; Kaplan–Meier cancer-free curves per tertile; multivariate log-rank test (the mentor's convention) | A visual endpoint. The p-value tests "are the curves different", not "how well does it rank". |
| **95% CI** | Event-stratified bootstrap (1,000 resamples; cases and controls resampled separately) | Honest uncertainty. |

**Uncertainty is driven by the number of cancers, not patients.** With 18 test
cancers the legacy AUC 0.687 had a CI of [0.54, 0.83], ±0.14. With 157 test
cancers the CI is roughly ±0.045 (≈ 1/√events scaling). That alone is a
reason to distrust the old numbers: an AUC of 0.60 and one of 0.75 were both
compatible with the 18-event test set.

---

## 7. What "stratification" means here (it means several things)

1. **Stratified sampling (the cohort).** The enriched cohort samples by
   outcome: all cases plus a fraction of controls. The legacy
   `subset_v1.csv` stratified again: 120 cases + 280 controls. The purpose is
   enough events to learn from. The price is biased absolute risk (section 2).
2. **Stratified vs unstratified splits.** A *stratified* split keeps the event
   rate equal across train/val/test. The mentor's split was made on the full
   manifest (4% everywhere), so after enrichment it is **not** stratified:
   33% train vs 7.7% val/test. We keep it anyway, for comparability with the
   mentor's and the radiomics results and because it is patient-disjoint.
   The consequence is that the prevalence the model trains on differs from
   test. That is harmless for ranking metrics and fatal for 0.5 thresholds.
3. **Event-stratified subsampling (learning curve).** When training on 25%
   or 50% of train, cases and controls are subsampled separately, so the
   event rate stays 33% and only the *amount* of data changes.
4. **Event-stratified bootstrap.** Each bootstrap replicate keeps exactly 157
   test cancers, so CI width reflects score noise, not random variation in
   how many cancers were drawn.
5. **Risk stratification (KM tertiles).** Patients are split by *predicted*
   score into low/mid/high groups. This is the clinical use case: "who should
   be screened more intensively?"
6. **Stratified (subgroup) analysis.** C is re-computed within soft vs sharp
   reconstruction kernels and within lead-time groups. If signal exists only
   in one stratum, that stratum explains the signal.
7. **Stratified Cox** (not used here): a Cox model with a separate baseline
   hazard per stratum, e.g. per scanner. It is an alternative to adjusting
   for kernel as a covariate. We use covariate adjustment because it gives an
   interpretable coefficient.

---

## 8. Adjustment for age, sex and smoking: why, how, and do we need it?

**Why it matters: confounding.** A confounder causes both the input and the
outcome.
- **Age** raises lung-cancer risk steeply (NLST: about 8% per year) *and*
  changes every organ on CT. Lung density rises and emphysema progresses;
  bone density falls, especially in women after menopause.
- **Smoking status** raises risk *and* leaves marks in the lung (emphysema,
  bronchial wall thickening).
- **Sex** changes body and organ size and bone density. Its effect on lung
  cancer in NLST is small.

A CNN reading the image can therefore rank cancers above controls **just by
estimating age from the image**. That is real prediction, but it is not new
information: a questionnaire already knows the patient's age.

**Our two organs are the textbook case:**
- The *lungs* model can pick up age and smoking through density and emphysema,
  and also genuinely lung-specific signal: nodules, early tumours,
  parenchymal change.
- The *sternum* model has no route to lung cancer *except* through such
  confounders: bone density ≈ age/sex, sternum size ≈ body size/sex.
  **An unadjusted sternum C above 0.5 does not break the negative control if it
  disappears after adjustment.** If it survives adjustment, something else
  (scanner, kernel, body habitus) is leaking.

**How we adjust.** Post hoc, without retraining. The CNN is trained on the
image only, so its score means "what the image says". C4 then:
1. Fits a Cox model with CNN score + age + sex + smoking (+ kernel, + image
   covariates) and reads off the score's **adjusted HR per SD**. This asks
   whether the score still carries risk information *among patients of the
   same age, sex and smoking status*.
2. Compares test C of clinical-only vs clinical + CNN (**ΔC**, paired
   bootstrap). This asks whether the image adds to what the questionnaire
   already gives. The combined model is fit on the **val** set, because the
   CNN has memorised the training patients: train C is much higher than val
   C. Fitting the combination on train would give the CNN far too much
   weight. This is the same leakage family as the mentor's "cross-fitting"
   case study. Clinical-only models are fit on train as usual.
3. Reports the Spearman correlation of the score with each covariate: how
   much of the score *is* age, emphysema, kernel and so on.

**Do we need it here?**
- **For training: no.** Feeding age into the CNN would let it lean on age and
  make the image contribution harder to read.
- **For interpretation: yes, essential.** It is the only way to say whether
  the lungs signal is "lung biology or visible disease" rather than "an
  age/smoking detector", and whether the sternum control is really null.
- Even without the clinical table we can partly check this with **image-derived
  covariates**: lung volume, LAA-950 emphysema index, mean lung density,
  Perc15, sternum HU and volume, plus kernel. That is what C4 does now
  (section 9).

### Clinical covariates: source
The age/sex/smoking table (`prsn`) comes from the public TCIA IDC-780 package
(CC BY 4.0), downloaded into `data/external/` (gitignored). The mentor's copy
under `meta/nlst_780/` is owner-only. Regenerate with:

```bash
mkdir -p data/external/nlst_780 && cd data/external/nlst_780 && curl -sSLO \
  https://www.cancerimagingarchive.net/wp-content/uploads/package-nlst-780.2021-05-28.zip \
  && unzip -q package-nlst-780.2021-05-28.zip && cd nlst_780 \
  && unzip -q nlst780.idc.delivery.052821.zip && for z in *.csv.zip; do unzip -q "$z"; done
cd /faststorage/project/aura_thymus/student_pipeline
env/.venv/bin/python src/c0_build_cohort.py      # fills age/sex/race/cigsmok
```

`c4_evaluate.py` switches the clinical analyses on automatically when these
columns are filled. It only runs after the C2 training runs exist.

---

## 9. Results and why lungs and sternum differ

*Pending: filled in from `data/c4_*.csv` after `sbatch … submit_c2_train.sh final` and `c4_evaluate.py`.*

Planned decomposition of the "real difference" (old → new):

| step | model group | what changes | question answered |
|---|---|---|---|
| 1 | `old_ckpt@old_test` | – | reproduces the old 0.687 (127 pts / 18 cancers) |
| 2 | `old_ckpt` | test set 127 → 2,067 | was 0.687 real, or small-sample luck? |
| 3 | `legacy_subset` | retrain with 2,048 val patients for selection | how noisy is the old recipe itself? |
| 4 | `legacy_scale` | train 179 → 2,191 patients | effect of data size alone |
| 5 | `new_lc_*` → `new_main` | new preprocessing + Cox, at 179 / 548 / 1,096 / 2,191 train | effect of the fixes, and the learning curve |
| 6 | `new_bce`, `new_noaug` | one component removed | how much does the objective / augmentation matter? |
| 7 | `null` | labels permuted | the floor (≈ 0.5) |

"Why" analyses: lead time (events ≤ 1 y vs a 1-year landmark), kernel strata,
correlation of the score with lung volume / LAA-950 / density (lungs) and bone
HU / size (sternum), HR adjusted for these, a covariate-only baseline, and
Grad-CAM (`figs/c5_gradcam/`).
