# Progress log

Running log of what's been done, what was found, and decisions made along the
way. See `PLAN.md` for the staged design this follows.

## Environment

No `conda`/`module`/`pip` available system-wide on this cluster (front-end
node `fe-open-01`, base is `/usr/bin/python3` 3.9.21 with no site-packages).
`python3 -m venv` + `ensurepip` work fine, and the login node has outbound
internet access to PyPI, so:

```
python3 -m venv env/.venv
env/.venv/bin/pip install numpy nibabel matplotlib pandas
```

Installed so far: `numpy 2.0.2`, `nibabel 5.3.3`, `matplotlib 3.9.4`,
`pandas 2.3.3`. All scripts are run as `env/.venv/bin/python src/<script>.py`.
More packages (`torch`, `lifelines`, `scikit-survival`, `scikit-learn`) will
be added to this same venv as later stages need them.

Note: this is a login node, fine for lightweight I/O (visualization,
header-only checks) but heavy compute (CNN training) must go through Slurm,
same as the mentor's `code/submit_*.sh` pattern — not run directly here.

## Stage 0 — subset selection (`src/build_subset.py`)

Built `data/pid_lists/subset_v1.csv`: 400 patients (120 positive / 280
negative) sampled from `discovery_pipeline/enriched_cohort_pids.txt` ∩
`experiments/lcrisk_discovery/data/manifest.csv`, seed=42.

**Finding during Stage 1 QC, fixed here:** 3 of the initial 400 candidates
(`214975`, `205278`, `218012`) had only 5-22 axial slices in their baseline
CT (`derived/ct_1x1x2mm/{pid}_yr0.nii.gz`) — a few NLST series in this cohort
are partial-coverage scans, not full chest exams, and are useless for 3D
organ cropping. Added a `MIN_Z_SLICES = 80` QC gate (median in this cohort is
~156 slices, so this only rejects genuinely broken/partial scans) and
backfilled from the same stratified pools to keep 400/120/280. See
`data/pid_lists/README.md` for the current column list (now includes `n_z`).

## Stage 1 — visualize raw CT (`src/stage1_visualize_raw.py`)

Plotted HU histogram + mid sagittal/coronal/axial slices (soft-tissue window
WL40/WW400, same convention as `code/render_saliency_overlays.py`) for 5
subset patients (3 positive, 2 negative). Output: `figs/stage1_raw_ct/`.

Visually confirmed: correct orientation (via `nib.as_closest_canonical`),
expected bimodal HU histogram (air ~-850 HU, soft tissue ~0-100 HU), full
lung fields/heart/ribs/spine visible in the right anatomical places, spacing
consistently (1, 1, 2) mm as expected from the `ct_1x1x2mm` resampling. This
is also what caught the thin-volume issue above — worth doing before
trusting any later stage.

## Stage 2 — per-organ masking + visualization

Chose **7 organs** for the multi-organ sweep (`src/organs.py: ORGAN_SET`):
`anterior_mediastinum` (thymus proxy, custom sternum-anchored region — no
direct TotalSegmentator thymus label exists), `lungs` (union of the 5 TS lobe
labels — directly disease-relevant, a positive-control organ: if this doesn't
show signal in Stage 4, something is wrong with the pipeline), `heart`,
`aorta`, `liver`, `spleen` (immune-relevant, abdominal — distant from the
lung), and `sternum` (bone — a **negative control**: no biological reason to
carry lung-cancer signal, so if it ranks high in Stage 4 that's a red flag for
scanner/positioning confounding rather than real biology).

`src/organs.py` reuses the mentor's exact TotalSegmentator label map and the
sternum-anchored anterior-mediastinum geometry (`sternum_per_slice_roi`,
soft-tissue-only, largest-connected-component) copied/adapted from
`code/extract_organ_embeddings.py` and `code/prevascular_roi.py` — the pure
numpy/scipy parts only, since those files import torch/3DINO at module level
and we don't need that until Stage 3. Masking = bbox-crop + zero outside mask
to real air HU (-1000), **at native resolution** — deliberately not resized
to a fixed size yet, since that's a Stage-3 modeling decision (CNN input
size), not a data-correctness one.

`src/stage2_visualize_masks.py`: for the same 5 sample patients as Stage 1,
plots [whole-slice context with mask contour] + [mask-zeroed crop] per organ.
Visually verified on 2 patients (one event=1, one event=0) — all 7 organs
correctly located and plausibly sized (e.g. anterior_mediastinum is a small
triangle directly under the sternum; lungs/heart/liver/spleen/aorta/sternum
all match expected anatomy and rough dimensions). Output:
`figs/stage2_masks/{pid}_yr0_masks.png`.

Installed `scipy 1.13.1` into the venv (needed for the mediastinum mask's
distance-transform/connected-component geometry).

**Timing:** ~3s/patient for load + all 7 organs on the login node → ~20 min
for the full 400-patient subset. `src/cache_organ_crops.py` caches every
(patient, organ) crop to `data/organ_crops/{organ}/{pid}.npy` (float16,
native resolution) plus `data/organ_crops/manifest.csv` (which (pid, organ)
pairs actually got a mask — some organs can be missing/too small for a given
patient), launched in the background. Once done, Stage 3 reads directly from
this cache instead of recomputing masks per epoch (recomputing on the fly
would cost ~20 min/epoch, far too slow for iterative training).

## Stage 3 — simple CNN classifier (`src/stage3_cnn.py`)

Decision: no Slurm/GPU access confirmed yet, so Stage 3 is designed to be
**CPU-friendly on the login node** — a 2D-slice CNN, not 3D. For each organ
crop, picks the axial slice with the most in-mask coverage
(`best_slice_2d()`), HU-windows it (same soft-tissue/lung window convention
as Stage 2), resizes to 64x64, and trains a small 3-conv-block CNN
(`SimpleCNN2D`) to predict the binary lung-cancer event. Class imbalance
handled via `BCEWithLogitsLoss(pos_weight=...)`, not resampling. Train/val/
test split reused as-is from `subset_v1.csv`.

**Bug found and fixed during smoke-testing:** the first version's
`Dataset.__getitem__` reloaded the full native-resolution crop from disk and
recomputed the best-slice search on *every* access, i.e. every epoch — for
`lungs` (~17MB/patient at native res) that's ~120GB of repeated disk I/O over
40 epochs, and the first real run hung past a 2-minute timeout. Fixed by
precomputing every patient's 64x64 slice ONCE in `Dataset.__init__` and
caching it in memory (the 400-patient x 64x64 float array is a few MB, no
issue) — `__getitem__` now just indexes into that cache. Verified fast
(seconds, not minutes) with a smoke test on `sternum` using partial cache data
— that result itself is not meaningful (only 125/400 patients were cached at
the time, and skewed almost entirely positive because the cache job processes
`subset_v1.csv` in row order and positives were listed first) and was
discarded before the real run.

**Known limitation to keep in mind for Stage 4 (documented, not fixed):** the
`split` column inherited from the mentor's `manifest.csv` is not stratified
*within* our 400-patient subset — event rates come out to train 46.7%
(84/180), val 19.4% (18/93), test 14.2% (18/127). Val/test each have only 18
positive cases, so any organ's held-out AUC will have a wide confidence
interval on this subset size. Not re-splitting, to keep results comparable to
the mentor's own train/val/test assignment.

Currently waiting on `cache_organ_crops.py` (background) to finish caching
all 400 patients before running the real (not smoke-test) `--organ all` pass.

**Full run (400 patients, all 7 organs) — results:** all 7 masks were
present for 397-400/400 patients (`data/organ_crops/manifest.csv`). Trained
one `SimpleCNN2D` per organ, 40 epochs, best-val-AUC checkpoint selected.

| organ | val AUC | test AUC | test confusion `[[tn,fp],[fn,tp]]` |
|---|---|---|---|
| anterior_mediastinum | 0.562 | 0.464 | `[[0,109],[0,18]]` (predicts all positive) |
| lungs | 0.523 | 0.482 | `[[109,0],[18,0]]` (predicts all negative) |
| heart | 0.562 | 0.512 | `[[109,0],[18,0]]` (predicts all negative) |
| aorta | 0.524 | 0.633 | `[[0,109],[0,18]]` (predicts all positive) |
| liver | 0.475 | 0.656 | `[[0,109],[0,18]]` (predicts all positive) |
| spleen | 0.454 | 0.648 | `[[0,109],[0,18]]` (predicts all positive) |
| sternum | 0.544 | 0.425 | `[[86,23],[15,3]]` (only one with varied predictions) |

Full run: `figs/stage3_cnn/{organ}/training_curves.png`, raw numbers in
`data/stage3_results.csv`.

## Stage 4 — organ ranking, WITH confidence intervals (`src/stage4_ranking.py`)

Computed Hanley-McNeil (1982) 95% CIs for each organ's test AUC (from the
confusion matrix's true-class row sums: n1=18 positive, n0=109 negative for
every organ). Output: `figs/stage4_ranking.png`.

```
organ                    AUC    95% CI            CI excludes 0.5?
liver                  0.656  [0.510, 0.802]   yes (barely)
spleen                 0.648  [0.502, 0.794]   yes (barely)
aorta                  0.632  [0.486, 0.779]   no
heart                  0.512  [0.366, 0.657]   no
lungs                  0.482  [0.339, 0.625]   no
anterior_mediastinum   0.464  [0.323, 0.606]   no
sternum                0.425  [0.288, 0.562]   no
```

**Honest read of this result (do not skip to Stage 5 on this):**

1. **5 of 7 organs are statistically indistinguishable from chance** — wide
   CIs all spanning 0.5, exactly what `PLAN.md`'s own Stage 4 verification
   criterion warned to watch for ("not all organs at chance-level AUC").
2. **liver and spleen barely exclude 0.5** (lower CI bound 0.510 and 0.502),
   but **this is 7 uncorrected comparisons** — at a nominal 95% CI, getting
   ~2/7 "significant" purely by chance is unsurprising, not evidence of real
   signal. This is the same multiple-comparison trap the mentor's own
   `discovery_pipeline/PLAN.md` flags for ribs ("scattered non-replicating
   signal ... consistent with multiple-comparison noise, not real biology").
   No correction (Bonferroni etc.) would survive here.
3. **`lungs` — the organ we expected, as a positive control, to show the
   clearest signal (lung cancer is literally in the lung) — is at chance
   (AUC 0.482) and collapsed to predicting every patient negative.** This is
   the most important finding: a real positive control failing suggests the
   limitation is in the *method* (single best-coverage 2D slice per patient,
   ~180 training images, a small from-scratch CNN, 18 positives in
   val/test), not that organs genuinely carry no signal.
4. Several organs' predicted probabilities collapsed to an almost-constant
   value (confusion matrices show all-one-class predictions at the 0.5
   threshold) — consistent with the CNN not learning a real discriminative
   signal at all from this little data/this little of each volume, rather
   than learning a subtle-but-real one.

**This is a legitimate, useful first result for a "start simple" pipeline —
it demonstrates the mechanics end-to-end and honestly shows the naive
baseline isn't there yet — but it means Stage 5's KM curve should NOT be
built on `liver` as if it were a validated finding.** Options going forward
(paused here for a decision, not a technical blocker):
  a. Build Stage 5 mechanically anyway (e.g. on `liver`) purely as a
     pipeline-completeness demo, clearly labeled as non-validated.
  b. Strengthen Stage 3 first: use several/all axial slices per organ (a real
     3D CNN, or a 2.5D multi-slice input) instead of one 2D slice, since
     single-slice is almost certainly throwing away the signal that made
     `lungs` fail as a positive control.
  c. Jump to Stage 6 (benchmark against the mentor's frozen 3DINO
     embeddings) to check whether a stronger, pretrained feature extractor
     recovers the expected `lungs`/`anterior_mediastinum` signal that our
     from-scratch CNN missed — informative either way.

**Decision: option (b)** — the user has Slurm GPU access on this cluster
(GenomeDK, `gpu-l40s`/`gpu-h200` partitions), so we went straight to a real 3D
CNN instead of a 2D-slice compromise.

## Stage 3v2 — real 3D CNN, on GPU (`src/stage3_cnn3d.py`, `src/submit_stage3_cnn3d.sh`)

Same architecture family as the 2D version, extended to 3D (`Simple3DCNN`:
4 conv3d blocks with BatchNorm3d + global pool + FC), fed the **whole**
mask-zeroed organ crop resized to 64³ (trilinear) instead of one 2D slice.

Environment note: this cluster account has no conda/module system (see
top of this file) — the mentor's own GPU jobs (`code/submit_totalseg_fullres.sh`,
`code/submit_extract_embeddings.sh`) use `conda activate <env>`, which isn't
available to us, so `submit_stage3_cnn3d.sh` activates our own venv instead
(`source env/.venv/bin/activate`). Had to reinstall torch as a CUDA build
(`pip install torch`, no `--index-url .../cpu` this time) — the earlier
CPU-only install from the Stage 3v1 decision would have failed silently or
fallen back to CPU on a GPU node. Followed the cluster's GPU docs
(genome.au.dk/docs/computing-with-gpus) for the sbatch flags (`--partition
gpu-l40s`, `--gpus 1`, `--account aura_thymus`) — matches the mentor's own
scripts exactly — and staged `data/organ_crops/` (14 GB) to `$TMPDIR` first,
per the docs' recommendation (avoids repeated small-file reads over the
shared filesystem, relevant given the docs' "<75% GPU utilization after 2h
gets auto-cancelled" policy).

Before handing the sbatch script to the user, ran a CPU-only smoke test
(`--device cpu`, sternum organ) to catch bugs without spending GPU time —
confirmed the code path (loading, resizing, training loop) ran without
crashing, but it was still mid-training on the login node when the real GPU
job finished first (22s/organ on an L40S vs. never finishing in 20+ min on
CPU for one organ). Once the GPU result was in hand, killed it — note this
took two steps: `TaskStop` only detached the harness's tracking of the
background task, the underlying process (verified via `ps aux`) kept running
and accumulating CPU time on the shared login node until explicitly `kill`ed
by PID. Worth remembering for next time: `TaskStop` on a plain backgrounded
shell command is not guaranteed to kill the process tree.

**GPU run: `jobinfo` summary** — COMPLETED, exit 0:0, 3m54s wall time (2h
reserved), 87% GPU utilization, 14G staged to `$TMPDIR` in 20s. Real result,
not a smoke test:

| organ | val AUC | test AUC | 95% CI (Hanley-McNeil) | test confusion `[[tn,fp],[fn,tp]]` |
|---|---|---|---|---|
| **lungs** | 0.681 | **0.687** | **[0.543, 0.831]** | `[[95,14],[12,6]]` |
| **anterior_mediastinum** | 0.582 | **0.650** | **[0.504, 0.796]** | `[[7,102],[0,18]]` |
| aorta | 0.621 | 0.626 | [0.479, 0.773] | `[[96,13],[16,2]]` |
| sternum | 0.672 | 0.607 | [0.459, 0.755] | `[[109,0],[18,0]]` |
| liver | 0.673 | 0.541 | [0.395, 0.688] | `[[104,5],[17,1]]` |
| heart | 0.586 | 0.480 | [0.337, 0.623] | `[[89,20],[18,0]]` |
| spleen | 0.655 | 0.457 | [0.316, 0.598] | `[[0,109],[0,18]]` |

`figs/stage4_ranking_3d.png` (via `src/stage4_ranking.py --results-csv
data/stage3_3d_results.csv --out figs/stage4_ranking_3d.png`).

**This is a materially different and more encouraging result than the 2D
baseline, and worth reading carefully rather than just picking the top row:**

1. **`lungs` — our pre-declared positive control — now shows a real,
   CI-excludes-0.5 signal (0.687, CI [0.543, 0.831])**, with a confusion
   matrix that actually catches true positives (`12→6` split, not collapsed
   to one class like the 2D version's `18→0`). This is the key validation:
   it confirms the earlier 2D failure was a **method** limitation
   (single-slice information loss), not that these organs carry no signal —
   exactly as hypothesized when we paused after Stage 4v1.
2. **`anterior_mediastinum` (the thymus proxy — the project's actual
   scientific target) also shows a CI that excludes 0.5, though barely**
   (lower bound 0.504, essentially touching the boundary). This is
   encouraging for the immune-surveillance hypothesis this whole project is
   about, but should be reported cautiously: unlike `lungs`, this wasn't a
   pre-declared control, it's the primary hypothesis, tested at n=18
   positives — a genuinely small sample. Its confusion matrix
   (`[[7,102],[0,18]]`) shows the model leans heavily toward predicting
   positive (catches all 18 true positives but only 7/109 negatives) —
   AUC as a ranking metric is threshold-independent so this doesn't
   invalidate the AUC, but the 0.5-threshold behavior is poorly calibrated
   and would need fixing before using this as an actual risk score (relevant
   for Stage 5).
3. The other 5 organs remain at chance, same as the 2D run — no new false
   signal appeared, which is reassuring (the 3D upgrade didn't just make
   everything look better indiscriminately).
4. Multiple-comparison caveat still stands for `anterior_mediastinum`
   specifically (1 of 7 organs, barely-excluding CI) — `lungs` is on firmer
   footing since it was named as a positive control *before* running Stage 3,
   not picked post-hoc from the ranking.

## Stage 5 — Kaplan-Meier curves (`src/stage5_km_curve.py`)

Follows the mentor's exact KM convention (`discovery_pipeline/src/km_analysis.py`):
3D CNN risk score (sigmoid output) on the **held-out test set only** (val was
already used for checkpoint selection, so it's not clean held-out anymore) ->
tie-safe tertile split (`pd.qcut(risk.rank(method="first"), 3)`) ->
`lifelines.KaplanMeierFitter` per tertile -> `multivariate_logrank_test` for
a trend p-value.

**`lungs` result (run directly on the login node, CPU inference):** test
n=127, 18 events. Tertiles: Low risk (n=42, **1 event**), Mid risk (n=42, 8
events), High risk (n=43, 9 events). **Logrank trend p=0.037.**
`figs/stage5_km_curve/lungs_km.png` — clean separation, especially the Low
risk group (near-flat survival curve, almost no events) vs. Mid/High (which
overlap each other more than they separate from one another, but both
clearly separate from Low). Encouraging and consistent with the Stage 4 AUC
result, though still a small-sample result (single test split, no repeated
CV) — reported as encouraging, not as a proven finding.

**Performance note:** this one organ took ~10 min on the login node vs. the
GPU job's 22s/organ for full training — inference here is cheap, the
bottleneck is reading+resizing ~127 crops per organ sequentially over the
shared filesystem on a CPU-contended login node (no dedicated cores, unlike
an sbatch allocation's `-c 8`). Rather than run the remaining 6 organs
sequentially at that rate (~1h), packaged Stage 5 as a quick GPU job instead:
`src/submit_stage5_km.sh` (same `$TMPDIR`-staging pattern as Stage 3v2),
handed to the user to submit via `sbatch src/submit_stage5_km.sh`.

## Stage 6 — Grad-CAM saliency maps (`src/stage6_saliency.py`)

User asked to see "which parts of the image led to the result" and to write
it out as a NIfTI/similar for visualization. That's **Grad-CAM**
(Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017) — no new
model needed, just inspection of the checkpoints Stage 3v2 already saved.
Scoped to `lungs` + `anterior_mediastinum` only, per the user's explicit
narrowing (the validated positive control and the actual scientific target,
not all 7 organs).

**Method:**
1. Hook `Simple3DCNN.features[12]` (the last `Conv3d(64→128)` before
   `AdaptiveAvgPool3d`) for forward activations `A` and backward gradients
   `dY/dA`.
2. `alpha_k = mean_spatial(dY/dA_k)`; `CAM = ReLU(sum_k alpha_k * A_k)`,
   shape `[8,8,8]`, normalized per-patient by its own max.
3. Upsample the 8³ CAM directly to the crop's *native* resolution (trilinear,
   `align_corners=False`, same convention as `resize_and_window()`).
4. **Reproject into full-CT space.** Found during planning: the crop's
   bounding-box offset (`mask_zeroed_crop`'s `(lo, hi)` return value) is
   discarded at its only call site in `cache_organ_crops.py` — never cached
   anywhere. So the bbox is recomputed per patient by reloading the original
   CT+seg and rerunning `organ_mask()` + the same bbox arithmetic — checked
   with a hard assertion that the recomputed shape matches the cached crop's
   shape before trusting the reprojection.
5. Paste into a zero volume shaped like the full CT, save as
   `nib.Nifti1Image(vol, affine=ct_img.affine)` — the *canonical* CT's own
   affine (not identity), so it opens correctly aligned against the original
   scan in any NIfTI viewer.
6. PNG montage per patient (CT + CAM alpha-blended, sequential `"hot"`
   colormap since Grad-CAM is non-negative — unlike the mentor's signed IG
   in `code/organ_saliency.py`, which uses diverging `RdBu_r`), axial slice
   at peak CAM energy, matching `code/render_saliency_overlays.py`'s
   HU-window conventions.

Scope: 5 highest-risk + 5 lowest-risk test-set patients per organ (reusing
`stage5_km_curve.risk_scores_for_test()`), 20 patients total.

**Smoke-tested on the login node (CPU, 2 patients: one event=1, one
event=0)** before handing off the sbatch job:
- Bbox-recomputation assertion passed silently for both patients.
- Output `.nii.gz` shape (260,260,144) and affine matched the source CT
  exactly (`derived/ct_1x1x2mm/118669_yr0.nii.gz`) — confirmed with
  `nib.load(...).shape`/`.affine` on both.
- Visual check of the montage: overlay is spatially correct (sits right
  over the lungs, not shifted), but fairly coarse/smeared — expected, since
  an 8³ CAM upsampled ~30x will never have crisp edges. The highest-risk
  smoke-test patient's CAM concentrated more in the mediastinum/between the
  lungs than on any specific parenchymal nodule; noted here as an honest
  observation, not something to touch up before showing the real batch's
  results.
- Full-test-set patient selection (`risk_scores_for_test`) itself is slow on
  the login node (~10 min/organ, same bottleneck as Stage 5's login-node
  run) — this is why Stage 6 also goes through `sbatch` despite not needing
  a GPU computationally; `submit_stage6_saliency.sh` stages only the 2
  needed organs' crop subfolders to `$TMPDIR` (not all 7).
- Smoke-test output files deleted afterward so the real run starts clean.

**Job history (2 submissions, both real findings, not hidden):**
1. `gpu-l40s`, `--time 00:15:00` (job 64798970) — sat **PENDING** the whole
   time (`squeue` reason: `Resources`). Checked `sinfo`/`gnodes`: `gpu-l40s`
   is exactly **one node** (`gn-1002`) cluster-wide, shared with the plain
   `gpu` partition, and it was fully occupied (all 7 GPUs marked busy).
   Since this script doesn't actually need a GPU (Simple3DCNN is 4 conv
   layers, trivial compute), switched to `short` (44 CPU nodes, 12h limit,
   confirmed via `sacctmgr show assoc` that the account isn't restricted to
   specific partitions) instead of waiting on the single contended GPU node.
2. `short`, `--time 00:15:00` (job 64799090) — started immediately (good),
   but hit the time limit and got killed (`sacct`: `TIMEOUT`) partway
   through `anterior_mediastinum`. Root cause: staging the 8GB of `lungs` +
   `anterior_mediastinum` crops to `$TMPDIR` (~800 small `.npy` files over
   the network filesystem) alone took **10.5 of the 15 minutes**. `lungs`
   fully completed (10 patients + overview) in the remaining time;
   `anterior_mediastinum` got through 7/10 patients before being cancelled
   — no manifest CSV was written (that only happens once, after all organs
   finish), but every patient's individual `.nii.gz`/`.png` that *did*
   complete was valid and saved.
3. `short`, `--time 00:45:00` (job 64802239) — **COMPLETED**, 15:15 used out
   of 45:00 reserved. All 20 patients (both organs) processed; manifest
   written; both `_overview.png` figures produced.

**Bug found reviewing the real run's output (not the smoke test):** the
`anterior_mediastinum_overview.png` had row-2 (low-risk) panel titles
overlapping into row-1's images — `fig.tight_layout(rect=(0,0,1,0.94))`
wasn't giving the 2-row grid enough vertical room. Fixed the same way as
the earlier Stage 2 QC-figure overlap bug: `h_pad=3.0` + taller figsize +
`suptitle(y=0.995)`. Regenerated both `_overview.png` files directly from
the already-saved `.nii.gz`+source CT (no need to rerun the CNN/Grad-CAM/
bbox-recompute pipeline — the full-CT-space CAM was already on disk) via a
throwaway script, not a full sbatch resubmission.

**Visual review of the real (non-smoke-test) results:**
- `lungs`: high-risk patients' hottest CAM region is often central/
  mediastinal rather than clearly inside lung parenchyma, and the crop
  bounding box isn't always symmetric across both lungs (e.g. pid 102167's
  box leans toward one side) — consistent with the earlier note that an
  8³ CAM upsampled ~30x will be coarse, and a reminder that this is "what
  correlates with the model's score," not a verified tumor localizer.
  Low-risk patients' CAMs still light up somewhere (Grad-CAM highlights
  what drove *whatever* the logit was, including a confidently-low one —
  not "nothing," which is expected behavior, not a bug).
- `anterior_mediastinum`: the crop bounding box is tightly and consistently
  centered on the mediastinal ROI across all 10 patients (mask geometry
  behaving correctly), and within it the CAM concentrates on a well-defined
  rounded structure near the great vessels for most high-risk patients.
  4/5 high-risk patients saturate at risk≈1.00 and the low-risk group
  ranges 0.05-0.46 — a clean-looking split in this sample, but this organ's
  Stage 4 AUC CI [0.504, 0.796] barely excludes chance, so this is reported
  as an interesting pattern worth a closer look, not a confirmed result.

## Next: once the user runs `submit_stage5_km.sh` for the remaining 6
organs' KM curves, review all outputs together (Stage 5 + Stage 6) and
decide whether to pursue Stage 7 (benchmark against the mentor's frozen
3DINO embeddings).
