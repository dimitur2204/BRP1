# Learning pipeline → Kaplan-Meier survival curve (multi-organ CNN sweep)

## Context

The mentor's `discovery_pipeline/` already produces organ-level lung-cancer
risk scores via a sophisticated stack: TotalSegmentator → frozen 3DINO-ViT
embeddings → adversarially-deconfounded CoxNet/RSF models → `lifelines` KM
curves, run on the full ~26k-patient NLST cohort. That stack is a black box to
someone new to the project — the goal here is a separate, smaller, from-first-
principles pipeline that reaches the *same kind of output* (a KM curve, and a
ranking of which organs carry lung-cancer signal) using models the user
already understands (a plain CNN classifier), so the underlying concepts are
visible and inspectable at every step, not just the end result.

Three requirements drove the design (from the user directly):
1. Visualize the data at every pipeline stage, not just the final curve.
2. Start with simple, familiar models (a CNN) before anything like frozen
   embeddings + Cox models — the mentor's stack becomes an optional later
   benchmark, not the starting point.
3. Work on a deliberately small, self-selected NLST subset (mirroring the
   mentor's own precedent of starting with the small IDC-780 package before
   scaling up), rather than the full cohort or the full multi-terabyte
   `derived/` tree.

Decisions confirmed with the user:
- **Multi-organ sweep from the start** (not single-organ-first) — the goal is
  literally to identify *which* organs are informative, so the CNN classifier
  is trained per-organ across ~5–10 organs from the outset, not just added as
  a later ranking step.
- **Segmentation is a given, off-the-shelf tool, not a learning target.**
  Reuse the mentor's precomputed TotalSegmentator masks
  (`derived/totalseg_fullres/{pid}_yr0/seg.nii.gz`) rather than retraining a
  segmenter. `totalseg_fullres` has tens of thousands of per-patient folders
  — the fix is to never enumerate/copy that whole tree; only ever look up the
  handful of `{pid}_yr0` folders belonging to our chosen subset, by pid,
  directly.
- **Subset size: ~150–400 patients, stratified by event**, not the full
  cohort and not the mentor's exact 780-patient scale. NLST incidence is low
  (the mentor's full cohort had ~1,061 events / 26,254 patients, ~4%), so a
  random sample this size would contain almost no positives — the subset must
  be stratified to guarantee enough events to learn from and to split into
  legible KM groups.

## Status: Stages 0-2 done, Stage 3 (CNN) not started

See `PROGRESS.md` for the detailed running log (what was built, what was
found, decisions made). Summary:

- **Stage 0** ✅ `data/pid_lists/subset_v1.csv` — 400 patients (120 positive /
  280 negative), stratified from `enriched_cohort_pids.txt` ∩ `manifest.csv`,
  verified to have a usable CT + TotalSegmentator mask *and* enough z-coverage
  (a QC gate added after Stage 1 caught 3 partial-scan outliers).
- **Stage 1** ✅ `figs/stage1_raw_ct/` — raw CT sanity-checked for 5 sample
  patients (orientation, HU range, spacing all correct).
- **Stage 2** ✅ `figs/stage2_masks/`, `data/organ_crops/` — 7-organ masking
  (anterior_mediastinum, lungs, heart, aorta, liver, spleen, sternum) visually
  verified, full-cohort crops cached.
- **Stage 3v1** ✅ (2D baseline, superseded) `figs/stage3_cnn/`,
  `data/stage3_results.csv` — CPU-friendly 2D-slice CNN. Result: 5/7 organs at
  chance, and the `lungs` positive control also failed — see `PROGRESS.md`.
  This motivated Stage 3v2 rather than trusting these numbers.
- **Stage 3v2** ✅ (real 3D CNN, GPU) `src/stage3_cnn3d.py`,
  `src/submit_stage3_cnn3d.sh`, `data/stage3_3d_results.csv` — user confirmed
  Slurm GPU access (`gpu-l40s`), so the full 3D crop (not one slice) is fed to
  a small 3D CNN, run via `sbatch`. Ran successfully (87% GPU util, 4 min).
- **Stage 4** ✅ `figs/stage4_ranking_3d.png` — **`lungs` (pre-declared
  positive control) and `anterior_mediastinum` (the thymus proxy — this
  project's actual hypothesis) both show test AUC CIs that exclude chance**
  (0.687 [0.543,0.831] and 0.650 [0.504,0.796] respectively); the other 5
  organs remain at chance. `lungs` clearing as a positive control is the key
  validation that Stage 3v1's failure was a single-2D-slice method
  limitation, not absence of signal. See `PROGRESS.md` for the full read,
  including the caveats on `anterior_mediastinum`'s borderline CI and
  uncalibrated 0.5 threshold.
- **Stage 5/6** — not started. Natural next step: KM curve using `lungs`
  and/or `anterior_mediastinum` as the risk score.

## New project folder (this one)

```
student_pipeline/
├── PLAN.md                 (this file)
├── env/                    (env setup notes/lockfile — fresh env, see below)
├── data/
│   ├── pid_lists/           subset_v1.csv + README.md  [done]
│   └── organ_crops/         per-organ cropped volumes, populated in Stage 2
├── src/                    (data loading, CNN model, training loop, KM code)
├── notebooks/              (stage-by-stage exploration/visualization)
└── figs/                   (QC + result plots, one subfolder per stage)
```

Read-only with respect to `derived/` and `discovery_pipeline/` — this project
never writes back into either.

Environment: create a fresh, modern env (e.g. `python 3.11`, `torch`,
`nibabel`, `numpy`, `pandas`, `lifelines`, `scikit-survival`, `matplotlib`,
`scikit-learn`) rather than reusing any of the three existing envs in the
repo — they're each pinned/tuned for their own frozen tooling and mutually
incompatible (`totalseg` env: py3.10 unpinned; 3DINO env: py3.9;
`open_thymus_segmentator` env: py3.8, `torch==1.12.0`).

## Staged pipeline

**Stage 0 — Subset selection & audit.** ✅ Done — see `data/pid_lists/`.

**Stage 1 — Visualize raw data (3–5 patients from the subset).**
Load raw CT (`derived/ct_1x1x2mm/{pid}_yr0.nii.gz`) and its segmentation with
`nibabel`; plot HU histograms and axial/coronal/sagittal grayscale slices
using the mentor's HU-windowing convention from `code/render_saliency_overlays.py`
(soft-tissue WL40/WW400 → display range [-160,240]; lung WL-600/WW1500 →
[-1350,150]). Pure sanity check on orientation/spacing/HU range before any
modeling. Output: `figs/stage1_raw_ct/*.png`.

**Stage 2 — Per-organ masking + visualization.**
Pick ~5–10 organ labels from the TotalSegmentator label set already resolved
in `code/extract_organ_embeddings.py` (e.g. thymus/anterior-mediastinum via
the sternum-anchored region logic in `code/prevascular_roi.py`, plus heart,
lungs, liver, aorta — a mix of "expected relevant" and "control" organs). For
each, bbox-crop and zero outside the mask (same corrected approach as
`extract_organ_embeddings.py`'s `_mask_only_crop`, not the leaky bbox-only
variant), resize to a fixed input size. Adapt `code/prevascular_roi.py`'s
`qc_figure` panel style (axial/coronal/sagittal + mask contour overlay) for QC
on a handful of cases per organ first, then run over the full subset. Output:
per-organ cropped volumes cached to `data/organ_crops/{organ}/{pid}.npy` +
`figs/stage2_masks/{organ}/*.png`.

**Stage 3 — Simple CNN classifier, trained per organ.**
Model: a small 3D CNN (3–4 conv blocks + pooling + FC, binary lung-cancer
event classifier) per organ, or a 2D-slice CNN on the crop's central/peak
axial slice as a faster warm-up before the 3D version. Deliberately simple
and supervised end-to-end — no frozen foundation model — so training curves,
confusion matrices, and Grad-CAM-style saliency overlays (reusing the
heatmap-over-CT convention from `code/render_saliency_overlays.py`) are all
directly interpretable. Train/val/test split reuses the `split` column
already in `subset_v1.csv`/`manifest.csv`. Loop this over all ~5–10 chosen
organs using the same subset and same CNN architecture. Output: one model +
metrics per organ in `src/`, training curves and Grad-CAM montages in
`figs/stage3_cnn/{organ}/*.png`.

**Stage 4 — Organ ranking.**
Compare the per-organ CNNs from Stage 3 by held-out AUC (with k-fold CV if
the subset supports it) — a simple, interpretable proxy for "which organ
carries lung-cancer signal." Output: a bar chart of per-organ AUC with CI
(`figs/stage4_ranking.png`) plus the top organ(s)' saliency montage.

**Stage 5 — Kaplan-Meier curve.**
Take the best-ranked organ's (or top-2/3 pooled) CNN output probability as a
continuous risk score. Split into tertiles with `pd.qcut` on a tie-safe rank
(exactly as `discovery_pipeline/src/km_analysis.py` does), fit
`lifelines.KaplanMeierFitter` per tertile, run `multivariate_logrank_test` for
a trend p-value. If the subset doesn't have enough events per tertile for a
legible curve, extend it (more negatives/positives from the same sampling
frame, see `data/pid_lists/README.md`) rather than changing method. Output:
`figs/stage5_km_curve.png` with risk table, matching the mentor's plotting
style.

**Stage 6 (stretch, optional) — Benchmark against the mentor's approach.**
Reuse precomputed `derived/dino_organ_emb/{pid}_yr0.npz` embeddings for the
same student subset + a plain CoxNet (no deconfounding) to see how far the
simple CNN approach falls short of the full discovery pipeline — a graduation
point, not a required step.

## Critical files to reuse / reference (read-only)

- `discovery_pipeline/src/data.py` — pid/stem convention (`{pid}_yr0`), cohort
  join pattern (manifest ↔ prsn clinical ↔ scanner), label allowlist logic.
- `discovery_pipeline/src/km_analysis.py`, `km_per_organ.py` — canonical
  `lifelines` KM pattern (qcut → KaplanMeierFitter → logrank test).
- `code/prevascular_roi.py` — sternum/lung/heart-anchored ROI logic and its
  `qc_figure` visualization panel.
- `code/extract_organ_embeddings.py` — TotalSegmentator label→name map,
  corrected mask-and-zero crop approach (`_mask_only_crop`), HU normalization.
- `code/render_saliency_overlays.py` — HU display windows, heatmap-over-CT
  montage style.
- `experiments/lcrisk_discovery/data/manifest.csv` — reuse `time`/`event`
  columns as-is; don't re-derive the outcome.
- `discovery_pipeline/enriched_cohort_pids.txt` — sampling frame for Stage 0.

## Verification (once implementation starts)

- Stage 1–2: manual visual review of QC montages for the first 3–5 patients
  per organ (correct orientation, mask actually covers the intended organ).
- Stage 3: per-organ train/val loss curves should converge without
  diverging; sanity-check a few Grad-CAM overlays land inside the organ mask,
  not on background.
- Stage 4: ranking bar chart should show separation with visible confidence
  intervals, not all organs at chance-level AUC (if so, revisit subset size
  or organ choice before proceeding).
- Stage 5: KM curves should show visually separated tertile survival curves
  with a `multivariate_logrank_test` p-value reported (not necessarily
  significant given the small subset — report honestly either way).
