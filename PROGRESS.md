# PROGRESS: lungs 3D CNN on NLST (restart)

Running log. Plan/status: `PLAN.md`. Design: `docs/pipeline.md`.

## 2026-10-05: restart from scratch

**Deleted (user request).**
- Tracked: all `src/`, `data/pid_lists/` (enriched cohort + legacy subset),
  `data/legacy_400/`, every c2–c4 result CSV/JSON, `figs/`, and
  `docs/cnn3d_explained.md`.
- Gitignored: `data/cnn_cache/` (38 GB), `data/organ_crops/` (8 GB),
  `data/saliency/`, `runs/`, `models/*.pt`, `logs/*`.
- Kept: `env/`, `data/external/` (the public IDC-780 clinical package, which is
  the outcome source), `learn/` (user choice), and `experiments/` (untracked
  thymus-heart branch code, not part of this project).

**User decisions.** Lungs only via TotalSegmentator; Cox loss; new cohort and split.

**Facts found.**
- `prsn` has 53,452 participants. 26,254 have a baseline series in the IDC
  scan list, of which 25,861 have the resampled CT on disk.
- TotalSegmentator masks exist for 16,618 of those. Every cancer case has one,
  only ~63% of controls do. The event rate in the candidates is therefore
  6.3%, not ~4%.
- Our outcome rebuilt from `prsn` equals the scan list's time/event exactly
  (time from randomisation). We measure time from the T0 scan instead
  (`− scr_days0`, missing for 145 → 0). No event has time ≤ 0. 124 controls
  have no follow-up after T0 and are excluded.
- 27% of cancers are diagnosed within 1 year of T0.
- Lung extent in a 30-patient sample: max 353×286×308 mm, median
  292×204×264 mm. 3/30 masks touch the first/last slice, and 2 were truncated
  scans (158 mm, 48 mm).

**S1 run:** 16,543 candidates / 1,047 events (`data/candidates_attrition.csv`).

**Smoke tests (login node, CPU).**
- `cox_loss` equals a brute-force Breslow sum with ties (5 random trials,
  ≤ 1e-6). Minimising it gives β 0.670415 vs sksurv Breslow 0.670415.
- S2: 1.1–1.7 s/pid. The 40-pid chunked → merged array is identical to the
  direct run. The QC figure looks right: lungs centred, 5 mm dilated rim, 0
  outside. A large mask hole in event 217121 was checked on the native CT: it
  is hilar vessels, not a missed tumour.
- `augment`: an off-centre cube keeps its Z and its XY radius (33.0 → 33.0
  voxels) under a 15.5° rotation, so rotation is axial-only and unsheared.
  The identity transform reproduces the input (max diff 2e-5).
- S3 → S4 (`--smoke`, 2 epochs) → S5 run end to end. Re-running S4 gives
  identical predictions.
- Fixed during smoke testing: `harrell_c` returns NaN (not an exception) when
  there are no events or no comparable pairs; time-dependent AUC is NaN for
  horizons outside a split's follow-up.

**S2 array run (job 1681218):** all 40 tasks done with empty stderr; median 1.0 s/pid
(max 16 s). 16,515/16,543 ok. The 28 failures are all `ValueError: empty lung mask`
(TotalSegmentator found no lobes) and S3 excludes them. Chunks total 39.0 GB
apparent size; `du` shows ~0.4 GB because the filesystem compresses mostly-zero data.

**Next:** user: `s2_preprocess.py --merge` → `--qc` →
`s3_cohort.py` → `sbatch src/submit_s4_train.sh` → `s5_evaluate.py --run baseline`.

## 2026-10-26: reading → next direction, D1 coded

**Read:** DrasCLR (anatomy-aware contrastive SSL, lung CT), 3DINO (3D DINOv2,
public ViT-L weights), MAE vs JEPA for 3D brain MRI (objective must match the
signal; frozen probes good for weak signals). Summary + plan:
`docs/directions.md`.

**Facts found.**
- 3DINO weights: `huggingface.co/AICONSlab/3DINO-ViT`, file
  `3dino_vit_weights.pth`, gated, CC BY-NC-ND 4.0. Inference = percentile
  scaling to [−1, 1]. `get_intermediate_layers(..., reshape=True)` gives
  patch maps. Position encoding is interpolated for non-cubic inputs, so
  144×128×128 → 9×8×8 tokens works without resizing. Repo pins torch 2.0 /
  Python 3.9 / xformers 0.0.18.
- 3DINO's pretraining set includes 500 random NLST patients (images only).

**Built (untested, no data/GPU on this machine):** `src/s6_embed_3dino.py`,
`src/submit_s6_embed.sh`, `src/s7_probe.py` (linear Cox via `cox_loss` +
L2, L-BFGS, l2 by val C, writes the S4 run layout so S5 is unchanged).
`docs/pipeline.md` §8. `.gitignore`: `data/embeddings/`, `external/`,
`models/*.pth`.

**Next:** user: licence check → clone + weights → S6 smoke (`--limit 8`) →
full S6 → S7 for 4 feature sets + clinical → S5 on each.

