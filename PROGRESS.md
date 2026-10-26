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

## 2026-10-26 (cluster): S6/S7 smoke tests

- Branch `ssl-foundation-probe` checked out. The staged DINOv2-3D-Med work
  (s7_ssl_cache … s12, `data/features/`, `env/.venv-dino`, `data/cache/rounds_*`)
  is **not used** (user: don't reuse old caches). S6 reads only
  `data/cache/lungs_u8.npy` (S2 merge 2026-10-05). Checked: (16,543, 144, 128, 128) uint8,
  and all 16,345 cohort rows point to the right pid in `lungs_index.csv`.
- Env: `env/.venv` (torch 2.8.0+cu128) + `pip install omegaconf` (adds PyYAML,
  antlr4; nothing upgraded). fvcore/iopath are not needed; xformers is
  optional (falls back to plain attention).
- S6: `--device` flag added. Weights are now loaded with **strict** key matching
  (3DINO's loader is `strict=False` and would silently run on random weights).
  CPU smoke, 4 patients: loads strictly, ~5 s/patient on CPU, 3.1 GB RSS. Features finite;
  between-patient cosine of [CLS] 0.69–0.79 (not collapsed). Tokens
  (4, 576, 1024) float16. Masked input: 79 % of voxels at −1 (outside-lung air).
  Lung-token counts 93–133 of 576.
- S7: the first version (ridge, L-BFGS) **failed the synthetic check**: age
  hidden in 1,023 noise columns → val C 0.50 at every l2 up to 1e5 (age alone
  0.585), train C 0.81 on pure noise. Replaced by an elastic-net Cox path
  (sksurv): noise → val C 0.506; noise + age → 0.585 with 1 non-zero;
  clinical-only runs; 4,099 features fit in ~1 min.

## 2026-10-26: S6 run (masked input)

- Job 6295025 failed at start: `--save-tokens` opened its file before
  `data/embeddings/` existed (directory was only created at the end). Fixed
  (mkdir at start), re-smoked on CPU into a new directory.
- **Job 6295033** ✅ gpu-short, L40S, 8 min 50 s wall (inference 471 s, ~29 ms/patient).
  MaxRSS 33.6 GB against `--mem 32g` (memmap pages of the cache count), so the
  limit was raised to 48g.
- Verified `data/embeddings/3dino.npz`: pid and row order identical to `cohort.csv`
  (16,345). `cls` / `patch` / `last4` are finite, with 0 constant dims and all rows unique.
  Between-patient cosine median 0.75 / 0.94 / 0.78 (1st–99th pct 0.58–0.84 for
  cls). PCs for 90 % variance (2,000 pts): 246 / 132 / 265, so not collapsed.
  The first 4 patients match the CPU fp32 smoke at cosine ≥ 0.9998 (bf16 on GPU).
  `3dino.tokens.npy` (16345, 576, 1024) float16: 26 sampled patients, incl.
  first/last, all finite, none all-zero.

## 2026-10-26: S7 probes on masked 3DINO features (val only)

Elastic-net Cox (l1 0.5), penalty by val C. Val split 2,451 / 155 events.

| run | val C | non-zero / features |
|---|---|---|
| probe_3dino_cls | 0.7257 | 120 / 1024 |
| probe_3dino_patch | 0.7191 | 78 / 1024 |
| **probe_3dino_cls_patch** | **0.7291** | 143 / 2048 |
| probe_3dino_last4 | 0.7232 | 95 / 4096 |
| probe_clinical (age, sex, cigsmok) | 0.6206 | 2 / 3 |
| ref: `cosine` CNN | 0.688 | |

All four 3DINO sets beat the CNN on val by 0.03–0.04, and are within 0.01 of each other
(noise at 155 events). Val-chosen set = `cls_patch` (rule fixed before running). The val
numbers are optimistic (penalty chosen on val). ⬜ user: `probe_3dino_cls_patch_clin`, then
S5 on cls_patch / cls_patch_clin / clinical.

## 2026-10-26: S5 test results, frozen 3DINO (masked input)

All three runs completed (the NaN in the printout is `km_logrank_p` on the val row,
which S5 fills only for test, by design).

| run | val C | **test C [95% CI]** | test C after 1 y | test AUC 1/2/3/5 y |
|---|---|---|---|---|
| probe_3dino_cls_patch | 0.729 | **0.647 [0.609, 0.688]** | 0.653 | 0.63/0.60/0.63/0.64 |
| probe_3dino_cls_patch_clin | 0.735 | **0.675 [0.638, 0.713]** | 0.679 | 0.66/0.64/0.67/0.67 |
| probe_clinical | 0.621 | **0.653 [0.612, 0.693]** | 0.653 | 0.65/0.66/0.66/0.65 |
| ref `cosine` CNN | 0.688 | 0.648 [0.604, 0.689] | 0.653 | 0.63/0.63/0.62/0.65 |
| ref clinical + `cosine` η | | 0.675 | | |

- Frozen 3DINO alone = the supervised CNN on test (0.647 vs 0.648), and both are slightly
  below clinical alone (0.653). 3DINO + clinical = clinical + CNN (0.675).
- The val→test drop (0.729 → 0.647) is not mostly selection: the CNN dropped
  0.688 → 0.648, while clinical *rose* 0.621 → 0.653. So the val split is easier for
  image scores and harder for clinical ones (±0.04 CIs at 155 events each).
  Selecting among 4 feature sets × 50 penalties adds a little on top.
- Val Spearman: 3DINO η vs CNN η 0.66; vs age 0.25, lung volume 0.26,
  lung mean HU −0.19, slice thickness 0.04. The two image scores mostly carry the same
  signal (as in S6 saliency: lung size/inflation), not scanner artefacts.
- Not computed: paired ΔC (3DINO + clinical vs clinical) with a CI. This branch's S5
  has no paired comparison, and test predictions are only read by S5.

