# Lungs 3D CNN on NLST: baseline from scratch

Branch `cnn3d-lungs-cox` (from `cnn3d-lungs-sternum` 7382553), 2026-10-05. All previous code, cohorts,
caches and results were deleted (they are in git history up to `7382553`).
Design and rationale: `docs/pipeline.md`.

## Goal

A working, simple 3D CNN that reads the lungs on the NLST baseline CT and
predicts time to lung cancer (Cox loss). It is a base to build on, so there is
no hyperparameter grid.

## Decisions (user, 2026-10-05)

- Input: lungs only, from TotalSegmentator masks. This limits the cohort to
  patients with masks (all cancers, ~63% of controls).
- Outcome: time-to-event, Cox partial likelihood.
- New cohort and a fresh random split. Nothing from the old pipeline is reused.
- `learn/` (teaching workspace, git-excluded) is kept.

## Stages

- **S1 candidates** ✅ `src/s1_candidates.py` → 16,543 pts / 1,047 events.
- **S2 preprocess** ✅ array run (job 1681218): 16,515 ok, 28 empty lung masks. ✅ merged.
- **S3 cohort** ✅ 16,345 pts / 1,034 events; train 11,442/724, val 2,451/155, test 2,452/155.
- **S4/S5 run 1 `baseline`** ✅ (job 1683009): constant lr 3e-4, early stop at epoch 17,
  best epoch 5. Test C 0.614 [0.570, 0.660], below age alone (0.630). Underfits.
- **S4/S5 run 2 `cosine`** ✅ (job 1688465): peak lr 1e-3, warm-up 3 epochs, cosine decay,
  80 epochs. Best epoch 57. Test C 0.648 [0.604, 0.689], above age alone (0.630).
- **S5 sex/smoking adjustment** ✅ run on `cosine` (2026-10-06): HR/SD 1.64 [1.41, 1.91] →
  1.59 [1.37, 1.86] adjusted. ✅ `--adjust sex,cigsmok,age,lung_ml,lung_mean_hu` (2026-10-17): HR/SD
  1.64 → 1.37 [1.15, 1.62], p 0.0004. CNN adds risk info beyond lung size/density/age (age ρ 0.23 with η). ⬜ Time-dependent AUCs have no CI yet (bootstrap them like C; not built).
- **S4 run 3: seeds + training variants** (code 2026-10-17): new `s4_train.py` flags `--min-lr`,
  `--ema`, `--select last`, `--events-per-batch` (all default to the old behaviour).
  ✅ 40-epoch screen on `gpu-short` (2 epb4 runs cut at epoch 37 by the time limit; compared
  at epoch 37): ema 0.674 ± 0.006 vs epb4 0.695 ± 0.037 val C. Leaning epb4, not proven; EMA ≈ no
  effect. ⬜ **user decision:** final recipe + seeds. ⬜ Train it at 80 epochs (train split), S5.
  That is the baseline SSL must beat.
- **S4 final runs** `epb4_s0–2` (80 epochs, epb4 + EMA + last): s0 ✅ **overfits** (full-train C
  0.86 vs val 0.66; val loss worse than constant by epoch 80). s1/s2 ✅ confirm it (full-train
  0.97/0.90 vs val 0.70/0.67).
  → Recommended: keep `cosine` as the baseline; ⬜ optional `cosine_s1/s2` for seed spread.
- **S1b mask-selection check** ✅ (2026-10-17): kept vs dropped controls balanced on every
  clinical/scanner variable (|SMD| < 0.05, CV AUC 0.50–0.51); only processing order (old split,
  pid) differs. IPW weights 0.95–1.09 → not needed. Masked controls ≈ a random subset.
- **S6 saliency** ✅ on `cosine` (2026-10-17): saliency sits in the posterior lung bases in
  everyone (RLL 2.3×, LLL 1.6× their volume share; upper lobes ~0.1×), events ≈ controls; no
  cancer-lobe localisation (p 0.24 / 0.47, hit 0.24 vs 0.21 chance). A lung size/inflation score,
  not a nodule detector. ✅ Added discrimination (S5, 2026-10-17): test C clinical 0.654 → clinical + η 0.675,
  ΔC +0.021 [−0.017, +0.058] (not significant, 155 events); η alone 0.648. SSL target: ΔC CI > 0.
- **Supervisor meeting** (prep 2026-10-16): results + limitations + questions in PROGRESS.

## Open items

- Discussed, not built: select the epoch by val loss; a reference Cox model of
  lung volume + age.

# Next: self-supervised pretraining (DINOv2-3D-Med), branch `ssl-dino`

Branch `ssl-dino` (from `cnn3d-lungs-cox` cd6940c), 2026-10-18. Background and findings in
PROGRESS (2026-10-18).

## Goal

Pretrain a 3D ViT with DINOv2 ([AIM-Harvard/DINOv2-3D-Med](https://github.com/AIM-Harvard/DINOv2-3D-Med),
no released weights) on NLST lungs **without labels and without test patients**, then predict time to
cancer from its features. Bar to clear (S5, test, 155 events): `cosine` η alone C 0.648; clinical
0.654 → clinical + η 0.675, ΔC +0.021 [−0.017, +0.058]. Success = clinical + DINO with a ΔC CI > 0,
or clearly above `cosine`.

## Decisions (user, 2026-10-18)

- DINOv2-3D-Med trained by us, not CT-FM (CT-FM's pretraining set is 128,037/~148k NLST scans, so
  it has seen our test scans).
- Input: the existing S2 cache (2.5 mm, 144×128×128, lung-masked uint8). Same input as `cosine`, so
  the comparison isolates "self-supervised vs supervised". At 2.5 mm the whole lung is one forward
  pass (patch 8³ → 18×16×16 = 4,608 tokens), so one embedding per patient, no tiling/pooling.
  Finer-scale (nodule-level) crops are a later variant, only if this works.

## Stages (proposed; scripts follow the `s{N}_*.py` / `submit_s{N}_*.sh` style)

- **D0 setup** ✅ (2026-10-19) `bash src/setup_dino_env.sh` (login node): uv 0.8.22, repo at
  `77bde4d` in `ext/`, `env/.venv-dino` (Python 3.11.13, torch 2.7.1+cu126, 4.9 GB). Repo tests 24/25
  (1 known upstream test/code mismatch, cosmetic). PRIMUS-M builds on CPU: 146.4 M parameters;
  96³ → 1,729 tokens, whole 144×128×128 lung → 4,609 tokens (incl. CLS).
- **D1 data** ✅ code (2026-10-19). `src/s7_ssl_cache.py` + `submit_s7_ssl_cache.sh`: yr1/yr2 of
  train + val pids through the unchanged S2 `preprocess` (yr0 output re-checked bit-identical),
  same S3 image QC → `data/ssl_scans.csv`. `--list` ✅ 20,959 scans to do.
  ✅ run (job 5098861, 8–12 min/task) + merge: 20,938/20,959 ok (21 empty lung masks); QC drops
  98 short z, 56 small, 15 clipped → `data/ssl_scans.csv` **34,719 scans** of 13,893 pids
  (train 11,442 / 8,634 / 8,563, val 2,451 / 1,799 / 1,830 for yr0 / yr1 / yr2).
  `src/dino_data.py` (`LungViews`): random 128³ cube → 2 global 96³ (2.0–3.3 mm) + 4 local 24³
  (2.0–4.0 mm) views with the repo augmentations; views < 30% lung are redrawn. QC ✅
  `figs/d1_views.png` (64 yr0 scans): view lung fraction median 0.42 global / 0.63 local, ~50% of
  views needed a redraw, 1.2 s CPU per scan. `configs/dino/nlst_2p5mm.yaml`, `src/dino_callbacks.py`.
- **D2 smoke + throughput** ✅ run (jobs 5098862/3, gpu-short L40S, 2 h, epoch 40/100, yr0 only):
  `d2_m` PRIMUS-M 146 M, batch 4: 0.57 s/batch (~25k scans/h), 3% data wait, 33.7 GB.
  `d2_s` 384-dim/12 layers 22 M, batch 8: 0.63 s/batch (~45k scans/h), 25% data wait, 36.8 GB.
  **Collapse warning** in both: DINO-head outputs nearly identical across scans (Gram off-diag
  0.97–0.99), effective rank 1.5–3.4 (of max 8/16 samples), teacher entropy falling to 0.15–0.2.
  Backbone CLS less so (off-diag 0.1–0.5). Monitors use only one batch (8–16 samples): noisy.
  ⬜ Decide with D4/D5 on the d2 checkpoints vs a random-init backbone (PROGRESS 2026-10-19).
- **D3 pretrain** ⬜ the same script on `gpu` (12 h jobs), chained with
  `--dependency=afterany` and resumed from `last.ckpt`. Keep checkpoints at intervals so feature
  quality can be probed against pretraining length. Stop early if D5 on an intermediate checkpoint
  is no better than random-init features.
- **D4 features** ⬜ `src/s9_dino_features.py`: teacher backbone, full 144×128×128 volume at
  2.5 mm, for **all** cohort patients → CLS token + mean patch token per patient, saved in
  cohort-row order (`data/features/{run}/{ckpt}.npy`). Also a random-init backbone (control).
- **D5 linear probe** ⬜ `src/s10_dino_probe.py`: penalised Cox on the features (fit on train,
  penalty chosen by val C), writes `runs/{name}/pred.csv` in the S4 format, so
  `s5_evaluate.py` runs unchanged (incl. `--added-only`). Compare on the same split: random-init
  features, DINO checkpoints, `cosine`, clinical. Test is read only by S5.
- **D6 fine-tune (optional)** ⬜ the backbone + a linear Cox head trained with `cox_loss`
  (S4-style, low lr, layer decay), vs the same ViT trained from scratch.

## Open items (D0)

Checked on GenomeDK 2026-10-18:
- No Python ≥ 3.11, no `uv`, no pip for the system python3 (3.9.25) → D0 installs the uv release
  binary; uv downloads its own Python.
- Later screening rounds, cohort pids with CT + `seg.nii.gz` (2026-10-19): train yr1 8,687 / yr2
  8,613, val 1,819 / 1,840 (test 1,877 / 1,894, never used). Pretraining pool train + val: 13,893
  (yr0 only) → **34,852** scans with yr1/yr2. ⬜ **user decision:** yr0 only first, or add yr1/yr2
  (needs S2 on ~21k more scans into a separate cache; S2 ran at ~1 s/scan, so a 40-task array is well under an hour plus queueing, and ~50 GB of disk).
- `gpu-h200` (2 nodes × 4 H200, max 7 days) has AllowAccounts=ALL, so it should accept
  `aura_thymus`; confirm with the D2 job. `gpu` / `gpu-l40s` also allow 7 days (S4 used 12 h by choice).
- No W&B account (user) → `WANDB_MODE=disabled`, Lightning `CSVLogger`; check the repo's collapse
  callbacks log to it.
- ⬜ GPU budget for D3.
