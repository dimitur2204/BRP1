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
PROGRESS (2026-10-18). Run folders regrouped 2026-10-20 (`docs/pipeline.md`, Pipeline):
`d2_s` / `d2_m` are now `trial_2h/small` / `trial_2h/big`.

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
- **D4 features** code ✅ (2026-10-19) `src/s9_dino_features.py` + `submit_s9_dino_features.sh`:
  teacher backbone rebuilt from the run's launch.json, whole 144×128×128 yr0 volume, [CLS, mean patch
  token] for all 16,345 cohort patients → `data/features/{run}/{ckpt}.npy`; `random` = same backbone,
  random weights (control). CPU smoke ✅. ✅ run on d2 checkpoints (jobs 5120546–8, 17–34 min).
- **D5 linear probe** code ✅ `src/s10_dino_probe.py`: z-score (train) → elastic-net Cox (l1 0.5,
  50 penalties) on train → penalty by val C → `runs/probe_*/pred.csv` (S4 format, S5 unchanged);
  `--summary` = val C of every probe + `cosine` val C 0.688. No test metric. Synthetic check ✅
  (age + lung volume + noise: val 0.581 vs 0.61 for a Cox on the 2 true columns; pure noise 0.511):
  the probe is conservative; checkpoints and the random control pay the same price.
  ✅ **d2 probes (val C, 2026-10-20):** d2_s random 0.636 → ep 9/19/29/39 0.640/0.648/0.648/0.658;
  d2_m random 0.649 → 0.636/0.616/0.633/0.645. Refs: clinical-5 0.636, cosine 0.688. ΔC vs random
  d2_s +0.022 [−0.008, +0.056], d2_m −0.004 [−0.037, +0.028]; d2_s ep 39 vs cosine −0.030
  [−0.067, +0.004]. Random features ≈ lung size/density (ρ ±0.5). Not yet better than random.
  ✅ decided (user, 2026-10-20): first the authors' recipe at their run length on the small model,
  default vs anti-collapse, then the winner on the big model.
- **D2b small model, authors' length** ⬜ **user: 2 jobs** (`docs/pipeline.md` §8):
  `runs/dino/small_150ep/{default,stable}`, 34,719 scans, batch 8 × 15 = 120 per step (authors'),
  150 epochs × 5,000 scans = 750k samples, ~15 h each on an L40S. Config dry-run ✅. Then S9
  (random + checkpoints at epochs 29/59/89/119/149) and S10 as for the 2 h trial.
- **D6 fine-tune** code ✅ (2026-10-26) `src/s12_dino_finetune.py` + `submit_s12_dino_finetune.sh`
  (`docs/pipeline.md` §10). Decided (user): `stable` epoch-39 teacher; 30 epochs; scratch run
  without layer decay. Runs: `finetune/dino_ep39` vs `finetune/scratch` (same recipe, random
  weights), reference `cosine`. CPU smoke ✅ (64³ crop). Timing job 6287322: out of GPU memory at batch 32 → gradient
  checkpointing on by default. Cluster folder is now `plamen_pipeline` (PROGRESS 2026-10-26).
  ✅ `finetune/dino_ep39` (2026-10-26): val C 0.691 (cosine 0.687, probe 0.658); test η 0.654,
  clinical + η 0.681, ΔC +0.027 [−0.009, +0.063] (cosine +0.021) → ≈ CNN, not significant.
  ✅ `finetune/scratch` (2026-10-27): did not learn (train C 0.53, val C 0.583, η sd 0.03; test η 0.571, ΔC −0.001) →
  DINO init is what makes the ViT work. Goal (user): general lung features, no nodule detection;
  judge by ΔC over clinical (PROGRESS 2026-10-26).

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

# Next: thymus hypothesis (2026-10-20)

Idea (user): if SSL "learns the anatomy" (as in AIM-Harvard's thymus system), pretrain on
thymic-bed crops instead of lungs. Thymus is not in TotalSegmentator → their thymic-bed segmenter
(`vprudente/open_thymus_segmentator`, weights Zenodo 10.5281/zenodo.18355851). Expected lung-cancer
signal is small (E5: TH HR 0.91/SD adjusted, non-adeno only), so success is judged on (1) SSL
features predicting the published score, (2) HR/SD significance, (3) all-cause death, not on a
lung-cancer ΔC CI > 0.

- **T0 published score on our cohort** ✅ (2026-10-20) `src/s11_thymus_score.py` → `runs/thymus_score/`.
  HR/SD 0.878 [0.817, 0.943] + clinical (train + val, 864 events), non-adeno only (0.853 vs adeno 0.950).
  Val C alone 0.548; ΔC over clinical +0.004 [−0.006, +0.013]; over clinical + η 0.000 (LR p 0.57).
  → A thymus model cannot raise lung-cancer C here; no death data in public prsn.
  ⬜ **user decision:** stop, or continue T1–T3 for the anatomy question only (probe = published score).
- **T1 segmenter env + array** ⬜ (after T0)
- **T2 thymic-bed crop cache** (50×50×75 mm, 1 mm, soft-tissue window, unmasked) ⬜
- **T3 SSL on crops** (DINO with smaller views, and/or their SwAV recipe) ⬜ → S9/S10 + score probe

# Next: clinical covariates in the training loss, branch `cnn-clinical-cox`

Branch `cnn-clinical-cox` (from `ssl-dino`), 2026-10-27. Design: `docs/pipeline.md` §11. Idea from
2026-10-26 (PROGRESS): put the clinical covariates into the training Cox model so the CNN learns
only the risk they do not explain (goal: general lung features with ΔC over clinical > 0).

- **C1 code** ✅ (2026-10-27) `s4_train.py --clinical age,sex,cigsmok,lung_ml,lung_mean_hu
  --clinical-mode {offset,joint}` (`model.ClinicalCox`), otherwise the `cosine` recipe. Input: the
  2.5 mm `lungs_u8.npy` cache (as every run so far, not the new 1.4 mm cache). CPU smoke: PROGRESS.
- **C2 runs** ⬜ **user:** `runs/clinical_cox/offset` and `runs/clinical_cox/joint` (commands in
  §11), then S5 on both. Compare test ΔC with `cosine` (+0.021) and `finetune/dino_ep39` (+0.027).
- ⬜ Same idea for the DINO ViT fine-tune (S12), if C2 helps.
