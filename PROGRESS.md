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

**S3 (user ran):** 16,345 pts / 1,034 events. Lung z-extent < 200 mm removed 165 pts
(12 events). Volume and clipping removed 5 more. Splits: train 11,442/724, val
2,451/155, test 2,452/155, each with a 6.3% event rate.

**Run 1 `baseline` (job 1683009, L40S):** constant lr 3e-4, patience 12.
- Loading took 40 s. Each epoch took ~59 s, with ~45/357 batches skipped (no events).
- Early stop at epoch 17, best epoch 5 (val C 0.674). The best val loss came at
  epoch 14 (7.580 vs 7.718 for η = 0).
- **Underfitting:** train loss 3.407 → 3.36. Train C (eval mode) is 0.52–0.62,
  *below* val C. Val C jumps ±0.04 per epoch.
- Test (S5): C 0.614 [0.570, 0.660], C after 1 y 0.623, AUC 1/3/5 y 0.59/0.61/0.62.
  Age alone gets C 0.630 on test, so the CNN does not beat age yet.
- KM tertiles: 30/51/74 events, log-rank p 8e-5.
- Val 0.674 → test 0.614 is mostly the optimism of picking the max of 17 noisy val Cs.
- **What it learned:** η has Spearman ρ 0.65 with lung volume, −0.45 with mean
  lung HU and 0.13 with age (same on train and val). It is a "large,
  low-density lungs" score, not a nodule detector.

**Run 2 recipe (code, 2026-10-05):** `s4_train.py` gains `--schedule cosine` (new default: peak
lr 1e-3, linear warm-up over 3 epochs, cosine to 0, set per batch; skipped batches also
advance the schedule so it ends on time), default 80 epochs, `--patience 0` (= no early
stopping). Per-epoch `lr` is logged, and `curves.png` has a third panel for it. A finished
run directory is no longer overwritten without `--overwrite`. Checks: the schedule reaches
1e-3 at epoch 3, 5.3e-4 at epoch 40 and ~0 at the last step. A CPU smoke run (3 epochs)
works, and the overwrite guard works. The old recipe is
`--schedule constant --lr 3e-4 --epochs 60 --patience 12`.

**Run 2 `cosine` (job 1688465, user ran S4 + S5):**
- All 80 epochs ran, at ~59 s/epoch. Best val C was at epoch 57 (0.688). The
  final epochs were flat (val C 0.687).
- **No longer underfitting:** train loss fell 3.41 → 3.17. Train C (eval) rose
  to 0.71, above val C. Val loss reached 7.47 vs 7.718 for η = 0 (excess −0.25,
  vs −0.14 for baseline). Val η SD grew 0.14 → 0.73.
- Test: C 0.648 [0.604, 0.689] (baseline 0.614), C after 1 y 0.653,
  AUC 1/2/3/5 y 0.63/0.63/0.62/0.65. Age alone gets C 0.630.

**S5 adjustment for sex + smoking (2026-10-06, user request):** `s5_evaluate.py` gains
`--adjust` (default `sex,cigsmok`) and writes three outputs:
- `km_test_strata.png`: KM by η tertile in each sex × smoking panel;
- `km_test_adjusted.png`: unadjusted KM next to curves from a Cox model
  `tertile + covariates` on test, standardised to the test population, with
  200× bootstrap bands;
- `adjusted_hr.csv`: HR per val-SD of η and mid/high vs low tertile, unadjusted
  vs adjusted.

Test-split cells: F/former 503 pts/23 events, F/current 534/39, M/former
745/37, M/current 670/56.

Check: a synthetic score made of 0.8·male + 0.8·current smoker + noise, run in
scratch, gives HR/SD 1.13 → 0.91 adjusted, with smoking HR ≈ 1.9 in both
models. The figures render correctly. Runtime is ~1.5 min at 20/30 bootstraps;
the defaults (1,000 C bootstraps + 200 curve bootstraps) take longer.

**S5 adjusted run (user, 2026-10-06):** `runs/cosine/eval/adjusted_hr.csv`: HR/SD 1.64
[1.41, 1.91] unadjusted → 1.59 [1.37, 1.86] adjusted for sex + cigsmok. High vs low third
3.59 → 3.34. Within every sex × smoking stratum the high third had 2–6× the low third's
diagnosed share. Covariate rows (male 1.16, cigsmok 1.40) are not interpreted (Table 2
fallacy). Teaching course on this: `learn-adjustment/` (git-excluded).

**Gaps found 2026-10-06/16:**
- Age is not adjusted for. It is a confounder (val: ρ 0.23 with η; diagnosed 4.6% at
  55–59 vs 8.9% at 65–69).
- Time-dependent AUCs are point estimates only, with no CI. Only C has a bootstrap
  percentile CI (1,000 resamples of the test patients, no refit).
- Clinical vs clinical + CNN (added predictive value) has not been run.
- The s4 cosine schedule and s5 adjustment code are still uncommitted.

**Next:** user: `s5_evaluate.py --run cosine --adjust sex,cigsmok,age` → supervisor meeting.

## 2026-10-17: can the cosine recipe be improved before SSL?

**Cluster state.** Code on `/faststorage/project/aura_thymus/student_pipeline` was identical
(md5) to this copy; only `baseline` and `cosine` runs exist (run by dimitarnizamov); no jobs queued.
Access from Claude: ssh ControlMaster host `genomedk` in `~/.ssh/config` (user logs in once).

**Analysis of `cosine` history.**
- The low-lr tail is harmless and nearly useless: val C ≥ 0.68 from epoch ~45 (lr ≈ 4e-4);
  in the last 20 epochs (lr < 1e-4) val C stayed 0.681–0.687, train C 0.70–0.71. A floor would
  at most save GPU time. The decay to 0 makes the last epoch ≈ best (0.687 vs 0.688).
- Val C 95% CI is ±0.045 (0.645–0.732): one run cannot resolve config changes of ~0.01.
  Seed spread is unknown → seeds needed before any tuning claim.
- Batch composition: P(0 events | 32 pts, 6.3%) ≈ 12.5% (matches ~45/357 skipped), P(1) ≈ 27%:
  ~40% of slots give ≤ 1 case → noisy Cox gradient. Likeliest real gain.
- Train C 0.707 vs val 0.687: little overfitting; the score is mostly lung volume/density, so
  the ceiling is input/architecture (2.5 mm, no nodule scale), which SSL is meant to address.

**Code (`s4_train.py`).** New flags, all defaulting to the old behaviour: `--min-lr` (cosine
floor), `--ema D` (EMA of weights; eval/select/save use it; `val_c_raw` logged), `--select
{val_c,last}`, `--events-per-batch N` (N events + rest non-events per batch, cycles reshuffled;
events seen ~2×/epoch at N=4). New `gpu_gb` column (peak GPU memory). curves.png shows raw vs EMA
val C and the selected epoch.

**Checks (local scratch venv, CPU, synthetic 40-patient cache).** Default sampler gives the
identical batches to the old loop for the same seed (11,442 pts). `--events-per-batch 4`: 357
batches, all exactly 4 events/32 pts, 9,996 distinct controls/epoch, 1.96 draws per event. lr
with floor 0 ends at 3.27e-12 (= `cosine` history); floor 1e-5 ends at 1e-5. EMA lerp moves by the
expected amount. Smoke runs (default; `--ema --select last --events-per-batch --min-lr`) finish,
write history/model/pred; `select last` saves the final epoch; bad flag values are rejected;
overwrite guard still works.

**Experiment, revised (user decision: 12 GPU-h of 80-epoch runs was too much).** A 40-epoch
screen: `e40_ema_s0–2` (`--epochs 40 --ema 0.999 --select last`) vs `e40_epb4_s0–2` (same +
`--events-per-batch 4`). The only difference is events per batch; EMA vs raw comes from
`val_c_raw`. 6 jobs × ~50 min ≈ 5 GPU-h. Why 40 epochs is acceptable: in `cosine` val C was
already ~0.68 at epoch 45 with lr still 4e-4, so it was not limited by training length; config
ranking should hold, absolute C may be slightly lower. Rejected: training on a subset of train
(only 724 events; it changes the per-batch event count being tested). Then the winner gets the
full 80 epochs on the train split. Test is untouched until then. Adding val to training would
need S5 changes (val sets tertile cuts and the SD scale), so not now.

**GenomeDK GPUs (checked 2026-10-17, https://genome.au.dk/docs/computing-with-gpus/).**
`gpu` = `gpu-l40s` = gn-1002, 7 L40S, all 7 allocated (another user, 3-day jobs). `gpu-short` =
gn-1001, 8 L40S, idle, 2 h max. `gpu-h200`: 2 × 4 H200, busy. GPU jobs averaging < 75% GPU
utilisation after 2 h are cancelled (check with `jobinfo <jobid>`). The screen is submitted to
`gpu-short` with `-p gpu-short -t 01:30:00` on the sbatch command line.
Files copied to the cluster (old copies backed up locally first; only my edited lines differed).

**40-epoch screen results (jobs 4528879–84, user ran, `gpu-short`, `-t 01:30:00`).**
- Timeouts: all 6 loaded data at once, so each load took **1,763 s** (29 min, vs 423 s for
  `cosine` alone). `e40_ema_s0/s1` finished all outputs ~2 s before the limit (state TIMEOUT but
  complete). `e40_epb4_s0/s1` were killed at epoch 37/40 without a model.pt (history saved). epb4
  epochs are ~67 s vs ~60 s (no skipped batches = 12% more steps). Peak GPU memory 6.9 GB at batch
  32 → batch 64+ fits easily on an L40S.
- Compared at epoch 37 (the last epoch all have). In complete runs val C changes by ≤ 0.0024 from
  epoch 37 to 40, so epoch 37 is a fair proxy.

  | run | val C ep37 | raw weights | val loss − null | train C |
  |---|---|---|---|---|
  | e40_ema_s0 | 0.669 | 0.666 | −0.187 | 0.690 |
  | e40_ema_s1 | 0.673 | 0.672 | −0.187 | 0.608 |
  | e40_ema_s2 | 0.680 | 0.682 | −0.217 | 0.672 |
  | e40_epb4_s0 | 0.664 | 0.662 | −0.168 | 0.722 |
  | e40_epb4_s1 | 0.736 | 0.739 | −0.428 | 0.729 |
  | e40_epb4_s2 | 0.685 | 0.679 | −0.268 | 0.731 |

  ema: 0.674 ± 0.006; epb4: 0.695 ± 0.037 (median 0.685).
- Reading: (1) epb4 fits train consistently better (train C 0.73 vs 0.66, all 3 seeds), but its
  val C is mixed: one seed much better (0.736), one worse, one slightly better. Without s1 the means
  are equal. Leaning epb4, not proven. (2) EMA does nothing measurable at the end of a cosine
  schedule (EMA − raw within ±0.006): the lr is ~0 there, so the weights barely move anyway.
  (3) The last 10 epochs are flat (val C SD 0.001–0.007), so `--select last` is safe. (4) 40
  epochs cost ~0.013 val C vs the 80-epoch `cosine` (0.687, one seed). (5) Seed spread of the
  current recipe is small (SD 0.006), but epb4's is large (0.037).
- Fix for next submissions: `-t 02:00:00` (gpu-short max) and stagger starts with `--begin` so
  the 27 GB loads don't overlap.

**Final runs:** user submitted `epb4_s0–2` (80 epochs, epb4 + EMA + select last) on `gpu-short`,
staggered by 15 min.

**User question (2026-10-17): optimise AUC instead, because the cohort is enriched?** Answer given:
C and AUC are pairwise ranking measures, so random control subsampling doesn't change them in
expectation (only PPV/calibration/net benefit depend on prevalence). The real risk is
non-random selection of which controls have masks. Changing the loss doesn't address that, and
Cox is already a ranking loss; time-dependent AUC is already reported. → wrote S1b.

## 2026-10-17: S1b mask-selection check + S6 saliency (code, not run on real data yet)

**S1b `src/s1b_mask_selection.py`.** Population = S1 just before the mask step. Controls with
vs without a mask: SMD per covariate (clinical / scanner / admin), 5-fold CV AUC of predicting
the mask, IPW weights (clinical + scanner model, trimmed 1–99%) in `data/mask_ipw.csv`.
Facts found while writing it: IDC prsn has no pack-years/BMI. `scr_res0` 4–6 = positive screen
(dictionary). The scan list's old split: 60% of its train controls have masks vs 69% of
val/test → masks were generated in an order. Synthetic check (3,000 pts, mask built to depend
on age + one kernel): flagged exactly those (SMD 0.42, 0.44; tube current 0.10 by chance); admin
AUC 0.49; IPW mean age of kept controls 64.71 vs all controls 64.72 (unweighted 65.91).

**S6 `src/s6_saliency.py` + `submit_s6_saliency.sh`.** Occlusion (40 mm cube, 20 mm stride,
median-lung fill so the lung volume is unchanged) + Grad-CAM at features[4] (18×16×16). All
val events + as many random val controls; figures for top/bottom-η events and top-η controls.
Lobe test: TotalSegmentator lobes resampled with S2's transform (box centre from
lungs_index.csv); rim → nearest lobe; cancer lobe from prsn loc* (1,779/2,058 cancers have
exactly one lobe). Enrichment of the cancer lobe vs the same lobe in controls / other-lobe
events (fair reference, since cancers and possibly the model favour upper lobes), split by
dx ≤ 1 y / > 1 y. Refuses `--split test`. Smoke: synthetic CT + lobe masks → real s2
`preprocess` → cache → s6 on CPU: orientation checks 100% pass, η check 6e-8, all outputs and
figures OK. The real orientation check will also confirm the TotalSegmentator label numbering
(10 LUL, 11 LLL, 12 RUL, 13 RML, 14 RLL), which was taken from memory.

**S1b run (user, 2026-10-17 20:38).** Kept controls (with mask) vs dropped (without), all with a
year-0 CT on disk; 16,543 kept (= S1 candidates), 1,047 events, all with masks.
- Clinical and scanner variables are balanced: every |SMD| < 0.05 (age 61.30 vs 61.37, male
  59.5 vs 58.1%, current smoker 47.4 vs 47.3%, positive T0 screen 26.1 vs 26.2%, follow-up 6.44 vs
  6.41 y, manufacturer/kernel/slice thickness/kVp/mA/diagnostic quality all within 0.05).
- Only the administrative variables differ: old split (train 66.9 vs 75.1%, SMD −0.18) and pid
  (SMD −0.12). Predicting "has mask": CV AUC 0.504 clinical, 0.510 scanner, 0.509 both, 0.566 admin.
- IPW weights for controls are 0.95–1.09 (ESS 15,485 of 15,496), so a weighted evaluation would
  change nothing. Not needed.
- Conclusion: on everything measured, the kept controls behave like a random ~63% sample. The
  masks follow the processing order (old split, pid), not patient or scanner traits. So C and
  time-dependent AUC on this cohort should transfer to the full NLST CT-arm population with a
  baseline CT. Absolute risk (6.3% vs ~4% events) still isn't calibrated. Can't rule out selection
  on unmeasured traits (pack-years, emphysema, nodules are not in the IDC prsn table), but scanner
  and screen-result balance make it unlikely.

**Why val C > train C in e40_ema_s1 (user question, 2026-10-17).** The plotted train C uses a fixed
2,000-patient train subset (drawn by the seed). From pred.csv: e40_ema_s1 subset C 0.609 vs
full-train C 0.655 (random 2k subsets 0.656 ± 0.024; this one is ~2 SD low), val 0.672 (within
val's ±0.02). Normal-shuffle runs: full-train C ≈ val C (s0 0.666/0.667, s2 0.676/0.681,
cosine 0.696/0.688) → almost no overfitting. **epb4 does overfit:** e40_epb4_s2 full-train 0.730
vs val 0.683 (events seen ~2×/epoch → memorised). Also: e40_ema_s1 collapsed to C ≈ 0.52 at
epoch 13 and recovered (instability at high lr). Candidates: gradient clipping; full-train C at
the end of s4.

**epb4_s0 (job 4618510, 80 epochs, epb4 + EMA + last; 1:48 wall, completed).** Strong overfitting:
train C (eval, subset) 0.65 → 0.87, full-train C 0.859 vs val 0.660 (selected = last). Val C
flat ~0.665–0.670 from epoch 16. Val loss − null: best −0.20 at epoch 32, +0.025 at epoch 80
(worse than a constant score); val η SD 0.5 → 1.39 (overconfident). Each train event is seen
~2×/epoch, ~160× in 80 epochs → memorised. **Decision (recommended to user):** drop
`--events-per-batch`; keep the `cosine` recipe (val 0.688, full-train 0.696, test 0.648) as the
baseline. Run S6 saliency on `cosine`. Optionally `cosine_s1/s2` for the seed spread.

**epb4_s1/s2 (jobs 4618511/12, completed 1:37).** Overfitting confirmed in all three seeds:
full-train C / val C (last) 0.859/0.660, 0.972/0.696, 0.904/0.665 (cosine 0.696/0.688); val loss −
null at the end +0.03, +0.43, +0.10 (worse than a constant score). Mean val C 0.674 vs cosine
0.688. → `--events-per-batch` dropped; `cosine` stays the baseline.

**S6 saliency on `cosine` (job 4622037, 19.9 min for 313 val patients).** η check 3e-5. Lobe
orientation checks: left_is_left 1.0, upper_is_up 0.987 → TotalSegmentator labels 10–14 =
LUL/LLL/RUL/RML/RLL confirmed.
- **The model looks at the posterior lung bases in everyone.** Occlusion enrichment (share of
  positive saliency / volume share), median over 313 patients: RLL 2.34, LLL 1.55, RML 0.19,
  RUL 0.12, LUL 0.10. Mean maps of events and controls are nearly identical: peak at the
  posterior (dependent) bases / costophrenic recesses just above the diaphragm, right > left.
  Grad-CAM agrees. Per-patient figures: highest-η cancers (all upper-lobe cancers) light up the
  posterior bases, not the tumour lobe.
- **No cancer-lobe localisation:** 141 events with a lobe. Occlusion cancer-lobe enrichment
  median 0.148 vs same lobe in controls 0.124 (p 0.24) / other-lobe events 0.124 (p 0.47); top-lobe
  hit rate 0.24 vs chance 0.21. Grad-CAM 0.97 vs 0.89 (p 0.73), hit rate 0.13 vs 0.21. Diagnosed ≤ 1 y
  (n 33, likely visible at T0): 0.103 vs 0.124 (p 0.10, wrong direction). Nothing points at tumours.
- η vs per-lobe enrichment (Spearman): RLL +0.28, RML −0.41, RUL −0.33: higher-risk patients put
  even more weight on the right base.
- Reading: consistent with η ~ lung volume (ρ 0.65) / low density (−0.45). How far the lungs reach
  into the posterior costophrenic recesses (inspiration depth, hyperinflation) and how dense the
  dependent bases are fit a "large, hyperinflated lungs" score, a smoking/COPD-type risk marker,
  not a nodule detector. Caveat: occlusion measures sensitivity, not mechanism, and median fill
  also changes density in the denser dependent bases.
- Implication for SSL: SSL at the same 2.5 mm whole-lung input may only learn this global signal
  better. Before SSL: a reference Cox model (lung_ml + lung_mean_hu + age) to see what the CNN adds
  beyond lung size/density.

**S5 adjusted for sex, cigsmok, age, lung_ml, lung_mean_hu (user, 2026-10-17).** Test split.
η per val-SD: HR 1.64 [1.41, 1.91] unadjusted → **1.37 [1.15, 1.62], p 0.0004** adjusted (sex +
cigsmok alone: 1.59). log-HR 0.495 → 0.313: ~37% shared with the covariates, ~63% remains. High vs
low tertile 3.59 → 2.33 [1.43, 3.79]. Covariates (context only, Table 2 fallacy): age 1.074/y,
cigsmok 1.47, lung_ml 1.0002/mL (~1.22/L, p 0.04), lung_mean_hu 1.0045/HU (p 0.11). Reading: the CNN
carries risk information beyond lung size and mean density. With the S6 saliency (lung bases),
likely candidates are basal shape/diaphragm position/inspiration depth or the pattern of density;
not tumours. Limits: linear adjustment (nonlinear volume/density effects possible); mean HU is a
crude emphysema proxy (LAA-950 not computed); association, not added discrimination. The
clinical vs clinical + η C comparison is still not built. Files: `runs/cosine/eval/` = this
5-covariate version; `runs/cosine/eval_sex_cigsmok/` = the 2026-10-06 sex+cigsmok version (user copy).

**S5 added value over clinical variables (2026-10-17, `--added-only --out eval_added`, run on the
login node; `runs/cosine/eval_added/`).** New in `s5_evaluate.py`: Cox models fitted on val
(train would overweight η), scored on test; paired bootstrap (1,000×) of test patients for ΔC.
Clinical = age, sex, cigsmok, lung_ml, lung_mean_hu (no NaN in cohort.csv).

| test | C [95% CI] | C after 1 y | AUC 1/2/3/5 y |
|---|---|---|---|
| clinical | 0.654 [0.612, 0.696] | 0.651 | 0.663 / 0.646 / 0.674 / 0.669 |
| η alone | 0.648 [0.608, 0.688] | 0.653 | 0.631 / 0.625 / 0.621 / 0.645 |
| clinical + η | 0.675 [0.636, 0.713] | 0.679 | 0.662 / 0.649 / 0.661 / 0.680 |
| **Δ (clinical+η − clinical)** | **+0.021 [−0.017, +0.058]**, p_boot(Δ≤0) 0.135 | +0.028 | −0.001 / +0.003 / −0.014 / +0.011 |

On val (fit split): η per SD HR 1.82 [1.53, 2.17] in the combined model, LR test p < 1e-4 (optimistic:
val chose the epoch). Reading: η alone ≈ the 5-variable clinical model; adding η gives +0.02 C on test
(consistent with the adjusted HR 1.37), but with 155 test events the CI includes 0, so the
added discrimination is not established on test. A significant HR with a non-significant ΔC is
common (ΔC is an insensitive measure). Target for SSL: clinical + η clearly above 0.654 with a ΔC CI
that excludes 0. Power is limited by 155 test events (detectable ΔC ~0.04). Checks (synthetic, local):
η with information not in the clinical variables → ΔC +0.035 [0.020, 0.050]; η built from the clinical
variables only → ΔC −0.003 [−0.006, +0.001], p 0.93. Also new: `--out` (no overwrite of eval/),
`--added-only`, `--clinical`, `--fit-split`.

## 2026-10-18: self-supervised pretraining, branch `ssl-dino`

New branch `ssl-dino` from `cnn3d-lungs-cox` (cd6940c). Plan: PLAN.md "Next: self-supervised
pretraining".

**CT-FM rejected as the main model: it was pretrained on NLST.** Pai et al., arXiv 2501.09001,
Suppl. Table S5: `nlst` = 128,037 of ~148,000 pretraining scans ("a majority of the dataset is
comprised of the NLST cohort"). The IDC selection query (Suppl. S1) filters only on CT, slice
thickness 1–5 mm, > 50 slices, no localiser; no collection is excluded. So CT-FM has seen our test
scans (no labels, but not unseen images).

**DINOv2-3D-Med (pinned `77bde4d`, 2026-08-04), read in full.** No released weights; README:
"has not been extensively benchmarked"; eight `configs/experiments/` files deal with collapse
(lower lr, faster centring, smaller head), so collapse happened to the authors too. Defaults:
PRIMUS-M (EVA ViT, 864-dim, depth 16, 12 heads, patch 8³, RoPE + learned abs. position embedding),
96³ global / 24³ local views, 4 local views, 4 GPUs × batch 5 × accumulation 6, 150 epochs of
250 steps, fp16, W&B logging, 1 mm isotropic NIfTI via MONAI. The backbone takes any input size
(abs. position embedding resampled, RoPE built per D/H/W), so a crop-trained model can embed the
whole 144×128×128 lung. `configs/predict.yaml` is stale (refers to modules that no longer exist);
`scripts/eval/extract_features_radchestct.py` (CLS token from the student backbone, whole volume
resized to 160³) is the usable template for D4. Requires Python ≥ 3.11 → separate venv.

**Decided (user):** train DINOv2-3D-Med ourselves on the 2.5 mm S2 cache, no test patients in
pretraining. Why 2.5 mm: the ViT does not know the spacing, only consistency matters; at 2.5 mm
the whole lung fits one pass, so there is one embedding per patient and no crop pooling. Cost: 4–6
mm nodules are ~2 voxels, the same view as `cosine` (which S6 showed learns lung size/inflation).

**Cluster checks (2026-10-18).** No Python ≥ 3.11 or `uv` on GenomeDK (python3 3.9.25).
`{pid}_yr1/yr2` CT + TotalSegmentator folders exist (pid 100002). `gpu-h200`: 2 nodes × 4 H200,
AllowAccounts=ALL, max 7 days. No W&B account → CSV logging.

**D0 setup ✅ (2026-10-19).** `src/setup_dino_env.sh`: system python3 has no pip → uv 0.8.22
release binary (sha256-checked) into `~/.local/bin`; repo cloned at `77bde4d` into `ext/`
(gitignored); `uv sync --frozen` → `env/.venv-dino` (Python 3.11.13, torch 2.7.1+cu126, 4.9 GB;
uv Python 41 MB under `env/uv-python`). Repo unit tests: 24/25 pass; `test_project_import_failure_is_actionable`
fails because `scripts/run.py` does not wrap the import error the test expects (upstream test ahead
of code; cosmetic). CPU check: PRIMUS-M (864-dim, depth 16, patch 8³) = 146.4 M parameters; a 96³
crop → 1,729 tokens, the whole 144×128×128 S2 lung → 4,609 tokens, so the whole-lung embedding
works as planned. 146 M parameters for 14–35k scans is large → compare a smaller ViT in D2.
Note: the cluster tree is a shared group folder (owner dimitarnizamov), not a git checkout;
files are copied with scp.

**Later screening rounds (2026-10-19)**, cohort pids, CT / CT + seg.nii.gz: train yr1 10,410 / 8,687,
yr2 10,015 / 8,613; val yr1 2,230 / 1,819, yr2 2,156 / 1,840; test yr1 2,253 / 1,877, yr2 2,187 / 1,894.
Train + val pool: 13,893 → 34,852 scans with masks.

## 2026-10-19: D1 data + S8 launcher (code, CPU-checked)

**S7** (`src/s7_ssl_cache.py`): `s2_preprocess.preprocess(pid, rnd="yr0")` gained a round argument
(default unchanged; 100002/100004 yr0 re-run = bit-identical to the S2 cache). Smoke: 4 yr1/yr2
scans ok, 1.4–1.9 s/scan. `--list` on the cluster: 20,959 scans (train yr1 8,687, yr2 8,613; val
1,819 / 1,840). Merge applies the S3 image QC and writes `data/ssl_scans.csv` (yr0 rows from
cohort.csv), asserting no test pid.

**D1 views** (`src/dino_data.py`). The repo's RandomResizedCrop3D crops side = scale × input side,
then resizes; its default (96³ input, global scale 0.5–1.0) would upsample to 1.25–2.5 mm. We crop
a random 128³ cube and use global 0.6–1.0 → 96³ (2.0–3.3 mm), local 0.15–0.30 → 24³ (2.0–4.0 mm).
Lung voxels are ~20% of the 144×128×128 grid (the mediastinum is 0), so views < 30% lung are
redrawn. QC on 64 yr0 scans: 195/384 views redrawn; final lung fraction median 0.42 (global) / 0.63
(local), none < 0.3; 1.22 s CPU per scan (6 views) → 12 data workers ≈ 10 scans/s. The figure looks
right (both lungs in global views, pleura/vessels in local views); Gaussian smoothing blurs some
views noticeably (repo default, kept).

**Repo quirks found by CPU runs of the full config** (tiny ViT, login node):
- `run.py` applies overrides after `parse()`, so `%`-macro copies ignore them: `hidden_size=192`
  left the backbone at 864-dim; `trainer#log_every_n_steps=4` left the callbacks at 50. S8 updates
  before parsing (verified: 1.4 M-parameter backbone, callbacks fire at step 4).
- `run.py` cannot resume → S8 resumes from `ckpt/last.ckpt` (verified) and refuses a relaunch whose
  config files/overrides differ from `launch.json` (verified).
- Warm-up is 10 epochs of the run (`configure_optimizers`), so `max_epochs` ≥ 11.
- Collapse callbacks only log to W&B → `src/dino_callbacks.py` subclasses log to the CSVLogger
  (verified: effective_rank, teacher_entropy(_normalized), gram offdiag mean/std, Gram PNGs).
- 3D RoPE: `embed_dim / num_heads` must be divisible by 6 (864/12 = 72 ok; 384/6 not; 384/4 ok).
- LearningRateMonitor writes one column per parameter group (layer decay) → wide metrics.csv.

**D2 jobs to submit (user)**, both throwaway (2 h limit, yr0-only scans until the S7 merge; do not
resume them after the merge, the scan list is not part of launch.json):
    sbatch src/submit_s8_dino_pretrain.sh --run-name d2_m trainer#max_epochs=100
    sbatch src/submit_s8_dino_pretrain.sh --run-name d2_s trainer#max_epochs=100 \
        hidden_size=384 lightning_module#backbone#eva_depth=12 lightning_module#backbone#eva_numheads=4 \
        data_module#batch_size=8
Read: `grep throughput logs/s8_*.out` (sec_per_batch, data_wait_frac, gpu_peak_gb), `jobinfo`
(GPU utilisation), `runs/dino/d2_*/metrics.csv` (losses, effective rank, entropy, Gram).

## 2026-10-19: S7 run + D2 results

**S7** (job 5098861, 40 × 8–12 min) + merge: 20,938/20,959 yr1/yr2 scans ok (21 `empty lung
mask`). Image QC: 98 lung z < 200 mm, 56 lung < 1,500 mL, 15 clipped > 2%. `data/ssl_scans.csv`:
34,719 scans of 13,893 pids (train yr0/yr1/yr2 11,442 / 8,634 / 8,563; val 2,451 / 1,799 / 1,830).
`rounds_u8.npy` 49.4 GB apparent. Median 0.97 s/scan.

**D2** (jobs 5098862 `d2_m`, 5098863 `d2_s`; gpu-short L40S 46 GB; both started before the merge,
so yr0 only; both hit the 2 h limit at epoch 40 of 100 = 1,667 optimiser steps, as planned;
checkpoints at epochs 9/19/29/39 kept in `runs/dino/d2_*/ckpt/`).

| | d2_m PRIMUS-M | d2_s 384-dim, depth 12, 4 heads |
|---|---|---|
| parameters | 146.4 M | 22.4 M |
| batch × accumulation | 4 × 6 | 8 × 6 |
| s / batch | 0.57 | 0.63 |
| scans / h | ~25k | ~45k |
| data wait | 3% (GPU-bound) | 25% (CPU-bound) |
| GPU peak | 33.7 GB | 36.8 GB |

Collapse monitors (every 150 steps; each computed on ONE batch: 8 (m) / 16 (s) global CLS tokens):
- DINO-head output (`student_cls_token`) Gram off-diagonal 0.97–0.99 whenever measured on it:
  nearly the same output for every scan. The callback then falls back to the backbone CLS (off-diag
  0.1–0.5, std 0.2–0.4), so the logged Gram numbers alternate between the two.
- Effective rank (on the head output): d2_m 5.7 → 1.5–2.6 (max 8); d2_s 12.1 → 1.4–3.4 (max 16).
- Teacher entropy (normalised) 0.87 → 0.2 (m) / 0.15 (s) by step 1,500–1,650.
- Losses fall (total 21.6 → ~7–10) but noisily; with a peaked teacher a falling DINO loss is
  not evidence of learning.
Reading: signs of (partial) collapse of the DINO head; the backbone CLS is not constant. Plausible
cause: lungs at 2.5 mm, centred and masked, look much alike across patients, so cross-view
matching is easy to satisfy with little information. The authors' `configs/experiments/` collapse
recipes (lower lr, faster centring, smaller head, tighter clipping) exist for this. The monitors
are too noisy (8–16 samples) to settle it. Decisive test: features of the d2 checkpoints vs a
random-init backbone in a Cox probe (D4 + D5), before spending days on D3.

## 2026-10-19: D4 features + D5 probe (code)

**S9** (`src/s9_dino_features.py`): checkpoints store the teacher as `model.teacher_backbone.*`
(294 tensors); the backbone is rebuilt from the run's `launch.json` and loaded strict. Whole yr0
volume (144×128×128 → 18×16×16 tokens), bf16, [CLS, mean patch] → 2 × embed_dim per patient, cohort
order, all splits (no labels). One pass over the data serves all requested checkpoints. CPU smoke
(d2_s, 8 patients): random-init features nearly identical across patients (mean pairwise
correlation 0.989), epoch-39 features much less so (0.675) -- the backbone is not constant.

**S10** (`src/s10_dino_probe.py`): elastic-net Cox (sksurv Coxnet, l1_ratio 0.5, 50 alphas down to
1% of alpha_max) on train, penalty by val C, eta for all patients → `runs/probe_*/pred.csv`.
Synthetic check (val): plain Cox on age + lung_ml 0.611 (age alone 0.585, lung_ml 0.593); the
same 2 columns + 0.5 SD noise among 126 noise columns → probe 0.581 (2 non-zero); ridge Cox on all
128 columns 0.555; 128 noise columns → 0.511. The noise columns cost ~0.03 C at 724 train events,
so absolute probe C is conservative; comparisons (checkpoint vs random) are like for like.

**D4 jobs to submit (user)** -- d2_m split in two (146 M params, one forward per checkpoint):
    sbatch src/submit_s9_dino_features.sh --run d2_s --ckpts random "dino_epoch=009" "dino_epoch=019" "dino_epoch=029" "dino_epoch=039"
    sbatch src/submit_s9_dino_features.sh --run d2_m --ckpts random "dino_epoch=009" "dino_epoch=019"
    sbatch --begin=now+10minutes src/submit_s9_dino_features.sh --run d2_m --ckpts "dino_epoch=029" "dino_epoch=039"

## 2026-10-20: D4/D5 on the d2 checkpoints -- not yet better than random weights

**S9** jobs 5120546 (d2_s, 5 ckpts, 17 min), 5120547/8 (d2_m, 3 + 2 ckpts, 34 / 28 min): 16,345
patients each, no constant dims. Median per-feature SD: random 0.088 (s) / 0.089 (m) vs trained
0.19–0.22 (s) / 0.28–0.36 (m).

**S10 probes** (val C; val 2,451 pts / 155 events; penalty chosen on val, so all slightly optimistic,
equally for every row):

| features | random | ep 9 | ep 19 | ep 29 | ep 39 |
|---|---|---|---|---|---|
| d2_s (768 dims) | 0.636 (24 nz) | 0.640 (137) | 0.648 (15) | 0.648 (16) | 0.658 (8) |
| d2_m (1,728 dims) | 0.649 (30) | 0.636 (63) | 0.616 (14) | 0.633 (21) | 0.645 (95) |

Val references (Cox on train, same split): age 0.585, lung_ml 0.593, clinical 5 (age, sex,
cigsmok, lung_ml, lung_mean_hu) 0.636; `cosine` (supervised CNN) 0.688.

Paired bootstrap over val patients (1,000×): d2_s ep39 − random +0.022 [−0.008, +0.056];
d2_m ep39 − random −0.004 [−0.037, +0.028]; d2_s ep39 − cosine −0.030 [−0.067, +0.004].
Spearman of probe η with lung_ml / lung_mean_hu / age (val): s random +0.52 / −0.49 / +0.10;
s ep39 +0.48 / −0.38 / +0.07; m random +0.53 / −0.47 / +0.14; m ep39 +0.32 / −0.31 / +0.16;
cosine +0.18 / −0.17 / +0.23.

Reading: random-weight features already reach the clinical level because they encode lung
size/density (global statistics of the masked volume survive any random projection). After 40
short epochs (yr0 only), the small model trends upward but is not significantly above random; the
big model shows no gain, consistent with the D2 collapse monitors. Trained features lean less on
lung size (m: ρ 0.53 → 0.32) without being more predictive. Neither reaches cosine.
Proposed next (not started): small model, full 100 epochs on all 34,719 scans, default recipe vs
the repo's anti-collapse recipe (exp07: base_lr 0.002, center_momentum 0.9, gradient_clip_val 1.0,
projection_dim 16384), ~5 h each; same S9/S10 against random. If neither beats random, change the
input (finer-scale crops) rather than the recipe.

## 2026-10-20: run folders regrouped; small-model runs at the authors' length prepared

**Regrouped (user request; only plamenmichev's folders moved; `runs/baseline`, `runs/cosine` are
dimitarnizamov's and are referenced everywhere, so they stay):**
`runs/{e40_ema_s*, e40_epb4_s*, epb4_s*}` → `runs/cosine_improved_trial/`;
`runs/dino/d2_s`, `d2_m` → `runs/dino/trial_2h/small`, `big`; `data/features/d2_*` →
`data/features/trial_2h/{small,big}`; `runs/probe_d2_{s,m}_{ckpt}_both` →
`runs/dino_probe/trial_2h/{small,big}/{ckpt}_both`. The `run` field in the moved probe and feature
JSONs was rewritten (old name kept as `run_old`). `s10 --summary` reproduces the 2026-10-20 table.
S10's default output is now `runs/dino_probe/{run}/{ckpt}_{use}`. The d2 runs' launch.json still
says run_name d2_*, so they cannot be resumed (not needed).

**Order (user, 2026-10-20):** the authors' recipe first (small model), then the big model with the
winning recipe. `trial_2h/big` already was the authors' default recipe, but with 24 instead of 120
scans per optimiser step (1 vs 4 GPUs) and 40 short epochs (40k samples) instead of 750k. The two
runs below match the authors' batch and length on one GPU; `stable` adds their exp07 anti-collapse
settings. Dry-run (`--print-config`) confirms every override: max_epochs 150,
limit_train_batches 625, accumulation 15, batch 8, 20 workers, 384-dim / 12 layers / 4 heads,
default base_lr 0.004 / centre momentum 0.99 / clip 3.0 / projection 65,536 vs stable 0.002 / 0.9 /
1.0 / 16,384. lr = base_lr × √(120/1024). GPU nodes have 128 CPUs (gpu: 7 L40S, gpu-h200: 2 × 4
H200), so -c 24 / 20 workers per job (d2_s waited 25% on 12 workers). Commands: docs/pipeline.md §8.
