**Always read `PLAN.md` (staged plan + status) and `PROGRESS.md` (detailed
running log of what was built/found/decided) before starting work.** Update
both after any real progress — don't duplicate their content here.

Current project (branch `cnn3d-lungs-cox`, restarted from scratch
2026-10-05): a simple baseline **3D CNN on the lungs** (TotalSegmentator
lobes) of the NLST baseline CT → time to lung cancer (Cox loss), stages
`S1`–`S5` in `PLAN.md`. Every design choice is explained in
`docs/pipeline.md`; keep it in sync with the code. The previous pipeline
(enriched cohort, lungs/sternum, c0–c5) was deliberately deleted and must not
be reused; it is on branch `cnn3d-lungs-sternum` (`7382553`). The heart-radiomics project
lives on branches `heart-radiomics` / `thymus-heart-aging`.

- Do not run sbatch scripts or anything on Slurm, do the scripts and tell me to run them.
- `env/.venv` — a Python **3.9** venv (gitignored, ~5.5GB), *not* one of
  the repo's other conda envs (those are pinned to older/incompatible
  tooling). Regenerate: `python3 -m venv env/.venv && env/.venv/bin/pip
  install numpy==1.26.4 nibabel matplotlib pandas scipy torch scikit-learn
  lifelines scikit-survival`. Importing torch from BeeGFS takes ~1 min on the
  login node.
- Never enumerate/`ls`/`find` the `derived/totalseg_fullres/` or
  `derived/ct_1x1x2mm/` trees (tens of thousands of entries) — only ever look
  up specific `{pid}_yr0` paths by name.
- Clinical CSVs under `meta/nlst_780/` are owner-only (`mer`). Don't work
  around this. The outcome/clinical source is the public TCIA IDC-780 `prsn`
  table in `data/external/` (gitignored; regeneration command in
  `docs/pipeline.md` §6). `nlst.csv` is IDC series metadata only; its
  `SeriesDescription` encodes kernel, kVp, mA, slice thickness.
- `src/nlst.py` — all paths and the input definition (lung labels, 2.5 mm
  grid, HU window, uint8 encoding, QC thresholds, split seed). `src/model.py`
  — `LungCNN`, `cox_loss`, `harrell_c`, `augment`.
- `src/s{N}_*.py`, `src/submit_s{N}_*.sh` — pipeline stages; follow the CLI
  style of the existing scripts.
- `data/cohort.csv` (S3) is the source of truth for training/evaluation:
  pid, cache `row`, split, time (days from T0 scan), event, covariates, QC.
- Test-set discipline: S4 writes test predictions, but only `s5_evaluate.py`
  reads them.
- `data/cache/`, `runs/`, `data/external/`, `env/`, `models/*.pt`,
  `logs/*.out`/`*.err` are gitignored — see `.gitignore` for regeneration.
- `learn/` (git-excluded via `.git/info/exclude`) is a teaching workspace
  that was written for the old code; leave it alone unless asked.
