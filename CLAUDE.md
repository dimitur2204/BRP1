**Always read `PLAN.md` (staged plan + status) and `PROGRESS.md` (detailed
running log of what was built/found/decided) before starting work.** Update
both after any real progress — don't duplicate their content here.

Current project (branch `cnn3d-lungs-sternum`): the 3D CNN retrained at scale
on **lungs (positive control) and sternum (negative control)** →
lung-cancer risk (Cox loss, BCE ablation, KM), stages `C0`–`C5` in `PLAN.md`.
Every design choice is explained in `docs/cnn3d_explained.md`; keep it in
sync with the code. The heart-radiomics project lives on branches
`heart-radiomics` / `thymus-heart-aging`; the original 7-organ CNN pipeline
is in git history (commit `eb3b89b`).

- Do not run sbatch scripts or anything on Slurm, do the scripts and tell me to run them.
- `env/.venv` — a Python **3.9** venv (gitignored, ~5.5GB), *not* one of
  the repo's other conda envs (those are pinned to older/incompatible
  tooling). Regenerate: `python3 -m venv env/.venv && env/.venv/bin/pip
  install numpy==1.26.4 nibabel matplotlib pandas scipy torch scikit-learn
  lifelines scikit-survival`. Importing torch from BeeGFS takes ~1 min on the
  login node.
- Never enumerate/`ls`/`find` the full `derived/totalseg_fullres/` tree
  (~55k folders) — only ever look up specific `{pid}_yr0` paths by name.
- Clinical CSVs (`meta/nlst_780/*.csv`: age, sex, smoking) are owner-only
  (`mer`). Don't work around this. The public TCIA IDC-780 package into
  `data/external/` is the intended source, but downloading it needs the
  user's explicit go-ahead (command in `docs/cnn3d_explained.md` §8). Until
  then `cnn_cohort_v1.csv` has empty clinical columns and C4 skips the
  clinical adjustment. `nlst.csv` is IDC series metadata only; its
  `SeriesDescription` encodes kernel, kVp, mA, slice thickness.
- `src/organs.py` — label map (lungs = lobes 10–14, sternum = 116), HU
  windows (`WINDOW`, and `LEGACY_WINDOW` for reproducing the old model),
  mask/crop geometry, `load_patient`. `src/cnn3d.py` — `CNN3D`,
  `cox_ph_loss`, cache loading, `harrell_c`.
- `src/c{N}_*.py`, `src/submit_c{N}_*.sh` — pipeline stages; follow the CLI
  style of the existing scripts.
- `data/pid_lists/cnn_cohort_v1.csv` (C0; enriched cohort, 6,306 pids,
  pid-identical to the radiomics `heart_cohort_v1`) is the source of truth.
  `data/pid_lists/subset_v1.csv` (400 pids) defines the legacy model's subset
  (`in_subset_v1` column).
- Test-set discipline: every C2 run writes test predictions, but only
  `c4_evaluate.py` reads them, and never for the `grid` tuning runs.
- `data/cnn_cache/`, `runs/`, `data/external/`, `data/organ_crops/`,
  `data/saliency/`, `env/`, `models/*.pt`, `logs/*.out`/`*.err` are gitignored
  — see `.gitignore` for exact regeneration commands.
