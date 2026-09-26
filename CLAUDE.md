**Always read `PLAN.md` (staged plan + status) and `PROGRESS.md` (detailed
running log of what was built/found/decided) before starting work.** Update
both after any real progress — don't duplicate their content here.

Current project: heart radiomics → lung-cancer risk (DeepSurv + Cox/RSF
baselines → KM), stages `R0`–`R7` in `PLAN.md`. The earlier multi-organ CNN
pipeline (`src/stage{1..6}_*`) is finished; its PLAN/PROGRESS are in git
history (commit `eb3b89b`).

- Do not run sbatch scripts or anything on Slurm, do the scripts and tell me to run them.
- `env/.venv` — a Python **3.9** venv (gitignored, ~5.5GB), *not* one of
  the repo's other conda envs (those are pinned to older/incompatible
  tooling). Regenerate: `python3 -m venv env/.venv && env/.venv/bin/pip
  install numpy nibabel matplotlib pandas scipy torch scikit-learn lifelines
  pyradiomics==3.1.0 SimpleITK scikit-survival neuroCombat shap`
  (pyradiomics 3.1.0 ships a cp39 wheel — don't bump Python without checking).
- Never enumerate/`ls`/`find` the full `derived/totalseg_fullres/` tree
  (~55k folders) — only ever look up specific `{pid}_yr0` paths by name.
- Clinical CSVs (`meta/nlst_780/*.csv`: age, sex, smoking, ctab findings)
  are owner-only (`mer`) — unreadable until the mentor grants access. Don't
  work around this (e.g. via the world-readable `.zip` copies). `nlst.csv` is
  IDC series metadata only (no age/sex/smoking); its `SeriesDescription`
  encodes acquisition params (kernel, kVp, mA, slice thickness).
- `src/organs.py` — shared organ definitions, TotalSegmentator label map
  (heart = 51), mask/crop geometry (reused verbatim from the mentor's `code/`
  scripts).
- `src/r{N}_*.py`, `src/submit_r{N}_*.sh` — radiomics stages; follow the
  CLI style of the existing `src/stage*` scripts.
- `data/pid_lists/heart_cohort_v1.csv` (built in R0; enriched cohort, 6,373
  pids) is the source of truth for the radiomics project.
  `data/pid_lists/subset_v1.csv` (400 pids) belongs to the old CNN pipeline.
- `data/organ_crops/`, `data/saliency/`, `env/`, `models/*.pt`,
  `logs/*.out`/`*.err` are gitignored (regenerable/large) — see `.gitignore`
  for exact regeneration commands.
