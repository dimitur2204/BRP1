**Always read `PLAN.md` (staged plan + status) and `PROGRESS.md` (detailed
running log of what was built/found/decided) before starting work.** Update
both after any real progress — don't duplicate their content here.

- Do not run sbatch scripts or anything on Slurm, do the scripts and tell me to run them.
- `env/.venv` — a fresh Python 3.11 venv (gitignored, ~5.5GB), *not* one of
  the repo's other conda envs (those are pinned to older/incompatible
  tooling). Regenerate: `python3 -m venv env/.venv && env/.venv/bin/pip
  install numpy nibabel matplotlib pandas scipy torch scikit-learn lifelines`.
- Never enumerate/`ls`/`find` the full `derived/totalseg_fullres/` tree
  (~55k folders) — only ever look up specific `{pid}_yr0` paths by name.
- `src/organs.py` — shared organ definitions, TotalSegmentator label map,
  mask/crop geometry (reused verbatim from the mentor's `code/` scripts).
- `src/stage{1..6}_*.py`, `src/submit_stage{3,5,6}_*.sh` — the staged
  pipeline; each stage script takes `--organ`/`--device`/`--crops-dir` in
  the same CLI style, read the most recent one before writing a new stage.
- `data/pid_lists/subset_v1.csv` — the single source of truth for which
  400 patients are in scope; every later stage reads from it.
- `data/organ_crops/`, `env/`, `models/*.pt`, `logs/*.out`/`*.err` are
  gitignored (regenerable/large) — see `.gitignore` for exact regeneration
  commands.
