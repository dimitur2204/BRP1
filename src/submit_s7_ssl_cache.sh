#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH -c          2
#SBATCH --mem       8g
#SBATCH --time      01:30:00
#SBATCH --array     0-39
#SBATCH --job-name  s7_ssl_cache
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/s7_%A_%a.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/s7_%A_%a.err

# S7 -- yr1/yr2 lung volumes of train + val patients for SSL pretraining (CPU only),
# same input definition as S2. 40 tasks x ~520 scans x ~1-1.5 s ~= 10-15 min each.
# Finished chunks are skipped, so failed tasks can be resubmitted with --array=<ids>.
# Before:  env/.venv/bin/python src/s7_ssl_cache.py --list
# After:   env/.venv/bin/python src/s7_ssl_cache.py --merge

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
cd "$PROJECT/src"
python s7_ssl_cache.py --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$SLURM_ARRAY_TASK_COUNT"
