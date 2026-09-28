#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH -c          4
#SBATCH --mem       16g
#SBATCH --time      01:00:00
#SBATCH --array     0-19
#SBATCH --job-name  c1_cache
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/c1_cache_%A_%a.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/c1_cache_%A_%a.err

# C1 -- cache lungs + sternum CNN inputs (new iso + legacy 64^3) for the
# 6,306-patient cohort. 20 tasks x ~315 pids x ~2.9 s/pid ~= 15-20 min each
# (CPU only; no GPU needed). Afterwards, on the login node:
#   env/.venv/bin/python src/c1_cache_volumes.py --merge
#   env/.venv/bin/python src/c1_cache_volumes.py --qc

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
cd "$PROJECT/src"
python c1_cache_volumes.py --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$SLURM_ARRAY_TASK_COUNT"
