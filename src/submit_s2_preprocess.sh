#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH -c          2
#SBATCH --mem       8g
#SBATCH --time      01:30:00
#SBATCH --array     0-39
#SBATCH --job-name  s2_preprocess
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/s2_%A_%a.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/s2_%A_%a.err

# S2 -- lung volumes for the s1 candidates (CPU only). 40 tasks x ~415 pids
# x ~1.5 s/pid ~= 10-20 min each. Finished chunks are skipped, so failed
# tasks can be resubmitted with --array=<ids>. Afterwards, on the login node:
#   env/.venv/bin/python src/s2_preprocess.py --merge
#   env/.venv/bin/python src/s2_preprocess.py --qc
#   env/.venv/bin/python src/s3_cohort.py

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
cd "$PROJECT/src"
python s2_preprocess.py --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$SLURM_ARRAY_TASK_COUNT"
