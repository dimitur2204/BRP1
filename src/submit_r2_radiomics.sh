#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH --array     0-19
#SBATCH -c          2
#SBATCH --mem       8g
#SBATCH --time      01:00:00
#SBATCH --job-name  r2_radiomics
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/r2_radiomics_%A_%a.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/r2_radiomics_%A_%a.err

# R2 -- heart radiomics + cardiac features for the 6,306-pid cohort
# (data/pid_lists/heart_cohort_v1.csv), split into 20 array tasks of ~315 pids.
#
# CPU only (pyradiomics is single-threaded numpy/C). Local test measured
# ~2.6 s/patient => ~14 min of compute per task; --time 01:00:00 leaves
# generous headroom for network-filesystem reads (each patient reads its own
# CT + seg directly from derived/ by name -- no $TMPDIR staging needed, there
# is no shared cache to copy, and the tree is never enumerated).
#
# Each task rewrites its chunk CSV after every patient, so a timeout keeps the
# partial work; failed patients are recorded with status=error:..., not
# dropped. Re-running one task (e.g. --array 7) just overwrites that chunk.
#
# After ALL tasks finish, merge (on the login node, takes seconds):
#   env/.venv/bin/python src/r2_extract_radiomics.py --merge
#
# For the R6 controls, change --organ below (lungs / sternum).

set -euo pipefail

PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"

source "$PROJECT/env/.venv/bin/activate"
cd "$PROJECT/src"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT on $(hostname)"
python r2_extract_radiomics.py --organ heart \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "$SLURM_ARRAY_TASK_COUNT"
echo "[$(date)] done."
