#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH --array     0-39
#SBATCH -c          1
#SBATCH --mem       6g
#SBATCH --time      00:45:00
#SBATCH --job-name  r2_fullarm
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/r2_fullarm_%A_%a.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/r2_fullarm_%A_%a.err

# R2 (full CT arm) -- cardiac features only (calcium proxy, pericardial fat,
# heart volume, core noise + qc_* flags; no pyradiomics) for the 9,872 TH-scored
# NLST CT-arm pids outside the enriched cohort that have a CT + TotalSegmentator
# mask (data/pid_lists/heart_cohort_fullarm.csv, built by
# `src/r0_build_cohort.py --pids-file ... --exclude-cohort ...`).
# The other ~9k TH-scored pids have a CT but no TotalSegmentator mask yet.
#
# Local test: ~2 s/patient => ~250 pids/task ~ 9 min; --time 00:45:00 is headroom
# for network-filesystem reads. Chunks are rewritten after every patient.
#
# After ALL tasks finish (login node, seconds):
#   env/.venv/bin/python src/r2_extract_radiomics.py --merge --tag fullarm \
#       --cohort data/pid_lists/heart_cohort_fullarm.csv
# -> data/radiomics/heart_features_fullarm.csv (analysis: experiments/E6, to be written)

set -euo pipefail

PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"

source "$PROJECT/env/.venv/bin/activate"
cd "$PROJECT/src"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT on $(hostname)"
python r2_extract_radiomics.py --organ heart --cardiac-only --tag fullarm \
    --cohort "$PROJECT/data/pid_lists/heart_cohort_fullarm.csv" \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "$SLURM_ARRAY_TASK_COUNT"
echo "[$(date)] done."
