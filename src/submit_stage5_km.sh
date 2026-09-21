#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu-l40s
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       32g
#SBATCH --time      00:30:00
#SBATCH --job-name  stage5_km
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/stage5_km_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/stage5_km_%j.err

# Stage 5 -- Kaplan-Meier curves for all 7 organs, from the Stage 3v2 3D CNN
# checkpoints. Inference-only (no training), but staged via sbatch anyway
# because reading+resizing ~127 crops/organ over the shared filesystem on a
# CPU-contended login node took ~10 min for ONE organ (see PROGRESS.md) --
# a dedicated node's 8 cores + GPU makes this a ~1-2 min job for all 7.

set -euo pipefail

PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"

echo "[$(date)] staging organ crops to \$TMPDIR ($TMPDIR)..."
cp -r "$PROJECT/data/organ_crops" "$TMPDIR/organ_crops"
echo "[$(date)] staging done: $(du -sh $TMPDIR/organ_crops | cut -f1)"

source "$PROJECT/env/.venv/bin/activate"

python "$PROJECT/src/stage5_km_curve.py" --organ all --crops-dir "$TMPDIR/organ_crops"

echo "[$(date)] done."
