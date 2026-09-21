#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu-l40s
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       32g
#SBATCH --time      02:00:00
#SBATCH --job-name  stage3_cnn3d
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/stage3_cnn3d_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/stage3_cnn3d_%j.err

# Stage 3v2 -- 3D CNN training for all 7 organs, on GPU.
#
# Uses the project's own venv (env/.venv), NOT conda -- this cluster account
# has no conda/module system set up (unlike the mentor's discovery_pipeline
# envs), so we built a self-contained venv instead (see PROGRESS.md).
#
# Stages data/organ_crops/ (14 GB, dominated by `lungs` at 7.5 GB) to
# $TMPDIR first, per the cluster's GPU docs recommendation -- avoids reading
# ~2800 small/medium files repeatedly over the shared BeeGFS filesystem
# during dataset construction, which is CPU-side work that would otherwise
# leave the GPU idle and risk the docs' "<75% GPU utilization after 2h" auto
# -cancellation policy (unlikely to matter at this small scale/short runtime,
# but it's what the docs recommend, so we do it).

set -euo pipefail

PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"

echo "[$(date)] staging organ crops to \$TMPDIR ($TMPDIR)..."
cp -r "$PROJECT/data/organ_crops" "$TMPDIR/organ_crops"
echo "[$(date)] staging done: $(du -sh $TMPDIR/organ_crops | cut -f1)"

source "$PROJECT/env/.venv/bin/activate"

echo "[$(date)] torch/cuda check:"
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "[$(date)] starting training..."
python "$PROJECT/src/stage3_cnn3d.py" --organ all --crops-dir "$TMPDIR/organ_crops"

echo "[$(date)] done."
