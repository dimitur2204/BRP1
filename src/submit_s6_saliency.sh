#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu-short
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       48g
#SBATCH --time      02:00:00
#SBATCH --job-name  s6_saliency
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/s6_%x_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/s6_%x_%j.err

# S6 -- saliency maps (occlusion + Grad-CAM) of one trained run on val patients.
# All arguments are passed to s6_saliency.py:
#   sbatch src/submit_s6_saliency.sh --run epb4_s0
#   sbatch src/submit_s6_saliency.sh --run epb4_s0 --pids 123456,234567 --patch 8 --stride 4  # finer maps
# Output: runs/{run}/saliency_val/. gpu-short allows 2 h; ~155 events + 155 controls
# take ~1 h (occlusion ~2k forward passes per patient + reading each lobe mask).
# (Only the val volumes, ~6 GB, are held in host RAM.)

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
cd "$PROJECT/src"
python s6_saliency.py "$@"
