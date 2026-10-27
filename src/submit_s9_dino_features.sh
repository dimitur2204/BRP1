#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu-short
#SBATCH --gpus      1
#SBATCH -c          4
#SBATCH --mem       32g
#SBATCH --time      02:00:00
#SBATCH --job-name  s9_feat
#SBATCH --output    /faststorage/project/aura_thymus/plamen_pipeline/logs/s9_%x_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/plamen_pipeline/logs/s9_%x_%j.err

# S9 -- DINOv2 features for all cohort patients. All arguments go to s9_dino_features.py:
#   sbatch src/submit_s9_dino_features.sh --run trial_2h/small --ckpts random "dino_epoch=009" ...
# One pass over the 16,345 yr0 volumes (~38 GB read) serves all listed checkpoints; each
# extra checkpoint adds one forward pass per patient. Finished checkpoints are skipped on
# resubmission. Progress lines show the speed; split checkpoints over jobs if 2 h is short.
# Then (login node): env/.venv/bin/python src/s10_dino_probe.py --run <run> --ckpt <ckpt>

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/plamen_pipeline
mkdir -p "$PROJECT/logs"
export WANDB_MODE=disabled
export PYTHONPATH="$PROJECT/src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PROJECT/env/.venv-dino/bin/python" "$PROJECT/src/s9_dino_features.py" "$@"
