#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       96g
#SBATCH --time      08:00:00
#SBATCH --job-name  s12_ft
#SBATCH --output    /faststorage/project/aura_thymus/plamen_pipeline/logs/s12_%x_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/plamen_pipeline/logs/s12_%x_%j.err

# S12 -- fine-tune the DINO ViT with a Cox loss (or the same ViT from scratch). All arguments
# are passed to s12_dino_finetune.py:
#   sbatch src/submit_s12_dino_finetune.sh --run-name finetune/dino_ep39 \
#       --dino-run small_150ep/stable --ckpt "dino_epoch=039"
#   sbatch --begin=now+15minutes src/submit_s12_dino_finetune.sh --run-name finetune/scratch \
#       --dino-run small_150ep/stable --ckpt random --layer-decay 1.0 --lr-backbone 1e-3
# Then: env/.venv/bin/python src/s5_evaluate.py --run finetune/dino_ep39 (and finetune/scratch)
# 30 epochs x ~7-8 min (L40S) ~ 4 h; on an H200 (-p gpu-h200 before the script name) ~2-2.5 h.
# A timing check fits gpu-short: sbatch -p gpu-short -t 00:45:00 ... --epochs 2 --run-name finetune/timing
# Stagger jobs (--begin) so the ~27 GB train loads don't overlap.
# GenomeDK cancels GPU jobs averaging < 75% GPU utilisation after 2 h (check: jobinfo <jobid>).

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/plamen_pipeline
mkdir -p "$PROJECT/logs"
export WANDB_MODE=disabled
export PYTHONPATH="$PROJECT/src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PROJECT/env/.venv-dino/bin/python" "$PROJECT/src/s12_dino_finetune.py" "$@"
