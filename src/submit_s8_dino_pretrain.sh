#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu-short
#SBATCH --gpus      1
#SBATCH -c          16
#SBATCH --mem       64g
#SBATCH --time      02:00:00
#SBATCH --job-name  s8_dino
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/s8_%x_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/s8_%x_%j.err

# S8 -- DINOv2 pretraining. All arguments are passed to s8_dino_pretrain.py.
# D2 smoke/throughput (gpu-short, < 2 h; the defaults above):
#   sbatch src/submit_s8_dino_pretrain.sh --run-name trial_2h/big trainer#max_epochs=100
# D3 full run: longer partition/time on the command line, before the script name, e.g.
#   sbatch -p gpu-h200 -t 3-00:00:00 src/submit_s8_dino_pretrain.sh --run-name d3_... ...
# Re-submitting the same command resumes from runs/dino/{run}/ckpt/last.ckpt.
# 16 CPUs: 3D view augmentation costs ~1.2 s of CPU per scan (D1 QC), 12 data workers + headroom.
# Watch: grep throughput logs/s8_*.out (data_wait_frac), jobinfo <jobid> (GPU utilisation;
# GenomeDK cancels GPU jobs averaging < 75% after 2 h).

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
export WANDB_MODE=disabled
export PYTHONPATH="$PROJECT/src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PROJECT/env/.venv-dino/bin/python" "$PROJECT/src/s8_dino_pretrain.py" "$@"
