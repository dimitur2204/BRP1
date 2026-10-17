#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       96g
#SBATCH --time      12:00:00
#SBATCH --job-name  s4_train
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/s4_%x_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/s4_%x_%j.err

# S4 -- train one lungs 3D CNN run. All arguments are passed to s4_train.py:
#   sbatch src/submit_s4_train.sh --run-name cosine                # current defaults, seed 0
#   sbatch src/submit_s4_train.sh --run-name cosine_s1 --seed 1
# Then: env/.venv/bin/python src/s5_evaluate.py --run cosine
# A finished run directory is never overwritten (new --run-name or --overwrite).
# Runs under 2 h (e.g. --epochs 40, ~50 min) can go to the gpu-short partition (8 L40S,
# 2 h max, usually free); sbatch options go before the script name:
#   sbatch -p gpu-short -t 02:00:00 src/submit_s4_train.sh --run-name e40_ema_s0 --epochs 40 ...
# The ~27 GB load takes ~7 min alone but ~29 min when 6 jobs load at once: stagger several jobs
# with --begin=now+10minutes, +20minutes, ...
# GenomeDK cancels GPU jobs averaging < 75% GPU utilisation after 2 h (check: jobinfo <jobid>).
# (~27 GB of train volumes are held in host RAM, hence --mem 96g.)

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
cd "$PROJECT/src"
python s4_train.py "$@"
