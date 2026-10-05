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
#   sbatch src/submit_s4_train.sh                                  # runs/baseline, seed 0
#   sbatch src/submit_s4_train.sh --seed 1 --run-name baseline_s1
# Then: env/.venv/bin/python src/s5_evaluate.py --run baseline
# (~27 GB of train volumes are held in host RAM, hence --mem 96g.)

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
cd "$PROJECT/src"
python s4_train.py "$@"
