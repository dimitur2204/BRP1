#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       32g
#SBATCH --time      06:00:00
#SBATCH --job-name  s6_embed
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/s6_%x_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/s6_%x_%j.err

# S6 -- frozen 3DINO-ViT features for the whole cohort (inference only).
# Arguments are passed to s6_embed_3dino.py:
#   sbatch src/submit_s6_embed.sh                     # -> data/embeddings/3dino.npz
#   sbatch src/submit_s6_embed.sh --save-tokens       # + ~19 GB of patch tokens
# Then, on the login node:
#   env/.venv/bin/python src/s7_probe.py --feature-set cls --run-name probe_3dino_cls
#   env/.venv/bin/python src/s5_evaluate.py --run probe_3dino_cls
# The 6 h limit is a guess (16k ViT-L forward passes of 576 tokens); check the
# s/patient printed in the log.

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
cd "$PROJECT/src"
python s6_embed_3dino.py "$@"
