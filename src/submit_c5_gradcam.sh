#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH -c          4
#SBATCH --mem       24g
#SBATCH --time      00:45:00
#SBATCH --job-name  c5_gradcam
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/c5_gradcam_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/c5_gradcam_%j.err

# C5 -- Grad-CAM for the headline lungs + sternum models (final/main seed 1),
# 5 highest + 5 lowest scored test patients each. CPU is enough (20 forward+
# backward passes); run after `sbatch ... submit_c2_train.sh final` and
# `python src/c4_evaluate.py` (c4's CIs are used for the "interpret with
# caution" caption).

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
cd "$PROJECT/src"
python c5_gradcam.py --device cpu
