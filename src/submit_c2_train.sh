#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition gpu-short
#SBATCH --gpus      1
#SBATCH -c          8
#SBATCH --mem       48g
#SBATCH --time      02:00:00
#SBATCH --job-name  c2_train
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/c2_%x_%A_%a.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/c2_%x_%A_%a.err

# C2 -- 3D CNN training, one array task = up to 3 runs of one organ (the
# organ's cache is loaded to the GPU once per task). Run set = $1.
#
#   sbatch --array=0-11 src/submit_c2_train.sh grid
#   env/.venv/bin/python src/c3_select.py            # writes data/c3_chosen_configs.json
#   sbatch --array=0-$(( $(env/.venv/bin/python src/c2_train.py --set final --count) - 1 )) \
#          src/submit_c2_train.sh final
#   sbatch --array=0-3 src/submit_c2_train.sh curves  # figure-only re-runs (diagnostics, per-epoch ckpts)
#   env/.venv/bin/python src/c2_plot_curves.py        # -> figs/c2_training_curves/, data/c2_curves_spikes.csv
#
# (`python src/c2_train.py --set <set> --count` prints the number of tasks.)
# Finished runs (runs/<set>/<run_id>/result.json) are skipped, so a task that
# hits the 2 h gpu-short limit can simply be resubmitted with the same command.

set -euo pipefail
SET=${1:?usage: sbatch --array=0-N src/submit_c2_train.sh grid|final|curves}
PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"
source "$PROJECT/env/.venv/bin/activate"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
cd "$PROJECT/src"
python c2_train.py --set "$SET" --task "$SLURM_ARRAY_TASK_ID"
