#!/bin/bash
#SBATCH --account   aura_thymus
#SBATCH --partition short
#SBATCH -c          8
#SBATCH --mem       32g
#SBATCH --time      00:45:00
#SBATCH --job-name  stage6_saliency
#SBATCH --output    /faststorage/project/aura_thymus/student_pipeline/logs/stage6_saliency_%j.out
#SBATCH --error     /faststorage/project/aura_thymus/student_pipeline/logs/stage6_saliency_%j.err

# Stage 6 -- Grad-CAM saliency maps for `lungs` + `anterior_mediastinum`
# (the validated positive control and the actual scientific target). Doesn't
# need a GPU (Simple3DCNN is 4 conv layers, 20 patients total, inference +
# one backward pass -- trivially fast on CPU). Originally requested
# gpu-l40s to mirror Stage 3/5's convention, but that partition has exactly
# ONE node cluster-wide, so the job sat PENDING behind whoever else was
# using it -- switched to `short` (44 nodes, 12h limit), which has far more
# capacity for a plain CPU job like this one.
#
# Still submitted via sbatch rather than run on the login node for I/O
# reasons: it re-reads the *original* full-resolution CT+seg files from
# derived/ for each selected patient (that's new -- Stages 3/5 only ever
# touched the small cached crops), and per project convention that kind of
# read goes through a dedicated node's $TMPDIR staging, not the login node.
# We only stage the 2 organs' crop subfolders (not all 7), and only ever
# read 20 individual {pid}_yr0 files from derived/ directly -- never
# enumerate/copy that tree.
#
# First run (job 64799090) hit the original 00:15:00 limit: staging ~8GB of
# crops (~800 small .npy files over the network filesystem) alone took
# 10.5 minutes, leaving too little time to finish both organs. Bumped to
# 00:45:00 -- `lungs` fully completed in the remaining time, so this budget
# is generous, not tight.

set -euo pipefail

PROJECT=/faststorage/project/aura_thymus/student_pipeline
mkdir -p "$PROJECT/logs"

echo "[$(date)] staging lungs + anterior_mediastinum crops to \$TMPDIR ($TMPDIR)..."
mkdir -p "$TMPDIR/organ_crops"
cp -r "$PROJECT/data/organ_crops/lungs" "$TMPDIR/organ_crops/lungs"
cp -r "$PROJECT/data/organ_crops/anterior_mediastinum" "$TMPDIR/organ_crops/anterior_mediastinum"
echo "[$(date)] staging done: $(du -sh $TMPDIR/organ_crops | cut -f1)"

source "$PROJECT/env/.venv/bin/activate"

python "$PROJECT/src/stage6_saliency.py" --organ default --crops-dir "$TMPDIR/organ_crops"

echo "[$(date)] done."
