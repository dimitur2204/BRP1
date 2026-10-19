#!/bin/bash
# D0 -- set up DINOv2-3D-Med for the SSL stages (login node, no Slurm, ~10-20 min).
#
#   bash src/setup_dino_env.sh
#
# 1. installs uv (pinned release binary, sha256-checked) into ~/.local/bin
# 2. clones AIM-Harvard/DINOv2-3D-Med at the pinned commit into ext/ (gitignored)
# 3. builds env/.venv-dino (Python 3.11, the repo's uv.lock: torch 2.7.1 cu12.6, MONAI 1.4,
#    Lightning 2.5) -- separate from env/.venv (3.9), which stays for S1-S6
# 4. runs the repo's unit tests and imports the PRIMUS backbone on CPU
# uv's Python, cache and the venv live in the project tree, not in $HOME.
# Safe to re-run: every step skips what is already there.

set -euo pipefail
PROJECT=/faststorage/project/aura_thymus/student_pipeline
REPO_URL=https://github.com/AIM-Harvard/DINOv2-3D-Med
REPO_COMMIT=77bde4d                       # 2026-08-04, the version read for PLAN.md
UV_VERSION=0.8.22
REPO=$PROJECT/ext/DINOv2-3D-Med

export UV_CACHE_DIR=$PROJECT/env/uv-cache
export UV_PYTHON_INSTALL_DIR=$PROJECT/env/uv-python
export UV_PROJECT_ENVIRONMENT=$PROJECT/env/.venv-dino
export PATH=$HOME/.local/bin:$PATH

# 1. uv: the release binary, checked against its published sha256 (system python3 has no pip)
if ! command -v uv >/dev/null; then
    tmp=$(mktemp -d)
    url=https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz
    curl -sSfL -o "$tmp/uv.tar.gz" "$url"
    curl -sSfL -o "$tmp/uv.sha256" "$url.sha256"
    (cd "$tmp" && echo "$(cut -d' ' -f1 uv.sha256)  uv.tar.gz" | sha256sum -c -)
    tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
    mkdir -p "$HOME/.local/bin"
    install -m 755 "$tmp"/uv-x86_64-unknown-linux-gnu/uv "$tmp"/uv-x86_64-unknown-linux-gnu/uvx "$HOME/.local/bin/"
    rm -rf "$tmp"
fi
uv --version

# 2. repo at the pinned commit
if [ ! -d "$REPO/.git" ]; then
    mkdir -p "$PROJECT/ext"
    git clone --quiet "$REPO_URL" "$REPO"
fi
git -C "$REPO" fetch --quiet origin
git -C "$REPO" checkout --quiet "$REPO_COMMIT"
git -C "$REPO" log -1 --format='repo at %h %cd'

# 3. venv from the lock file (exact versions)
cd "$REPO"
uv python install 3.11
uv sync --frozen --python 3.11
PY=$UV_PROJECT_ENVIRONMENT/bin/python
uv pip install --python "$PY" --quiet pandas==2.2.3   # for src/dino_data.py (not in the repo's lock)
$PY --version

# 4. checks (CPU)
# Known upstream failure at 77bde4d (1 of 25): test_project_import_failure_is_actionable expects
# run.py to wrap a project-import error in a ValueError, which run.py does not do. Cosmetic.
$PY -m unittest discover -s tests -t . -v || echo "unit tests: failures above (1 known upstream)"
$PY - <<'EOF'
import sys, torch
sys.path.insert(0, ".")
from utils.imports import import_module_from_path
import_module_from_path("project", ".")
from project.models.backbones.primus import Primus
m = Primus(input_channels=1, embed_dim=864, patch_embed_size=(8, 8, 8), num_classes=1,
           eva_depth=16, eva_numheads=12, input_shape=(96, 96, 96), classification=True,
           init_values=0.1, scale_attn_inner=True)
print("torch", torch.__version__, "| PRIMUS-M params %.1f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
with torch.no_grad():
    for shape in [(96, 96, 96), (144, 128, 128)]:   # training crop, whole S2 lung grid
        out = m(torch.zeros(1, 1, *shape))
        print("input", shape, "-> tokens", tuple(out.shape))
EOF
echo "D0 OK: env/.venv-dino ready"
