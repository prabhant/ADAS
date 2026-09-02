#!/usr/bin/env bash
set -euo pipefail

module load 2025 Miniconda3/25.5.1-1
export CONDARC=/dev/null

ENV_PREFIX="${ADAS_ENV_PREFIX:-$HOME/.conda/envs/adas-snellius}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "$ENV_PREFIX" ]]; then
    conda env update --prefix "$ENV_PREFIX" --file "$REPO_ROOT/environment-snellius.yml" --prune
else
    conda env create --prefix "$ENV_PREFIX" --file "$REPO_ROOT/environment-snellius.yml"
fi

# Snellius' system libstdc++ is older than the one the env's libicu needs, which
# breaks sqlite3 (and therefore vLLM) unless the env's own lib dir is searched first.
mkdir -p "$ENV_PREFIX/etc/conda/activate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/zz-ld-library-path.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

conda run --prefix "$ENV_PREFIX" python -c \
    "import sqlite3, openai, torch, transformers, vllm; print('ADAS environment ready; CUDA build:', torch.version.cuda)"

echo "Activate with: module load 2025 Miniconda3/25.5.1-1 && conda activate $ENV_PREFIX"
