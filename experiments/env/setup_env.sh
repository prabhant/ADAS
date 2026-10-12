#!/usr/bin/env bash
# Create the uv virtual environment for the ADAS scaling study.
# Run on a LOGIN NODE -- compute nodes have no internet.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ADAS_VENV:-$REPO_ROOT/.venv}"
PYTHON_VERSION="${ADAS_PYTHON_VERSION:-3.11}"

module load 2025 2>/dev/null || true

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_ROOT"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Search-loop dependencies (the stock ADAS requirements plus config parsing).
uv pip install \
    "openai>=1.40" \
    "backoff" \
    "numpy" \
    "pandas" \
    "scipy" \
    "tqdm" \
    "blobfile" \
    "pyyaml"

# Serving. Pin vLLM: the gpt-oss response format and the reasoning-effort
# control are version-sensitive, and Stage 0 validates both against this pin.
uv pip install "vllm==${VLLM_VERSION:-0.10.1}"

# Analysis layers. CPU-only, installed here so the genoa jobs need no network.
uv pip install \
    "zss" \
    "networkx" \
    "scikit-learn" \
    "sentence-transformers" \
    "matplotlib" \
    "statsmodels" \
    "shap"

python - <<'PY'
import importlib
required = ["openai", "numpy", "pandas", "scipy", "tqdm", "yaml",
            "zss", "networkx", "sklearn", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
print("missing:", missing if missing else "none")
raise SystemExit(1 if missing else 0)
PY

echo "environment ready at $VENV_DIR"
echo "activate with: source $VENV_DIR/bin/activate"
