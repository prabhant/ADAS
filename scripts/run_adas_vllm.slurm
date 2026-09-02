#!/usr/bin/env bash
#SBATCH --job-name=adas-vllm
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-%x-%j.out

set -euo pipefail

module load 2025 Miniconda3/25.5.1-1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ADAS_ENV_PREFIX:-$HOME/.conda/envs/adas-snellius}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
ADAS_SCRIPT="${ADAS_SCRIPT:-_mmlu/search.py}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

cd "$REPO_ROOT"
mkdir -p results "${TMPDIR:-/tmp}/$USER/hf"

export HF_HUB_CACHE="/projects/2/managed_datasets/hf_cache_dir"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/$USER/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADAS_MODEL="$MODEL"
export OPENAI_API_KEY="local-vllm"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

conda run --no-capture-output --prefix "$ENV_PREFIX" \
    vllm serve "$MODEL" \
    --download-dir "$HF_HUB_CACHE" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    --generation-config vllm &
VLLM_PID=$!

conda run --no-capture-output --prefix "$ENV_PREFIX" python - <<'PY'
import os
import time
import urllib.request

url = os.environ["OPENAI_BASE_URL"].removesuffix("/v1") + "/health"
for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                print("vLLM is ready")
                break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError(f"vLLM did not become ready at {url}")
PY

# Extra ADAS arguments can be supplied after the script name, for example:
# sbatch --export=ALL,MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
#   scripts/run_adas_vllm.slurm --n_generation 1 --valid_size 8 --max_workers 8
conda run --no-capture-output --prefix "$ENV_PREFIX" \
    python "$ADAS_SCRIPT" --model "$MODEL" "$@"
