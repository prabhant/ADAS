#!/bin/bash
# Launch the meta and worker vLLM servers inside one SLURM job, on separate
# ports and disjoint GPU sets.
#
# Sourced by the stage SLURM scripts; can also be run directly for debugging.
#
# Required environment:
#   META_MODEL, WORKER_MODEL   Hugging Face ids or absolute paths
#   META_TP, WORKER_TP         tensor-parallel size per role
# Optional:
#   META_PORT (8001), WORKER_PORT (8000), MAX_NUM_SEQS (256), LOG_DIR
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ADAS_PYTHON="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
ENV_PREFIX="$(dirname "$(dirname "$ADAS_PYTHON")")"
VLLM_BIN="$ENV_PREFIX/bin/vllm"

# The system libstdc++ on Snellius provides CXXABI up to 1.3.13; the conda env's
# ICU needs 1.3.15, so importing sqlite3 -- and therefore the whole xgrammar
# structured-output backend -- fails unless the env's own lib is found first.
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

META_MODEL="${META_MODEL:-}"
WORKER_MODEL="${WORKER_MODEL:?WORKER_MODEL must be set}"
META_TP="${META_TP:-1}"
WORKER_TP="${WORKER_TP:-1}"
META_PORT="${META_PORT:-8001}"
WORKER_PORT="${WORKER_PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
META_MAX_LEN="${META_MAX_LEN:-131072}"
WORKER_MAX_LEN="${WORKER_MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiments/results/_serving_logs}"

mkdir -p "$LOG_DIR"

# Compute nodes have no internet, and login-shell variables are not inherited
# into Snellius jobs, so the cache is pinned here rather than assumed.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}"
export HF_HOME="${HF_HOME:-/scratch-shared/$USER/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
mkdir -p "$HF_HOME"

# Model load from the shared filesystem is slow for large checkpoints. $TMPDIR is
# node-local scratch on nodes requested with --constraint=scratch-node, and is
# wiped at job end, so it holds the compile cache only, never results.
if [[ "${STAGE_TO_NVME:-0}" == "1" && -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
    export VLLM_CACHE_ROOT="$TMPDIR/vllm"
    mkdir -p "$VLLM_CACHE_ROOT"
    echo "staging vLLM cache on node-local scratch: $VLLM_CACHE_ROOT"
fi

VLLM_PIDS=()
cleanup() {
    for pid in "${VLLM_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

serve() {
    local role="$1" model="$2" port="$3" tp="$4" max_len="$5" gpus="$6"
    echo "launching $role: $model on GPUs [$gpus], tp=$tp, port=$port"
    # vLLM 0.28 replaced --guided-decoding-backend with --structured-outputs-config,
    # which takes JSON. Passing the old flag aborts the server at startup.
    CUDA_VISIBLE_DEVICES="$gpus" \
    "$VLLM_BIN" serve "$model" \
        --served-model-name "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --download-dir "$HF_HUB_CACHE" \
        --tensor-parallel-size "$tp" \
        --max-model-len "$max_len" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --enable-prefix-caching \
        --structured-outputs-config "{\"backend\": \"${GUIDED_BACKEND:-xgrammar}\"}" \
        --generation-config vllm \
        > "$LOG_DIR/${role}_${port}.log" 2>&1 &
    VLLM_PIDS+=("$!")
}

# Worker takes the low-numbered GPUs, meta the rest.
worker_gpus="$(seq -s, 0 $((WORKER_TP - 1)))"
serve worker "$WORKER_MODEL" "$WORKER_PORT" "$WORKER_TP" "$WORKER_MAX_LEN" "$worker_gpus"

health_args=(--worker-url "http://127.0.0.1:${WORKER_PORT}/v1")
if [[ -n "$META_MODEL" ]]; then
    meta_gpus="$(seq -s, "$WORKER_TP" $((WORKER_TP + META_TP - 1)))"
    serve meta "$META_MODEL" "$META_PORT" "$META_TP" "$META_MAX_LEN" "$meta_gpus"
    health_args+=(--meta-url "http://127.0.0.1:${META_PORT}/v1")
fi

"$ADAS_PYTHON" "$REPO_ROOT/experiments/serving/health_check.py" \
    "${health_args[@]}" \
    --timeout "${HEALTH_TIMEOUT:-2400}" \
    --check-json \
    ${CHECK_REASONING_EFFORT:+--check-reasoning-effort}

echo "servers ready"
export ADAS_WORKER_BASE_URL="http://127.0.0.1:${WORKER_PORT}/v1"
export ADAS_META_BASE_URL="http://127.0.0.1:${META_PORT}/v1"
export OPENAI_API_KEY="local-vllm"

# When sourced, hand control back with the servers still running.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    trap - EXIT
    export ADAS_VLLM_PIDS="${VLLM_PIDS[*]}"
    return 0
fi

wait
