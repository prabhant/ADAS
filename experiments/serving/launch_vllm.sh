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

# gpu_h100 nodes are shared by up to four jobs, so two array tasks of the same
# stage can land on one node. With fixed ports the second server fails to bind,
# its health check then finds the FIRST task's server on localhost, and the whole
# cell runs against another cell's models -- silently, and attributed to the wrong
# pair. Derive the ports from the job id instead, and step past anything already
# listening.
_port_free() {
    "$ADAS_PYTHON" - "$1" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}
_pick_port_pair() {
    local base=$(( 8000 + ( ${SLURM_JOB_ID:-$$} % 1800 ) * 2 ))
    local tries=0
    while (( tries < 50 )); do
        if _port_free "$base" && _port_free "$((base + 1))"; then
            echo "$base"
            return 0
        fi
        base=$(( base + 2 ))
        (( base > 11998 )) && base=8000
        tries=$(( tries + 1 ))
    done
    echo "could not find a free port pair" >&2
    return 1
}
_PORT_BASE="$(_pick_port_pair)"
WORKER_PORT="${WORKER_PORT:-$_PORT_BASE}"
META_PORT="${META_PORT:-$((_PORT_BASE + 1))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
META_MAX_LEN="${META_MAX_LEN:-131072}"
WORKER_MAX_LEN="${WORKER_MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiments/results/_serving_logs}"

mkdir -p "$LOG_DIR"

# Compute nodes have no internet, and login-shell variables are not inherited
# into Snellius jobs, so the cache is pinned here rather than assumed.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
mkdir -p "$HF_HOME"

# flashinfer JIT-compiles its sampling kernels on first use. torch here is built
# against CUDA 13.0 while the newest module on the system is 12.9.1, so there is
# no matched toolchain to compile them with. With none at all the server loads
# the model, captures CUDA graphs, and only then dies on the first sampled token.
# The PyTorch sampler is numerically equivalent and marginally slower.
export VLLM_USE_FLASHINFER_SAMPLER=0

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
    local path="${7:-}" parser="${8:-}"
    local extra=()
    [[ -n "$parser" ]] && extra+=(--reasoning-parser "$parser")
    echo "launching $role: $model on GPUs [$gpus], tp=$tp, port=$port${parser:+, reasoning parser $parser}"
    # vLLM 0.28 replaced --guided-decoding-backend with --structured-outputs-config,
    # which takes JSON. Passing the old flag aborts the server at startup.
    # A path is served under the canonical name, so the identity check still matches.
    CUDA_VISIBLE_DEVICES="$gpus" \
    "$VLLM_BIN" serve "${path:-$model}" \
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
        "${extra[@]}" \
        > "$LOG_DIR/${role}_${port}.log" 2>&1 &
    VLLM_PIDS+=("$!")
}

# Worker takes the low-numbered GPUs, meta the rest.
worker_gpus="$(seq -s, 0 $((WORKER_TP - 1)))"
serve worker "$WORKER_MODEL" "$WORKER_PORT" "$WORKER_TP" "$WORKER_MAX_LEN" "$worker_gpus" \
    "${WORKER_MODEL_PATH:-}" "${WORKER_REASONING_PARSER:-}"

health_args=(--worker-url "http://127.0.0.1:${WORKER_PORT}/v1" --expect-worker-model "$WORKER_MODEL")
if [[ -n "$META_MODEL" ]]; then
    meta_gpus="$(seq -s, "$WORKER_TP" $((WORKER_TP + META_TP - 1)))"
    serve meta "$META_MODEL" "$META_PORT" "$META_TP" "$META_MAX_LEN" "$meta_gpus" \
        "${META_MODEL_PATH:-}" "${META_REASONING_PARSER:-}"
    health_args+=(--meta-url "http://127.0.0.1:${META_PORT}/v1" --expect-meta-model "$META_MODEL")
fi

# The client only ever sees a health timeout, so the actual cause -- OOM, a bad
# flag, a JIT compile failure -- is only in the server's own log.
if ! "$ADAS_PYTHON" "$REPO_ROOT/experiments/serving/health_check.py" \
        "${health_args[@]}" \
        --timeout "${HEALTH_TIMEOUT:-2400}" \
        --check-json \
        ${CHECK_REASONING_EFFORT:+--check-reasoning-effort}; then
    for log in "$LOG_DIR"/worker_"$WORKER_PORT".log "$LOG_DIR"/meta_"$META_PORT".log; do
        [[ -f "$log" ]] || continue
        echo "--- tail of $log ---" >&2
        tail -60 "$log" >&2
    done
    exit 1
fi

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
