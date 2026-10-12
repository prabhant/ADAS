#!/bin/bash
#SBATCH --job-name=adas_sanity_01_worker
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=experiments/results/_slurm/sanity_01_worker_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_01_worker_%j.err

# Sanity 01: can one worker model be served and answer.
#
# One GPU, the smallest ladder model, a handful of tokens. The walltime is a cap
# rather than a cost -- billing follows actual runtime, and the job exits as soon
# as the checks pass, typically in three to five minutes of which most is weight
# loading from the shared filesystem.
#
# What this settles, in order of how expensive it is to discover later:
#   1. the model resolves out of the managed cache with HF_HUB_OFFLINE=1
#   2. vLLM 0.28 accepts the flags launch_vllm.sh passes
#   3. response_format json_object actually returns parseable JSON
#   4. guided decoding works, or falls back cleanly

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
VLLM="$(dirname "$PY")/vllm"
export PYTHONPATH="$REPO_ROOT"

# Default modules are disabled on Snellius; load by full version, and do it
# before setting LD_LIBRARY_PATH so the conda env stays in front of Lmod's paths.
command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || true
module load 2025 2>/dev/null || true
module load CUDA/12.9.1 2>/dev/null || true
echo "nvcc: $(command -v nvcc || echo 'not available')"

# System libstdc++ provides CXXABI_1.3.13; the conda env's ICU wants 1.3.15, so
# sqlite3 and hence the xgrammar backend fail to import without this.
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Login-shell variables are not inherited, so the cache is configured here.
# HF_HOME must not point into the managed cache: it is read-only and the hub
# writes locks into it.
export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HOME="/scratch-shared/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export OPENAI_API_KEY=local-vllm
# flashinfer JIT-compiles its sampler against torch, which is built here for CUDA
# 13.0 while the newest module is 12.9.1. With no toolchain it died after loading
# the model; with a mismatched one it would be worse. The PyTorch sampler is
# numerically equivalent and marginally slower.
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$HF_HOME"

MODEL="${SANITY_WORKER_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
PORT=8000
LOG="$REPO_ROOT/experiments/results/_slurm/sanity_01_vllm_${SLURM_JOBID:-manual}.log"

cd "$REPO_ROOT"
echo "=== node $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== serving $MODEL on port $PORT ==="
"$VLLM" serve "$MODEL" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --download-dir "$HF_HUB_CACHE" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --structured-outputs-config '{"backend": "xgrammar"}' \
    > "$LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

echo "=== waiting for /health (tail of $LOG on failure) ==="
if ! "$PY" experiments/serving/health_check.py \
        --worker-url "http://127.0.0.1:${PORT}/v1" \
        --timeout 900 --check-json \
        --report "experiments/results/_sanity/01_worker.json"; then
    echo "--- vLLM log tail ---"
    tail -60 "$LOG"
    exit 1
fi

echo "=== one real completion through the project's own client ==="
ADAS_SANITY_MODEL="$MODEL" ADAS_SANITY_URL="http://127.0.0.1:${PORT}/v1" "$PY" - <<'PYTHON'
import os

from experiments.adas_ext.model_router import RoleClient

client = RoleClient(
    role="worker",
    model=os.environ["ADAS_SANITY_MODEL"],
    base_url=os.environ["ADAS_SANITY_URL"],
    api_key="local-vllm",
    temperature=0.0,
    max_tokens=256,
)
reply = client.chat_json([
    {"role": "system", "content": "Reply only with JSON of the form {\"answer\": \"<number>\"}."},
    {"role": "user", "content": "Ada has 3 boxes with 7 pencils each. How many pencils? Reply as JSON."},
])
print("  parsed reply:", reply)
assert isinstance(reply, dict), "worker did not return a JSON object"
assert "21" in str(reply), f"expected 21 somewhere in {reply}"
print("  worker answered correctly and the JSON contract held")
PYTHON

echo "=== sanity 01 PASSED ==="
