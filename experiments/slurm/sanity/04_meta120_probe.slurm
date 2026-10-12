#!/bin/bash
#SBATCH --job-name=adas_sanity_04_meta120
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus=3
#SBATCH --time=00:45:00
#SBATCH --output=experiments/results/_slurm/sanity_04_meta120_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_04_meta120_%j.err

# Sanity 04: does gpt-oss-120b actually fit beside a worker on one node.
#
# NOT a five-minute job, and deliberately separate from 00-03 for that reason.
# The 120b repo is 183GB on disk and loading it across the shared filesystem
# dominates the runtime; the compute here is two short completions.
#
# This is the single riskiest assumption in the whole plan. models.yaml claims
# tensor_parallel_size 2 on the grounds that the mxfp4 weights are ~61GB and the
# 183GB figure is the repo total including a bf16 copy. If vLLM instead loads
# bf16, the meta agent needs three GPUs, every gpt-oss-120b cell becomes
# unschedulable beside a worker, and the top meta rung has to become
# nemotron-3-super-120b (already FP8, unambiguously 2 GPUs). Better to learn
# that from one 45-minute job than from a stage that dies at hour nineteen.
#
# Three GPUs are requested rather than two so the run survives either outcome
# and reports which one happened.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
VLLM="$(dirname "$PY")/vllm"
export PYTHONPATH="$REPO_ROOT"

command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || true
module load 2025 2>/dev/null || true
module load CUDA/12.9.1 2>/dev/null || true

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HOME="/scratch-shared/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO       # want the memory-profiling lines
export OPENAI_API_KEY=local-vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$HF_HOME"

MODEL="${SANITY_META_MODEL:-openai/gpt-oss-120b}"
TP="${SANITY_META_TP:-2}"
PORT=8001
LOG="$REPO_ROOT/experiments/results/_slurm/sanity_04_vllm_${SLURM_JOBID:-manual}.log"

cd "$REPO_ROOT"
echo "=== node $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== serving $MODEL at tensor-parallel-size $TP ==="
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))" \
"$VLLM" serve "$MODEL" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --download-dir "$HF_HUB_CACHE" \
    --tensor-parallel-size "$TP" \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    > "$LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

if ! "$PY" experiments/serving/health_check.py \
        --meta-url "http://127.0.0.1:${PORT}/v1" \
        --timeout 2400 --check-json --check-reasoning-effort \
        --report "experiments/results/_sanity/04_meta120.json"; then
    echo "--- vLLM log tail: look for OOM or a bf16 weight load ---"
    tail -80 "$LOG"
    echo
    echo "VERDICT: gpt-oss-120b does NOT serve at tp=$TP."
    echo "Switch the top meta rung to nemotron-3-super-120b in models.yaml."
    exit 1
fi

echo "=== weight format and memory footprint actually observed ==="
grep -iE 'mxfp4|quantiz|dtype|model weights take|KV cache' "$LOG" | head -20 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== reasoning effort is load-bearing, not merely accepted ==="
# Stage 2r's entire design assumes low and high produce different amounts of
# reasoning. If the two are indistinguishable, that arm measures nothing.
"$PY" - <<PYTHON
import os

from experiments.adas_ext.model_router import RoleClient

lengths = {}
for effort in ("low", "high"):
    client = RoleClient(
        role="meta", model="$MODEL",
        base_url="http://127.0.0.1:$PORT/v1", api_key="local-vllm",
        temperature=0.8, max_tokens=2048, reasoning_effort=effort,
    )
    reply = client.chat_json([
        {"role": "user", "content":
         'A farmer has 17 sheep; all but 9 run away. How many remain? '
         'Reply as JSON {"answer": "<number>", "reasoning": "<your working>"}'},
    ])
    lengths[effort] = len(str(reply.get("reasoning", "")))
    print(f"  {effort:<5} reasoning chars: {lengths[effort]}  answer: {reply.get('answer')}")

if lengths["high"] <= lengths["low"] * 1.2:
    print("  WARNING: reasoning_effort looks inert. Stage 2r would measure nothing.")
else:
    print("  reasoning_effort is load-bearing")
PYTHON

echo "=== sanity 04 PASSED: gpt-oss-120b serves at tp=$TP ==="
