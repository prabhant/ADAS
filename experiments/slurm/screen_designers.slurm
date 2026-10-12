#!/bin/bash
#SBATCH --job-name=adas_screen_designers
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=experiments/results/_slurm/screen_designers_%A_%a.out
#SBATCH --error=experiments/results/_slurm/screen_designers_%A_%a.err

# Designer screen: one candidate meta model per array task. Serves it, sends it
# the real ADAS meta prompt several times, and records whether it can design at
# all and what its first designs look like. No worker, no code execution.
#
# Two groups because #SBATCH cannot vary GPUs per array index:
#   sbatch --array=0-4 experiments/slurm/screen_designers.slurm 1gpu
#   sbatch --gpus=2 --cpus-per-task=32 --time=01:30:00 --array=0-3 \
#          experiments/slurm/screen_designers.slurm 2gpu
#
# The 2gpu group includes gpt-oss-120b at tensor-parallel 2, which also settles
# the feasibility question the old sanity 04 probe was written for.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
load_gpu_modules
activate_env

# mistral-7b-v0.3 and both phi-3 models do not load under transformers 5.x here
# (tokenizer conversion; longrope config) and are dropped.
SCREEN_GROUP="${1:-1gpu}"
case "$SCREEN_GROUP" in
    1gpu) DESIGNERS=(gpt-oss-20b qwen3.5-27b qwen3-14b qwen3-30b-a3b nemotron-3-nano-30b) ;;
    2gpu) DESIGNERS=(gpt-oss-120b nemotron-3-super-120b qwen3.5-35b-a3b mixtral-8x7b) ;;
    *) echo "unknown screen group '$SCREEN_GROUP'" >&2; exit 1 ;;
esac
KEY="${DESIGNERS[$SLURM_ARRAY_TASK_ID]}"

eval "$(python - "$KEY" <<'PY'
import os, shlex, sys, yaml
spec = yaml.safe_load(open(os.path.join(os.environ["REPO_ROOT"], "experiments/config/models.yaml")))["models"][sys.argv[1]]
for k, v in {"HF": spec["hf_id"], "TP": spec["tensor_parallel_size"], "NGPU": spec["gpus"],
             "MPATH": spec.get("model_path", ""), "PARSER": spec.get("meta_reasoning_parser", ""),
             "CTX": min(int(spec["max_model_len"]), 32768)}.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

GOT="$(nvidia-smi -L | wc -l)"
if (( NGPU > GOT )); then
    echo "$KEY needs $NGPU GPUs but this task has $GOT; submit it in the 2gpu group" >&2
    exit 1
fi

echo "=== designer $KEY ($HF) tp=$TP ctx=$CTX parser=${PARSER:-none} on $(hostname) ==="

# Served in launch_vllm.sh's single "worker" slot; the parser is the designer's,
# since in this job the model is being asked to design.
export WORKER_MODEL="$HF" WORKER_TP="$TP" WORKER_MAX_LEN="$CTX"
export WORKER_MODEL_PATH="$MPATH" WORKER_REASONING_PARSER="$PARSER"
export META_MODEL="" MAX_NUM_SEQS=16
source "$REPO_ROOT/experiments/serving/launch_vllm.sh"

echo "=== memory and weight footprint ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
grep -iE 'weights take|model loading took|KV cache|quantiz' "$LOG_DIR/worker_${WORKER_PORT}.log" | head -8 || true

OUT="$RESULTS_ROOT/screen_designers"
mkdir -p "$OUT"
python "$REPO_ROOT/experiments/serving/designer_probe.py" \
    --url "$ADAS_WORKER_BASE_URL" --model "$HF" --key "$KEY" \
    --n 6 --max-tokens 12288 --report "$OUT/$KEY.json"

echo "=== designer screen $KEY complete ==="
