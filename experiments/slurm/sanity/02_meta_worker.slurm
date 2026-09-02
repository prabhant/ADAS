#!/bin/bash
#SBATCH --job-name=adas_sanity_02_roles
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --time=00:25:00
#SBATCH --output=experiments/results/_slurm/sanity_02_roles_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_02_roles_%j.err

# Sanity 02: two models, two roles, two ports, disjoint GPUs.
#
# The whole study rests on meta and worker being genuinely separate models on
# one node. This is the cheapest configuration that tests it: gpt-oss-20b as
# meta on GPU 1, Mistral-7B as worker on GPU 0. The 120b meta is deliberately
# not used here -- see 04_meta120_probe.slurm, which cannot finish in five
# minutes because of load time alone.
#
# The failure this is really hunting: both roles silently resolving to the same
# endpoint, which would look like a working run and answer none of the questions.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
export PYTHONPATH="$REPO_ROOT"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HOME="/scratch-shared/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export OPENAI_API_KEY=local-vllm
mkdir -p "$HF_HOME"

export WORKER_MODEL="${SANITY_WORKER_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
export META_MODEL="${SANITY_META_MODEL:-openai/gpt-oss-20b}"
export WORKER_TP=1 META_TP=1
export WORKER_MAX_LEN=8192 META_MAX_LEN=16384
export MAX_NUM_SEQS=32
export CHECK_REASONING_EFFORT=1
export LOG_DIR="$REPO_ROOT/experiments/results/_slurm"

cd "$REPO_ROOT"
echo "=== node $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Reuse the production launcher rather than a bespoke one, so this exercises the
# code path the real stages take.
source experiments/serving/launch_vllm.sh

echo "=== endpoints ==="
echo "  worker: $ADAS_WORKER_BASE_URL -> $WORKER_MODEL"
echo "  meta:   $ADAS_META_BASE_URL   -> $META_MODEL"

"$PY" experiments/serving/health_check.py \
    --worker-url "$ADAS_WORKER_BASE_URL" \
    --meta-url "$ADAS_META_BASE_URL" \
    --check-json --check-reasoning-effort \
    --report "experiments/results/_sanity/02_roles.json"

echo "=== the two roles are really two different models ==="
"$PY" - <<'PYTHON'
import argparse
import os

from experiments.adas_ext.model_router import build_role_clients

args = argparse.Namespace(
    model=os.environ["META_MODEL"],
    worker_model=os.environ["WORKER_MODEL"],
    meta_base_url=os.environ["ADAS_META_BASE_URL"],
    worker_base_url=os.environ["ADAS_WORKER_BASE_URL"],
    meta_temperature=0.8, worker_temperature=0.5,
    meta_max_tokens=1024, worker_max_tokens=512,
    meta_reasoning_effort=None, worker_reasoning_effort=None,
    request_timeout=600,
)
clients = build_role_clients(args)
assert clients.meta.base_url != clients.worker.base_url, "both roles hit one endpoint"
assert clients.meta.model != clients.worker.model, "both roles use one model"

for role, client in (("worker", clients.worker), ("meta", clients.meta)):
    reply = client.chat_json([
        {"role": "system", "content": 'Reply only with JSON: {"model_says": "<one word>"}'},
        {"role": "user", "content": "Say the word ready."},
    ])
    print(f"  {role:<7} {client.model:<40} -> {reply}")
print("  routing is distinct across roles")
PYTHON

echo "=== meta model emits the agent-design schema the search loop parses ==="
"$PY" - <<'PYTHON'
import os

from experiments.adas_ext.model_router import AGENT_DESIGN_SCHEMA, RoleClient

client = RoleClient(
    role="meta",
    model=os.environ["META_MODEL"],
    base_url=os.environ["ADAS_META_BASE_URL"],
    api_key="local-vllm",
    temperature=0.8,
    max_tokens=1024,
)
reply = client.chat_json(
    [{"role": "user", "content":
      "Propose a trivial agent design. Reply as JSON with keys thought, name, code. "
      "The code must be a Python function 'def forward(self, taskInfo):' returning a string."}],
    schema=AGENT_DESIGN_SCHEMA,
)
missing = [key for key in ("thought", "name", "code") if key not in reply]
assert not missing, f"meta reply missing {missing}: {reply}"
print(f"  name: {reply['name']}")
compile(reply["code"], "<candidate>", "exec")
print("  meta produced schema-valid, syntactically-parseable code")
PYTHON

echo "=== sanity 02 PASSED ==="
