#!/bin/bash
#SBATCH --job-name=adas_sanity_06_base
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=experiments/results/_slurm/sanity_06_baseline_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_06_baseline_%j.err

# Sanity 06: the Stage 0.5 baseline path, end to end, at toy scale.
#
# Stage 0.5 blocks every later phase and its code has never executed. Rather
# than test a copy of it, this drives the real thing -- _common.sh resolves the
# real cell, launch_vllm.sh serves the real worker, run_search invokes the real
# baseline path -- with only the question count overridden.
#
# One GPU and a quarter-hour cap so it backfills quickly. Four questions prove
# nothing about accuracy and are not meant to; what is being tested is that the
# path runs, writes cot_baseline.json in the shape _common.sh expects, and that
# the value is then actually picked up by a downstream cell.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
source experiments/slurm/_common.sh
load_gpu_modules
activate_env

# Keep the toy result out of the real results tree: a 4-question baseline left
# where a later stage could read it would silently corrupt the informed prompt.
export RESULTS_ROOT="$REPO_ROOT/experiments/results/_sanity/s06"
rm -rf "$RESULTS_ROOT"
mkdir -p "$RESULTS_ROOT"

export ALLOW_MISSING_BASELINE=1

echo "=== node $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="

# s05_cot_mistral-7b: the smallest worker, so load time does not dominate.
load_cell "s05_cot_mistral-7b"
[[ "$CELL_CONDITION" == "solo_cot" ]] || { echo "expected solo_cot, got $CELL_CONDITION" >&2; exit 1; }

# Only the question count changes; every other argument is the real one.
export VALID_SIZE=4 N_REPEAT=1 MAX_WORKERS=4

echo "=== serving worker only ==="
META_MODEL="" source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
export ADAS_META_BASE_URL="$ADAS_WORKER_BASE_URL"

echo "=== flags this cell will actually run with ==="
search_args | tr '\n' ' '; echo

run_search

echo "=== the baseline landed where _common.sh looks for it ==="
python - <<'PYTHON'
import json
import os
import sys

path = os.path.join(os.environ["RESULTS_ROOT"], "s05_cot_mistral-7b", "cot_baseline.json")
if not os.path.exists(path):
    sys.exit(f"MISSING {path} -- every informed search would refuse to start")

data = json.load(open(path))
for key in ("accuracy", "condition", "name", "worker_model", "per_question", "fitness"):
    if key not in data:
        sys.exit(f"cot_baseline.json has no {key!r}: {sorted(data)}")

print(f"  condition   {data['condition']} ({data['name']})")
print(f"  worker      {data['worker_model']}")
print(f"  accuracy    {data['accuracy']}")
print(f"  fitness     {data['fitness']}")
print(f"  questions   {[len(rep) for rep in data['per_question']]}")
assert isinstance(data["accuracy"], (int, float)), "accuracy is not numeric"
assert 0.0 <= data["accuracy"] <= 1.0, f"accuracy out of range: {data['accuracy']}"
PYTHON

echo "=== a downstream cell now picks the number up ==="
# The whole point of Stage 0.5: this is the value quoted to the informed meta agent.
load_cell "s1_m20_w7b" >/dev/null
unset ALLOW_MISSING_BASELINE
baseline="$(worker_cot_baseline)"
[[ -n "$baseline" ]] || { echo "worker_cot_baseline found nothing; the guard would block Phase 1" >&2; exit 1; }
echo "  s1_m20_w7b would be told its worker scores $baseline"

search_args | tr '\n' ' ' | grep -o -- '--worker_cot_accuracy [0-9.]*' \
    || { echo "flag not emitted despite a baseline being present" >&2; exit 1; }
echo

echo "=== sanity 06 PASSED ==="
