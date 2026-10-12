#!/bin/bash
#SBATCH --job-name=adas_crosseval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --time=12:00:00
#SBATCH --output=experiments/results/_slurm/crosseval_%A_%a.out
#SBATCH --error=experiments/results/_slurm/crosseval_%A_%a.err

# Cross-evaluation: the best agent from each source cell, run against every
# worker model.
#
# This is what separates "this design suits this worker" from "this design is
# better". An agent designed for a weak worker that also wins on a strong one is
# just a good agent; one that wins only on the worker it was designed for is
# evidence of genuine adaptation, which is the actual claim.
#
# Only the best validation agent per source cell is transferred. The full
# archive-by-worker matrix would cost more than the searches themselves, and the
# best agent is where the interesting claim lives.
#
# Selection is on validation fitness alone. Selecting on test and then reporting
# test would make the whole grid circular.
#
#   sbatch --array=0-2 experiments/slurm/crosseval.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
mkdir -p "$RESULTS_ROOT/crosseval"
load_gpu_modules
activate_env

TASK="${SLURM_ARRAY_TASK_ID:-0}"

# Read the cross-evaluation plan out of cells.yaml.
eval "$(python - "$TASK" <<'PYTHON'
import os
import shlex
import sys

import yaml

with open(os.path.join(os.environ["REPO_ROOT"], "experiments", "config", "cells.yaml")) as handle:
    config = yaml.safe_load(handle)
with open(os.path.join(os.environ["REPO_ROOT"], "experiments", "config", "models.yaml")) as handle:
    registry = yaml.safe_load(handle)

plan = config["crosseval"]
worker_key = plan["target_workers"][int(sys.argv[1])]
spec = registry["models"][worker_key]

print(f"TARGET_WORKER_KEY={shlex.quote(worker_key)}")
print(f"WORKER_KEY={shlex.quote(worker_key)}")
print(f"WORKER_MODEL={shlex.quote(spec['hf_id'])}")
print(f"WORKER_TP={spec['tensor_parallel_size']}")
print(f"WORKER_MAX_LEN={spec['max_model_len']}")
print(f"WORKER_MAX_TOKENS={spec['max_output_tokens']}")
print(f"SOURCE_CELLS={shlex.quote(' '.join(plan['source_cells']))}")
print(f"DOMAIN={shlex.quote(plan.get('domain', 'mgsm'))}")
print(f"TEST_SIZE={plan.get('test_size', 300)}")
print(f"N_REPEAT={plan.get('n_repreat', 5)}")
PYTHON
)"
export WORKER_MODEL WORKER_KEY WORKER_TP WORKER_MAX_LEN WORKER_MAX_TOKENS DOMAIN

echo "cross-evaluating on worker $TARGET_WORKER_KEY"

# No meta model is needed: nothing is designed here, only executed.
META_MODEL="" source "$REPO_ROOT/experiments/serving/launch_vllm.sh"

for source_cell in $SOURCE_CELLS; do
    src_archive="$RESULTS_ROOT/$source_cell/${source_cell}_run_archive.json"
    if [[ ! -f "$src_archive" ]]; then
        echo "skipping $source_cell: no archive at $src_archive" >&2
        continue
    fi

    pair_id="${source_cell}__on__${TARGET_WORKER_KEY}"
    pair_dir="$RESULTS_ROOT/crosseval/$pair_id"
    mkdir -p "$pair_dir"

    # Slice out the single best-on-validation agent into a one-entry archive,
    # which evaluate() then treats like any other run.
    python - "$src_archive" "$pair_dir/${pair_id}_run_archive.json" "$source_cell" "$TARGET_WORKER_KEY" <<'PYTHON'
import json
import sys

src, dst, source_cell, target_worker = sys.argv[1:5]
with open(src) as handle:
    archive = json.load(handle)

designs = [e for e in archive if e.get("code") and e.get("raw_median_accuracy") is not None]
if not designs:
    raise SystemExit(f"no scored designs in {src}")

best = max(designs, key=lambda e: e["raw_median_accuracy"])
best = dict(best)
best["designed_for"] = source_cell
best["evaluated_on"] = target_worker
best["source_validation_fitness"] = best.get("raw_median_accuracy")

with open(dst, "w") as handle:
    json.dump([best], handle, indent=4)
print(f"selected {best.get('name')} (validation {best['source_validation_fitness']})")
PYTHON

    ( cd "$REPO_ROOT/_${DOMAIN}" && python search.py \
        --evaluate_only \
        --domain "$DOMAIN" \
        --save_dir "$pair_dir" \
        --expr_name "$pair_id" \
        --run_id "$pair_id" \
        --worker_model "$WORKER_MODEL" \
        --worker_base_url "$ADAS_WORKER_BASE_URL" \
        --meta_base_url "$ADAS_WORKER_BASE_URL" \
        --worker_max_tokens "$WORKER_MAX_TOKENS" \
        --test_size "$TEST_SIZE" \
        --n_repreat "$N_REPEAT" \
        --shuffle_seed 0 \
        --max_workers 192 )
done

echo "cross-evaluation on $TARGET_WORKER_KEY complete"
