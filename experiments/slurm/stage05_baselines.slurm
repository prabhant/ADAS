#!/bin/bash
#SBATCH --job-name=adas_s05_baseline
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=experiments/results/_slurm/s05_baseline_%A_%a.out
#SBATCH --error=experiments/results/_slurm/s05_baseline_%A_%a.err

# Stage 0.5: the worker measured alone, with and without chain-of-thought.
#
# Blocks every later phase, for two separate reasons:
#
#   1. The solo-CoT number is what the informed meta agent is told its worker
#      already achieves. Without it the "informed" condition is informed minus
#      the one fact that matters, and the Stage 1b informed/blind contrast
#      measures nothing. _common.sh now refuses to start a search when the
#      baseline is missing rather than degrading quietly.
#   2. The bare-model number is the floor for Q1 and the leftmost point of the
#      Pareto front. A searched agent that does not beat it is not evidence of
#      anything.
#
# No search and no meta model: only the worker is served, so this is a two-GPU
# job at most and the eight cells run as one array.
#
#   sbatch --array=0-7 experiments/slurm/stage05_baselines.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
activate_env

load_cell "stage_index=0.5:${SLURM_ARRAY_TASK_ID}"

if [[ -z "${CELL_CONDITION:-}" || ( "$CELL_CONDITION" != "bare_model" && "$CELL_CONDITION" != "solo_cot" ) ]]; then
    echo "cell $CELL_ID has condition '${CELL_CONDITION:-none}', expected bare_model or solo_cot" >&2
    exit 1
fi

# Nothing here reads a baseline, and requiring one would deadlock: this is the
# stage that produces them.
export ALLOW_MISSING_BASELINE=1

# Worker only. Passing an empty META_MODEL stops launch_vllm.sh starting a
# second server, and the meta URL is pointed at the worker so the argument list
# stays uniform.
META_MODEL="" source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
export ADAS_META_BASE_URL="$ADAS_WORKER_BASE_URL"

run_search

out="$RESULTS_ROOT/$CELL_ID/cot_baseline.json"
if [[ "$CELL_CONDITION" == "solo_cot" ]]; then
    [[ -f "$out" ]] || { echo "solo_cot cell produced no $out" >&2; exit 1; }
    python -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f\"  {d['name']} on {d['worker_model']}: {d['accuracy']:.4f}  ({d['fitness']})\")
" "$out"
fi

echo "stage 0.5 cell $CELL_ID complete"
