#!/bin/bash
#SBATCH --job-name=adas_s2_main
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=20:00:00
#SBATCH --output=experiments/results/_slurm/s2_main_%A_%a.out
#SBATCH --error=experiments/results/_slurm/s2_main_%A_%a.err

# Stage 2: main dataset.
#
# One array task per cell within a stage. STAGE selects which stage, so the same
# script covers the main grid, the reasoning-effort arm, the reference
# conditions and the cost-penalised arm. STAGE is the first positional argument:
# Snellius does not export the submitting shell, and --export=ALL would copy
# whatever happens to be set there into every task.
#
#   sbatch --array=0-15 experiments/slurm/stage2_main.slurm
#   sbatch --array=0-3 experiments/slurm/stage2_main.slurm 2r
#   sbatch --array=0-5 experiments/slurm/stage2_main.slurm 2d
#   sbatch --array=0-3 experiments/slurm/stage2_main.slurm 1b
#
# The reference conditions are not optional. Random recombination and the
# no-reflection arm are what distinguish "the meta agent is designing" from
# "sampling from the archive with a mutation operator"; without them a cross-cell
# accuracy difference has no interpretation.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
load_gpu_modules
activate_env

STAGE="${1:-2}"
load_cell "stage_index=${STAGE}:${SLURM_ARRAY_TASK_ID}"
export ADAS_SELF_SCRIPT="experiments/slurm/stage2_main.slurm"

# A cell with every generation in its archive only has test evaluation left,
# which needs no designer. Submit such a cell with the worker's GPUs only.
if search_complete; then
    export ADAS_EVALUATE_ONLY=1
    echo "$CELL_ID: search complete; test evaluation only, no meta server"
fi

# 20 generations at 128 validation questions will not always fit in one 20h
# block. --requeue plus the existing resume logic chunks it against the 120h
# ceiling without new code.
if [[ -f "$RESULTS_ROOT/$CELL_ID/${CELL_ID}_run_archive.json" ]]; then
    echo "resuming $CELL_ID from existing archive"
    preserve_run_logs
fi

# The random-recombination arm needs no meta server at all; skip loading it.
if [[ "${CELL_CONDITION:-}" == "random_recombination" || "${ADAS_EVALUATE_ONLY:-0}" == "1" ]]; then
    META_MODEL="" source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
    export ADAS_META_BASE_URL="$ADAS_WORKER_BASE_URL"
else
    source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
fi

run_search
maybe_resubmit

echo "stage $STAGE cell $CELL_ID complete"
