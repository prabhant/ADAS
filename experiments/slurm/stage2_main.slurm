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
# conditions and the cost-penalised arm. Snellius does not inherit login-shell
# variables into jobs, so STAGE has to travel through --export:
#
#   sbatch --array=0-15 experiments/slurm/stage2_main.slurm
#   sbatch --export=ALL,STAGE=2r --array=0-3 experiments/slurm/stage2_main.slurm
#   sbatch --export=ALL,STAGE=2d --array=0-5 experiments/slurm/stage2_main.slurm
#   sbatch --export=ALL,STAGE=1b --array=0-3 experiments/slurm/stage2_main.slurm
#
# The reference conditions are not optional. Random recombination and the
# no-reflection arm are what distinguish "the meta agent is designing" from
# "sampling from the archive with a mutation operator"; without them a cross-cell
# accuracy difference has no interpretation.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
load_gpu_modules
activate_env

STAGE="${STAGE:-2}"
load_cell "stage_index=${STAGE}:${SLURM_ARRAY_TASK_ID}"
export ADAS_SELF_SCRIPT="experiments/slurm/stage2_main.slurm"

# 20 generations at 128 validation questions will not always fit in one 20h
# block. --requeue plus the existing resume logic chunks it against the 120h
# ceiling without new code.
if [[ -f "$RESULTS_ROOT/$CELL_ID/${CELL_ID}_run_archive.json" ]]; then
    echo "resuming $CELL_ID from existing archive"
fi

# The random-recombination arm needs no meta server at all; skip loading it.
if [[ "${CELL_CONDITION:-}" == "random_recombination" ]]; then
    META_MODEL="" source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
    export ADAS_META_BASE_URL="$ADAS_WORKER_BASE_URL"
else
    source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
fi

run_search
maybe_resubmit

echo "stage $STAGE cell $CELL_ID complete"
