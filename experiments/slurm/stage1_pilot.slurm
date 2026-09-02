#!/bin/bash
#SBATCH --job-name=adas_s1_pilot
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=20:00:00
#SBATCH --array=0-8
#SBATCH --output=experiments/results/_slurm/s1_pilot_%A_%a.out
#SBATCH --error=experiments/results/_slurm/s1_pilot_%A_%a.err

# Stage 1: the full 3x3 pilot, one seed, ten generations per cell.
#
# Array index maps to `array_index` in cells.yaml, so the grid is edited in YAML
# and never here.
#
# This stage is a redirect point, not a result. Read it for three things:
# whether cells differ at all, whether any cell is degenerate, and whether the
# measurement layers separate designs that look different by eye. If Stage 1 is
# uninformative, Stage 2 changes shape rather than simply running longer.
#
#   sbatch experiments/slurm/stage1_pilot.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
activate_env

load_cell "array_index=${SLURM_ARRAY_TASK_ID}"

# search() resumes from archive[-1]['generation'] and evaluate() resumes through
# eval_archive, so a requeued task picks up where it stopped with no extra code.
if [[ -f "$RESULTS_ROOT/$CELL_ID/${CELL_ID}_run_archive.json" ]]; then
    echo "resuming $CELL_ID from existing archive"
fi

source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
run_search

echo "stage 1 cell $CELL_ID complete"
