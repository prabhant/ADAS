#!/bin/bash
#SBATCH --job-name=adas_s3_transfer
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=20:00:00
#SBATCH --output=experiments/results/_slurm/s3_transfer_%A_%a.out
#SBATCH --error=experiments/results/_slurm/s3_transfer_%A_%a.err

# Stage 3: corner cells on MMLU.
#
# Separates conclusions about agent design from conclusions about arithmetic.
# MGSM is short-answer maths; MMLU is multiple-choice knowledge, so the useful
# design space is plausibly different -- ensembling and verification loops help
# far less when the answer is one of four letters. A worker-capability effect
# that appears on MGSM and vanishes on MMLU is a domain effect, and that
# distinction is worth the four extra cells.
#
#   sbatch --array=0-3 experiments/slurm/stage3_transfer.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
activate_env

load_cell "stage_index=3:${SLURM_ARRAY_TASK_ID}"

if [[ "$DOMAIN" != "mmlu" ]]; then
    echo "stage 3 expects domain mmlu, got $DOMAIN" >&2
    exit 1
fi

source "$REPO_ROOT/experiments/serving/launch_vllm.sh"
run_search

echo "stage 3 cell $CELL_ID complete"
