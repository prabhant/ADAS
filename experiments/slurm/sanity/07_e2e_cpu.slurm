#!/bin/bash
#SBATCH --job-name=adas_sanity_07_e2e
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=experiments/results/_slurm/sanity_07_e2e_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_07_e2e_%j.err

# Sanity 07: the CPU end-to-end test of the search loop and the analysis layer,
# against a scripted stub model with known answers. Run as a job rather than in
# a terminal so a dropped connection cannot kill it halfway.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
activate_env

python "$REPO_ROOT/experiments/tests/search_e2e.py"
