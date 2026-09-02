#!/bin/bash
#SBATCH --job-name=adas_s0_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=experiments/results/_slurm/s0_smoke_%j.out
#SBATCH --error=experiments/results/_slurm/s0_smoke_%j.err

# Stage 0: plumbing smoke test.
#
# Two generations, eight validation and eight test questions. This exists to
# prove the pipeline end to end -- role routing, JSON mode, guided decoding, the
# sandbox, instrumentation, resume -- before any expensive run.
#
# Nothing about accuracy is interpretable at this size and nothing here should
# be reported. For a faster and cheaper version of the same idea, see
# experiments/slurm/sanity/, which does not need a full stage configuration.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
activate_env

load_cell "s0_smoke"

# Both servers, disjoint GPUs, then block until healthy.
CHECK_REASONING_EFFORT=1 \
    source "$REPO_ROOT/experiments/serving/launch_vllm.sh"

# Record what the health check found: whether guided decoding works, whether
# json_object mode is honoured, and whether reasoning_effort is load-bearing
# rather than merely accepted.
python "$REPO_ROOT/experiments/serving/health_check.py" \
    --worker-url "$ADAS_WORKER_BASE_URL" \
    --meta-url "$ADAS_META_BASE_URL" \
    --check-json --check-reasoning-effort \
    --report "$RESULTS_ROOT/$CELL_ID/health_report.json"

# Sandbox guards must actually hold before untrusted code is executed at scale.
# Run as a module: the guards are installed in a "spawn" child, which re-imports
# __main__, so a heredoc parent cannot drive this.
python -W ignore::RuntimeWarning -m experiments.adas_ext.sandbox

run_search

echo "stage 0 complete: inspect $RESULTS_ROOT/$CELL_ID"
