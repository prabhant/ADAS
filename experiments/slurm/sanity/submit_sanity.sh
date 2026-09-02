#!/bin/bash
# Submit the sanity checks, in dependency order.
#
# Run from the repository root, on a login node:
#   ./experiments/slurm/sanity/submit_sanity.sh          # 00-03, the fast set
#   ./experiments/slurm/sanity/submit_sanity.sh --with-120b
#
# SLURM will not create the directories behind --output and --error, and a job
# whose output path does not exist fails at submission with a message that does
# not say so. They are created here.
#
# Jobs are chained with --dependency=afterok so a failure stops the chain rather
# than burning GPU hours on a configuration already known to be broken. Snellius
# does not inherit login-shell variables into jobs, so anything a job needs is
# passed explicitly through --export.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p experiments/results/_slurm experiments/results/_sanity

WITH_120B=0
[[ "${1:-}" == "--with-120b" ]] && WITH_120B=1

EXPORTS="ALL"
[[ -n "${ADAS_PYTHON:-}" ]] && EXPORTS="$EXPORTS,ADAS_PYTHON=$ADAS_PYTHON"

submit() {
    local script="$1" dep="${2:-}"
    local args=(--export="$EXPORTS" --chdir="$REPO_ROOT")
    [[ -n "$dep" ]] && args+=(--dependency=afterok:"$dep")
    sbatch "${args[@]}" "$script" | awk '{print $NF}'
}

echo "submitting from $REPO_ROOT"

JOB00=$(submit experiments/slurm/sanity/00_cpu_env.slurm)
echo "  00 cpu env         $JOB00   (rome, ~2 min, SBU weight 1)"

JOB01=$(submit experiments/slurm/sanity/01_worker_load.slurm "$JOB00")
echo "  01 worker load     $JOB01   (gpu_h100 x1, ~4 min)  after $JOB00"

JOB02=$(submit experiments/slurm/sanity/02_meta_worker.slurm "$JOB01")
echo "  02 two roles       $JOB02   (gpu_h100 x2, ~5 min)  after $JOB01"

JOB03=$(submit experiments/slurm/sanity/03_mini_search.slurm "$JOB02")
echo "  03 mini search     $JOB03   (gpu_h100 x2, ~8 min)  after $JOB02"

if [[ "$WITH_120B" == "1" ]]; then
    JOB04=$(submit experiments/slurm/sanity/04_meta120_probe.slurm "$JOB02")
    echo "  04 gpt-oss-120b    $JOB04   (gpu_h100 x3, ~30 min) after $JOB02"
else
    echo
    echo "  04 gpt-oss-120b    skipped. It cannot finish in five minutes -- loading"
    echo "     a 183GB repo dominates the runtime. Re-run with --with-120b to settle"
    echo "     whether the top meta rung is schedulable beside a worker."
fi

echo
echo "watch:   squeue -u \$USER"
echo "output:  experiments/results/_slurm/"
