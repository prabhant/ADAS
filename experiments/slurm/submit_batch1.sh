#!/bin/bash
# Submit batch 1 and record every job id, so the batch can be inspected after a
# dropped connection with experiments/slurm/status.sh.
#
#   bash experiments/slurm/submit_batch1.sh
#
# Refuses to run twice: resubmitting after a reconnect would double the cost.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p experiments/results/_slurm

BATCH="experiments/results/_slurm/batch1.tsv"
if [[ -s "$BATCH" && "${FORCE:-0}" != "1" ]]; then
    echo "batch 1 was already submitted (see $BATCH); set FORCE=1 to submit again" >&2
    exit 1
fi
: > "$BATCH"

COMMIT="$(git rev-parse --short HEAD)"
[[ -z "$(git status --porcelain -- _mgsm _mmlu experiments/adas_ext experiments/analysis experiments/serving experiments/slurm experiments/config)" ]] \
    || { echo "commit code changes first: results must map to a commit" >&2; exit 1; }

submit() {
    local label="$1"; shift
    local jid
    jid="$(sbatch --parsable "$@" 2>/dev/null | cut -d';' -f1)"
    printf '%s\t%s\n' "$label" "$jid" >> "$BATCH"
    printf '  %-16s %s\n' "$label" "$jid"
}

echo "submitting batch 1 at commit $COMMIT"
submit e2e_cpu        experiments/slurm/sanity/07_e2e_cpu.slurm
submit stage0_smoke   experiments/slurm/stage0_smoke.slurm
submit worker_screen  --export=ALL,STAGE=0.5s --array=0-14 experiments/slurm/stage05_baselines.slurm
submit designers_1gpu --export=ALL,SCREEN_GROUP=1gpu --array=0-7 experiments/slurm/screen_designers.slurm
submit designers_2gpu --export=ALL,SCREEN_GROUP=2gpu --gpus=2 --cpus-per-task=32 --time=01:30:00 \
                      --array=0-3 experiments/slurm/screen_designers.slurm

{
    echo
    echo "### $(date '+%Y-%m-%d %H:%M') - batch 1 submitted at commit $COMMIT"
    echo
    echo '```'
    cat "$BATCH"
    echo '```'
    echo
    echo "Inspect with: bash experiments/slurm/status.sh"
} >> RUNLOG.md
echo "recorded in $BATCH and RUNLOG.md"
