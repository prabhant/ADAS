#!/bin/bash
# Submit a named batch and record every job id, so it can be inspected after a
# dropped connection with experiments/slurm/status.sh.
#
#   bash experiments/slurm/submit_batch.sh batch1b
#
# Refuses to resubmit a batch: doing so after a reconnect would double the cost.
# Never uses --export: jobs configure themselves, and parameters are positional.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p experiments/results/_slurm

NAME="${1:?usage: submit_batch.sh <batch name>}"
BATCH="experiments/results/_slurm/${NAME}.tsv"
if [[ -s "$BATCH" && "${FORCE:-0}" != "1" ]]; then
    echo "$NAME was already submitted (see $BATCH); set FORCE=1 to submit again" >&2
    exit 1
fi

[[ -z "$(git status --porcelain -- _mgsm _mmlu experiments/adas_ext experiments/analysis \
          experiments/serving experiments/slurm experiments/config experiments/env experiments/tests)" ]] \
    || { echo "commit code changes first: results must map to a commit" >&2; exit 1; }
COMMIT="$(git rev-parse --short HEAD)"

submit() {
    local label="$1"; shift
    local jid
    jid="$(sbatch --parsable "$@" 2>/dev/null | cut -d';' -f1)"
    [[ -n "$jid" ]] || { echo "submission of $label failed" >&2; exit 1; }
    printf '%s\t%s\n' "$label" "$jid" >> "$BATCH"
    printf '  %-16s %s\n' "$label" "$jid"
}

: > "$BATCH"
echo "submitting $NAME at commit $COMMIT"
case "$NAME" in
    batch1b)
        # Batch 1 again, for what failed or was lost; see RUNLOG.md.
        submit e2e_cpu        experiments/slurm/sanity/07_e2e_cpu.slurm
        submit stage0_smoke   experiments/slurm/stage0_smoke.slurm
        submit designers_1gpu --array=0-4 experiments/slurm/screen_designers.slurm 1gpu
        submit designers_2gpu --gpus=2 --cpus-per-task=32 --time=01:30:00 --array=0-3 \
                              experiments/slurm/screen_designers.slurm 2gpu
        ;;
    batch1c)
        # The two designers batch1b could not serve: nemotron-super (engine-ready
        # timeout during a 14-min load) and mixtral (chat template rejects system).
        submit designers_2gpu --gpus=2 --cpus-per-task=32 --time=01:30:00 --array=1,3 \
                              experiments/slurm/screen_designers.slurm 2gpu
        ;;
    batch2)
        # Production Stage 0.5 baseline for the campaign worker on MMLU (cell index 8).
        submit mmlu_baseline  --array=8 experiments/slurm/stage05_baselines.slurm 0.5
        ;;
    batch3)
        # Re-measure with the broader MMLU answer extractor; e2e covers the shared seed cache.
        submit e2e_cpu        experiments/slurm/sanity/07_e2e_cpu.slurm
        submit mmlu_baseline  --array=8 experiments/slurm/stage05_baselines.slurm 0.5
        ;;
    batch4)
        # Panel prerequisites: e2e (evaluation deadline/resume) and the shared seed archive.
        submit e2e_cpu        experiments/slurm/sanity/07_e2e_cpu.slurm
        submit panel_seeds    --array=0 experiments/slurm/stage05_baselines.slurm panel
        ;;
    batch5)
        # The designer panel. Refuses to start without what every cell needs.
        [[ -f experiments/results/s05_cot_mmlu_mistral-7b/cot_baseline.json ]] \
            || { echo "MMLU baseline missing: inspect batch3 first" >&2; rm -f "$BATCH"; exit 1; }
        compgen -G "experiments/results/_seed_cache/*__mmlu__valid128__shuffle0__rep1__*.json" >/dev/null \
            || { echo "shared seed archive missing: inspect batch4 first" >&2; rm -f "$BATCH"; exit 1; }
        submit designers_1gpu --array=1-5 --gpus=2 --cpus-per-task=32 \
                              experiments/slurm/stage2_main.slurm panel
        submit designers_2gpu --array=6-8 --gpus=3 --cpus-per-task=48 \
                              experiments/slurm/stage2_main.slurm panel
        ;;
    *)
        echo "unknown batch '$NAME'" >&2; rm -f "$BATCH"; exit 1 ;;
esac

{
    echo
    echo "### $(date '+%Y-%m-%d %H:%M') - $NAME submitted at commit $COMMIT"
    echo
    echo '```'
    cat "$BATCH"
    echo '```'
    echo
    echo "Inspect with: bash experiments/slurm/status.sh"
} >> RUNLOG.md
echo "recorded in $BATCH and RUNLOG.md"
