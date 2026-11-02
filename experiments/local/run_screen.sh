#!/usr/bin/env bash
#
# run_screen.sh
# Phase 1 screen of the Copilot arm (experimental_plan_copilot.md): every
# worker x domain cell, one at a time, because the rate limit is per account.
# CoT first, then the seed agents, then bare. Finished cells are skipped, and
# the loop stops if the gateway halts on its credit cap (runner exit code 3).
#
#   bash experiments/local/run_screen.sh
#   SCREEN_DOMAINS="gpqa" SCREEN_WORKERS="gpt-6-luna" bash experiments/local/run_screen.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${ADAS_PYTHON:-.venv/bin/python}"
RESULTS="${ADAS_COPILOT_RESULTS_ROOT:-experiments/results_copilot}"
WORKERS="${SCREEN_WORKERS:-claude-haiku-5.5 gpt-6-luna gpt-5.6-luna}"
DOMAINS="${SCREEN_DOMAINS:-mgsm mmlu drop gpqa}"

cell_done() {
    local cell="$1" kind="$2"
    case "${kind}" in
        cot) [[ -f "${RESULTS}/${cell}/cot_baseline.json" ]] ;;
        bare) [[ -f "${RESULTS}/${cell}/bare_baseline.json" ]] ;;
        seeds)
            [[ -f "${RESULTS}/${cell}/${cell}_run_archive.json" ]] || return 1
            "${PYTHON}" -c '
import json, sys
archive = json.load(open(sys.argv[1]))
sys.exit(0 if len(archive) == 7 and all("fitness" in e for e in archive) else 1)
' "${RESULTS}/${cell}/${cell}_run_archive.json"
            ;;
    esac
}

main() {
    local kind domain worker cell status
    # One assertion for the whole screen, so the Mac cannot sleep between cells.
    if command -v caffeinate >/dev/null 2>&1; then
        caffeinate -s -i -m -w "$$" &
    fi
    for kind in cot seeds bare; do
        for domain in ${DOMAINS}; do
            for worker in ${WORKERS}; do
                cell="cp1_${kind}_${domain}_${worker}"
                if cell_done "${cell}" "${kind}"; then
                    echo "$(date '+%H:%M:%S') skip ${cell} (done)"
                    continue
                fi
                echo "$(date '+%H:%M:%S') run  ${cell}"
                status=0
                "${PYTHON}" experiments/local/run_cell.py "${cell}" --quiet || status=$?
                if (( status == 3 )); then
                    echo "gateway halted on ${cell}; stopping the screen"
                    return 3
                fi
                (( status == 0 )) || echo "WARNING: ${cell} exited ${status}"
            done
        done
    done
    echo "$(date '+%H:%M:%S') screen complete"
}

main "$@"
