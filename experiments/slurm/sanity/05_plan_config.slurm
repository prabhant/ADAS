#!/bin/bash
#SBATCH --job-name=adas_sanity_05_plan
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=experiments/results/_slurm/sanity_05_plan_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_05_plan_%j.err

# Sanity 05: does the experimental plan actually resolve into runnable jobs.
#
# Sanity 00 checked the environment. This checks the layer above it: every cell
# in cells.yaml, every stage's array range, and every flag the config layer
# generates. All of it is CPU-only and takes about a minute, against searches
# that run for twenty hours -- a flag typo caught here costs nothing, and the
# same typo caught at runtime costs a node-day.
#
# The check that earns its keep is the last one: argument lists produced by
# _common.sh are parsed by search.py's own parser. Nothing else catches drift
# between the two, and they are edited independently.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

# search_args creates each cell's output directory, so point the whole check at
# a scratch tree; inspecting the plan should not litter the real results root
# with 52 empty directories.
SANITY_RESULTS="$(mktemp -d)"
trap 'rm -rf "$SANITY_RESULTS"' EXIT
export ADAS_RESULTS_ROOT="$SANITY_RESULTS"

source experiments/slurm/_common.sh
activate_env

echo "=== node $(hostname), job ${SLURM_JOBID:-none} ==="

echo "=== every cell resolves, and array ranges match cell counts ==="
python - <<'PYTHON'
import os
import subprocess
import sys
from collections import Counter

import yaml

root = os.environ["REPO_ROOT"]
with open(os.path.join(root, "experiments", "config", "cells.yaml")) as handle:
    cells = yaml.safe_load(handle)["cells"]

by_stage = Counter(str(c["stage"]) for c in cells)
print(f"  {len(cells)} cells across {len(by_stage)} stages")
for stage, count in sorted(by_stage.items()):
    print(f"    stage {stage:<4} {count:>2} cells  ->  --array=0-{count - 1}")

# The array directives that are hard-coded in the stage scripts.
declared = {
    "0.5": ("stage05_baselines.slurm", 8),
    "1": ("stage1_pilot.slurm", 9),
}
bad = []
for stage, (script, expected) in declared.items():
    actual = by_stage.get(stage, 0)
    status = "ok" if actual == expected else "MISMATCH"
    print(f"  [{status}] {script} declares {expected}, cells.yaml has {actual}")
    if actual != expected:
        bad.append(script)
if bad:
    sys.exit(f"array ranges out of sync: {bad}")
PYTHON

echo "=== stage 0.5 covers every worker that later stages use ==="
python - <<'PYTHON'
import os
import sys

import yaml

root = os.environ["REPO_ROOT"]
with open(os.path.join(root, "experiments", "config", "cells.yaml")) as handle:
    cells = yaml.safe_load(handle)["cells"]

# A searched cell needs a solo-CoT baseline for its worker, at its reasoning
# effort. Anything missing here would stop that cell at submission time.
have = set()
for cell in cells:
    if cell.get("condition") == "solo_cot":
        have.add((cell["worker"], cell.get("worker_reasoning_effort")))

need = set()
for cell in cells:
    if str(cell["stage"]) == "0.5" or not cell.get("meta"):
        continue
    if cell.get("informed_meta") is False:
        continue
    need.add((cell["worker"], cell.get("worker_reasoning_effort")))

missing = need - have
for worker, effort in sorted(need, key=str):
    tag = f"{worker}" + (f" ({effort})" if effort else "")
    print(f"  [{'ok' if (worker, effort) in have else 'MISSING'}] baseline for {tag}")
if missing:
    sys.exit(f"searched cells whose worker has no solo_cot baseline: {sorted(missing, key=str)}")
PYTHON

echo "=== the missing-baseline guard actually fires ==="
# search_args calls `exit 1` when it refuses, which terminates the subshell, so
# the subshell's own status is the signal. Testing with `if` also stops the
# outer `set -e` treating the expected failure as a script error.
export ADAS_META_BASE_URL=http://x/v1 ADAS_WORKER_BASE_URL=http://y/v1
if ( unset ALLOW_MISSING_BASELINE
     load_cell "stage_index=2:0" >/dev/null
     search_args >/dev/null 2>"$SANITY_RESULTS/guard_err" ); then
    echo "  GUARD DID NOT FIRE: an informed search would have run with no baseline" >&2
    exit 1
fi
echo "  [ok] guard refused: $(head -1 "$SANITY_RESULTS/guard_err")"

echo "=== every cell builds a flag list that search.py accepts ==="
export ADAS_META_BASE_URL=http://x/v1 ADAS_WORKER_BASE_URL=http://y/v1
export ALLOW_MISSING_BASELINE=1
ARGS_DIR="$SANITY_RESULTS/args"
mkdir -p "$ARGS_DIR"
n=0
for cell_id in $(python -c "
import os, yaml
p = os.path.join(os.environ['REPO_ROOT'], 'experiments', 'config', 'cells.yaml')
print('\n'.join(c['id'] for c in yaml.safe_load(open(p))['cells']))
"); do
    load_cell "$cell_id" >/dev/null
    # One argument per line, matching how run_search reads them with mapfile -t.
    # Flattening onto one line would silently drop empty-valued arguments.
    search_args > "$ARGS_DIR/$cell_id"
    n=$((n + 1))
done
echo "  built argument lists for $n cells"

python - "$ARGS_DIR" <<'PYTHON'
import os
import shlex  # noqa: F401
import sys

root = os.environ["REPO_ROOT"]
sys.path.insert(0, os.path.join(root, "_mgsm"))
sys.path.insert(0, os.path.join(root, "_mmlu"))

import search as mgsm_search  # noqa: E402

parser = mgsm_search.build_parser()

expected_flag = {
    "bare_model": ("--baseline", "bare"),
    "solo_cot": ("--baseline", "cot"),
    "random_recombination": ("--search_mode", "random"),
    "no_reflection": ("--no_reflection", None),
    "empty_archive": ("--init_archive", "empty"),
}

import yaml
with open(os.path.join(root, "experiments", "config", "cells.yaml")) as handle:
    conditions = {c["id"]: c.get("condition", "") for c in yaml.safe_load(handle)["cells"]}

failures = []
checked = 0
args_dir = sys.argv[1]
for cell_id in sorted(os.listdir(args_dir)):
    with open(os.path.join(args_dir, cell_id)) as handle:
        argv = handle.read().splitlines()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        failures.append(f"{cell_id}: search.py rejected its own generated flags: {argv}")
        continue
    checked += 1

    condition = conditions.get(cell_id, "")
    if condition in expected_flag:
        flag, value = expected_flag[condition]
        if flag not in argv:
            failures.append(f"{cell_id}: condition {condition} did not emit {flag}")
        elif value is not None and argv[argv.index(flag) + 1] != value:
            failures.append(f"{cell_id}: {flag} was {argv[argv.index(flag) + 1]!r}, expected {value!r}")

    # A cell that searches needs somewhere to search to; one that does not must
    # not be given a meta model it will never call.
    is_baseline = condition in ("bare_model", "solo_cot")
    if not is_baseline and args.target_generations < 1:
        failures.append(f"{cell_id}: target_generations={args.target_generations}")
    if is_baseline and args.model:
        failures.append(f"{cell_id}: baseline cell was given a meta model {args.model!r}")

print(f"  {checked} argument lists parsed cleanly by search.py")
if failures:
    print("\n".join("  " + f for f in failures))
    sys.exit(1)
PYTHON

echo "=== sanity 05 PASSED ==="
