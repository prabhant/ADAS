#!/bin/bash
# Shared helpers for the stage SLURM scripts.
#
# Sourced, never executed. Resolves a cell id from experiments/config/cells.yaml
# into the environment variables the stage scripts and launch_vllm.sh expect, so
# the grid is edited in one place (YAML) rather than in six shell scripts.
set -euo pipefail

# SLURM_SUBMIT_DIR is the reliable anchor inside a job: BASH_SOURCE is resolved
# against a copy of the script in SLURM's spool directory on some configurations.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="$REPO_ROOT/experiments/config"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/experiments/results}"
export RESULTS_ROOT
mkdir -p "$RESULTS_ROOT/_slurm"

# Snellius does not export login-shell variables into jobs, so the model cache
# has to be configured here rather than assumed. HF_HOME must stay off the
# managed cache: that path is read-only and the hub writes locks into it.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}"
export HF_HOME="${HF_HOME:-/scratch-shared/$USER/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
mkdir -p "$HF_HOME"

activate_env() {
    # Absolute interpreter first: `conda activate` needs shell hooks that a
    # non-interactive job shell has not sourced, and failing over to the system
    # python would give a confusing ImportError instead of a clear message.
    local candidate="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
    if [[ -x "$candidate" ]]; then
        export PATH="$(dirname "$candidate"):$PATH"
        # Snellius' system libstdc++ tops out at CXXABI_1.3.13 while the conda
        # env's ICU needs 1.3.15. Without this, importing sqlite3 fails, which
        # takes the xgrammar structured-output backend down with it.
        export LD_LIBRARY_PATH="$(dirname "$(dirname "$candidate")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.venv/bin/activate"
    elif [[ -n "${ADAS_CONDA_ENV:-}" ]]; then
        # shellcheck disable=SC1091
        source activate "$ADAS_CONDA_ENV"
    else
        echo "no usable python; set ADAS_PYTHON or run experiments/env/setup_env.sh" >&2
        exit 1
    fi
    python -c "import openai, tqdm, yaml" \
        || { echo "environment is missing dependencies; run experiments/env/setup_env.sh" >&2; exit 1; }
    echo "python: $(command -v python) ($(python --version 2>&1))"
}

# Emit `KEY=value` lines for one cell: CELL_*, META_*, WORKER_* and the stage
# defaults merged with per-cell overrides.
_resolve() {
    python - "$1" <<'PYTHON'
import shlex
import sys
import os

import yaml

config_dir = os.path.join(os.environ["REPO_ROOT"], "experiments", "config")
with open(os.path.join(config_dir, "cells.yaml")) as handle:
    cells_config = yaml.safe_load(handle)
with open(os.path.join(config_dir, "models.yaml")) as handle:
    registry = yaml.safe_load(handle)

target = sys.argv[1]
selector, _, value = target.partition("=")
if selector == "array_index":
    cell = next((c for c in cells_config["cells"] if str(c.get("array_index")) == value), None)
elif selector == "stage_index":
    stage, index = value.split(":")
    matching = [c for c in cells_config["cells"] if str(c.get("stage")) == stage]
    cell = matching[int(index)] if int(index) < len(matching) else None
else:
    cell = next((c for c in cells_config["cells"] if c["id"] == target), None)

if cell is None:
    sys.stderr.write(f"no cell matching {target!r}\n")
    raise SystemExit(2)

stage = str(cell["stage"])
merged = dict(cells_config["common"])
merged.update(cells_config["stage_defaults"].get(stage, {}))
merged.update({k: v for k, v in cell.items() if v is not None})

def emit(key, value):
    if value is None or value is True:
        value = "1" if value is True else ""
    elif value is False:
        value = "0"
    print(f"{key}={shlex.quote(str(value))}")

emit("CELL_ID", cell["id"])
emit("CELL_STAGE", stage)
emit("CELL_SEED", cell.get("seed", "a"))
emit("CELL_CONDITION", cell.get("condition", ""))
emit("DOMAIN", merged.get("domain", "mgsm"))

for name, prefix in (("meta", "META"), ("worker", "WORKER")):
    key = cell.get(name)
    if not key:
        emit(f"{prefix}_MODEL", "")
        continue
    spec = registry["models"][key]
    emit(f"{prefix}_KEY", key)
    emit(f"{prefix}_MODEL", spec["hf_id"])
    emit(f"{prefix}_TP", spec["tensor_parallel_size"])
    emit(f"{prefix}_MAX_LEN", spec["max_model_len"])
    emit(f"{prefix}_MAX_TOKENS", spec["max_output_tokens"])
    if spec.get("stage_to_node_nvme"):
        emit("STAGE_TO_NVME", 1)

emit("N_GENERATION", merged.get("n_generation", 10))
emit("VALID_SIZE", merged.get("valid_size", 128))
emit("TEST_SIZE", merged.get("test_size", 300))
emit("N_REPEAT", merged.get("n_repreat", 1))
emit("MAX_ATTEMPTS", merged.get("max_attempts", 3 * int(merged.get("n_generation", 10) or 1)))
emit("DEBUG_MAX", merged.get("debug_max", 3))
emit("MAX_WORKERS", merged.get("max_workers", 192))
emit("SHUFFLE_SEED", merged.get("shuffle_seed", 0))
emit("META_TEMPERATURE", merged.get("meta_temperature", 0.8))
emit("WORKER_TEMPERATURE", merged.get("worker_temperature", 0.5))
emit("INFORMED_META", merged.get("informed_meta", True))
emit("COST_LAMBDA", merged.get("cost_lambda", 0.0))
emit("WORKER_REASONING_EFFORT", cell.get("worker_reasoning_effort", ""))
emit("META_REASONING_EFFORT", cell.get("meta_reasoning_effort", ""))
PYTHON
}

load_cell() {
    local resolved
    resolved="$(_resolve "$1")"
    while IFS= read -r line; do
        [[ -n "$line" ]] && eval "export $line"
    done <<< "$resolved"
    echo "cell: $CELL_ID  stage=$CELL_STAGE  meta=${META_MODEL:-none}  worker=$WORKER_MODEL"
}

# Measured solo-CoT accuracy from Stage 0.5, fed to the meta prompt so it is
# calibrating against its actual worker rather than against gpt-3.5.
worker_cot_baseline() {
    local path="$RESULTS_ROOT/s05_cot_${WORKER_KEY}/cot_baseline.json"
    [[ -f "$path" ]] || return 0
    python -c "import json,sys; print(json.load(open(sys.argv[1]))['accuracy'])" "$path"
}

# Assemble the argument list common to every search invocation.
search_args() {
    local out_dir="$RESULTS_ROOT/$CELL_ID"
    mkdir -p "$out_dir"

    local args=(
        --domain "$DOMAIN"
        --save_dir "$out_dir"
        --expr_name "$CELL_ID"
        --run_id "$CELL_ID"
        --model "${META_MODEL:-}"
        --worker_model "$WORKER_MODEL"
        --meta_base_url "$ADAS_META_BASE_URL"
        --worker_base_url "$ADAS_WORKER_BASE_URL"
        --meta_max_tokens "${META_MAX_TOKENS:-8192}"
        --worker_max_tokens "${WORKER_MAX_TOKENS:-4096}"
        --meta_temperature "$META_TEMPERATURE"
        --worker_temperature "$WORKER_TEMPERATURE"
        --n_generation "$N_GENERATION"
        --target_generations "$N_GENERATION"
        --max_attempts "$MAX_ATTEMPTS"
        --debug_max "$DEBUG_MAX"
        --valid_size "$VALID_SIZE"
        --test_size "$TEST_SIZE"
        --n_repreat "$N_REPEAT"
        --max_workers "$MAX_WORKERS"
        --shuffle_seed "$SHUFFLE_SEED"
        --cost_lambda "$COST_LAMBDA"
    )

    [[ "${INFORMED_META:-1}" == "1" ]] && args+=(--informed_meta) || args+=(--blind_meta)
    [[ -n "${WORKER_REASONING_EFFORT:-}" ]] && args+=(--worker_reasoning_effort "$WORKER_REASONING_EFFORT")
    [[ -n "${META_REASONING_EFFORT:-}" ]] && args+=(--meta_reasoning_effort "$META_REASONING_EFFORT")

    local baseline
    baseline="$(worker_cot_baseline || true)"
    [[ -n "$baseline" ]] && args+=(--worker_cot_accuracy "$baseline")

    case "${CELL_CONDITION:-}" in
        random_recombination) args+=(--search_mode random) ;;
        no_reflection)        args+=(--no_reflection) ;;
        empty_archive)        args+=(--init_archive empty) ;;
    esac

    printf '%s\n' "${args[@]}"
}

run_search() {
    mapfile -t args < <(search_args)
    ( cd "$REPO_ROOT/_${DOMAIN}" && python search.py "${args[@]}" )
}
