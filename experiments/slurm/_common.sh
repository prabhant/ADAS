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
# Namespaced on purpose: a bare RESULTS_ROOT exported by a throwaway command in
# the submitting shell once leaked into jobs and sent their results to a temp dir.
RESULTS_ROOT="${ADAS_RESULTS_ROOT:-$REPO_ROOT/experiments/results}"
export RESULTS_ROOT
mkdir -p "$RESULTS_ROOT/_slurm"

# Snellius does not export login-shell variables into jobs, so the model cache
# has to be configured here rather than assumed. HF_HOME must stay off the
# managed cache: that path is read-only and the hub writes locks into it.
#
# Home rather than /scratch-shared: scratch has been observed deleting hours-old
# files platform-side, and the Layer 3 embedding model lives here, where an
# offline analysis job on genoa can still find it.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
mkdir -p "$HF_HOME"

# Snellius disables default modules, so anything needed at runtime must be asked
# for explicitly and by full version.
#
# Call this BEFORE activate_env. Lmod prepends to LD_LIBRARY_PATH, and
# activate_env has to end up in front of it so the conda env's libstdc++ still
# wins the CXXABI_1.3.15 fight.
#
# Version skew worth knowing: torch here is built against CUDA 13.0 and the
# newest module on the system is 12.9.1. This gives nvcc for tooling that wants
# a toolchain present, but is NOT a matched pair for JIT-compiling kernels that
# link against torch -- which is why the flashinfer sampler stays off in
# launch_vllm.sh rather than being re-enabled now that nvcc exists.
load_gpu_modules() {
    if ! command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh 2>/dev/null || return 0
    fi
    module load "${ADAS_MODULE_YEAR:-2025}" 2>/dev/null || true
    module load "${ADAS_CUDA_MODULE:-CUDA/12.9.1}" 2>/dev/null || true
    echo "modules: $(module -t list 2>&1 | grep -v '^Currently Loaded' | tr '\n' ' ')"
    echo "nvcc:    $(command -v nvcc || echo 'not available')"
}

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
    # Preflight the whole import graph, not just three packages: a broken repo
    # state should fail in seconds here rather than after a model has loaded.
    python -c "
import openai, tqdm, yaml, numpy, pandas
import experiments.adas_ext, experiments.analysis.extract_features
import sqlite3  # xgrammar's structured-output backend depends on it
" || { echo "environment failed import preflight; run experiments/env/setup_env.sh" >&2; exit 1; }
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
    emit(f"{prefix}_MODEL_PATH", spec.get("model_path", ""))
    emit(f"{prefix}_SERVE_ARGS", spec.get("serve_args", ""))
    emit(f"{prefix}_TP", spec["tensor_parallel_size"])
    emit(f"{prefix}_MAX_LEN", spec["max_model_len"])
    emit(f"{prefix}_MAX_TOKENS", spec["max_output_tokens"])
    # Workers keep the setup their Stage 0.5 baselines were measured under.
    emit(f"{prefix}_REASONING_PARSER",
         spec.get("meta_reasoning_parser", "") if prefix == "META" else "")
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
#
# Reasoning effort is part of the worker's identity: gpt-oss-20b at low and high
# effort are different workers and have different baselines, so the effort-
# specific directory is preferred when one exists.
worker_cot_baseline() {
    # MGSM cells predate other domains and carry no domain in their id.
    local tag="" candidates=()
    [[ "${DOMAIN:-mgsm}" != "mgsm" ]] && tag="${DOMAIN}_"
    [[ -n "${WORKER_REASONING_EFFORT:-}" ]] \
        && candidates+=("$RESULTS_ROOT/s05_cot_${tag}${WORKER_KEY}_${WORKER_REASONING_EFFORT}/cot_baseline.json")
    candidates+=("$RESULTS_ROOT/s05_cot_${tag}${WORKER_KEY}/cot_baseline.json")
    local path
    for path in "${candidates[@]}"; do
        if [[ -f "$path" ]]; then
            python -c "
import json, sys
d = json.load(open(sys.argv[1]))
if d.get('domain') != sys.argv[2]:
    sys.exit(f\"{sys.argv[1]} is a {d.get('domain')} baseline; refusing it for {sys.argv[2]}\")
print(d['accuracy'])" "$path" "${DOMAIN:-mgsm}"
            return
        fi
    done
    return 0
}

# Assemble the argument list common to every search invocation.
search_args() {    local out_dir="$RESULTS_ROOT/$CELL_ID"
    mkdir -p "$out_dir"

    local args=(
        --domain "$DOMAIN"
        --save_dir "$out_dir"
        --expr_name "$CELL_ID"
        --run_id "$CELL_ID"
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

    # Baseline and random-recombination cells have no meta model. Emitting an
    # empty --model would hand the meta client a blank model name rather than
    # leaving it unused.
    [[ -n "${META_MODEL:-}" ]] && args+=(--model "$META_MODEL")
    [[ "${INFORMED_META:-1}" == "1" ]] && args+=(--informed_meta) || args+=(--blind_meta)
    [[ -n "${WORKER_REASONING_EFFORT:-}" ]] && args+=(--worker_reasoning_effort "$WORKER_REASONING_EFFORT")
    [[ -n "${META_REASONING_EFFORT:-}" ]] && args+=(--meta_reasoning_effort "$META_REASONING_EFFORT")

    local baseline
    baseline="$(worker_cot_baseline || true)"
    if [[ -n "$baseline" ]]; then
        args+=(--worker_cot_accuracy "$baseline")
    elif [[ "${INFORMED_META:-1}" == "1" && -n "${META_MODEL:-}" && "${ALLOW_MISSING_BASELINE:-0}" != "1" ]]; then
        # Without this the run would silently proceed as "informed minus the
        # accuracy number", quietly invalidating the Stage 1b informed/blind
        # contrast while still producing a healthy-looking archive.
        echo "no Stage 0.5 baseline for worker '${WORKER_KEY}${WORKER_REASONING_EFFORT:+ (${WORKER_REASONING_EFFORT})}'." >&2
        echo "run experiments/slurm/stage05_baselines.slurm first, or set ALLOW_MISSING_BASELINE=1 to accept a weaker informed condition." >&2
        exit 1
    fi

    case "${CELL_CONDITION:-}" in
        bare_model)           args+=(--baseline bare) ;;
        solo_cot)             args+=(--baseline cot) ;;
        random_recombination) args+=(--search_mode random) ;;
        no_reflection)        args+=(--no_reflection) ;;
        empty_archive)        args+=(--init_archive empty) ;;
    esac

    local deadline
    deadline="$(_search_deadline_epoch || true)"
    [[ -n "$deadline" ]] && args+=(--deadline_epoch "$deadline")

    printf '%s\n' "${args[@]}"
}

# Unix time at which the search should stop starting new generations. A whole
# generation must fit inside the margin, so it is generous by default.
_search_deadline_epoch() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
    local end margin
    margin="${ADAS_WALLTIME_MARGIN_SECONDS:-1800}"
    end="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
           | grep -oE 'EndTime=[^ ]+' | head -1 | cut -d= -f2)"
    [[ -n "$end" ]] || return 1
    local epoch
    epoch="$(date -d "$end" +%s 2>/dev/null)" || return 1
    echo $(( epoch - margin ))
}

# Continue a cell that stopped on the walltime deadline rather than on its
# generation budget. The archive is written after every generation, so the new
# job resumes from it instead of restarting.
maybe_resubmit() {
    local summary="$RESULTS_ROOT/$CELL_ID/$CELL_ID/budget_summary.json"
    [[ -f "$summary" ]] || return 0
    python - "$summary" <<'PY' || return 0
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("stopped_on_deadline") else 1)
PY
    local script="${ADAS_SELF_SCRIPT:-}"
    if [[ -z "$script" ]]; then
        echo "$CELL_ID stopped on the walltime deadline but ADAS_SELF_SCRIPT is unset; resubmit by hand" >&2
        return 0
    fi
    local opts=(--chdir="$REPO_ROOT")
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && opts+=(--array="$SLURM_ARRAY_TASK_ID")
    echo "$CELL_ID hit the walltime deadline; resubmitting to continue"
    sbatch "${opts[@]}" "$script" ${STAGE:+"$STAGE"}
}

run_search() {
    mapfile -t args < <(search_args)
    ( cd "$REPO_ROOT/_${DOMAIN}" && python search.py "${args[@]}" )
}
