#!/bin/bash
# Check the managed cache and download the one model that is not in it.
# LOGIN NODE ONLY -- compute nodes have no internet, and every job script sets
# HF_HUB_OFFLINE=1.
#
#   ./experiments/env/download_models.sh                # only what is missing
#   ./experiments/env/download_models.sh embedding      # one model
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANAGED_CACHE="/projects/2/managed_datasets/hf_cache_dir"
PROJECT_CACHE="${ADAS_HF_CACHE:-/projects/2/$USER/hf_cache_dir}"
ADAS_PYTHON="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$ADAS_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Every model in the grid turned out to be in the managed cache once the repo
# ids were checked against it rather than assumed. The corrections that mattered:
# there is no Qwen3-4B, the catalogue carries Qwen3.5-27B rather than a
# Qwen3-27B-Instruct, Nemotron repos are prefixed NVIDIA-, and
# Mistral-Small-3.2-24B is absent (the judge is Mixtral-8x7B-Instruct instead).
# Only the sentence-transformers embedding model still needs fetching.
declare -A SELF_DOWNLOAD=(
    [embedding]="sentence-transformers/all-MiniLM-L6-v2"
)

declare -A MANAGED=(
    [gpt-oss-20b]="openai/gpt-oss-20b"
    [gpt-oss-120b]="openai/gpt-oss-120b"
    [nemotron-3-super-120b]="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    [qwen3.5-27b]="Qwen/Qwen3.5-27B"
    [phi-4]="microsoft/phi-4"
    [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.2"
    [judge]="mistralai/Mixtral-8x7B-Instruct-v0.1"
)

mkdir -p "$PROJECT_CACHE"
export HF_HUB_CACHE="$PROJECT_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if ! command -v huggingface-cli >/dev/null 2>&1; then
    export PATH="$(dirname "$ADAS_PYTHON"):$PATH"
fi
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not on PATH. Set ADAS_PYTHON to the environment's python." >&2
    exit 1
fi

check_managed() {
    local key="$1" repo="$2"
    local dirname="models--${repo//\//--}"
    if [[ -d "$MANAGED_CACHE/$dirname" ]]; then
        echo "  [managed] $key -> $MANAGED_CACHE/$dirname"
    else
        echo "  [MISSING] $key ($repo) is not in the managed cache."
        echo "            Either download it here or open a servicedesk ticket."
    fi
}

download() {
    local key="$1" repo="$2"
    local dirname="models--${repo//\//--}"
    if [[ -d "$PROJECT_CACHE/$dirname" ]]; then
        echo "  [have] $key -> $PROJECT_CACHE/$dirname"
        return
    fi
    echo "  [get ] $key ($repo)"
    huggingface-cli download "$repo" --cache-dir "$PROJECT_CACHE" --resume-download
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
    targets=("${!SELF_DOWNLOAD[@]}")
fi

echo "Managed models (checked, not downloaded):"
for key in "${!MANAGED[@]}"; do
    check_managed "$key" "${MANAGED[$key]}"
done

echo
echo "Self-download targets into $PROJECT_CACHE:"
for key in "${targets[@]}"; do
    if [[ -n "${SELF_DOWNLOAD[$key]:-}" ]]; then
        download "$key" "${SELF_DOWNLOAD[$key]}"
    else
        echo "  [skip] $key is not a self-download target"
    fi
done

echo
echo "Set both caches in job scripts:"
echo "  export HF_HUB_CACHE=\"$MANAGED_CACHE:$PROJECT_CACHE\""
