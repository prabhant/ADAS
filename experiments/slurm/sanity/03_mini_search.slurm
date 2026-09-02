#!/bin/bash
#SBATCH --job-name=adas_sanity_03_search
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --time=00:30:00
#SBATCH --output=experiments/results/_slurm/sanity_03_search_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_03_search_%j.err

# Sanity 03: the whole pipeline end to end, at toy scale.
#
# One generation, four validation questions, four test questions. Nothing about
# the accuracy is interpretable and nothing here should ever be reported -- the
# point is that every artifact the analysis depends on gets written, and that
# untrusted generated code executes inside the sandbox rather than beside it.
#
# Checks the artifacts by name at the end, because a search that "succeeds"
# while silently writing no call log would invalidate Layer 2 and Layer 4 and
# would not be noticed until analysis, several thousand SBU later.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
export PYTHONPATH="$REPO_ROOT"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HOME="/scratch-shared/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export OPENAI_API_KEY=local-vllm
mkdir -p "$HF_HOME"

export WORKER_MODEL="${SANITY_WORKER_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
export META_MODEL="${SANITY_META_MODEL:-openai/gpt-oss-20b}"
export WORKER_TP=1 META_TP=1
export WORKER_MAX_LEN=8192 META_MAX_LEN=16384
export MAX_NUM_SEQS=32
export LOG_DIR="$REPO_ROOT/experiments/results/_slurm"

CELL_ID="sanity_search"
OUT_DIR="$REPO_ROOT/experiments/results/$CELL_ID"

cd "$REPO_ROOT"
# Start from clean state so a stale archive cannot make a broken run look fine.
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "=== node $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
source experiments/serving/launch_vllm.sh

echo "=== toy search: 1 generation, 4 valid, 4 test ==="
cd "$REPO_ROOT/_mgsm"
"$PY" search.py \
    --domain mgsm \
    --save_dir "$OUT_DIR" \
    --expr_name "$CELL_ID" \
    --run_id "$CELL_ID" \
    --model "$META_MODEL" \
    --worker_model "$WORKER_MODEL" \
    --meta_base_url "$ADAS_META_BASE_URL" \
    --worker_base_url "$ADAS_WORKER_BASE_URL" \
    --meta_max_tokens 4096 \
    --worker_max_tokens 1024 \
    --n_generation 1 \
    --target_generations 1 \
    --max_attempts 3 \
    --debug_max 2 \
    --valid_size 4 \
    --test_size 4 \
    --n_repreat 1 \
    --max_workers 8 \
    --shuffle_seed 0 \
    --informed_meta

cd "$REPO_ROOT"
echo "=== every artifact the analysis depends on exists ==="
"$PY" - "$OUT_DIR" "$CELL_ID" <<'PYTHON'
import json
import os
import sys

out_dir, cell_id = sys.argv[1], sys.argv[2]
run_dir = os.path.join(out_dir, cell_id)

required = {
    os.path.join(out_dir, f"{cell_id}_run_archive.json"): "archive (Layer 1, 1b, 3, 5)",
    os.path.join(run_dir, "budget_summary.json"): "budget and failure taxonomy (Q6)",
    os.path.join(run_dir, f"{cell_id}_token_usage.json"): "token accounting (cost axis)",
    os.path.join(run_dir, f"{cell_id}_call_log.jsonl"): "per-call dataflow (Layer 2)",
}
missing = [f"{path} -- {why}" for path, why in required.items() if not os.path.exists(path)]
for path, why in required.items():
    print(f"  [{'ok' if os.path.exists(path) else 'MISSING'}] {os.path.basename(path)}  ({why})")
if missing:
    print("\n".join(missing))
    sys.exit(1)

with open(os.path.join(out_dir, f"{cell_id}_run_archive.json")) as handle:
    archive = json.load(handle)
designs = [e for e in archive if e.get("code")]
print(f"  archive holds {len(archive)} entries, {len(designs)} with code")

scored = [e for e in designs if e.get("raw_median_accuracy") is not None]
assert scored, "no design carries a fitness; scoring never ran"
for entry in scored:
    vector = entry.get("valid_per_question")
    assert vector, f"{entry['name']} has no per-question vector; Layer 4 would be empty"
    print(f"  {entry['name']:<28} fitness={entry['raw_median_accuracy']:.3f} "
          f"vector_len={len(vector)} debug_attempts={entry.get('debug_attempts')}")
# Q7/Q8/Q9 capture: designs must be addressable campaign-wide, and the archive
# the designer actually saw must be recorded rather than inferred later.
assert all(e.get("design_id") for e in designs), \
    f"designs without design_id: {[e['name'] for e in designs if not e.get('design_id')]}"
generated = [e for e in designs if isinstance(e.get("generation"), int)]
assert generated, "no design was generated; only seeds are present"
for entry in generated:
    assert entry.get("archive_context") is not None, \
        f"{entry['name']} has no archive_context; Q8 archive utilisation would be inferred"
    print(f"  {entry['design_id']}  saw {len(entry['archive_context'])} archive entries, "
          f"reasoning={'yes' if entry.get('meta_reasoning') else 'not exposed'}")
with open(os.path.join(run_dir, "budget_summary.json")) as handle:
    print("  budget:", json.load(handle))
PYTHON

echo "=== the analysis layers consume what the search just wrote ==="
"$PY" - <<'PYTHON'
import os

from experiments.analysis.aggregate import load_campaign
from experiments.analysis.dataflow_graph import profiles_for_run
from experiments.analysis.extract_features import extract_features

root = os.path.join(os.environ["PYTHONPATH"], "experiments", "results", "sanity_search")
campaign = load_campaign(os.path.dirname(root))
designs = [e for archive in campaign.values() for e in archive if e.get("code")]
print(f"  loaded {len(designs)} designs back through the analysis loader")
for entry in designs:
    print(f"  {entry['name']:<28} {extract_features(entry['code'])['features']}")

profiles = profiles_for_run(os.path.join(root, "sanity_search"))
print(f"  reconstructed {len(profiles)} runtime dataflow profiles")
PYTHON

echo "=== sanity 03 PASSED ==="
