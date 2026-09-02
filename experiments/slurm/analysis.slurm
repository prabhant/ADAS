#!/bin/bash
#SBATCH --job-name=adas_analysis
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/results/_slurm/analysis_%j.out
#SBATCH --error=experiments/results/_slurm/analysis_%j.err

# Analysis: CPU only, genoa partition.
#
# Every measurement layer consumes artifacts already written by the searches, so
# nothing here loads a model and the whole analysis re-runs for free whenever a
# measure is revised. genoa bills at SBU weight 1.0 against gpu_h100's 192, which
# makes iterating on the analysis effectively costless -- the reason the layers
# were designed to be post-hoc in the first place.
#
# The judge- and embedding-backed layers (thought-code alignment, Layer 3 prompt
# semantics) need a GPU and are therefore opt-in, not run here.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/experiments/slurm/_common.sh"
activate_env

FIGURE_DIR="${FIGURE_DIR:-$RESULTS_ROOT/_figures}"
AGGREGATE_PATH="$RESULTS_ROOT/_aggregate/campaign.json"
mkdir -p "$FIGURE_DIR" "$RESULTS_ROOT/_aggregate"

python - <<'PYTHON'
import json
import os
import sys

sys.path.insert(0, os.environ["REPO_ROOT"])

import yaml

from experiments.analysis import aggregate as agg
from experiments.analysis import figures
from experiments.analysis.attribution import (
    attribution_table, build_design_table, group_by_worker, variance_check,
)
from experiments.analysis.contamination import scan_archive, prior_archive_overlap
from experiments.analysis.distances import structural_distance
from experiments.analysis.lineage import build_lineage

repo_root = os.environ["REPO_ROOT"]
results_root = os.environ["RESULTS_ROOT"]
figure_dir = os.environ["FIGURE_DIR"]
aggregate_path = os.path.join(results_root, "_aggregate", "campaign.json")

with open(os.path.join(repo_root, "experiments", "config", "cells.yaml")) as handle:
    cells_config = yaml.safe_load(handle)

# The seven seed agents are the novelty reference point. Loading them from the
# domain prompt module keeps the reference identical to what search actually saw.
sys.path.insert(0, os.path.join(repo_root, "_mgsm"))
from mgsm_prompt import get_init_archive  # noqa: E402

seed_codes = [entry["code"] for entry in get_init_archive() if entry.get("code")]
print(f"{len(seed_codes)} seed agents loaded as the novelty reference")

payload = agg.aggregate(results_root, seed_codes, cells_config, aggregate_path)
print(f"aggregated {len(payload['cells'])} cells -> {aggregate_path}")

for cell in payload["cells"]:
    if cell["degenerate"]["is_degenerate"]:
        # Kept and reported. A meta model that cannot produce working code is a
        # result about that model, not a run to be quietly dropped.
        print(f"  DEGENERATE {cell['cell_id']}: {'; '.join(cell['degenerate']['reasons'])}")

campaign = agg.load_campaign(results_root, cells_config)
all_designs = [entry for archive in campaign.values() for entry in archive]
design_rows = build_design_table(all_designs)

# Attribution is fitted per worker tier and never pooled: the complexity that
# suits a 4B worker is probably not the one that suits a 27B worker.
stats = agg.fit_pooled_stats(campaign, seed_codes,
                             os.path.join(results_root, agg.POOLED_STATS_FILENAME))
attribution = {}
for worker, rows in group_by_worker(design_rows).items():
    check = variance_check(rows)
    if not check["usable"]:
        print(f"  skipping attribution for {worker}: fitness range {check['range']:.3f}")
        attribution[worker] = {"skipped": check}
        continue
    designs = [e for e in all_designs
               if (e.get("worker_model") or "unknown") == worker and e.get("code")]
    lineage = build_lineage(designs, lambda a, b: agg._safe_structural(a, b, stats))
    rows_by_index = {i: row for i, row in enumerate(build_design_table(designs))}
    attribution[worker] = {
        "variance": check,
        "table": attribution_table(rows, lineage, rows_by_index),
    }

with open(os.path.join(results_root, "_aggregate", "attribution.json"), "w") as handle:
    json.dump(attribution, handle, indent=2, default=str)

# Contamination stays cheap: name and description matching against published
# architectures, plus collisions with the prior archives shipped in results/.
contamination = {
    cell_id: {
        "published_match": scan_archive(archive),
        "prior_archive_overlap": prior_archive_overlap(archive, os.path.join(repo_root, "results")),
    }
    for cell_id, archive in campaign.items()
}
with open(os.path.join(results_root, "_aggregate", "contamination.json"), "w") as handle:
    json.dump(contamination, handle, indent=2, default=str)

signatures = agg.designer_signatures(results_root, seed_codes, cells_config)
with open(os.path.join(results_root, "_aggregate", "designer_signatures.json"), "w") as handle:
    json.dump(signatures, handle, indent=2, default=str)

# The failure taxonomy is a result in its own right, not just telemetry: a weak
# meta model failing on JSON formatting is a different claim from one failing on
# semantics.
failure_histograms = {}
for cell in payload["cells"]:
    cell_id = cell["cell_id"]
    path = os.path.join(results_root, cell_id, cell_id, "budget_summary.json")
    if os.path.exists(path):
        with open(path) as handle:
            failure_histograms[cell_id] = json.load(handle).get("failure_histogram", {})
    else:
        failure_histograms[cell_id] = {}

written = figures.render_all(
    aggregate_path, figure_dir,
    design_rows=design_rows,
    signatures=signatures,
    failure_histograms=failure_histograms,
    crosseval_rows=agg.crosseval_matrix(results_root),
)
print("figures written:")
for path in written:
    print(f"  {path}")
PYTHON

echo "analysis complete: $FIGURE_DIR"
