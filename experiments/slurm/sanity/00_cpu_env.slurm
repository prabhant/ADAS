#!/bin/bash
#SBATCH --job-name=adas_sanity_00_cpu
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=experiments/results/_slurm/sanity_00_cpu_%j.out
#SBATCH --error=experiments/results/_slurm/sanity_00_cpu_%j.err

# Sanity 00: everything that does not need a GPU.
#
# Runs on rome at SBU weight 1.0 rather than gpu_h100 at 192, so this costs
# essentially nothing and should be the first thing that ever fails.
#
# Snellius does not export login-shell variables into jobs and disables default
# modules, so every variable and path this job needs is set explicitly below.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
export PYTHONPATH="$REPO_ROOT"
# System libstdc++ provides CXXABI_1.3.13; the conda env's ICU wants 1.3.15.
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_ROOT"

echo "=== node: $(hostname)  job: ${SLURM_JOBID:-none} ==="
echo "=== interpreter ==="
"$PY" --version
"$PY" -c "import openai, tqdm, yaml, numpy, pandas; print('core deps ok')"
"$PY" -c "import sqlite3; print('sqlite3 ok (xgrammar backend depends on it)')"

echo "=== config parses and is internally consistent ==="
"$PY" - <<'PYTHON'
import os
import sys

import yaml

root = os.environ["PYTHONPATH"]
config = os.path.join(root, "experiments", "config")
with open(os.path.join(config, "models.yaml")) as handle:
    registry = yaml.safe_load(handle)
with open(os.path.join(config, "cells.yaml")) as handle:
    cells = yaml.safe_load(handle)
with open(os.path.join(config, "framework_api.yaml")) as handle:
    yaml.safe_load(handle)

known = set(registry["models"])
missing = set()
for cell in cells["cells"]:
    for role in ("meta", "worker"):
        key = cell.get(role)
        if key and key not in known:
            missing.add((cell["id"], role, key))
for key in cells["crosseval"]["target_workers"]:
    if key not in known:
        missing.add(("crosseval", "worker", key))

if missing:
    for entry in sorted(missing):
        print(f"  UNKNOWN MODEL {entry}")
    sys.exit(1)
print(f"{len(cells['cells'])} cells reference {len(known)} registered models, all resolvable")

# The node budget is the constraint that actually bites: meta and worker share
# one four-GPU node, so any pairing needing more than four is unschedulable.
by_key = registry["models"]
for cell in cells["cells"]:
    meta, worker = cell.get("meta"), cell.get("worker")
    gpus = (by_key[meta]["gpus"] if meta else 0) + (by_key[worker]["gpus"] if worker else 0)
    if gpus > 4:
        print(f"  INFEASIBLE {cell['id']}: needs {gpus} GPUs on one node")
        sys.exit(1)
print("every cell fits in one 4-GPU node")
PYTHON

echo "=== managed cache holds every model the grid asks for ==="
"$PY" - <<'PYTHON'
import os
import sys

import yaml

root = os.environ["PYTHONPATH"]
with open(os.path.join(root, "experiments", "config", "models.yaml")) as handle:
    registry = yaml.safe_load(handle)

cache = registry["defaults"]["managed_cache"]
missing = []
for key, spec in registry["models"].items():
    if spec.get("source") != "managed":
        continue
    path = os.path.join(cache, "models--" + spec["hf_id"].replace("/", "--"))
    status = "ok" if os.path.isdir(path) else "MISSING"
    print(f"  [{status}] {key} -> {spec['hf_id']}")
    if status == "MISSING":
        missing.append(key)

if missing:
    print(f"not in the managed cache: {missing}")
    sys.exit(1)
PYTHON

echo "=== search modules import under both domains ==="
( cd _mgsm && "$PY" -c "import search; print('mgsm search imports')" )
( cd _mmlu && "$PY" -c "import search; print('mmlu search imports')" )

echo "=== sandbox guards actually hold ==="
# Must run as a module, not a heredoc: multiprocessing "spawn" re-imports
# __main__ in the child, and a parent read from stdin has no importable
# __main__, so the child would die before the guards were installed.
"$PY" -W ignore::RuntimeWarning -m experiments.adas_ext.sandbox

echo "=== analysis layers run against the seed archive ==="
"$PY" - <<'PYTHON'
import os
import sys

root = os.environ["PYTHONPATH"]
sys.path.insert(0, os.path.join(root, "_mgsm"))
from mgsm_prompt import get_init_archive

from experiments.analysis.behavioral import bootstrap_median_ci
from experiments.analysis.designer_signature import hallucinated_apis, load_framework_api
from experiments.analysis.distances import PooledStats, seed_pairwise_percentile
from experiments.analysis.extract_features import extract_features
from experiments.analysis.lineage import detect_motifs

seeds = [entry for entry in get_init_archive() if entry.get("code")]
codes = [entry["code"] for entry in seeds]
api = load_framework_api()
for entry in seeds:
    features = extract_features(entry["code"])["features"]
    bad = hallucinated_apis(entry["code"], api)["total"]
    print(f"  {entry['name']:<28} agents={features['n_agents_static']} "
          f"calls={features['n_call_sites']} halluc={bad}")

stats = PooledStats.fit(codes)
print(f"  pooled stats over {stats.n} designs; degeneracy threshold "
      f"{seed_pairwise_percentile(codes, stats):.3f}")
print(f"  bootstrap CI smoke: {bootstrap_median_ci([1,0,1,1,0,1], n_resamples=200)}")
PYTHON

echo "=== sanity 00 PASSED ==="
