#!/bin/bash
# Patches to the conda environment that this cluster needs. Idempotent; rerun
# after any reinstall of the packages below. Login node only, with no jobs
# queued -- never modify an environment a running job may be importing from.
#
#   bash experiments/env/patch_env.sh
set -euo pipefail

PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# flashinfer.comm is imported by vLLM at startup whenever tensor parallelism is
# above 1. fd_exchange.py annotates with array.array[int], which only evaluates
# on Python 3.12+; this env is 3.11, so every tp>=2 server died at startup.
# Deferring annotation evaluation makes the annotation inert.
target="$SITE/flashinfer/comm/fd_exchange.py"
if [[ -f "$target" ]] && ! grep -q '^from __future__ import annotations' "$target"; then
    sed -i '1i from __future__ import annotations' "$target"
    echo "patched $target"
else
    echo "already patched or absent: $target"
fi
