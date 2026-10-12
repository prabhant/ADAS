---
name: snellius-verified
description: Authoritative, empirically-verified Snellius facts for this workspace - partitions, modules, CUDA/torch version skew, vLLM 0.28 behaviour, and known silent-failure traps. Overrides the vendored snellius-* skills wherever they disagree.
when_to_use: Any time you write, submit or debug a SLURM job on Snellius, serve a model with vLLM, or reason about partitions, modules, storage or GPU sizing in this workspace.
---

# Snellius: verified facts

Everything here was measured on this cluster, not read from documentation. Where
the vendored `snellius-core` / `snellius-slurm` / `snellius-vllm-core` skills
disagree, **this file wins** — several of their reference tables predate the
2024 hardware refresh.

## Partitions (from `sinfo`, verified)

The vendored skills list `thin`, `gpu`, `fat`. **Those partitions do not exist.**
Submitting to them fails immediately.

| Partition | Nodes | Shape | Use |
|-----------|-------|-------|-----|
| `rome` | tcn[4-524] | 128 cores, 0 GPU | CPU work, SBU weight 1 |
| `genoa` | tcn[527-1263] | 192 cores, 0 GPU | CPU work, cheapest |
| `gpu_h100` | gcn[73-160] | 64 cores, 4x H100 94GB | 16 cores per GPU |
| `gpu_a100` | gcn[6-72] | 4x A100 | older GPUs |
| `fat_rome`, `fat_genoa` | | high memory | |
| `himem_4tb`, `himem_8tb` | | very high memory | |
| `gpu_mig`, `gpu_vis`, `staging`, `cbuild` | | special purpose | |

- Minimum allocation is 1/4 node. Single-node jobs may take 1/4, 1/2, 3/4 or all.
- Max walltime 120h. Default walltime is 10 minutes — always set `--time`.
- **Billing follows actual runtime, not the walltime cap**, so generous caps are free.
  Request only the GPUs you use: a tensor-parallel-1 model on `--gpus=2` wastes half.
- `gpu_h100` bills at SBU weight 192 against `genoa`'s 1. Put anything CPU-only on genoa.

## Modules

Default modules are **disabled**. `module avail` shows almost nothing until a year
module is loaded, and full versions are mandatory.

```bash
module load 2025            # year modules: 2023, 2024, 2025
module load CUDA/12.9.1
```

- CUDA by year: `2023` → 12.1.1, 12.4.0 · `2024` → 12.6.0 · `2025` → 12.8.0, 12.9.1
- In non-interactive shells `module` may be undefined: `source /etc/profile.d/lmod.sh`
- **Load modules BEFORE prepending anything to `LD_LIBRARY_PATH`.** Lmod prepends, so
  loading afterwards pushes CUDA paths ahead of your environment's libraries.

## Environment inheritance — a silent trap

> "On Snellius we are not exporting environment variables which are set by the
> user within the bash session."

`VAR=x sbatch job.sh` **silently does nothing**. Pass parameters as positional
script arguments (`sbatch job.slurm 0.5s` -> `$1`). Every variable a job needs
must be exported inside the job script.

**Never use `--export=ALL`.** It copies the *whole* submitting shell into the job.
Here, a `RESULTS_ROOT=$(mktemp -d)` left by an earlier verification command in
the same terminal sent a whole batch's results to a temp dir that existed only
on the compute nodes. Default submission (no `--export`) is clean; use
namespaced variables (`ADAS_RESULTS_ROOT`) for anything a script reads from the
environment.

Also: `--output`/`--error` directories must exist *before* submission, and the
`#SBATCH` block must directly follow `#!/bin/bash` with no interruption.

## Storage

| Path | Notes |
|------|-------|
| `$TMPDIR` | node-local scratch, needs `#SBATCH --constraint=scratch-node`, wiped at job end |
| `/scratch-shared/$USER` | shared scratch |
| `/projects/2/managed_datasets/hf_cache_dir` | managed HF model cache, **read-only** |

Node-local scratch is `$TMPDIR`, **not** `/scratch-node`. Never point `HF_HOME` at
the managed cache — it is read-only and the HF hub writes lock files into it.
Verify model repo ids by listing the cache; do not assume they exist.

## This environment (conda + vLLM)

Python: `$HOME/.conda/envs/adas-snellius/bin/python` (3.11). torch **2.13.0+cu130**,
vLLM **0.28.0**, transformers 5.16.1.

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

**Required.** The system `libstdc++` provides only `CXXABI_1.3.13` while the env's
ICU needs `1.3.15`; without this `import sqlite3` fails, which takes vLLM's xgrammar
structured-output backend and the entire `vllm` CLI down with it.

**CUDA version skew:** torch is built for CUDA 13.0 and the newest module is 12.9.1.
`nvcc` is therefore available for tooling but is *not* a matched pair for
JIT-compiling kernels that link torch.

## vLLM 0.28 specifics

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

**Required.** flashinfer JIT-compiles its sampling kernels on first use and needs
`nvcc` matched to torch. The failure is maximally misleading: the model loads,
weights resolve, CUDA graphs capture to 100%, *then* the engine dies on the first
sampled token and the client only sees a `/health` timeout.

Other 0.28 changes that break older recipes:

| Old | New |
|-----|-----|
| `--guided-decoding-backend xgrammar` | `--structured-outputs-config '{"backend":"xgrammar"}'` |
| `extra_body={"guided_json": schema}` | `response_format={"type":"json_schema","json_schema":{...}}` |

The `guided_json` change is a **silent** failure: the request model sets
`extra="allow"`, so the old key is accepted, ignored, and enforces nothing. No error
is raised and the schema simply never applies.

`vllm serve --help` fails on login nodes ("Failed to infer device type") because there
is no GPU. Introspect `vllm.engine.arg_utils.EngineArgs` instead. Request models live
in `vllm/entrypoints/openai/chat_completion/protocol.py`.

Measured on gpt-oss-20b: `reasoning_effort` is load-bearing and `reasoning_content` is
exposed. On Mistral-7B the field is *accepted but inert* — vLLM ignores unknown keys,
so always test whether a control is load-bearing, never whether it is accepted.
`reasoning_tokens` in usage reports 0; that cost lands inside `completion_tokens`.

## Debugging habits that paid off here

- A job that reports `COMPLETED` with exit 0 can still be wrong. Check that the
  *numbers* are plausible, not just that artifacts exist.
- When a served model fails, the useful error is in the vLLM log, not the client
  traceback — the client only ever sees a health timeout. Always tail the server log.
- Chain sanity jobs with `--dependency=afterok:` so one failure stops the rest.
- Prefer short walltimes for probe jobs; they backfill and start far sooner.

## Serving on shared nodes

`gpu_h100` nodes are **shared by up to four jobs**, so two array tasks of the same
stage can land on one node and see the same `localhost`.

- **Never hardcode server ports.** Derive them from `SLURM_JOB_ID` and step past
  anything already listening. With fixed ports the second task fails to bind, then
  finds the *first* task's server and silently runs an entire cell against the
  wrong models — recorded under the wrong configuration.
- **A 200 from `/health` is not enough.** It only proves something is listening.
  Cross-check `/v1/models` against the model you asked for. SLURM does not always
  reap a previous job's server, and a stale process answering `/health` in ~0s
  (versus a normal multi-minute cold start) is the tell.

## Long jobs

- Compute a deadline from `scontrol show job $SLURM_JOB_ID` `EndTime` minus a margin
  large enough for one unit of work, stop cleanly before it, and resubmit. Being
  SIGKILLed mid-generation loses that generation; stopping between them resumes.
- Persist state after every unit of work so a resubmission continues rather than
  restarts.

## Storage durability

- `/scratch-shared` has been observed deleting hours-old files platform-side. Keep
  venvs and caches on `/projects` (if you have a writable project dir) or `$HOME`,
  never on scratch.
- `$HOME` here is 200 GiB and lightly used; it is also where a self-downloaded HF
  model lands by default, so pointing `HF_HOME` at it keeps offline analysis jobs
  working on CPU partitions.

## JIT compilation on compute nodes (general pattern)

flashinfer, DeepGEMM and similar backends JIT-compile CUDA kernels **on the first
real request** — startup and warmup do not touch them, so failures surface minutes
in, long after the model looks healthy. Prefer torch-native backends. If JIT is
unavoidable, point `CUDA_HOME` at the *wheel-bundled* toolkit rather than a system
module, and note that pip splits CUDA across per-library wheels, so headers need
`CPATH` and libraries need `LIBRARY_PATH`.

## Serving specific models with this env (vLLM 0.28, transformers 5.16, Python 3.11)

Measured in batch 1 (2026-10-12):

| Model | Status |
|---|---|
| gpt-oss-20b, qwen3-14b, qwen3-30b-a3b, nemotron-3-nano-30b | serve at tp=1; design 5-6/6 |
| qwen3.5-27b as designer | serves; health JSON probe needs >=4096 tokens (thinks first with `--reasoning-parser qwen3`) |
| any tp>=2 model | died at startup until `experiments/env/patch_env.sh`: `flashinfer/comm/fd_exchange.py` uses `array.array[int]`, a Python 3.12-only annotation, imported by vLLM whenever tp>1 |
| mistral-7b-v0.3 | fails: transformers 5.x cannot build its fast tokenizer |
| phi-3-mini/medium-128k | fail: transformers 5.x longrope config wants `original_max_position_embeddings` in `rope_parameters` |
| mixtral-8x7b | managed cache `refs/main` points at a weightless snapshot; serve the explicit snapshot path |
| gemma-3-12b-it, Llama-3.2-3B | cache dirs empty to this account (license-gated) |
| qwen3-14b as WORKER | returns empty `answer` fields under JSON grammar without a reasoning parser; unusable as a worker |

Containers: the vendored `snellius-vllm-core` skill serves via the official
`vllm/vllm-openai` image under Apptainer, which sidesteps these Python-env issues.
A colleague reported Apptainer failing cluster-wide ("maximum number of user
namespaces exceeded"), which is why this project serves natively. If the native
env keeps accumulating patches, retest the container route.

**A server is only proven by a real request.** `/health` passes long before
lazily-JIT-compiled kernels (sampling, all-reduce) are first exercised.

## Fail fast on a dead server

The health gate must watch the server *process*, not only poll its URL: a server
that dies at startup otherwise burns the whole health timeout (40 min here,
on every failed task). A dead child stays a zombie until the job shell reaps it,
so `os.kill(pid, 0)` reports it alive -- check `/proc/<pid>/stat` state instead.
