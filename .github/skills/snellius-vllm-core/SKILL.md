---
name: snellius-vllm-core
description: Shared logic for deploying vLLM inference endpoints on Snellius via Apptainer, driven by recipes.vllm.ai. Prerequisite skill loaded by snellius-vllm-strict and snellius-vllm-relaxed — do not invoke directly.
when_to_use: Loaded automatically as a prerequisite when snellius-vllm-strict or snellius-vllm-relaxed is used.
---

# Snellius vLLM Core

Shared recipe-lookup, interview, and config-generation logic for serving models with vLLM on Snellius. Not meant to be invoked on its own — `snellius-vllm-strict` and `snellius-vllm-relaxed` load this skill and differ only in what they do when a model has no published recipe (see "Step 2: Look up the recipe" below).

**Prerequisites:** Load `snellius-core`, `snellius-slurm`, and `snellius-containers` skills.

---

## Overview

Given a model name, produce a `./vllm-deployments/<model_slug>/` folder containing:
- `job.sh` — SLURM batch script that pulls (if needed) the `vllm/vllm-openai` container and runs `vllm serve` under Apptainer with `--nv`.
- `smoke_test.sh` — a plain `curl` OpenAI-compatible chat-completion request.
- `README.md` — the chosen configuration, why, and deploy/test instructions.

This is an interview: ask one question at a time, always offer the recipe's own recommendation as the default, and always offer "figure out the best choice for me" as a valid answer.

---

## Step 1: Identify the model

Ask the user which model they want to serve (a Hugging Face id like `zai-org/GLM-4.7`, or a plain name).

## Step 2: Look up the recipe

The scripts referenced below (`scripts/lookup_recipe.py`, `scripts/render_job.py`) live in this skill's own directory — resolve them relative to wherever this SKILL.md was loaded from (its base directory), not relative to your working directory.

Run:

```bash
python3 scripts/lookup_recipe.py "<model query>"
```

This prints JSON with a `status` field: `found`, `ambiguous`, or `not_found`.

**Do not use `WebFetch` to check whether a recipe exists.** `WebFetch` summarizes page content through a small model, which can paraphrase or invent entries. `lookup_recipe.py` reads the raw `models.json` / per-model `.json` from recipes.vllm.ai directly — it is the only source of truth for this check.

- `found` → continue to Step 3 using the output's `recipe` field as the source of defaults.
- `ambiguous` → show the `candidates` list, ask the user to pick one, then continue to Step 3.
- `not_found` → **this is the branch point.** `snellius-vllm-strict` and `snellius-vllm-relaxed` each define what happens here; follow whichever wrapper skill's instructions invoked this one.

## Step 3: Interview

Ask these questions one at a time. For each, show the recipe's recommendation (when one exists) and offer "you choose" as an explicit option.

1. **GPU partition & count** — `gpu_a100` (4× A100 40GB, no native FP8/NVFP4) or `gpu_h100` (4× H100 94GB, native FP8, no NVFP4). See `references/hardware-quant-matrix.md`. GPU count sets `--tensor-parallel-size`.
2. **Quantization** — offer the recipe's quant variants, filtered to what the chosen partition supports per the matrix. If the recipe's top recommendation isn't supported on the chosen hardware, say so and offer the best supported fallback.
3. **KV cache dtype** — recipe's `--kv-cache-dtype` default vs. override (e.g. `auto`, `fp8`).
4. **Context length & batching** — `--max-model-len` and `--max-num-batched-tokens`, using the recipe's tuning notes (throughput vs. latency) if present.
5. **Speculative decoding** — on/off, only if the recipe defines a method (e.g. MTP).
6. **Tool-call / reasoning parser flags** — only if the recipe defines them (e.g. `--tool-call-parser`, `--reasoning-parser`).
7. **Job mechanics** — job name, walltime, port (default `8000`).

## Step 4: Draft and approve

Assemble the full `vllm serve` command and show it to the user along with a summary of every choice made and why. Do not write any files until the user approves the draft. If they want changes, go back to the relevant interview question.

## Step 5: Generate files

Write the approved answers to a temp JSON file matching this shape:

```json
{
  "job_name": "glm47-fp8",
  "partition": "gpu_h100",
  "gpu_count": 4,
  "cpus": 32,
  "mem": "256G",
  "walltime": "01:00:00",
  "port": 8000,
  "vllm_tag": "latest",
  "model_id": "zai-org/GLM-4.7-FP8",
  "model_slug": "glm-4-7-fp8",
  "tensor_parallel_size": 4,
  "extra_flags": "--trust-remote-code --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45",
  "notes": "Human-readable summary of what was chosen and why.",
  "verified": true,
  "sources": []
}
```

Then render it:

```bash
python3 scripts/render_job.py /tmp/vllm-answers.json ./vllm-deployments/<model_slug>/
```

`verified: false` (with `sources` populated) only applies to the relaxed skill's no-recipe fallback path — it stamps the generated `README.md` as unverified. Otherwise always `true`.

## Step 6: Offer the smoke test

After files are written, ask once:

> "Want me to submit this as a real job on Snellius and hit it with `smoke_test.sh` to confirm it actually serves? This uses GPU SBUs (`gpu_a100`: 128 SBU/GPU-hour, `gpu_h100`: 192 SBU/GPU-hour) for however long the job runs — even a quick test costs a few minutes of GPU time."

- **Yes:**
  ```bash
  ssh snellius 'mkdir -p ~/vllm-deployments/<model_slug>'
  rsync -avz ./vllm-deployments/<model_slug>/ snellius:~/vllm-deployments/<model_slug>/
  ssh snellius 'cd ~/vllm-deployments/<model_slug> && sbatch job.sh'
  ```
  Poll with `ssh snellius 'squeue -u $USER'` until the job is `R`, find the allocated node, and run the smoke test against it:
  ```bash
  ssh snellius 'squeue -u $USER -h -o "%N"'
  ssh snellius 'ssh <node> "cd ~/vllm-deployments/<model_slug> && ./smoke_test.sh localhost:<port>"'
  ```
  Report the response, then tear the job down:
  ```bash
  ssh snellius 'scancel <job_id>'
  ```
- **No:** skip. `smoke_test.sh` is already saved for them to run manually later.

---

## Reference

- [Hardware/quant compatibility matrix](references/hardware-quant-matrix.md)
- [recipes.vllm.ai](https://recipes.vllm.ai/) — source of truth for model recipes
