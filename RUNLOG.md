# ADAS on Snellius - run log

Append-only. Newest entries at the bottom. If the connection drops, start here.

## How to resume

```bash
cd /home/psingh/ADAS
cat RUNLOG.md                           # what was done, what is pending
bash experiments/slurm/status.sh        # live state of the latest batch
```

SLURM jobs survive a dropped connection; only commands typed into the terminal
do not. Anything long-running is therefore submitted as a job, never run inline.

## State before batch 1 (2026-10-12)

- Code committed at `581ea1b` (scoring, accuracy statistic, port collisions,
  structured outputs, walltime handoff, e2e test, analysis fixes).
- Verified: sanity 00-03, 05, 06; Stage 0.5 baselines (job 26414481) valid.
- Stage 0.5 CoT accuracy on MGSM: mistral-7b 14.2%, phi-4 82.7%,
  qwen3.5-27b 81.2%, gpt-oss-20b low 86.2% / high 90.9%.
  phi-4 and qwen3.5-27b are statistically indistinguishable; three of four
  workers are near ceiling.
- Direction agreed: exploratory study. Broaden the designer (meta) axis across
  model families before deepening; pick a worker with headroom.

## Batch 1 plan (5 jobs, inspect before submitting more)

1. CPU end-to-end test of the search loop and analysis layer (rome)
2. stage0_smoke - first run of the production stage path (2 GPUs)
3. Worker screen - CoT baselines for candidate workers, MGSM and MMLU
4. Designer screen, 1-GPU designers - can each model design at all
5. Designer screen, 2-GPU designers - includes gpt-oss-120b, which also
   settles the tp=2 feasibility question the old sanity 04 probe was for

## Log

### 2026-10-12 - pre-flight findings (no GPU used)

- gpt-oss-120b weights as served are mxfp4, 65 GB -> tensor-parallel 2 fits.
  The "183 GB" that worried us was the whole repo including bf16 copies.
- gemma-3-12b-it and Llama-3.2-3B-Instruct: cache dirs exist but are empty to
  this account (license-gated). Dropped; Phi-3-mini stands in as the small model.
- Mixtral-8x7B: `refs/main` points at a snapshot with NO weights; the weights are
  in snapshot 1e637f2d... Served from that explicit path.
- Nemotron-3, Qwen3.5, Phi-3 all load natively under transformers 5.16; no
  --trust-remote-code (the repos' custom code may not even run on 5.x).
- Reasoning parsers in vLLM 0.28: qwen3, nemotron_v3, openai_gptoss. Thinking
  models need one as designers, or the JSON grammar forbids thinking before the
  first `{`. Enabled for the META role only, so worker baselines stay valid.

### 2026-10-12 - BUG FIXED: analysis fields leaked into the designer's prompt

`get_prompt` / `get_reflexion_prompt` serialised whole archive entries, so the
meta agent saw per-question vectors, other designs' reasoning traces and the
worker's identity. Effects had it run: blind-mode meta was not blind (Stage 1b
invalid), the method was no longer ADAS, and one generated design nearly doubled
the prompt (22.9k -> 43.6k chars), overflowing 32k-context designers within a few
generations. Now only thought/name/code/fitness/generation are shown (as stock
ADAS). e2e test now asserts no analysis field reaches the designer.
