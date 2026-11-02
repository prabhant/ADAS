---
title: ADAS on GitHub Copilot models
description: Experimental plan for re-running the ADAS benchmarking campaign on models served through the GitHub Copilot SDK and billed in GitHub AI Credits
ms.date: 2026-10-28
ms.topic: concept
---

## Verdict

The campaign can run on Copilot models. The GitHub Copilot SDK (`github-copilot-sdk` 1.0.24,
runtime 1.0.101) supports everything Meta Agent Search needs:

* choosing the model and reasoning effort per session
* replacing the system prompt, with tools off and permissions denied
* strict JSON-schema output for the designer
* per-call token, reasoning-token and cost reporting

Usage is billed per token in GitHub AI Credits (1 credit = $0.01) at each model's
published rate. Five differences from the vLLM campaign are recorded as deviations
rather than hidden:

* Temperature cannot be set. The seeds' 0.8 (CoT-SC, Debate) and 0.1 (final decision)
  settings, and any temperature a designer chooses, have no effect.
* The runtime could add its own prompt, tool definitions or calls to each request. The
  Phase 0 audit has to show that a call bills only our prompt.
* Rate limits are unpublished, while a run makes roughly 100k to 400k worker calls.
* Vendors publish no parameter counts, so designer "size" (Q8) becomes tier and price
  within a model family.
* Hosted models can change over time; every call logs the served model id and the
  provider call id.

Scope agreed for this round: one designer (Claude Haiku 5.5), Copilot models in both
roles, runs on a local machine, and a 50,000-credit budget. The hybrid arm (Copilot
designer, Snellius vLLM worker) and further designers are deferred.

## How the Copilot path works

```text
run_cell.py --starts--> copilot_gateway.py (127.0.0.1, one port per role)
     |                       |  OpenAI /v1/chat/completions -> Copilot SDK session
     +--runs--> _<domain>/search.py --RoleClient--> gateway --> Copilot model
                     +-- sandboxed agent code --> worker port only
```

The gateway (`experiments/serving/copilot_gateway.py`) speaks the OpenAI chat API that
`RoleClient` already uses, so search, sandbox, instrumentation and every
`_<domain>/search.py` run unchanged. It enforces four things:

* Each port serves an explicit model allowlist, and the sandbox may reach only the
  worker port, so generated code cannot call the designer model.
* Every session has a replaced system prompt, no tools, denied permissions and no
  session store, and is deleted after use.
* Each call is appended to a ledger (`copilot_ledger.jsonl` per cell) with tokens,
  reasoning tokens, cache tokens, cost, served model, conversation mode and the
  requested temperature.
* At the cell or global credit cap, and on authentication failure, the gateway holds
  further requests instead of failing them, and the runner stops the search. A failed
  worker call would otherwise be scored as a wrong answer and silently corrupt fitness.

Spend is charged as the larger of Copilot's reported cost and an estimate from the
prices in `experiments/config/copilot_models.yaml`, so a stale price or a unit error
can only make the budget stricter.

### Verified SDK behaviour

Offline tests and a real-runtime check without credentials established these facts:

* Structured output supports only strict `json_schema`; there is no plain JSON mode.
  Worker replies are therefore requested as text, and the gateway extracts the JSON
  object (bare, fenced or after a preamble). The designer schema is sent closed
  (`additionalProperties: false`, all fields required).
* The stored Copilot CLI login is visible only to the SDK's `copilot-cli` client mode,
  which is therefore the default. The isolated `empty` mode requires its own runtime
  home and then needs a token in `COPILOT_GITHUB_TOKEN`.
* With credentials missing, the runtime fails in under 0.1 s with "No GitHub OAuth
  token or Copilot HMAC key provided". SDK 1.0.24's structured wait ignores that error
  and sits out its timeout, so the gateway watches for session errors itself.
* The designer's reflexion and debug turns continue one SDK session; only the new user
  turn is sent. The model sees its own raw reply, where ADAS echoes `str(dict)`.

## Questions and what this arm adds

The research questions are unchanged (`questions.md`). Hosted models add three things
the vLLM campaign could not measure cheaply:

* Cost in money: every design carries `credits_per_question`, so Q1 and Q2 can put a
  cheap worker plus a discovered scaffold against a frontier model used plainly, at
  matched spend.
* Reasoning effort with the model fixed (Q4), on hosted models that expose it.
* Designers from 2026 frontier families (Q7, Q8), with the contamination scan first:
  these designers have very likely read the ADAS paper.

## Phases

Run every command from the repository root inside the environment described in
[README.md](README.md#github-copilot-models-experimental). Each cell resumes when rerun.

### Phase 0: build, verify, gate (at most 1,000 credits)

1. Run the model inventory and fix any id it reports missing in
   `experiments/config/copilot_models.yaml`:

   ```bash
   python experiments/serving/copilot_probe.py inventory
   ```

2. Run the overhead audit. This is a gate; do not start searches unless it passes:

   ```bash
   python experiments/serving/copilot_probe.py audit --model claude-haiku-5.5
   ```

   It fails on more than one model call per request, sub-agent or runtime-initiated
   calls, more than 100 fixed input tokens (minimal prompt and regression intercept),
   or a substituted model.

3. Build the capability matrix for the designer and the worker candidates:

   ```bash
   python experiments/serving/copilot_probe.py capabilities \
       --models claude-haiku-5.5,gpt-6-luna,gpt-5.6-luna --honor_max_tokens
   ```

   It records JSON compliance, schema support, whether reasoning effort changes output
   tokens, whether a reasoning trace comes back, whether `max_tokens` is honoured, and
   reported against estimated cost. Pass `--honor_max_tokens` to the runner only if
   the matrix shows it holds.

4. Sweep throughput and set `max_workers` and `max_concurrency` in
   `copilot_cells.yaml` to the recommendation:

   ```bash
   python experiments/serving/copilot_probe.py throughput --model claude-haiku-5.5
   ```

5. Run a two-generation smoke search, then reconcile its ledger against the GitHub
   billing dashboard:

   ```bash
   python experiments/local/run_cell.py cp0_smoke
   ```

Gate: proceed only if the audit passes and throughput reaches about 1,000 calls per
minute. Below that, switch Phase 2 to the lean protocol (`test_top_k: 5`).

### Phase 1: worker and domain screen (at most 4,000 credits)

Scaffolds only matter where a worker leaves headroom. In the vLLM campaign MGSM, MMLU
and DROP left little for strong workers, so GPQA (now on the instrumented pipeline)
joins the screen. The screen covers three workers (Claude Haiku 5.5, GPT-6 Luna, and
GPT-5.6 Luna, all at their default effort) on four domains: bare and CoT at three
passes, plus the seven seed agents on validation. GPT-5.6 Luna replaces GPT-5.4 nano,
which this account is not offered, as the likely-weaker worker.

```bash
python experiments/local/run_cell.py --list
for d in mgsm mmlu drop gpqa; do for w in claude-haiku-5.5 gpt-6-luna gpt-5.6-luna; do
  for c in cot seeds bare; do python experiments/local/run_cell.py cp1_${c}_${d}_${w} --quiet; done
done; done
```

The selection rule is fixed before reading any result:

1. Keep pairs whose CoT accuracy is between 30% and 85%.
2. Pick the pair with the most seed agents at CoT + 5 points or more.
3. Break ties toward Claude Haiku 5.5, which makes the Phase 2 cells self-design cells
   like the MGSM panel's Qwen3-30B-A3B.

`python experiments/local/screen_report.py` prints the screen table and applies this
rule. Write its pick into `selection` in `experiments/config/copilot_cells.yaml`
(`worker`, `domain`, the other two workers as `transfer_workers`), then run the
production baseline the informed designer is told:

```bash
python experiments/local/run_cell.py cp1_baseline_cot_<domain>_<worker>
```

### Phase 2: Claude Haiku 5.5 designer runs (at most 15,000 credits)

Two replicates on the ADAS panel protocol, unchanged: 30 generations, 128 validation
and 800 test questions (GPQA: 32 x 5 and 166), 90 attempts, 3 debug rounds, the shared
seed cache, and an informed designer. Every design gets a test score, so the
validation-to-test analysis stays comparable to batches 5 and 8.

```bash
python experiments/local/run_cell.py cp2_claude-haiku-5.5_<domain>_<worker>_a
python experiments/local/run_cell.py cp2_claude-haiku-5.5_<domain>_<worker>_b
```

With one designer, the spread between the two replicates is the noise floor for every
later designer comparison. The pre-registered comparison is the design selected on
validation against the best seed and against Self-Consistency, with a paired bootstrap
over test questions, matching the batch 5 and 8 tables.

### Phase 3: questions a hosted setup answers cheaply (at most 20,000 credits)

In priority order:

1. Random recombination (`cp3_random_*`): 40 generations with no designer, seeds and
   the top five designs tested. It is the design-space baseline for Q7 and is never cut.
2. Plain frontier baselines (`cp3_plain_*`): CoT with GPT-6.1 Sol, Claude Sonnet 5.5 and
   Claude Opus 5.5 on the Phase 2 test questions. Together with Phase 2 this gives the
   accuracy against credits-per-question Pareto front for Q1 and Q2. No search is needed.
3. Reasoning effort (`cp3_effort_*`), only if the capability matrix shows effort is
   not inert for `selection.effort_worker`. These cells are disabled until then. Run
   both effort baselines first, then the two 20-generation searches.
4. Transfer (`cp3_transfer_*`): the best-on-validation design of replicate a, beside
   CoT, Self-Consistency and Debate, evaluated under the other two screened workers (Q5).

Cut order if the budget tightens: 3, then 4, then the test set of 1. Never cut 1 or 2.

### Phase 4: analysis (no credits)

* Run the contamination scan before reading any design.
* Run the existing analysis layers on the search cells with
  `python experiments/local/analyze_copilot.py <cells...> --domain <domain>`. It stages
  the named cells (the results root also holds screen seeds and test-only archives),
  runs aggregate, contamination, motifs, designer signatures and figures, and adds the
  panel comparison: the design selected on validation against Self-Consistency and the
  best seed, paired bootstrap over test questions. Temperature features are inert;
  `meta_params_total` stays empty and tier stands in for size. The prompt-semantics
  layer and the LLM judge need the Snellius models and are not run locally.
* Write results to a new `findings_copilot.md`. `visualization/build_site.py` parses the
  `findings.md` panel tables by position and expects eight designers in a fixed order,
  so a Copilot table there would break the site.

## Budget

| Phase | Cap (credits) | Purpose |
|-------|---------------|---------|
| 0 | 1,000 | probes, gate, smoke |
| 1 | 4,000 | screen, production baseline, seed cache |
| 2 | 15,000 | two designer runs |
| 3 | 20,000 | random reference, crossover, effort, transfer |
| Reserve | 10,000 | reruns, or the next designers |

Per-run estimates from the batch 5 and 8 figures (37 designs including seeds, about 8k
worker tokens per question per design, roughly 80% input), at the Claude Haiku 5.5 and
GPT-6 Luna rates ($0.10 input and $0.50 output per million tokens):

| Item | Tokens | Credits |
|------|--------|---------|
| Worker, full panel protocol (all designs on 128 + 800 questions) | about 275M | 4,500 to 7,000 |
| Worker, lean protocol (seeds and top 5 on 300 test questions) | about 65M | 1,200 to 1,700 |
| Designer, 30 generations (prompt grows to about 60k tokens) | about 4M in, 0.4M out | 50 to 300 |

Each cell's `max_credits` in `copilot_cells.yaml` enforces its share, and
`global_max_credits` caps the sum of every ledger. Phase 0 replaces these estimates with
measured cost per call. After it, an extra designer costs roughly 5,000 to 7,000 credits
on the full protocol, or about 1,500 on the lean one.

## Deviations register

Carry these into `findings_copilot.md`:

* Temperature is inert. Prompts stay identical to the vLLM panels, and the requested
  temperature is logged per call.
* The designer sees its raw prior reply instead of ADAS's `str(dict)` echo.
* Worker JSON is extracted from text, because the SDK has no JSON mode. The gateway also
  recovers Python-literal dicts and single-quoted dicts with unescaped apostrophes
  (Claude Haiku 5.5's habit); the ledger flags each extraction.
* Output caps are provider defaults: `max_tokens` had no effect on any model in Phase 0.
  Truncation shows up as `finish_reason: length`.
* Call starts are paced to stay under the account rate limit: 200 per minute in Phase 1,
  175 per Claude Haiku 5.5 cell from Phase 2 on (about 420 combined across parallel cells
  was safe), 30 for Claude Sonnet 5.5, and up to 300 for a cell running alone. After a
  rate-limit error the gateway waits at least 30 s, because Copilot can bill rate-limited
  calls for the attempts the runtime retried internally (339 credits in one unpaced
  Sonnet run; Haiku's rate-limit errors cost nothing).
* Claude Opus 5.5 is not measured: Copilot rejects every request carrying the ADAS worker
  system prompt with "400 The requested model is not supported".
* The gateway retries a call for up to 50 minutes, so a sleeping or offline laptop
  delays the run instead of scoring questions as failed.
* Designer size is tier and price, not parameter count.
* Replicates come from service nondeterminism; there is no sampling seed.
* Hosted models can change version during the campaign; the ledger pins what served
  each call.

## Replication with GPT-6 Luna (pre-registered 2026-10-30)

Written before any of its credits were spent. The Haiku study's answer to "does ADAS
help on DROP" rests on one worker, one designer and two replicates; this round asks
whether it holds with GPT-6 Luna as the worker, with two designers, and when the
search itself runs in parallel.

### Design

* Worker: GPT-6 Luna, the only other screened worker on a domain inside the 30-85% CoT
  band (DROP: CoT 62.6% validation, 69.7% test; format-CoT 92.9% test). Same 128
  validation and 300 test questions, same seeds, same lean test (the seeds and the 5
  best designs by validation), 15 generations per search.
* Designers: GPT-6 Luna designing for itself (the faithful replication) and Claude Haiku
  5.5 designing for GPT-6 Luna (designer held fixed, so a difference is the worker's).
* Search modes, both with 15 designs:
  * sequential: the ADAS loop, each proposal sees every earlier result;
  * batch: 3 proposals per round from one archive snapshot, designed and evaluated
    concurrently (`_drop/search.py --proposals_per_round 3`), 5 rounds. The designer
    learns from results every 3 designs instead of every design.
* Two replicates of each designer x mode, so 8 searches. All run at once under one
  account-wide pace (`pace_rates`, a slot file shared by every gateway), with
  `experiments/local/run_parallel.py`.
* Controls: format-CoT and CoT on test twice (with the Phase 3.4 cell, three runs: the
  re-run noise floor); random recombination, 40 generations; worker effort low and high
  (validation baselines, the selected self-design re-tested, and one 15-generation
  self-design search at each); transfer of the selected self-design to Claude Haiku 5.5
  and GPT-5.6 Luna. The plain frontier models are worker-independent and not repeated.

Cells: `copilot_cells.yaml`, group `luna` (the `cpl_*` cells and the GPT-6 Luna baselines).
Budget: 12,000 credits; `global_max_credits` is 21,000 (8,724 spent before this round).

### Gate (L0), before the group runs

1. Throughput: a concurrency sweep for GPT-6 Luna, and GPT-6 Luna and Claude Haiku 5.5
   at once, to learn whether the two providers draw on separate rate limits, and
   whether GPT-6 Luna's rate-limited calls are billed. `pace_rates` is set from this.
2. `cpl_smoke_seq_gpt-6-luna`: GPT-6 Luna as designer (never used as one): schema
   replies, Reflexion turns continued, debug rounds.
3. `cpl_smoke_batch3_claude-haiku-5.5`: the batch path through the real gateway.

### Comparisons

Selection per search: best validation, ties to the earliest generation. Paired bootstrap
over the 300 test questions within a cell; differences between cells are read against
the format-CoT re-run spread.

1. Headline: each selected design vs Self-Consistency. Replicates if at least +5 points
   in most of the 8 searches.
2. Format: each selected design vs format-CoT. The Haiku conclusion (the gain is answer
   format) replicates if the intervals include 0 or lie below it.
3. Designer: GPT-6 Luna vs Claude Haiku 5.5 designing for GPT-6 Luna (selected test,
   validation distribution, design character), and Haiku designing for GPT-6 Luna vs the
   transferred Haiku design (83.3%, Phase 3.4).
4. Batch vs sequential at 15 designs: best and median validation, selected test, motifs
   and novelty, duplicate proposals within a round, wall-clock per search.
5. Best of replicates: within each designer x mode, the design with the best validation
   across both replicates, and its test score.
6. Random recombination: does it stay within the seed range, as with Haiku?
7. Effort: low vs high for CoT, format-CoT and the design; each effort search vs
   format-CoT at that effort.
8. Transfer: the GPT-6 Luna self-design on Claude Haiku 5.5 and GPT-5.6 Luna, vs
   format-CoT there.
9. Replicate variance (Q8) across the 8 searches.

Cut order if credits or time run short: the Haiku batch searches, then the effort
searches, then random recombination's test set. The sequential searches and the
format-CoT controls are never cut. Results go to `findings_copilot_luna.md`.

## Deferred

* The hybrid arm: a Copilot designer with the Snellius Qwen3-30B-A3B worker on MGSM,
  reusing the batch 8 seed cache. It needs either outbound HTTPS from compute nodes or
  an SSH tunnel to the vLLM worker.
* Further designers, as a tier ladder within one family (for example GPT-6.1 Terra,
  GPT-6.1 Sol), funded from the reserve. GPT-6 Luna as designer is part of the
  replication above.

## Verification status

Offline, without credentials or credits:

* `experiments/tests/copilot_gateway_test.py` covers JSON extraction, schema closing,
  session continuation, error mapping through `RoleClient`, retries, partial billing,
  the allowlist, budget and authentication holds, and the probe gates.
* `experiments/tests/copilot_e2e.py` runs the runner, gateway and real `search.py` with a
  scripted provider. It checks exact accuracies, continued designer sessions, credit
  accounting, halt and resume, and the test-only cells.
* `experiments/tests/search_e2e.py` passes for MGSM, DROP and the GPQA port, including
  the lean protocol.
* Against the real runtime with no credentials, `cp0_smoke` halts within a second,
  stops the search, and records no score.

With the Copilot login (2026-10-29), Phases 0 to 3 ran and the Phase 4 layers ran on
every search cell; results are in [findings_copilot.md](findings_copilot.md). In short:
the overhead audit passed for all three workers, and reported cost matches the price
table. Sustained throughput is about 200 calls per minute per cell, so the throughput
gate selected the lean protocol. No hand-designed seed beat CoT by 5 points on any
worker or domain. On the pre-registered pick (DROP with Claude Haiku 5.5), the designer
found +13 to +15 points over Self-Consistency, all of it answer format: a one-line format
instruction on CoT matches it. The campaign spent 8,724 credits.
