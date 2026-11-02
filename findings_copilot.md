---
title: ADAS on GitHub Copilot models, findings
description: Results of the Copilot SDK arm of the ADAS study, Phases 0 to 3 (feasibility and cost, worker and domain screen, designer runs, and the controls that explain them)
ms.date: 2026-10-29
ms.topic: reference
---

## Summary (2026-10-29)

Phases 0 to 3 of [experimental_plan_copilot.md](experimental_plan_copilot.md) are done,
with one designer (Claude Haiku 5.5) on DROP, and the Phase 4 analysis layers have run on
every search cell. 8,724 AI credits ($87.24) of the 50,000 budget are spent.

* The Copilot path works and bills exactly what it claims. A call bills our prompt
  plus at most about 45 tokens, makes one model call, and Copilot's reported cost
  equals the published per-token price to five decimals.
* On today's lightweight hosted models, hand-designed scaffolds add nothing. Across
  three workers and four ADAS domains (72 seed-agent results), the seed agents average
  -0.1 points against chain-of-thought, and none beats it by 5 points.
* ADAS reproduces. Claude Haiku 5.5, designing for itself on DROP, found agents 13 to 15
  points above Self-Consistency on test in both replicates. Random recombination of the
  seeds with no designer found nothing (-1.8 points), so the gain is the designer's.
* The whole gain is answer format. CoT told to answer in the F1 metric's own form
  (format-CoT, one call) scores 94.8%, level with the searched designs (+0.3 and -1.4,
  intervals include zero) at a seventh to a thirteenth of their cost. Moved to two
  OpenAI workers, the searched design keeps its lead over the hand-designed agents but
  falls 7 to 10 points behind format-CoT.
* Once the answer format is right, neither bigger models nor more reasoning pay.
  GPT-6.1 Sol adds 1.6 points at 8 times the cost and Claude Sonnet 5.5 nothing at 23
  times. Worker effort from low to high changes CoT, format-CoT and the searched design
  by under a point. Searches run at low and high effort land at or below format-CoT.
* The same designer with the same prompt wrote programmatic designs in one replicate and
  prompt-only designs in the other, at the same accuracy (Q8). Separately run agents
  vary by up to about 3 points on identical questions from service nondeterminism.
* Claude Opus 5.5 could not be measured: Copilot rejects the ADAS worker prompt for it.

Raw data: `experiments/results_copilot/` (ledgers, archives, runner logs, probe reports
in `_probes/`, analysis reports in `_analysis/phase2` and `_analysis/phase3`). Screen
table: `python experiments/local/screen_report.py`. Search-cell panel tables:
`python experiments/local/analyze_copilot.py <cells...> --domain drop --name <name>`.

## Phase 0: what a Copilot call is and costs

### Account and models

The account (Copilot Business/Enterprise, via the stored Copilot CLI login) offers all
the registered models except GPT-5.4 nano, which GPT-5.6 Luna replaces as the
likely-weaker worker. The SDK's own price table (`models.list`, AI credits per million
tokens) matches `copilot_models.yaml` exactly for every model used. The login is visible
only to the SDK's `copilot-cli` client mode, which is now the default.

### Overhead audit: passed for all three workers

| Worker | Minimal prompt billed (input tokens) | Model calls per request | Reported cost = price-table estimate |
|--------|-------------------------------------|-------------------------|--------------------------------------|
| claude-haiku-5.5 | 54 | 1 | yes, every call |
| gpt-6-luna | 44 | 1 | yes, every call |
| gpt-5.6-luna | 44 | 1 | yes, every call |

The minimal prompt ("Reply with one word." / "Say OK.") is about 10 tokens of content,
so the fixed overhead is at most about 45 tokens: no hidden system prompt and no tool
definitions. The runtime labels our own calls `initiator: "user"`; there were no
sub-agent or runtime-initiated calls and no model substitution.

### Capabilities (worker prompt on MGSM)

| Worker | Usable JSON | Strict schema | Effort low / default / high: output tokens (reasoning) | Latency low / high |
|--------|-------------|---------------|--------------------------------------------------------|--------------------|
| claude-haiku-5.5 | 100% after extraction | yes | 145 (36) / 265 (149) / 351 (229) | 3.0 / 4.3 s |
| gpt-6-luna | 100% | yes | 181 (116) / 210 (167) / 287 (247) | 4.3 / 6.0 s |
| gpt-5.6-luna | 100% | yes | 122 (67) / 178 (120) / 253 (207) | 4.4 / 6.1 s |

* Reasoning effort changes token use on all three, so the Q4 effort arm is viable.
* All three reason by default; reasoning tokens are most of the output cost.
* `max_tokens` is not honoured by any of them (the capability override has no effect),
  so output caps are provider defaults.
* A reasoning trace came back for GPT-6 Luna on a short probe, and for Claude Haiku 5.5
  in the designer calls of the smoke search.

### Answer format: Claude Haiku 5.5 needed two parser fixes

The ADAS worker prompt displays the requested fields as a Python dict repr. Without a
provider JSON mode, Claude Haiku 5.5 often answers in that form (`{'answer': 'C'}`),
fenced or bare, and sometimes with apostrophes left unescaped inside single-quoted
values (`'officials' personalities'`), which neither JSON nor Python can parse. Those
replies became empty, wrong answers: 2 to 9% of Haiku's answers in every domain, while
the OpenAI models had none. The gateway now recovers both forms (unit-tested on the
verbatim replies; 0 of 96 unparseable afterwards, from 4 of 48). The Haiku screen cells
were rerun; the first run is kept as `_superseded_v1_*`.

| Haiku CoT | Before the fix | After |
|-----------|----------------|-------|
| MMLU | 87.8% | 95.6% |
| GPQA | 81.2% | 85.4% |
| DROP (F1) | 72.7% | 73.6% |
| MGSM | 92.7% | 93.0% |

This mirrors batch 2 of the vLLM campaign, where answer format was a quarter of
Mistral-7B's gap on MMLU: answer-format compliance has to be measured before any
capability claim.

A bare JSON number (`42`) instead of an object, seen from GPT-6 Luna under the bare
prompt, used to crash `LLMAgentBase.query`'s fallback and void the whole evaluation.
`RoleClient` now treats it as a format failure, scored as an empty answer like any
other.

### Throughput and running conditions

| Concurrency | Calls/min | p50 latency | Rate-limited |
|-------------|-----------|-------------|--------------|
| 32 | 260 | 3.7 s | 0 |
| 64 | 494 | 5.8 s | 0 |
| 128 | 533 | 8.8 s | 10% |

* Sustained runs near 500 calls/min drew short rate limits ("reset in under a minute",
  tens per cell, all recovered by retries). Cells now pace call starts at 200 per minute.
* Long stalls in the first screen pass (every call waiting about 15 minutes) were the
  Mac sleeping, not throttling: `caffeinate -i` did not hold with a 1-minute sleep
  timer, and the machine ran only in maintenance wakes. The runner now holds
  `caffeinate -s` (prevents system sleep on AC power).

### Smoke search

`cp0_smoke` (Haiku designs for Haiku, MGSM, 8 questions, 2 generations): 518 calls,
14.75 credits, both designs valid with no debug round, every capture field filled, and
designer reflexion turns continued in one session. The first design translated the
problem before solving it, the motif the open designers also found on MGSM.

### Cost per question (validation, one agent)

| Agent | Credits per question (Haiku, MGSM) |
|-------|------------------------------------|
| Chain-of-Thought | 0.023 |
| Self-Consistency | 0.109 |
| LLM Debate | 0.179 |

GPQA costs about three times as much per question (longer questions, longer reasoning).
Claude prompts above about 700 tokens are written to the prompt cache at 1.25x the input
price and never read back for unique worker prompts, adding 5 to 10% per call.

## Phase 1: worker and domain screen

128 validation questions (GPQA: 32 x 3 for CoT and bare, 32 x 5 for seeds); CoT and bare
averaged over three passes, the six non-CoT seed agents over one. "Best seed" excludes
CoT.

| Domain | Worker | Bare | CoT | Best seed | Seeds >= CoT + 5 | Credits |
|--------|--------|------|-----|-----------|------------------|---------|
| MGSM | claude-haiku-5.5 | 94.3% | 93.0% | Self-Consistency 95.3% | 0/6 | 97 |
| MGSM | gpt-6-luna | 91.1% | 91.9% | Quality-Diversity 93.0% | 0/6 | 57 |
| MGSM | gpt-5.6-luna | 93.0% | 93.0% | LLM Debate 94.5% | 0/6 | 95 |
| MMLU | claude-haiku-5.5 | 94.8% | 95.6% | LLM Debate 96.1% | 0/6 | 105 |
| MMLU | gpt-6-luna | 93.5% | 94.8% | Self-Consistency 96.1% | 0/6 | 43 |
| MMLU | gpt-5.6-luna | 95.1% | 95.3% | Quality-Diversity 96.9% | 0/6 | 78 |
| DROP (F1) | claude-haiku-5.5 | 75.5% | 73.6% | Self-Refine 78.5% | 0/6 | 104 |
| DROP (F1) | gpt-6-luna | 64.3% | 62.6% | Dynamic Roles 65.9% | 0/6 | 61 |
| DROP (F1) | gpt-5.6-luna | 66.8% | 67.4% | Step-back 67.9% | 0/6 | 113 |
| GPQA | claude-haiku-5.5 | 86.5% | 85.4% | Self-Consistency 87.5% | 0/6 | 318 |
| GPQA | gpt-6-luna | 84.4% | 85.4% | LLM Debate 89.4% | 0/6 | 194 |
| GPQA | gpt-5.6-luna | 86.5% | 88.5% | LLM Debate 88.1% | 0/6 | 336 |

1. Hand-designed scaffolds add nothing on these workers. Over all 72 seed results the
   gain over CoT averages -0.1 points (median +0.1, range -7.9 to +4.9), half positive
   and half negative, which is what no effect plus noise looks like. By seed agent the
   means run from +0.4 (Debate, Quality-Diversity) to -1.6 (Step-back). The largest
   single gain is Self-Refine on DROP with Haiku, +4.9 F1.
2. Telling the model to reason buys nothing either: bare minus CoT ranges from -2.1 to
   +1.9 points (mean -0.1). These models reason internally whatever the prompt says,
   which is the situation Q1 and Q4 ask about.
3. Three of four ADAS domains are at the ceiling for lightweight models: MGSM 91 to 93%,
   MMLU 94 to 96%, GPQA Diamond 85 to 89%. Only DROP (F1 63 to 75%, partly a span-format
   ceiling) leaves room.
4. The pre-registered rule picks DROP with Claude Haiku 5.5: the three DROP pairs are
   the only ones with CoT in 30 to 85%, all have 0/6 seeds at CoT + 5, and the tie goes
   to Haiku. Production CoT baseline (128 x 5): 74.7% F1, 95% CI 71.9 to 77.4%.

Caveats: one pass per seed agent gives roughly +/-8 points per agent at 128 questions,
so single entries are noisy; the conclusion rests on the 72-result distribution, not
on any one cell. CoT and bare have three passes. The GPQA validation split is 32
questions.

## Phase 2: Claude Haiku 5.5 designs for itself on DROP

Two replicates on the panel protocol (30 generations, 128 validation questions), with
the lean test the Phase 0 gate prescribes: the seven seeds and the five best designs by
validation, on 300 test questions. Informed designer (CoT baseline 74.7% F1).

| Replicate | Generations (attempts) | Selected (gen) | Validation | Test | vs Self-Consistency [95% CI] | vs CoT | vs format-CoT | Worker calls / question | Credits / question |
|-----------|------------------------|----------------|------------|------|------------------------------|--------|---------------|-------------------------|--------------------|
| a | 30 (31) | Token-F1 Minimum Bayes Risk Selection (7) | 91.1% | 95.1% | +14.7 [+11.4, +18.3] | +14.4 | +0.3 [-1.0, +1.7] | 6.0 | 0.151 |
| b | 30 (30) | Evidence-Weighted Contrastive Selection (21) | 91.1% | 93.4% | +12.8 [+9.5, +16.3] | +12.3 | -1.4 [-3.2, +0.3] | 5.0 | 0.283 |

Seeds on the same test questions: 76.6 to 82.6% F1. Format-CoT (below): 94.8%.

1. The classic ADAS result reproduces: every one of the 60 designs scores above the seed
   mean on validation, the jump comes at generation 1 or 2, and the selected designs beat
   every hand-designed agent on test by 12 to 18 points.
2. The whole gain is answer format. Both replicates converged on answers in the DROP F1
   metric's own form: replicate a reimplemented the metric's normalisation (articles
   stripped, number words to digits) to vote among candidates, replicate b told every
   solver to "write numbers as bare digits ... copy spans verbatim ... output only the
   final answer". The stock CoT seed with only that answer instruction added (format-CoT,
   one call) scores 94.8%: +12.1 [+8.8, +15.4] over stock CoT, and level with the searched
   designs (+0.3 and -1.4, both intervals include zero) at a seventh to a thirteenth of
   their cost.
3. What the designs look like depends on the run more than on the designer (Q8). Same
   designer, same prompt: replicate a wrote programmatic designs (median 66 statements,
   structural depth 10, question-type routing in 11 of 30 designs), replicate b prompt-only
   ones (median 17 statements, depth 0, role specialisation in 29 of 30). Both reached the
   same test level.
4. Haiku is a competent designer. No hallucinated framework calls, every successful design
   worked on the first try, one design in 61 attempts was abandoned (a runtime exception).
   The contamination scan flags 5 of 30 (a) and 3 of 30 (b) designs, all as
   self-consistency, which they build on from the seed archive.
5. Cost: 1,257 and 1,577 credits; 42,829 and 33,173 model calls; about 5 hours each.

## Phase 3.1: random recombination, no designer (Q7)

Forty generations in which each new agent recombines archive members at random, with no
designer. Same worker, seeds, validation questions and lean test as Phase 2.

| Run | Designs | Above seed mean (validation) | Above best seed (validation) | Best validation | Selected, test | vs Self-Consistency [95% CI] | vs format-CoT [95% CI] |
|-----|---------|------------------------------|------------------------------|-----------------|----------------|------------------------------|------------------------|
| Random recombination | 40 | 18 | 0 | 78.3% | 79.6% | -1.8 [-4.7, +1.2] | -15.2 [-18.9, -11.6] |
| Designer, replicate a | 30 | 30 | 29 | 91.1% | 95.1% | +14.7 [+11.4, +18.3] | +0.3 [-1.0, +1.7] |
| Designer, replicate b | 30 | 30 | 30 | 91.1% | 93.4% | +12.8 [+9.5, +16.3] | -1.4 [-3.2, +0.3] |

* Recombination never leaves the seeds' range. Its best design on validation (78.3%)
  sits just under the best seed (Self-Refine, 78.5%), and the five tested designs score
  79.6 to 83.3% on test against 75.2 to 83.2% for the seeds re-tested in the same cell.
* The Phase 2 gain therefore comes from the designer, not from the archive loop: the
  designer wrote the answer formatting that no seed contains. A keyword scan (digits,
  verbatim, normalise, shortest, only the answer and similar) finds answer-format wording
  or answer-normalising code in none of the 40 recombinations, and in 29 and 30 of the
  30 designs of replicates a and b.
* The same seed on the same 300 questions scores up to 2.7 points apart between cells
  (Quality-Diversity 83.2% here, 80.5% in replicate a; CoT 80.7, 81.2, 82.0 and 82.8%
  across four cells): service nondeterminism. Differences of 1 to 3 points between
  separately run agents are within this noise; the paired intervals above account only
  for question sampling.
* Cost: 739 credits, 29,451 calls; 5 of 45 attempts raised runtime exceptions.

## Phase 3.2: frontier models used plainly, and the format control

Same 300 test questions; one call per question for both agents.

| Worker | CoT | Format-CoT | Format gain [95% CI] | Credits / question |
|--------|-----|------------|----------------------|--------------------|
| claude-haiku-5.5 | 82.8% | 94.8% | +12.1 [+8.8, +15.4] | 0.022 |
| gpt-6.1-sol | 83.8% | 96.5% | +12.7 [+9.2, +16.4] | 0.184 |
| claude-sonnet-5.5 | 87.5% | 94.1% | +6.6 [+4.1, +9.1] | 0.508 |

* Q1 and Q2: the cheapest point on the accuracy-cost front is the small model with the
  right answer format. GPT-6.1 Sol buys +1.6 points [+0.2, +3.3] over it at 8 times the
  cost; Claude Sonnet 5.5 buys nothing at 23 times. The searched designs, a small model
  plus an expensive scaffold, are dominated: no better than Haiku's format-CoT and below
  GPT-6.1 Sol's (-1.3 [-2.7, -0.2] and -3.0 [-5.0, -1.3]).
* Claude Opus 5.5 could not be measured: through Copilot it answers "400 The requested
  model is not supported" whenever the ADAS worker system prompt is present, while minimal
  and long plain prompts work.
* Rate-limited calls are billed: the runtime retries internally and Copilot charges each
  attempt's input. An unpaced first Sonnet attempt cost 339 credits in failed calls (340 of
  the campaign's first 5,824). Cells now pace calls, and the gateway waits at least 30 s
  after a rate-limit error.

## Phase 3.3: reasoning effort, worker fixed (Q4)

Claude Haiku 5.5 at low and high effort, the same 300 test questions, no search
(reasoning tokens per call: 54 low, 182 high).

| Agent | Low | Default | High | High minus low [95% CI] |
|-------|-----|---------|------|-------------------------|
| Chain-of-Thought | 82.5% | 82.8% | 83.0% | +0.5 [-2.7, +3.7] |
| Format-CoT | 94.5% | 94.8% | 94.8% | +0.3 [-1.2, +1.9] |
| Replicate a's design (6 calls) | 94.5% | 95.1% | 94.7% | +0.2 [-1.2, +1.5] |

On validation, CoT is 75.0, 74.7 and 74.5% F1 at low, default and high effort. Neither
internal reasoning (3.4 times the reasoning tokens) nor external scaffolding (6 calls
instead of 1) buys accuracy on this task once the answer format is right; they are not
substitutes or complements here, both are inert.

The pre-registered effort searches: the same designer at default effort, 20 generations,
with the worker at low or at high effort for every validation and test call. Each row is
compared with format-CoT at the same effort.

| Worker effort | Best validation | Median validation | Selected (gen) | Test | vs Self-Consistency [95% CI] | vs format-CoT [95% CI] | Credits / question |
|---------------|-----------------|-------------------|----------------|------|------------------------------|------------------------|--------------------|
| Low | 86.3% | 83.0% | Evidence-Grounded Contrastive Audit (5) | 90.8% | +11.3 [+7.9, +14.7] | -3.6 [-6.4, -1.1] | 0.127 |
| Default (Phase 2 a, b) | 91.1%, 91.1% | 89.2%, 88.4% | see Phase 2 | 95.1%, 93.4% | +14.7, +12.8 | +0.3, -1.4 | 0.151, 0.283 |
| High | 90.9% | 88.8% | Agreement-Gated Routed Reconciliation (15) | 94.8% | +15.6 [+12.2, +19.0] | 0.0 [-1.7, +1.5] | 0.128 |

* At high effort the search repeats Phase 2: a design about 15 points above the
  hand-designed agents and level with the one-call format control.
* The low-effort search ends 3.6 points below its format control, and its design is 3.9
  points [1.3, 6.7] below the high-effort search's. The worker is not the reason: at low
  effort format-CoT and replicate a's design both score 94.5%. What differs is the
  design this run produced. The selected low-effort design gives one generic answer
  instruction ("the shortest form matching the required answer type"). The high-effort
  one routes each question by answer type and asks for "a single number" or "the
  shortest exact span". With one run per effort level, and Phase 2's two replicates just
  as different in character, this is a run-level difference, not an effect of effort.
* Answer to Q4 for Claude Haiku 5.5 on DROP: worker effort changes neither what plain
  prompting scores nor what search finds.
* Cost: 745 credits (low) and 937 credits (high), against 1,257 and 1,577 for the
  default-effort replicates, which ran 30 generations.

## Phase 3.4: transfer of the selected design (Q5)

Replicate a's selected design (Token-F1 Minimum Bayes Risk Selection), designed for
Claude Haiku 5.5, evaluated under the other two screened workers.

| Worker | Design | CoT | Self-Consistency | Format-CoT | Design vs SC [95% CI] | Design vs format-CoT [95% CI] |
|--------|--------|-----|------------------|------------|-----------------------|-------------------------------|
| claude-haiku-5.5 (source) | 95.1% | 82.8% | 80.4% | 94.8% | +14.7 [+11.4, +18.3] | +0.3 [-1.0, +1.7] |
| gpt-6-luna | 83.3% | 69.7% | 69.2% | 92.9% | +14.1 [+11.1, +17.2] | -9.6 [-14.1, -5.3] |
| gpt-5.6-luna | 84.9% | 74.1% | 72.8% | 91.8% | +12.2 [+9.4, +15.0] | -6.8 [-10.8, -3.1] |

The design's lead over the hand-designed agents transfers intact, which by the usual
ADAS reading would count as a transferable discovery. Against the one-line format
control it does not transfer: on both OpenAI workers it is 7 to 10 points worse. Its
answer-normalising code fits the source worker's output habits; the plain instruction
fits every worker. The OpenAI workers' format gain is larger than Haiku's (+18 to +23
points over CoT).

## Spend

| Item | Credits |
|------|---------|
| Phase 0 probes | 23 |
| Phase 0 smoke search | 15 |
| Phase 1 screen (final cells) | 1,599 |
| Phase 1 production baseline | 14 |
| Superseded Haiku screen cells (before the parser fix) | 640 |
| Phase 2 designer runs (two replicates) | 2,834 |
| Phase 3.1 random recombination | 739 |
| Phase 3.2 plain frontier models and format control | 787 |
| Phase 3.3 effort baselines, contrasts and searches | 1,823 |
| Phase 3.4 transfer | 250 |
| Total | 8,724 |

The total is 17% of the 50,000-credit budget ($87.24). It includes about 23 credits of
DROP CoT calls lost to a `run_baseline` crash (fixed) and rerun, and 339 credits billed
for rate-limited calls in the first, unpaced Claude Sonnet 5.5 run (Phase 3.2). Totals
come from every cell's `copilot_ledger.jsonl` (`credits_charged`).

## Running it on a laptop: what we learned

* Rate limits. Claude Haiku 5.5 cells sustained about 420 calls per minute combined
  (175 per cell) with no errors; 533 per minute drew 10% rate-limit errors. Claude Sonnet
  5.5's limit is far lower, and its cells ran at 30 per minute. Copilot billed Sonnet's
  rate-limited calls (339 credits) but not Haiku's (396 errors, 0 credits). The gateway
  now waits at least 30 s after a rate-limit error.
* Sleep. `caffeinate -i` did not stop idle sleep; `caffeinate -s` (on AC power) does, and
  `run_cell.py` uses it. Closing the lid or running on battery still sleeps the Mac.
  Several outages of 15 to 97 minutes cost no questions: the gateway retries a call for
  up to 50 minutes and the runs carried on after wake. The queued calls then trip the
  rate limit for a few seconds.
* Pace changes. The pace is fixed when a cell's gateway starts. To change it, stop the
  runner with Ctrl+C or SIGINT and rerun the cell with `--max_calls_per_minute`; the
  search resumes from its archive and loses only the generation in progress. (During
  this campaign SIGTERM skipped the cleanup and orphaned the gateway and search; since
  2026-10-30 the runner treats SIGTERM like SIGINT.)
* Model quirks. Claude Opus 5.5 rejects the ADAS worker system prompt ("400 The requested
  model is not supported"). Claude Haiku 5.5 often answers in Python-dict syntax with
  unescaped apostrophes, which the gateway repairs. No model honoured `max_tokens`.

## Decisions needed before Phase 2

These were settled by the pre-registered rules on 2026-10-29: Phase 2 ran the lean
protocol (the throughput gate), and it ran at all because the original campaign's gate
says to continue when nothing beats the bare model. The original wording follows.

1. The throughput gate. The plan set about 1,000 calls/min as the bar for the full
   panel protocol and, below it, the lean protocol (seeds and the five best designs
   tested on 300 questions). Measured: about 490 calls/min in bursts, 200 sustained
   with pacing. By the rule, Phase 2 runs lean. Estimates for DROP with Haiku, per run:
   full protocol about 4,000 credits and 20+ hours; lean about 1,500 credits and 5 to 6
   hours.
2. Whether Phase 2 still asks the right question. With no seed agent ahead of CoT
   anywhere, a designer run on DROP asks whether automated design finds gains hand
   design did not. A null result is informative (it extends the vLLM panels' null to
   hosted models), and `experimental_plan.md` already says to lean into Q7, Q8 and Q6
   when nothing beats the bare model. The alternative is to spend the Phase 2 budget
   on Phase 3.2 (frontier models used plainly) and on more designers, where the
   character of designs (Q8) rather than their accuracy is the question.
