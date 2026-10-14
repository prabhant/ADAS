# Findings so far — batch 1 (2026-10-13)

What has been run: **screens, not searches.** Worker baselines (no search),
first-shot design proposals from 9 candidate designers (6 each; compiled and
inspected, not executed), and one 2-generation end-to-end search to prove the
pipeline. These results choose the setup for the search campaign; they are not
yet answers to Q1–Q9. Raw data: `experiments/results/s05_*/`,
`experiments/results/screen_designers/*.json`; full history in `RUNLOG.md`.

## 1. Workers: almost all near the ceiling

128 questions per cell. The MGSM cells for Mistral, Phi-4, Qwen3.5-27B and
gpt-oss-20b use 5 repeats; the rest use 3.

| Worker | MGSM bare | MGSM CoT | MMLU CoT |
|---|---|---|---|
| Mistral-7B-v0.2 | 6.6% | 14.2% | **46.4%** |
| Phi-4 | 31.4% | 82.7% | 90.1% |
| Qwen3.5-27B | 41.4% | 81.2% | 93.0% |
| Qwen3-30B-A3B | — | 76.3% | 85.2% |
| Nemotron-3-Nano-30B | — | 75.5% | 87.2% |
| gpt-oss-20b (low / high effort) | — | 86.2% / 90.9% | — |

- **Only Mistral-7B on MMLU (46%) is mid-range.** Mistral on MGSM is near the
  floor (14%); every other worker scores 75–93%, leaving a search little to
  improve.
- **The planned worker ladder has two distinguishable rungs, not three.**
  Phi-4 and Qwen3.5-27B tie under CoT (+1.4 points, 95% CI −3.6 to +6.6,
  paired bootstrap over questions).
- **Chain-of-thought helps strong workers far more than the weak one:** +51
  points for Phi-4, +40 for Qwen3.5-27B, +8 for Mistral.
- **Reasoning effort is load-bearing:** gpt-oss-20b high vs low is +4.7 points
  (CI +1.9 to +7.8), so the reasoning-effort arm (Q4) is viable.
- Qwen3-14B returned empty answers as a worker; its 13% / 27% reflect that
  failure, not capability.

## 2. Designers: all can design; designs differ in size and style

Each designer got the identical real ADAS prompt (seed archive; worker
documented as Mistral-7B at 14.2% on MGSM) and made 6 proposals at
temperature 0.8. Sorted by active parameters. Sizes are language-model
parameters counted from the checkpoints, embeddings included, so they differ
slightly from vendor figures (e.g. gpt-oss-20b is 3.6B active per OpenAI).

| Designer | Params total / active | Usable | Agents per design | Output tokens (incl. reasoning) | Unusable because |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 30.5B / 3.4B | 6/6 | 2.8 | 2,800 | — |
| Qwen3.5-35B-A3B | 34.7B / 3.5B | 6/6 | 3.3 | 3,900 | — |
| Nemotron-3-Nano-30B | 31.6B / 3.6B | 6/6 | 2.2 | 3,700 | — |
| gpt-oss-20b | 20.9B / 4.2B | 4/6 | 3.5 | 3,600 | syntax ×2 |
| gpt-oss-120b | 116.8B / 5.7B | 5/6 | 3.2 | 1,500 | syntax |
| Nemotron-3-Super-120B | 120.7B / 12.8B | 6/6 | 4.2 | 2,400 | — |
| Mixtral-8x7B (2023, no reasoning) | 46.7B / 12.9B | 3/6 | 2.7 | 600 | syntax ×2, wrong shape |
| Qwen3-14B | 14.8B / 14.8B | 6/6 | 5.5 | 1,900 | — |
| Qwen3.5-27B | 26.9B / 26.9B | 5/6 | 4.4 | 2,600 | syntax |

**Run-to-run noise is large.** The first (lost) batch ran four of these
designers with identical settings. From its logs: gpt-oss-20b was 6/6 usable
(now 4/6), Qwen3-14B averaged 3.8 agents (now 5.5), Nemotron-Nano used ~6.6k
output tokens (now 3.7k); Qwen3-30B was unchanged. Single differences in this
table are not meaningful.

**Q7 — what the designs look like.** All designers draw on the same core
repertoire: ensembling, self-critique and verification dominate. Two
behaviours stand out:

- **Designers adapt to the task.** 8 of 47 usable designs translate the
  problem into English before solving (5 of 9 designers; checked in the code,
  not just the name). Ten of MGSM's 11 languages are not English, and no seed
  agent does this.
- **Designers read the worker information.** 18 of 47 design rationales
  mention the worker or its 14.2% accuracy (5 of 9 designers; Nemotron-Super in
  all 6). This is the channel the informed-vs-blind comparison tests.

Many designs map to published techniques (self-consistency, verifiers,
program-aided solving); the contamination scan must run before any design is
called novel.

**Q8 — total size does not predict design size; active size might.** By total
parameters there is no pattern: gpt-oss-120b's designs are no bigger than
gpt-oss-20b's (3.2 vs 3.5 agents), and it writes fewer than half the tokens.
By active parameters, the three reasoning designers with ≥12B active propose
the most agents (4.2–5.5) and the five with under 6B the fewest (2.2–3.5). Treat
this as a hypothesis, not a finding:

- the split was spotted after seeing the data;
- the rerun moved Qwen3-14B by as much as the gap itself;
- Qwen3.5-35B's six proposals alone span 2 to 7 agents;
- code length shows no such pattern;
- Mixtral does not follow it.

**Q6 — what breaks.** Every model can act as a designer. Failures are Python
syntax errors in the generated code (unterminated strings, a stray brace,
missing indentation), not malformed JSON; invented framework calls are rare
(≤0.6 per design). Two compiled designs index an agent's output as a
dictionary (`info.content['name']`), which works only if the worker returns
nested JSON rather than text, so "usable" is an upper bound until designs run.
The older non-reasoning Mixtral is the weakest: 3/6, and all three usable
designs revolve around external resources. One genuinely calls sympy; the
other two route through "external tools" and "external knowledge" that are
just labels on ordinary LLM calls.

## 3. Problems fixed before collecting search data

Four bugs would each have invalidated results silently; all were fixed before
any search data existed:

1. **Scoring:** a worker answering with a JSON number instead of a string was
   scored wrong, which reads as a capability gap.
2. **Accuracy statistic:** the median of a per-question 0/1 vector was
   reported, which can only be 0 or 1.
3. **Prompt leak:** the designer saw analysis fields (worker identity, other
   designs' reasoning traces, per-question scores). This broke the blind
   condition and could overflow 32k-context designers within a few
   generations. The end-to-end test now guards against it.
4. **Silent no-op:** vLLM 0.28 accepts and ignores the old structured-output
   parameter (`guided_json`), so schemas were never enforced.

## 4. Cost signal

The 2-generation smoke search (gpt-oss-20b designing for Mistral-7B, 8
questions) took 15 min end to end, including loading both models. Each
generation used ~7.5k designer tokens and 10–43 s of worker evaluation, at
5.6–7.4 worker calls per question. A full Stage 1 cell (64 questions, 10
generations) has not been timed.

## 5. Decisions for the next batch (2026-10-13)

- **Worker:** Mistral-7B on MMLU, the only mid-range setting.
- **Size axis:** the final analysis reports designer size as both total and
  active parameters. Analysis rows now carry both. Across the eight reasoning
  designers the two are uncorrelated (r = −0.01 on log scale), so their
  effects can be told apart.
- **Mixtral stays the held-out judge**, so it is not a designer. The panel is
  the eight reasoning designers.
- **Baseline rerun (done, job 27980629):** Mistral-7B CoT on MMLU is 47.2%
  (95% CI 43.3–51.1%; 128 questions × 5), matching the screen's 46.4%. The
  baseline lookup is now domain-aware: it previously would have told an MMLU
  designer the MGSM number.
- **Answer format is a quarter of the gap.** 172 of 640 answers (27%) had no
  extractable letter and were scored wrong. Only 7 were failed calls; the rest
  are answers in a format the stock ADAS extractor rejects (it accepts a bare
  letter or `D)`; the saved example is `D. Mass and acceleration`). The
  failures recur on the same questions: 9 questions fail in all 5 passes, and
  61 of 128 fail at least once. Of the answers that were parsed, 64.5% are
  correct. So 47% measures knowledge and format compliance together, and a
  search can gain simply by fixing the output format.
- **Still open before searching:** calibrate the motif detector, which has
  tagged a voting ensemble as a critique loop.

## Caveats

Six first-shot proposals per designer describe first instincts, not what a
search converges to; given the run-to-run noise above, designer differences
are descriptive only. The worker screen's per-question data was lost
(accuracies survive in the logs). Mistral-7B-v0.3, both Phi-3 models, Gemma-3
and Llama-3.2 could not be served in this environment.

# Designer panel — batch 5 (2026-10-14)

Eight designers, one ADAS search each (30 generations target, Mistral-7B worker,
MMLU, 128 validation / 800 test questions, shared seed archive). Worker CoT
baseline 55.6%; the seven seed agents score 53–56% on test.

| Designer (total / active B) | Generations (attempts) | Stopped by | Best on validation | Its test score | Designs ≥ seed mean on test | Worker calls / question | Broken designs |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (31 / 3.4) | 19 (90) | output cap: 42% of calls truncated | 63.3% | test failed | 11/17 | 23.2 | 3/19 |
| Qwen3.5-35B-A3B (35 / 3.5) | 23 (90) | context overflow from gen 24 | 68.0% | 55.5% | 12/23 | 11.9 | 0/23 |
| Nemotron-3-Nano (32 / 3.6) | 22 (90) | context overflow from gen 23 | 54.7% | 54.5% | 0/22 | 6.1 | 14/22 |
| gpt-oss-20b (21 / 4.2) | 30 (31) | — | 64.8% | 56.4% | 14/29 | 9.8 | 6/30 |
| gpt-oss-120b (117 / 5.7) | 30 (31) | — | 64.1% | 54.5% | 12/30 | 8.2 | 0/30 |
| Nemotron-3-Super (121 / 12.8) | 24 (90) | context overflow from gen 25 | 65.6% | 56.5% | 24/24 | 31.4 | 0/24 |
| Qwen3-14B (15 / 14.8) | 30 (51) | — | 64.1% | 54.4% | 15/30 | 15.2 | 0/30 |
| Qwen3.5-27B (27 / 26.9) | 30 (34) | — | 65.6% | 55.9% | 19/30 | 7.0 | 0/30 |

"Broken" = below 30% on validation (chance is 25%).

1. **No designer beat the hand-designed agents.** Each designer's selected
   design versus Self-Consistency on the 800 test questions: −1.4 to +0.9
   points, every 95% CI includes zero. The best validation scores (63–68%)
   are what picking the best of 30 equally good designs on 128 noisy
   questions gives by chance (simulated median 64.1%, 90% range 61–68%).
   With a weak worker on knowledge-heavy MMLU there is little for a scaffold
   to add, consistent with ADAS reporting its smallest gains on MMLU.
2. **Validation is informative, the differences are just small.** Among the
   172 working designs, validation predicts test (Spearman 0.59), but true
   test accuracy spans only ~52–59.5%. Spending buys a little: designs using
   over 8k worker tokens per question average 55.8% on test, under 3k 54.0%.
3. **Designers differ far more in character than in results (Q7, Q8).**
   Worker cost per question varies 5× (6 to 31 calls). Nemotron-Super is the
   most expensive and the only designer whose every design ended at or above
   the seed average. The dense Qwen3-14B and Qwen3.5-27B build the largest
   designs (6.9 agents vs 4.2–5.1), consistent with the screen's active-size
   hypothesis; Nemotron-Super has the fewest agents but the most calls, so
   "design size" depends on whether agents or calls are counted. All
   designers converge on ensembling plus critique.
4. **Silent failure is designer-specific (Q6).** 14 of Nemotron-Nano's 22
   designs run without error yet score near chance: they wrap the worker's
   answer in extra parsing and voting that falls back to a wrong answer.
   gpt-oss-20b has 6 such designs; most designers have none.
5. **The protocol did not fit three designers (Q6).** Archived designs add
   1.7–2.6k prompt tokens each, twice the screen-based estimate, so three
   designers hit their 65,536-token window at generations 23–25 and failed
   every later attempt. Qwen3-30B-A3B's 4,096-token output cap truncated 42%
   of its calls. Later generations changed little where they ran: the best
   validation score was already reached by generation 19 for six of eight.

Caveats: one run per designer (the ADAS protocol), so designer differences are
descriptive; unequal generation counts until the incomplete runs are finished;
the motif detector still over-tags critique loops.
