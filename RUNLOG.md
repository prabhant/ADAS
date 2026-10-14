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

### 2026-10-12 13:37 - batch 1 submitted at commit 8b4068b

```
e2e_cpu	27933483
stage0_smoke	27933485
worker_screen	27933489
designers_1gpu	27933494
designers_2gpu	27933497
```

Inspect with: bash experiments/slurm/status.sh

### What to check before batch 2 (do not submit more until these are read)

| Job | Pass looks like | If it fails |
|---|---|---|
| e2e_cpu | ends `PASSED`, incl. "no analysis fields reach the designer" | code bug; fix before any search |
| stage0_smoke | `stage 0 complete`; budget has successes; read per-generation `wallclock_s` to cost Stage 1 | first full production path; read `.err` + `_serving_logs/` |
| worker_screen | 15 `s05s_*/cot_baseline.json`; want a worker at ~40-70% CoT | a failed task is a model that will not serve here |
| designers_1gpu / 2gpu | `screen_designers/<key>.json`; `usable_rate` = can it design at all (Q6) | per-model; gpt-oss-120b at tp=2 serving = feasibility answered |

Decisions batch 1 should settle: which worker (headroom), MGSM vs MMLU, which
designers go into the panel, whether gpt-oss-120b or nemotron-super is the
large designer.

Open, not blocking batch 1:
- Mixtral is both a designer candidate and the held-out judge. If it joins the
  panel, the judge must change (e.g. Qwen2.5-72B-Instruct, run separately).
- Meta output budget for 32k-context designers: prompt + max_tokens must fit;
  set per designer from the prompt_tokens the screen reports.

### 2026-10-12 - batch 1 inspected: mostly failed, five root causes

Results that did come back (numbers from job logs; the JSON files were lost, see 1):

| Worker CoT | MGSM | MMLU |
|---|---|---|
| mistral-7b | 14.2% (s05) | **46.4%** |
| phi-4 | 82.7% (s05) | 90.1% |
| qwen3.5-27b | 81.2% (s05) | 93.0% |
| qwen3-30b-a3b | 76.3% | 85.2% |
| nemotron-3-nano-30b | 75.5% | 87.2% |
| qwen3-14b | 13.3% (?) | 26.6% (? = chance) - suspicious, check samples |

| Designer | usable | agents | note |
|---|---|---|---|
| gpt-oss-20b | 6/6 | 3.2 | most varied; proposes code-execution designs |
| qwen3-14b | 6/6 | 3.8 | converges on verify-refine + self-consistency |
| qwen3-30b-a3b | 6/6 | 2.8 | equation/code-centred |
| nemotron-3-nano-30b | 5/6 | 2.2 | verbose (~6.6k tokens/proposal), 1 truncated |

Root causes:
1. Results landed in a login-node tmp dir: submit_batch1.sh used --export=ALL,
   which copied a stale `RESULTS_ROOT=$(mktemp -d)` from a verification command
   in the same terminal into every screen job. Never use --export=ALL.
2. e2e: `get_reflexion_prompt(None)` crashed after the prompt-leak fix, so no
   generation ever succeeded. The e2e test caught it (0 generated).
3. Every tp=2 model failed: vLLM imports flashinfer.comm at startup, and
   flashinfer/comm/fd_exchange.py uses `array.array[int]`, which needs
   Python 3.12; this env is 3.11.
4. qwen3.5-27b as designer: with the reasoning parser on, the health check's
   short JSON probe spent all tokens thinking -> content None -> "failed".
5. stage0_smoke: health_check wrote its report into a dir not yet created.
   Also: dead servers were polled for the full 2400 s (40 min) instead of
   failing fast - health check did not watch the server process.
Model-specific: mistral-7b-v0.3 tokenizer conversion error; phi-3 mini/medium
fail under transformers 5.x (longrope config missing
original_max_position_embeddings).

### 2026-10-12 - fixes for batch 1 failures (verified on CPU before resubmitting)

- health_check: watches the server process (zombie-aware via /proc) and fails
  in seconds instead of polling a dead server for 40 min; JSON probes now 4096
  tokens and use the production response_format json_schema; report dir created.
- get_reflexion_prompt(None) fixed in both domains (first generation).
- experiments/env/patch_env.sh: flashinfer fd_exchange.py gets
  `from __future__ import annotations` -> tp>=2 servers can start on Python 3.11.
  Applied to the conda env (no jobs were running). Re-run after any reinstall.
- No more --export anywhere: STAGE / screen group are positional args; results
  root only from ADAS_RESULTS_ROOT. submit_batch1.sh replaced by
  `submit_batch.sh <name>`.
- Dropped as not servable here: mistral-7b-v0.3, phi-3-mini, phi-3-medium.
  qwen3-14b dropped as a worker (empty answers), kept as a designer.

Worker screen NOT rerun: its numbers are recovered above. Key finding: the only
worker with headroom on either domain is mistral-7b on MMLU (46.4% CoT).

Next: batch1b = e2e, stage0_smoke, 1gpu designers (0-4, to recover the actual
designs), 2gpu designers (0-3). Same scope as batch 1, not new work.

### 2026-10-12 17:09 - batch1b submitted at commit 2be863d

```
e2e_cpu	27952332
stage0_smoke	27952334
designers_1gpu	27952339
designers_2gpu	27952342
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-12 - batch1b inspected: 9 of 11 tasks passed

- e2e PASSED, including "no analysis fields reach the designer".
- stage0_smoke COMPLETED: 2/2 generations succeeded first try; the full
  production path works. Cost signal: ~7.5k designer tokens and 10-43 s of
  worker evaluation per generation at 8 questions, ~6-7 worker calls/question.
- gpt-oss-120b serves at tp=2 (8 min incl. load) -> top designer is feasible.
- flashinfer patch confirmed: tp=2 models now start.

| Designer | usable | agents | dominant motifs |
|---|---|---|---|
| gpt-oss-120b | 5/6 | 3.2 | ensembling, critic loop |
| gpt-oss-20b | 4/6 | 3.5 | early exit, routing, verification, role specialisation |
| nemotron-3-nano-30b | 6/6 | 2.2 | critic loop, ensembling, early exit |
| qwen3-14b | 6/6 | 5.5 | majority voting, critic loop, decomposition |
| qwen3-30b-a3b | 6/6 | 2.8 | ensembling, verification |
| qwen3.5-27b | 5/6 | 4.4 | verification, ensembling, critic loop |
| qwen3.5-35b-a3b | 6/6 | 3.3 | critic loop, ensembling, routing, verification |

Unusable proposals are all real Python syntax errors (unterminated strings,
unmatched brace) - designer behaviour (Q6 data), which the search's debug loop
is there to repair. Not infrastructure.

Two failures:
- nemotron-3-super-120b: 128 GB loaded at ~34 s/shard (13m50s) from the shared
  filesystem; vLLM's engine-ready timeout is 600 s, so the API server gave up
  while loading. Its process stayed alive after the error, so the fail-fast
  PID check never fired and the job ran 50 min.
- mixtral-8x7b: its chat template rejects a system message (HTTP 400). The
  newer `main` snapshot's tokenizer template handles one; serve weights from
  the old snapshot with the tokenizer from the new one.

### 2026-10-12 18:51 - batch1c submitted at commit efcb48b

```
designers_2gpu	27963412
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-13 - batch1c complete; batch 1 done; writeup in findings.md

- batch1c: nemotron-3-super-120b (29 min, incl. ~14 min load) and mixtral-8x7b
  (with newer tokenizer) both completed. All 9 designers screened.
- Findings written up in `findings.md` (workers, designers, fixed bugs, cost,
  implications). Next batch not submitted - awaiting user decision on worker
  (recommend mistral-7b on MMLU), designer panel, and the judge if Mixtral stays.
- findings.md verified line by line against data. Corrections vs first draft:
  CIs clustered by question (phi-4 vs qwen3.5-27b +1.4 [-3.6, +6.6]); English-
  translation designs 8/47 from 5 designers (checked in code); batch 1's lost
  designer run is a valid replicate (same probe/prompt) and shows large
  run-to-run noise (qwen3-14b 3.8 -> 5.5 agents). Q8: active, not total,
  parameters may track design size - hypothesis only.
- Bookkeeping bug noticed: Stage 0.5 cot_baseline.json records `repeats: 1`
  though it holds 128 x 5 = 640 attempts. Not fixed yet.

### 2026-10-13 01:59 - batch2 submitted at commit 2113af9

```
mmlu_baseline	27980629
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-13 - decisions taken; batch2 contents

User decisions: worker = mistral-7b on MMLU; report designer size as BOTH total
and active params; Mixtral stays the held-out judge (not a designer; panel =
the 8 reasoning designers); rerun the Mistral MMLU baseline first.

Code changes at 2113af9 (before submitting batch2):
- BUG FIXED: worker_cot_baseline ignored the domain, so an MMLU search would
  have told the informed designer Mistral's MGSM number (14.2%). Lookup is now
  `s05_cot_<domain>_<worker>` for non-MGSM domains and refuses a file whose
  `domain` field does not match.
- New cell s05_cot_mmlu_mistral-7b (stage 0.5 index 8; 128 q x 5).
- Baseline JSONs now record `n_repreat` (fixes the misleading `repeats: 1`).
- models.yaml params_total/params_active are numbers (billions), one convention
  for all models: LM params incl. embeddings, excl. vision/MTP, counted from
  checkpoint headers. gpt-oss-20b 20.91/4.19, gpt-oss-120b 116.83/5.71 (OpenAI's
  3.6/5.1 exclude embeddings). Across the 8 designers log-total vs log-active
  r = -0.01. aggregate/attribution rows carry meta_params_total/_active.
- Mixtral roles -> [judge]; judge entry notes the model_path + tokenizer fix.

Found for the panel search (not yet fixed): designers with a 4096-token output
cap exceeded it in the screen - qwen3.5-35b 2/6 (up to 9.6k), nemotron-nano 2/6,
qwen3.5-27b 1/6. Raise their caps before searching.

When 27980629 finishes: `bash experiments/slurm/status.sh` shows the
s05_cot_mmlu_mistral-7b accuracy; expect ~46% (screen value 46.4%, 128 x 3).

### 2026-10-13 - batch2 inspected: baseline valid; answer-format issue found

- 27980629 COMPLETED in 2m44s. s05_cot_mmlu_mistral-7b = 47.2% (CI 43.3-51.1%),
  128 q x 5 (per-pass 45.3-49.2%). Consistent with the screen's 46.4%.
- 172/640 answers "error in q": no letter extracted, scored 0. Only 7 are failed
  calls (6 missing field, 1 JSON). The rest are format misses by the STOCK ADAS
  extractor (bare letter or "X)" only); example "D. Mass and acceleration".
  Systematic: 9 questions unparsed in all 5 passes, 61/128 in at least one.
  Accuracy among parsed answers 64.5%.
- Consequence: the search's fitness rewards format fixing as well as reasoning.
  Needs a user decision before the panel search: keep stock extractor
  (faithful ADAS) or accept "X." / "(X)" / "answer is X" and re-measure.

### 2026-10-13 13:55 - batch3 submitted at commit 9a043be

```
e2e_cpu	28003128
mmlu_baseline	28003131
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-13 - batch3 contents (user decisions on batch2 findings)

User decisions: accept more MMLU answer formats and re-measure; raise designer
output caps for qwen3.5-35b, qwen3.5-27b, nemotron-nano; all designers start
from the same evaluated seed archive.

Code at 9a043be:
- _mmlu/utils.extract_choice: bare letter, "(D)", "D.", "D) text", "D: text",
  "the answer is D", exact option text; stock "X)" rule last. Uppercase only,
  so "a"/"Apple" are never read as choices (29/29 unit cases incl. traps).
  Every miss (up to 200) is now saved in the baseline's samples for auditing.
- Old stock-extractor baseline moved to
  experiments/results/_superseded/s05_cot_mmlu_mistral-7b_stock_extractor_27980629.
- meta_max_output_tokens 12288 for the three designers (worker caps stay 4096).
- Shared seed cache at $RESULTS_ROOT/_seed_cache (all cells). New flags:
  --require_seed_cache (cell field require_seed_cache) refuses to evaluate seeds
  itself; --seeds_only (cell condition seed_eval) evaluates seeds then stops.
  Cache writes are atomic. e2e section F tests all three.
  The panel needs: one seed_eval cell first, then designer cells with
  require_seed_cache, chained by --dependency=afterok.

ADAS protocol (paper arXiv 2408.08435 + original code), for choosing ours:
30 iterations (ARC 25); ONE search run per domain, no repeated searches;
valid 128 / test 800 for MMLU, evaluated once (GPQA and ARC: 5 times);
debug up to 5 rounds in the paper (code default debug_max 3); seeds are 7
hand-designed agents; ~$300 per reasoning-domain run.

### 2026-10-13 14:38 - batch4 submitted at commit 9757607

```
e2e_cpu	28004675
panel_seeds	28004677
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-13 - batch3 e2e PASSED; user chose the ADAS protocol; batch4

- batch3 e2e 28003128 PASSED (4m19s) incl. section F (shared seed archive).
  Baseline re-measure 28003131 still PENDING at time of writing.
- User: "Follow ADAS". Panel = stage "panel" in cells.yaml: 30 generations,
  ONE run per designer, MMLU valid 128 / test 800, n_repreat 1, max_attempts 90,
  debug_max 3 (released code default; the paper says up to 5).
  Index 0 = pan_seeds_mistral-7b (seed_eval, 1 GPU); 1-5 = 1-GPU designers
  (2 GPUs with worker); 6-8 = 2-GPU designers (3 GPUs). Mixtral excluded (judge).
- Context: the archive prompt grows ~0.8-1.2k tokens per design -> ~30-41k by
  generation 30. Qwen3-14B / Qwen3-30B-A3B were served at 40,960 and would have
  overflowed (overflow = failed attempt, i.e. fewer generations). Designer role
  only: --hf-overrides rope_parameters YaRN x2 -> 65,536 (Qwen's documented
  method; vLLM config accepted offline; without it 65,536 is rejected, so a
  silent no-op is impossible). Caveat: static YaRN may slightly change short-
  context behaviour of these two designers. Others already >= 65,536.
- BUG FIXED (stock ADAS): evaluate() skipped a design whose test run raised, so
  on resume every later test result was matched to the wrong design. Now
  recorded with `test_error`. Test evaluation also stops at the walltime margin
  and maybe_resubmit resubmits until every design has a test record; it now
  keeps the job's --gpus/--cpus-per-task. e2e section G covers this.
- batch4 at 9757607: e2e 28004675, panel_seeds 28004677.
- NEXT: when 28003131 (baseline) and 28004677 (seeds) finish, inspect both, then
  `bash experiments/slurm/submit_batch.sh batch5` (8 designers; refuses to
  submit if the baseline or the shared seed archive is missing).

### 2026-10-13 22:56 - batch6 submitted at commit 5d9f1c2

```
e2e_cpu	28049888
mmlu_baseline	28049889
panel_seeds	28049890
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-13 22:50 - batch3/4 inspected; seed job failed; batch6

- 28003131 baseline (extractor v2): 54.7% (CI 50.8-58.6). 79/640 unreadable:
  75 free text incl. "D, Mass and acceleration" (letter + comma, not handled),
  4 empty. Moved to _superseded/s05_cot_mmlu_mistral-7b_extractor_v2_28003131.
- 28004675 e2e PASSED incl. section G (evaluation deadline/resume).
- 28004677 seeds FAILED after 6/7 seeds: "Dynamic Assignment of Roles" raised
  `'dict' object has no attribute 'lower'`. Mistral sometimes returns a nested
  JSON object for a field; the seed calls .lower() on it. Stock ADAS never hit
  this (GPT-3.5 returned strings). One crashing question fails the whole agent.
- FIX (5d9f1c2): LLMAgentBase converts every worker field to text (dict/list ->
  JSON text, null -> ""), applied identically to all designs. MMLU extractor
  also reads "D, text" and refuses multi-letter answers ("A, B or D", "A or B",
  "B/C"); 39/39 unit cases. Offline re-score of the saved misses: 56.6%; the
  43 left are "None of the above", refusals and multi-letter answers.
- Stale partial seed cache and its archive moved to _superseded/ so nothing
  computed under the old handling is reused. batch5 guard now requires all 7
  seeds in the cache, not just the file.
- batch6 at 5d9f1c2: e2e 28049888, mmlu_baseline 28049889, panel_seeds 28049890.
- NEXT: inspect all three, then `bash experiments/slurm/submit_batch.sh batch5`.

### 2026-10-14 01:23 - batch5 submitted at commit ec88ae3

```
designers_1gpu	28056328
designers_2gpu	28056329
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-14 01:25 - batch6 inspected (all OK); designer panel (batch5) submitted

- 28049889 baseline: s05_cot_mmlu_mistral-7b = 55.6% (CI 51.7-59.5), 53/640
  unreadable (was 172 with the stock extractor). This is the number the
  informed designers are told.
- 28049890 seeds: all 7 in the shared cache (valid 128, 1 pass):
  CoT 52.3, CoT-SC 58.6, Self-Refine 53.9, LLM Debate 58.6, Step-back 60.2,
  Quality-Diversity 55.5, Role Assignment 56.2. Unreadable 6-13 of 128 each.
- 28049888 e2e PASSED.
- batch5 at ec88ae3: designers_1gpu 28056328 (array 1-5: gpt-oss-20b,
  qwen3-14b, qwen3-30b-a3b, qwen3.5-27b, nemotron-nano; 2 GPU, 32 CPU) and
  designers_2gpu 28056329 (array 6-8: gpt-oss-120b, nemotron-super,
  qwen3.5-35b; 3 GPU, 48 CPU). 20 h each; resubmit themselves at the deadline.
- What to watch first: each task's .out shows "Initial Archive (cached)" x7
  (shared seeds used), then generation 1. The Qwen3 designers' server log must
  show max_model_len 65536 (YaRN); nemotron-super loads ~14 min.

### 2026-10-14 15:40 - designer panel (batch5) inspected

- All 8 tasks COMPLETED in 2-7 h (none hit the 20 h limit). Seeds loaded from
  the shared cache in every cell.
- Complete (30 generations): gpt-oss-20b, gpt-oss-120b, qwen3-14b, qwen3.5-27b.
- Incomplete, budget exhausted at 90 attempts:
  nemotron-nano 22, nemotron-super 24, qwen3.5-35b 23: context overflow every
  attempt from generations 23-25 (prompt 53-57k + output cap > 65,536).
  Archived designs add 1.7-2.6k tokens each, 2x the screen-based estimate.
  qwen3-30b-a3b 19: 71/170 designer calls truncated at the 4,096 output cap
  ("empty message body" 39, unterminated JSON 31).
- Result: no designer's selected design beats CoT-SC on the 800 test questions
  (-1.4 to +0.9, all CIs include 0). Best-validation 63-68% = best-of-30 noise
  on 128 questions (sim median 64.1%). Among 172 working designs val~test
  Spearman 0.59. Details in findings.md "Designer panel".
- NEXT (needs user decision): resume the 4 incomplete cells to 30 generations
  with max_model_len 131072 for nemotron-nano/super and qwen3.5-35b (native
  262k, no YaRN) and meta output cap 12288 for qwen3-30b-a3b; then CPU analysis.

### 2026-10-14 15:55 - batch7 submitted at commit e64318e

```
e2e_cpu	28082598
resume_1gpu	28082599
resume_2gpu	28082600
mgsm_screen	28082601
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-14 - user: "do step 1 and 3" -> batch7

Step 1, finish the panel (commit e64318e):
- meta_max_model_len 131072 for nemotron-nano, nemotron-super, qwen3.5-35b
  (native 262k; no YaRN; hybrid models need <=2.7 GB KV at 131k).
  meta_max_output_tokens 12288 for qwen3-30b-a3b (YaRN x2 at 65,536 kept, so its
  model behaviour is unchanged mid-run). Worker-role settings unchanged.
- Resumed cells keep their archive; preserve_run_logs copies the earlier
  per-run logs to part<N>_* in the same directory before the new job rewrites
  them (abandoned/overflow records of part 1 are kept). Analysis: token_usage
  and budget_summary must be summed over parts.
- Test evaluation: a crash on one question now scores that question wrong
  (counted in test_crashed_questions) instead of voiding the design's whole
  test score; search-time crashes still go to the debug loop as in ADAS.
  test_error entries are retried in place. This recovers 3 designs, incl.
  qwen3-30b's best. gpt-oss-20b (index 1) is re-run only for its test retry.
  Consistent with earlier scores: every design scored before had 0 crashes.
- e2e section H covers crash scoring and the retry.

Step 3, choose the next task: DROP/GPQA/ARC are not ported to the instrumented
pipeline, so screen workers on MGSM: the 7 seed agents on mistral-7b, phi-4,
qwen3-30b-a3b, nemotron-nano, qwen3.5-27b (stage mgsm_screen, panel settings:
valid 128, 1 pass), i.e. measure directly how much scaffolds add per worker.
The chosen worker's seed archive is then reused as is by an MGSM panel.

batch7: e2e 28082598; resume_1gpu 28082599 (array 1,3,5), resume_2gpu
28082600 (7,8), mgsm_screen 28082601 (0-4). All three afterok:e2e.

### 2026-10-14 17:11 - batch8 submitted at commit 966ed78

```
mgsm_baseline	28086604
designers_1gpu	28086605
designers_2gpu	28086606
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-14 17:19 - batch9 submitted at commit a50f5fa

```
e2e_cpu	28086789
drop_screen	28086790
```

Inspect with: bash experiments/slurm/status.sh

### 2026-10-14 - MGSM screen complete; MGSM panel (batch8); DROP ported (batch9)

MGSM worker screen (7 seed agents, valid 128, 1 pass), seeds >= CoT+5 points:
  qwen3-30b-a3b CoT 74.2 best 88.3 (5/6) | mistral-7b 10.9 / 28.1 (5/6) |
  qwen3.5-27b 82.8 / 91.4 (2/6) | phi-4 83.6 / 86.7 (0/6) | nemotron-nano 78.9 / 82.0 (0/6).
  -> worker for the MGSM panel: qwen3-30b-a3b (strongest broad scaffold gain).

batch8 at 966ed78: mgsm_baseline 28086604 (s05_cot_qwen3-30b-a3b, stage 0.5 idx 9,
  128 x 5) -> designers_1gpu 28086605 (panel_mgsm 0-4) + designers_2gpu 28086606
  (5-7), both afterok on the baseline. Same 8 designers and ADAS protocol;
  index 2 is the self-design cell (designer = worker = Qwen3-30B-A3B).
  Qwen3.5-27B designer context raised to 131072 (native; 8.6 GB KV).

batch9 at a50f5fa: DROP ported to the instrumented pipeline.
- _drop/search.py started as a copy of the reviewed _mmlu/search.py; only the
  data loading (load_drop()[1:-1], few-shot records excluded as in ADAS), the
  answer instruction (ADAS's) and scoring changed. DROP F1 is stored as 0-1.
- drop_prompt.py: worker documentation ("F1 score"), PROMPT_FIELDS leak fix,
  first-generation reflexion fix, LLMAgentBase uses WORKER_MODEL.
- BUG FIXED (stock ADAS DROP): bootstrap CI passed 0.025/0.975 to
  np.percentile (expects 0-100), so DROP "95% CIs" were the 0.025th-0.975th
  percentiles. Other domains were correct.
- e2e now runs for mgsm and drop (stub answers the gold span or "zzqx").
- e2e 28086789 -> drop_screen 28086790 (array 0-4, same 5 workers), afterok.
- Analysis note: DROP per-question scores are continuous F1, not 0/1; the
  behavioural layer's error-set (Jaccard) measures assume binary.
