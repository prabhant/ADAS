--
title: ADAS Meta-Agent and Worker-Model Scaling Study
description: Agent-executable experimental plan for measuring how meta-agent and worker-model capability jointly shape the agent architectures discovered by ADAS on the Snellius SLURM cluster
author: Prabhan Singh
ms.date: 2026-09-02
ms.topic: how-to
keywords:
  - adas
  - meta agent search
  - slurm
  - snellius
  - vllm
  - agentic systems
---

## Purpose

You are running an exploratory study on top of the ADAS (Automated Design of Agentic Systems) codebase. The aim is to find out what happens when meta-agent capability and worker-model capability vary independently, not to establish novelty or defend a hypothesis.

Read this entire document before making any change. The stages exist to sequence the work and to redirect effort when something turns out to be uninformative, not to gate a claim. A negative or messy result is a legitimate output here.

## What this study is and is not

This is an exploration. Design accordingly.

| This study does                                    | This study does not                              |
|----------------------------------------------------|--------------------------------------------------|
| Prefer breadth of conditions over seeds per cell    | Chase statistical significance                    |
| Treat surprises and failures as findings            | Require ADAS to beat its baselines to be useful   |
| Report descriptive trends and worked examples       | Make causal claims from three seeds               |
| Read the discovered agents and describe them        | Rank methods or claim state of the art            |
| Optimize for questions a practitioner asks today    | Optimize for a contribution narrative             |

One consequence worth internalizing: if ADAS turns out not to help at all with modern models, that is a useful and publishable-quality observation about the field, not a failed experiment. Several of the questions below are interesting precisely because the answer could be "scaffolding no longer matters much."

## Questions to explore

ADAS separates two model roles that this study varies independently.

| Role       | Function                                     | Currently hardcoded as              |
|------------|----------------------------------------------|-------------------------------------|
| Meta agent | Writes new agent code, reflects, debugs       | `args.model`, default `gpt-4o-2024-05-13` |
| Worker     | Executes inside the discovered agent          | `gpt-3.5-turbo-0125`                |

The original ADAS work was built on `gpt-3.5-turbo-0125`, a non-reasoning model from 2023. Nearly every technique in its seed archive, from self-consistency to debate to reflexion, exists to compensate for weak single-pass reasoning. The models available now are different in kind, which makes the following questions live rather than settled.

### Q1. Does scaffolding still pay when the model reasons internally

This is the question with the most immediate practical weight. Teams are still wrapping elaborate multi-agent pipelines around models that already deliberate internally. If external scaffolding is largely redundant against a reasoning model, a great deal of production complexity is unnecessary. Watch whether the discovered architectures thin out as internal reasoning increases.

### Q2. Where is the cost crossover

The real deployment decision is small model plus expensive scaffold versus large model plain, at matched spend. AFlow reports workflows that let smaller models match GPT-4o at 4.55% of its cost, which if it generalizes is the most economically consequential result in this literature. Build the Pareto front and locate the crossover.

### Q3. Does the search know when to stop

Given a strong worker, does the meta agent converge on simpler designs, or does it keep adding machinery because nothing penalizes it? Automated design that cannot recognize sufficiency is a different tool from one that can.

### Q4. Is external scaffolding a substitute or a complement for internal reasoning

Test-time compute can be spent inside the model as reasoning tokens or outside it as extra calls. Which buys more accuracy per token is genuinely open, and the answer likely depends on the task. This is the cleanest version of Q1 and it is measurable directly.

### Q5. Which direction does a design transfer

Designing under a cheap worker and deploying under an expensive one would be an attractive workflow if it worked. Check both directions, since they need not be symmetric.

### Q6. What actually breaks

Which models cannot function as meta agents, and why. Report the failure taxonomy as a primary result rather than as an appendix, since knowing the floor is practically useful to anyone choosing a model for this role.

### Q7. What do the discovered agents actually look like

Read them. The most valuable output of ADAS-style work is usually a specific surprising design that someone noticed, and no metric substitutes for looking.

### Q8. Does designer size change the character of the designs, not just their quality

Every question above varies the worker. This one varies the designer and asks what changes about the artifacts themselves when a small model rather than a large one is doing the designing.

It is a primary question, not an afterthought, and it is nearly free because the meta ladder is already a factor in the matrix. What it needs is its own metrics, because "scores higher" and "designs differently" are separate claims and the existing accuracy measures only address the first.

The interesting possibilities are qualitative rather than scalar. A small designer might recombine archive entries while a large one invents. It might write an impressive design description and then implement something much simpler. It might reach for one building block repeatedly while a larger model routes, decomposes, and verifies. It might ignore the archive entirely and wander, rather than building on what came before. None of these show up in a fitness number, and two designs with identical accuracy can differ on all of them.

See the designer signature layer in the analysis section for how to measure each.

### Q9. What predicts whether an agent performs well

Two halves. How does complexity relate to performance, and which components recur in high performers as against low performers.

This needs no additional runs. By the end of the campaign you will hold several hundred design and fitness pairs spanning multiple workers, designer tiers, and domains, which is a dataset in its own right.

Expect the answer to be conditional rather than absolute. The complexity that suits a 4B worker is probably not the complexity that suits a 27B one, which is why the curves below are drawn per worker tier and never pooled. A single global answer to "how much scaffolding is right" would be the less interesting outcome.

## Prior work and the conventions it sets

### A thread worth pulling

The ADAS authors report something close to Q1 and Q3, but as an unsystematic aside drawn from a single transfer table. In their discussion they note that the best GPT-3.5 agent on ARC uses a complex feedback mechanism, yet when transferred to stronger models an agent with a simpler feedback mechanism and more refinement wins. They attribute this to GPT-3.5 evaluating and refining poorly, so it requires elaborate scaffolding while stronger models do not.

That observation was made across a two-year-old capability gap. The gap between a 4B model and a reasoning model today is larger and differently shaped, which makes the thread worth pulling rather than merely confirming.

### Analysis conventions to adopt

Five practices recur across ADAS, AFlow, AgentSquare, GPTSwarm, and MaAS. Adopt all five rather than inventing alternatives, so results stay comparable to the literature.

Stepping-stone lineage is ADAS's most informative analysis, and it is a genealogy rather than a metric. The paper traces how a pattern appearing at iteration 3 became a stepping stone later designs reused, and how the best ARC agent combined three ideas that first surfaced at iterations 5, 11, and 12. See the lineage layer in the analysis section.

Median with bootstrap confidence intervals, over repeated evaluation. ADAS evaluates each agent five times, reports the median with a 95% bootstrap interval, and treats as tied any entry whose median falls inside the interval of the best. Use this convention verbatim.

Resource metrics reported alongside accuracy, never folded into it. MaAS tabulates prompt tokens, completion tokens, dollar cost, and wall-clock time, split between search and inference. AFlow's headline claim is that smaller models beat GPT-4o at 4.55% of the cost. Accuracy alone is not treated as a sufficient result in this field.

Cross-model and cross-dataset transfer tables. ADAS transfers its top three agents across both models and domains; MaAS does the same. The cross-evaluation grid already planned here is the standard robustness check.

Initialization ablation. ADAS Appendix I reruns search from an empty archive, finds it still beats hand-designed baselines, and finds that on math an empty start actually wins. This doubles as a control condition.

### Whether ADAS still works is itself one of the findings

The MaAS authors reproduce ADAS as a baseline and report it underperforming plain CoT on GSM8K, MATH, MultiArith, HumanEval, and MBPP, with negative deltas where ADAS reports large gains. AFlow similarly attributes ADAS's weakness to its linear archive and coarse experience representation, noting that dumping every past workflow into the prompt makes the meta agent behave "more like an explorer of infinite possibilities than a designer seeking the optimal workflow."

In a confirmatory study this would be a threat. Here it is one of the more interesting things you might find, and it connects directly to Q1: if ADAS helped mainly by compensating for weak 2023-era workers, its apparent decline against modern models is the story rather than a bug in your setup. Record it as an outcome, do not treat it as a reason to stop.

## Critical design decisions

### Use measured capability, not parameter count

The recent models available on Snellius are largely mixture-of-experts, where size is ambiguous between total and active parameters, and where orderings are non-monotonic. Plotting against parameter count would produce a misleading figure.

Use each model's solo Chain-of-Thought accuracy on the domain validation split as its capability coordinate. The pipeline already computes this for free as the seed-archive `Chain-of-Thought` entry, so it costs nothing extra. Report dense versus MoE as a covariate.

### The meta agent must be informed of its worker

The meta agent currently has no idea which model will execute its designs. The worker identity is baked into a documentation string shown to the meta agent at [_mmlu/mmlu_prompt.py](_mmlu/mmlu_prompt.py#L306), which claims the worker is `gpt-3.5-turbo-0125`.

You have to edit that string anyway to stop it lying. When you do, add an explicit statement of the worker model name and its measured CoT baseline accuracy. A blind meta agent can only adapt through fitness feedback, so leaving it blind would blunt Q3 and Q4.

That reasoning is an assumption, not a finding. Test it with a two-cell ablation in Stage 1 running both informed and blind variants. If blind matches informed, the meta agent was adapting through fitness feedback alone, which is itself worth reporting and would simplify every later cell.

### Equal successes, not equal attempts

See the budget control section under repository changes. This is the single most dangerous confound in the study.

### Accuracy-only fitness makes the cost question invisible

ADAS scores agents on accuracy alone, and its authors list multi-objective search as future work. Additional LLM calls mechanically buy accuracy, which is why self-consistency and debate work at all. An unconstrained search therefore drifts toward expensive agents no matter which meta model drives it.

For Q2 and Q3 this is not a nuisance confound to be eliminated, it is the phenomenon to be measured. An accuracy-only search that never simplifies is direct evidence about whether automated design recognizes sufficiency.

So instrument rather than only correct. Record tokens and calls for every agent, build the accuracy-versus-cost Pareto front per cell, and report accuracy at matched call budgets alongside raw accuracy.

Then add one contrast. MaAS optimizes utility minus lambda times cost with lambda in 1e-3, 5e-3, and 1e-2, and reports that larger lambda yields more cost-efficient solutions at some performance cost. Run a cost-penalized condition on the corner cells using that range. The comparison between penalized and unpenalized search is the cleanest available probe of Q3.

### Contamination is an interpretation caveat, not a blocker

The seed archive is seven named published methods, every ladder model has read those papers, ADAS is on GitHub, its discovered agents are printed in the paper's appendix, and this repository ships prior archives under `results/`. A model can reproduce them from memory, and newer models memorized more.

Because this study is not making a novelty claim, contamination degrades from a confound into a caveat. You still want to know whether a design was recalled or constructed, since that changes how you read it, but the finding does not hinge on the distinction.

Keep the cheap version. Probe each meta model cold, with no archive, asking what agent design patterns it knows, and keep a reference list of published architectures to check discovered designs against. Skip the formal recall-versus-discovery scoring unless something surprising turns up.

## Model shortlist

All managed models live under `/projects/2/managed_datasets/hf_cache_dir`. Login nodes have internet access, so anything absent can be downloaded into project space with your own Hugging Face token. Self-downloading also bypasses the SURF servicedesk ticket normally required for gated models.

### Meta agent ladder

Context length binds harder than size here. The meta prompt embeds the entire archive of agent code, reaching tens of thousands of tokens by generation 20. Anything under 64k context will fail mid-run and you will be measuring context overflow instead of design ability.

| Model        | Parameters          | Context | License    | H100 GPUs | Role in ladder             |
|--------------|---------------------|---------|------------|-----------|----------------------------|
| gpt-oss-20b  | 21B MoE, 3.6B active | 128k    | Apache 2.0 | 1         | Floor rung                 |
| Qwen-3.5-27B | 27B dense            | Long    | Apache 2.0 | 1         | Mid rung, dense control    |
| gpt-oss-120b | 117B MoE, 5.1B active | 128k   | Apache 2.0 | 2         | Top rung, same family as floor |

### Worker ladder

Worker prompts are short, so context is a non-issue. Instruction following and JSON compliance are what matter.

| Model                     | Parameters | License    | Source        | H100 GPUs |
|---------------------------|------------|------------|---------------|-----------|
| Qwen3-4B-Instruct         | ~4B dense  | Apache 2.0 | Self-download | 1         |
| Phi-4                     | 14B dense  | MIT        | Managed       | 1         |
| Qwen-3.5-27B              | 27B dense  | Apache 2.0 | Managed       | 1         |

Verify the exact Qwen3 small-model identifier on Hugging Face before committing. The small rung is the only model requiring download, and nothing recent, ungated, and small exists in the managed set.

### Excluded models

| Model                    | Reason for exclusion                                        |
|--------------------------|-------------------------------------------------------------|
| DeepSeek R1 and V3.2     | 642 to 700 GB, requires two full H100 nodes                  |
| Qwen-3.5-397B-A17B       | 752 GB, tensor parallel 8                                    |
| Phi-4 as meta agent      | 16k context, fails by mid-run. Acceptable as a worker only   |
| GPT-J, GPT-NeoX, GPT-2   | 2k context, obsolete                                         |
| Mistral 7B v0.1          | 8k context, 2023 vintage                                     |
| Llama 2, CodeLlama 2     | Gated and superseded                                         |
| A100 partition           | 40 GiB forces tensor-parallel splits on models that fit one H100 |

## Design matrix

| Factor    | Levels                                        | Notes                                     |
|-----------|-----------------------------------------------|-------------------------------------------|
| Meta      | gpt-oss-20b, Qwen-3.5-27B, gpt-oss-120b        | Three designer tiers, carries Q8          |
| Worker    | Qwen3-4B, Phi-4-14B, Qwen-3.5-27B              | Three capability tiers                    |
| Reasoning | Low and high effort on a gpt-oss worker        | Direct probe of Q1 and Q4                 |
| Domain    | MGSM primary, MMLU secondary                   | Cheapest loops, low floor-effect risk     |
| Seed      | Two replicates per cell                        | Distinct `--expr_name`, fixed data split  |

Two seeds rather than three is a deliberate exploratory trade. Spend the freed budget on more conditions, since with this few replicates you are describing trends either way and an extra cell teaches more than an extra seed.

The reasoning factor is the most interesting addition and the one most specific to 2026. The gpt-oss models expose a reasoning effort setting, so you can hold the worker model completely fixed and vary only how much it deliberates internally. If the discovered architecture thins out as internal reasoning rises, that is a direct answer to Q1 with the model identity controlled, which no cross-model comparison can give you. Verify in Stage 0 that your vLLM build exposes this control, since the whole factor depends on it.

There is no LLM sampling seed in this codebase. `--shuffle_seed` controls the data split only. Replicates arise from temperature, 0.5 for workers and 0.8 for the meta agent. Hold `--shuffle_seed` at 0 across all cells so every configuration sees identical validation and test items.

DROP, GPQA, and ARC are deliberately deferred. ARC in particular floors at zero for small workers and teaches nothing about architecture.

## Environment and cluster facts

| Property              | Value                                                        |
|-----------------------|--------------------------------------------------------------|
| GPU partition         | `gpu_h100`, 88 nodes, 4x H100 SXM5 94 GiB, 64 cores, 720 GiB |
| Minimum allocation    | One quarter node: 16 cores, 1 GPU, 180 GiB                    |
| SBU weight            | 192 per H100 GPU-hour, so 768 per full node-hour              |
| CPU partition         | `genoa`, SBU weight 1.0, use for all analysis                 |
| Maximum walltime      | 120 hours on all partitions                                   |
| Fast-track scheduling | Jobs of one hour or less                                      |
| Managed models        | `/projects/2/managed_datasets/hf_cache_dir`                   |

Compute nodes have no internet. Export `HF_HUB_OFFLINE=1` in every job script. Download models from a login node only.

Model load time from the shared filesystem is slow for large checkpoints. Twenty-two H100 nodes carry local NVMe at `/scratch-node`. Stage the 120b meta agent there when your allocation lands on one.

## Repository changes

Apply these to `_mgsm/search.py` and `_mmlu/search.py`, plus the two corresponding `*_prompt.py` files. Leave the other domains untouched.

### Client routing

Replace `client = openai.OpenAI()` with a factory that returns two distinct clients, one per role, pointed at local vLLM servers via `base_url` with a dummy API key. Thread the worker client through `LLMAgentBase` and the meta client through `get_json_response_from_gpt_reflect`.

### Arguments

Add `--worker_model`, `--meta_base_url`, `--worker_base_url`, `--run_id`, `--target_generations`, and `--max_attempts`. Remove the `choices=[...]` restriction on `--model`, which currently permits only three OpenAI models.

### Exception handling

The `backoff.on_exception(backoff.expo, openai.RateLimitError)` decorator is inert against vLLM, which never rate limits. It returns HTTP 400 on context overflow instead. Catch `openai.BadRequestError` and `openai.APIConnectionError`, and classify context overflow as its own logged failure category rather than folding it into code errors.

### Budget control

The search loop reads `for n in range(start, args.n_generation)` and, on failure, executes `n -= 1; continue`. Rebinding the loop variable inside a `for` loop over `range` does nothing, so the generation is silently skipped rather than retried. Weaker meta agents therefore finish with fewer archive entries from the same nominal budget, making search budget correlate with model capability.

Convert to a `while` loop that runs until `target_generations` successful entries land, capped at `max_attempts` total tries. Record the attempt count as its own outcome variable. This is a required control, not an optimization.

### Instrumentation

Several planned metrics have nowhere to live in the current code. Add all of the following before generating any data.

Persist `debug_attempts` per archive entry, along with the terminal failure reason when a generation is abandoned.

Persist abandoned generations rather than discarding them. Failed designs currently vanish, which biases every downstream metric toward the subset each model could successfully produce.

Add a thread-safe token accumulator keyed by generation and role. A commented-out cost calculation already exists in `get_json_response_from_gpt` and shows the intended shape.

Persist the per-question correctness vector. `evaluate_forward_fn` computes per-question outcomes and immediately collapses them into `acc_list` and a bootstrap confidence interval. Keeping the raw binary vector is a one-line change that enables the entire behavioral analysis layer.

Log every `LLMAgentBase.query` call: call identifier, agent name, role, output fields, iteration index, and the `author` field of every input `Info`. The `Info` namedtuple at [_mmlu/search.py](_mmlu/search.py#L20) already carries provenance, so this reconstructs the runtime dataflow graph at negligible cost.

Record the framework API surface as an explicit list, covering `LLMAgentBase`, its constructor arguments, the `Info` fields, and any domain-specific tool functions. The designer signature layer needs it to detect hallucinated calls, and writing it down once is easier than inferring it later.

### Safety

`evaluate_forward_fn` calls `exec()` on model-generated code with no isolation. This study runs far more untrusted code, from models less aligned than GPT-4o. Route execution through a restricted subprocess with no network access, a read-only filesystem view, and a wall-clock timeout. Treat this as a blocking requirement.

### Structured output

Confirm your vLLM build honors `response_format={"type": "json_object"}` through guided decoding, and prefer a schema-constrained backend. Test the gpt-oss models specifically. Their channel-separated response format with reasoning traces can interact badly with JSON mode, and both ends of the meta ladder depend on them.

### Throughput

`max_workers=48` was tuned for a rate-limited hosted API. Against local vLLM, raise it to match `--max-num-seqs` (start at 128 to 256). This is the highest-leverage single change for wall-clock time.

### Fairness controls

Both ends of the meta ladder are gpt-oss models, which emit reasoning traces that consume the same 4096 token budget as the code they produce. Holding `max_tokens` constant across models is therefore not a fairness control but its inverse, since gpt-oss receives materially less usable output budget than Qwen-3.5-27B. Budget completion tokens separately from reasoning tokens where the server exposes the split, otherwise calibrate per-model limits so usable output is equal. Log both counts either way, and log truncation events.

Cache the seed-archive evaluation per combination of worker, domain, and data split. It is currently re-evaluated on every run, injecting variance into the starting point of every search.

## Reference conditions

These are not statistical controls in the confirmatory sense. They are reference points that make the discovered agents interpretable, and each is cheap relative to a search run.

| Condition                   | Implementation                                          | What it tells you                                 |
|-----------------------------|---------------------------------------------------------|---------------------------------------------------|
| Best seed agent per worker  | Already computed during archive initialization           | Whether search moved at all, and from where        |
| Bare model, no scaffold     | Single call, no agent wrapper                            | The floor for Q1 and the anchor for the Pareto front |
| Random recombination        | Sample and splice archive entries with no meta reasoning | Whether reasoning or sampling drives any gain, and unbiased design samples for Q9 |
| No-reflection ablation      | Skip both Reflexion rounds in the search loop            | What the two reflection passes contribute          |
| Empty archive initialization | Follow ADAS Appendix I                                  | Whether gains come from the seeds or from search   |

The bare-model condition matters more than it looks. Q1 is meaningless without it, and it is nearly free to compute.

Random recombination deserves more runs than a pure control would need. It is the only condition that samples design space without optimization pressure, which makes it the best available source of low-performing designs, and Q9 cannot be answered well without them. Budget it as a data source rather than as a checkbox.

The empty-initialization condition carries a specific prior worth checking: ADAS found that on math, starting from an empty archive outperformed starting from the seeds. MGSM is the primary domain here, so that result may reappear.

## Operational definitions

Fix these before Stage 1, because several metrics depend on them being consistent across cells.

A cell is degenerate when more than half of its attempts fail terminally, or when the median novelty distance of its discovered agents falls below the 10th percentile of pairwise distances among the seed agents themselves. Degenerate cells are not discarded. They are the raw material for Q6, so record why they degenerated and keep them in the write-up.

The best agent is selected on validation fitness only. Selecting on test and then reporting test is circular, and the cross-evaluation grid depends on this selection being clean.

The failure taxonomy has seven categories: JSON parse error, Python syntax error, runtime exception, context overflow, wall-clock timeout, all-zero accuracy, and semantically wrong but running. Q6 is answered from the distribution across these categories. If parse errors dominate for a given model, that model is telling you about format compliance rather than design ability, which is itself worth reporting.

The generalization gap is validation fitness minus test fitness, tracked per agent. With 128 validation items and more than 20 candidates, selection overfitting is substantial and may differ systematically by meta model.

## Files to create

```text
experiments/
  config/
    models.yaml              Model registry: identifier, path, tensor-parallel size, context, eligible roles
    cells.yaml               Experiment grid: cell id, meta, worker, domain, seed, stage
  env/
    setup_env.sh             Create uv virtual environment, install dependencies
    download_models.sh       Login-node Hugging Face downloads into project space
  serving/
    launch_vllm.sh           Start meta and worker servers on separate ports and GPU sets
    health_check.py          Poll /health and /v1/models until both servers are ready
  adas_ext/
    __init__.py
    model_router.py          Per-role OpenAI client factory with base_url routing
    instrumentation.py       Thread-safe token, call, and debug logging
    sandbox.py               Restricted execution of generated agent code
    budget.py                Equal-successes loop control with attempt cap
  slurm/
    stage0_smoke.slurm       Single cell, tiny budget, plumbing validation
    stage1_pilot.slurm       Job array over the nine cells, one seed
    stage2_main.slurm        Job array over surviving cells by three seeds
    stage3_transfer.slurm    Corner cells on MMLU
    crosseval.slurm          Best agent per cell against every worker
    analysis.slurm           CPU-only, genoa partition
  analysis/
    extract_features.py      Layer 1: AST canonicalization and feature vector
    tree_distance.py         Layer 1b: Zhang-Shasha tree edit distance
    dataflow_graph.py        Layer 2: runtime DAG reconstruction and WL hashing
    prompt_semantics.py      Layer 3: instruction and role string embeddings
    behavioral.py            Layer 4: per-question outcome vectors
    lineage.py               Layer 5: stepping-stone genealogy and motif propagation
    designer_signature.py    Layer 6: thought-code alignment, hallucinated APIs, repertoire
    attribution.py           Complexity curves, component attribution, paired-delta fits
    contamination.py         Cold recall probe and published-design corpus matching
    distances.py             Distance combination with pooled z-scoring
    aggregate.py             Novelty, diversity, and complexity per cell
    figures.py               Plot generation
  results/
    <cell_id>/               Archives, traces, token logs, per-question vectors
```

## SLURM job structure

Each cell needs two vLLM servers inside one job. Request a full node so timing measurements are not perturbed by co-tenancy.

```bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --array=0-8
```

Within the job, split GPUs between roles using `CUDA_VISIBLE_DEVICES`, set `--tensor-parallel-size` per the model registry, launch both servers in the background, block on `health_check.py`, then run `search.py` against the two local endpoints.

Chunk long runs against the 120 hour ceiling using `--requeue`. No new code is needed: `search()` already resumes from `archive[-1]['generation']` and `evaluate()` resumes through `eval_archive`.

Run all analysis on `genoa` at SBU weight 1.0 rather than 192. The analysis is CPU-only and effectively free relative to the searches.

## Analysis: computing differences between produced agents

Every archive entry is a single `def forward(self, taskInfo)` function stored as a string, assembled from a small vocabulary of `LLMAgentBase` instantiations, agent calls, loops, branches, and literal instruction strings. The seed set in [_mmlu/mmlu_prompt.py](_mmlu/mmlu_prompt.py#L12-L230) spans the design space, from single-call `Chain-of-Thought` to four-role nested-loop `LLM Debate` to branch-on-output `Dynamic Assignment of Roles`.

A single scalar distance is the wrong instrument. Agents differ along four independent axes, and disagreement between the layers is itself a finding.

| Layer            | Question answered                          | Cost                    |
|------------------|--------------------------------------------|-------------------------|
| Structural       | Is the code shaped differently              | Free, static analysis   |
| Dataflow         | Do information paths differ at runtime      | Near-free, needs logging |
| Prompt semantics | Do the instructions say different things    | Cheap, needs embeddings |
| Behavioral       | Does it fail on different questions         | Free, one-line change   |

### Layer 1: Structural distance

Normalization determines whether this layer measures anything real. A weak meta agent can rename `Chain-of-Thought` to `Adaptive Reasoning Synthesizer`, rewrite every comment, and change nothing functional.

Parse with `ast`, which discards comments automatically. Then strip docstrings explicitly, alpha-rename local variables to canonical `v0, v1, ...` in first-appearance order, and replace string literals with a `STR` placeholder. Retain numeric literals, because `N = 3` versus `N = 5` is a genuine design difference.

Extract this feature vector:

| Feature                    | Extraction                                              |
|----------------------------|---------------------------------------------------------|
| `n_agent_sites`            | Count of `LLMAgentBase(...)` call nodes                  |
| `n_agents_static`          | Same, multiplied through comprehensions with literal iterables |
| `n_call_sites`             | Calls on names bound to agent objects                    |
| `max_loop_depth`           | Maximum nesting of `For` and `While`                     |
| `n_branches`               | Count of `If` nodes                                      |
| `has_dynamic_routing`      | An `If` whose test reads `.content` from an agent output |
| `n_distinct_roles`         | Literal strings passed to `role=`                        |
| `output_field_set`         | Union of output field names across agents                |
| `has_aggregation`          | `Counter`, `most_common`, or majority-vote patterns      |
| `has_context_accumulation` | `.extend(` or `.append(` onto an input list              |
| `n_temperatures`           | Distinct temperature literals                            |
| `n_statements`             | AST statement count                                      |

Use `n_statements` and never raw line count. Comment verbosity varies enormously between models and would otherwise dominate the complexity metric.

Compute distance as weighted Euclidean over features z-scored against pooled statistics from every run in the campaign, calculated once. Per-run normalization would make cells incomparable, defeating the purpose.

Complement this with normalized tree edit distance using `zss` over the canonicalized AST, divided by combined tree size. These functions run 30 to 60 AST nodes, so Zhang-Shasha is instant. Tree edit distance catches reorganizations the feature vector misses, and the feature vector catches semantics tree edit distance treats as superficial. Neither subsumes the other, so report both.

### Layer 2: Runtime dataflow graph

This is the strongest available signal and it is nearly free, because the codebase already carries provenance. The `Info` namedtuple is `('name', 'author', 'content', 'iteration_idx')` and `LLMAgentBase.__repr__` returns a unique agent identifier, so every piece of information records its producer.

From the call log described under instrumentation, reconstruct a directed acyclic graph in which nodes are LLM invocations and edges are information flow. Label nodes by role plus sorted output fields so a critic never matches a debater.

Static code does not determine this graph. `Self-Refine (Reflexion)` breaks early when its critic returns `True`, and `Dynamic Assignment of Roles` routes to one of four experts. Realized topology depends on the input, and only a runtime trace captures it.

Compare graphs with `networkx.weisfeiler_lehman_graph_hash` for exact bucketing and graph edit distance for a continuous measure on smaller comparisons.

This layer yields a bonus metric. Since each agent runs over hundreds of questions, you obtain a distribution of graphs rather than one. `Self-Consistency` produces an identical graph every time; `Reflexion` does not. Entropy over the WL hashes gives an adaptivity score separating fixed scaffolds from input-conditional ones, which is a useful complement to Q3 because it distinguishes a simple design from an adaptive one.

### Layer 3: Prompt semantics

A large share of each design lives in string literals: `cot_instruction`, `debate_instruction`, and the `role=` lists. Layer 1 deliberately blanked these, so handle them here.

Collect all instruction and role literals per agent, embed them, then match the two prompt sets with `scipy.optimize.linear_sum_assignment` and average the matched cosine distances, adding a fixed penalty per unmatched prompt.

Use a held-out embedding model that is not part of the experiment. Embedding with a ladder model risks it scoring its own phrasing as more self-similar, which would correlate with the exact axis under study.

### Layer 4: Behavioral distance

Two agents can look entirely different and behave identically. This layer checks the other three and costs the least.

Using the persisted per-question binary vectors, compute Hamming or Jaccard distance over error sets. This answers whether two agents fail on the same questions.

This layer is the reality check on the other three. If agents discovered under a 4B worker and under a 27B worker differ structurally but share near-identical error profiles, the structural difference is cosmetic, and reporting it as a design difference would be misleading.

### Layer 5: Stepping-stone lineage

The four distance layers describe pairs of agents. Lineage describes the search trajectory, and it is where ADAS found its most interesting result.

For each archive entry, identify its nearest predecessors under the Layer 1 and Layer 2 distances, and record which structural features appear for the first time at which generation. This produces a directed genealogy over the archive.

Report the generation at which each recurring motif first appears, tracking majority voting, critic loops, role specialization, problem decomposition, and verification steps. Then report how often later entries reuse it. A motif appearing once and never reused is noise. A motif that appears and then propagates is a stepping stone.

The cross-cell comparison becomes: do stronger meta agents produce more reusable stepping stones, or merely more one-off designs? That is closer to what people actually want to know about ADAS than any accuracy delta, and no published work measures it systematically.

One implementation note from MaAS. Rather than embedding raw prompt strings, they have an LLM generate a structured description of each operator and embed that description with MiniLM. Applying the same trick to Layer 3 and to motif identification gives more stable clusters than embedding raw instruction text.

### Layer 6: Designer signature

Layers 1 through 5 characterize the designs. This layer characterizes the designer, and it is what answers Q8. All five measurements are cheap, and all are computed from artifacts you are already saving.

Thought-code alignment is the sharpest of them. Every archive entry carries both a `thought`, which is a natural-language account of the intended design, and the `code` that was actually produced. A weaker designer frequently writes a plausible, even sophisticated thought and then implements something simpler or materially different. Score the gap with a held-out judge, giving it the canonicalized code and the thought with names stripped, and asking whether the code implements the design described. This measures whether a model can execute its own stated plan, which is distinct from whether the plan was any good, and I would expect it to separate tiers more cleanly than accuracy does.

Hallucinated API rate is a pure capability signal, cleanly separable from design quality. Count calls to methods that do not exist in the provided framework. Detect them by walking the AST for attribute calls on `self` and on agent objects, then comparing against the known framework surface. The ADAS meta prompt carries a "WRONG Implementation examples" section precisely because models invent APIs, so this failure mode is expected and worth quantifying rather than merely debugging away.

Archive utilization tests a specific published criticism. AFlow argues that dumping every past workflow into the prompt makes the ADAS meta agent behave "more like an explorer of infinite possibilities than a designer seeking the optimal workflow." Measure it per tier as the distance from each new design to its nearest archive predecessor, plus whether the `thought` explicitly references prior entries. A designer builds on the archive, an explorer wanders, and the two produce very different lineage graphs.

Motif repertoire is the set of distinct building blocks a tier reaches for across an entire run: ensembling, majority voting, critic loops, role specialization, decomposition, routing, verification, early exit. Report both repertoire size and composition. A designer that only ever ensembles is qualitatively different from one that routes and decomposes, even at identical accuracy, and this is the measurement most likely to produce a legible figure.

Trajectory shape asks whether a run converges or keeps wandering. Plot within-run diversity against generation index. A converging designer narrows toward a design family; a wandering one does not, and the distinction is invisible in any endpoint metric.

One caution specific to this layer. Debug attempts and hallucinated APIs measure coding competence, while archive utilization and motif repertoire measure design behavior. Keep them separate in the write-up. A model that writes buggy code is not necessarily a poor designer, and conflating the two would produce a tidier but less honest story.

### Deriving the reported metrics

Novelty against the seed archive is the minimum distance to any of the seven seed agents, reported separately per layer. The Layer 1 normalization is what makes this honest, since it is precisely what defeats rename-only innovation.

Within-run diversity is mean pairwise distance across archive entries. With 20 to 30 entries, resist anything more elaborate.

Keep an LLM judge as a cross-check only. Blind it by stripping `name` and `thought` and showing only canonicalized code, use a model outside the experiment, and seed the comparison set with known-identical pairs to measure the judge's own reliability.

Never score novelty from the `thought` field. It is the meta agent's self-authored description of its own design, and stronger models write more persuasive claims regardless of whether anything new happened. Use it for qualitative reading when writing up, nothing more.

### Analysis traps

Dead code inflates static counts, because models generate agents they never invoke. Compare `n_agents_static` against the runtime call log. The gap is itself a quality signal, and weaker meta agents should show a larger one.

Dynamic multiplicity breaks naive counting. A comprehension such as `[LLMAgentBase(...) for _ in range(N)]` is one AST node but N agents. Resolve `N` when it is a literal, otherwise mark it unknown and defer to runtime counts.

Distances must be pooled-normalized. Compute feature statistics once across the entire campaign, then apply them everywhere. Getting this wrong silently invalidates every cross-cell comparison.

Required libraries are `ast` and `collections` from the standard library, plus `zss`, `networkx`, `scipy`, and a sentence embedding model.

## Attribution: what makes an agent perform well

Q9 is answered from the pooled campaign rather than from any single cell, and it costs no extra compute.

### The complexity-performance curve

Plot fitness against complexity with one curve per worker tier. Never pool across workers, because the shape by tier is the finding. If the optimum sits at higher complexity for a 4B worker than for a 27B worker, that is Q1 and Q3 answered in one figure.

Three shapes are worth distinguishing.

| Shape       | Reading                                                                 |
|-------------|--------------------------------------------------------------------------|
| Monotonic   | More scaffolding always helps, and the search should have kept going      |
| Plateau     | Scaffolding helps to a point and then stops paying                        |
| Inverted U  | Past some point extra structure actively hurts                            |

An inverted U would be the most interesting outcome and is plausible, since every added hop introduces error accumulation, context dilution, and another place to fail. My expectation is a plateau or a mild inverted U peaking earlier for stronger workers, but the shape matters more than the direction and the data will settle it.

Always draw the cost-matched version beside the raw one. Complexity and token spend are close to collinear, so an uncontrolled curve may be showing nothing more than that more compute buys more accuracy.

### Component attribution

The obvious analysis is motif frequency in top-quartile versus bottom-quartile agents, and it is the misleading one, because motifs co-occur. Ensembling and final-decision aggregation nearly always travel together, so each will look important in isolation.

Use three methods and report where they disagree, since agreement across methods is far more persuasive than any single ranking.

Frequency contrast between top and bottom quartiles is simple and communicates well. Keep it for the figure, distrust it for the conclusion.

A regularized multivariate fit handles co-occurrence. Use L1 logistic regression or gradient-boosted trees with SHAP values over the Layer 1 feature vector against fitness.

Paired-delta analysis is the strongest of the three. For each parent-child pair in the lineage graph, compute the change in features and the change in fitness, then regress one on the other. Comparing genealogically related designs controls for most between-design confounding, in the way a within-subjects design does. This is precisely what AFlow's tree-structured experience log captures and ADAS does not: their appendix records entries like "Round 1 to Round 2, score improved from 0.4874 to 0.5244, added the Programmer operator," which is why their attribution reads convincingly. You can reconstruct the same structure from the lineage layer.

### Direct ablation, if budget allows

Take two or three high-performing agents and remove one motif at a time, then re-evaluate. No search is involved, so it is cheap per agent, and it is the only method here that supports a causal reading. If you have to choose between this and a more elaborate regression, choose this.

### Where the low performers come from

The archive is not a random sample of design space. Search optimizes for accuracy, so it moves away from poor designs and under-samples exactly the region attribution needs. Three sources fill the gap.

Abandoned generations, which the instrumentation now persists rather than discarding.

The random recombination condition, which samples without optimization pressure and is therefore the best source of unbiased design and fitness pairs.

The cross-evaluation grid, which supplies the same design evaluated under different workers and so separates a good design from a design that suits one particular worker.

### Statistical cautions

Archive entries are genealogically related, because later designs build on earlier ones. Treating them as independent samples will overstate confidence badly. Cluster by lineage, or at minimum state that the dependency exists and that the intervals are optimistic.

Check fitness variance per worker tier before fitting anything. A 27B worker on MGSM may sit near the top of the scale, and a compressed range makes attribution unstable and the curve shape unreadable.

Complexity is not one thing. Report the curve against at least two different complexity measures, such as runtime call count and structural depth, since they can diverge. An agent with one deep loop and an agent with five parallel calls are both "complex" in ways that need not behave alike.

## Staged execution

| Stage | Scope                          | Configuration                          | Purpose                                      |
|-------|--------------------------------|----------------------------------------|----------------------------------------------|
| 0     | One cell, MGSM                 | 2 generations, valid 8, test 8         | Plumbing, JSON compliance, sandbox, logging  |
| 0.5   | Bare model and solo CoT        | No search, valid 128, five repeats     | Capability axis and the Q1 anchor points     |
| 1     | Full 3x3 grid, 1 seed, MGSM    | 10 generations, valid 64, test 200     | First look, find where the interesting cells are |
| 1b    | Two cells, informed and blind  | Same as Stage 1                        | Does telling the meta agent its worker matter |
| 2     | Interesting cells, 2 seeds     | 20 generations, valid 128, test 300    | Main dataset                                 |
| 2r    | Reasoning effort low and high  | Fixed gpt-oss worker                   | Q1 and Q4 with model identity controlled     |
| 2c    | Reference conditions           | Matched to Stage 2                     | Anchors, see reference conditions            |
| 2d    | Cost-penalized corner cells    | Lambda from 1e-3 to 1e-2               | Q3, does search simplify when cost is priced |
| 3     | Corner cells, MMLU             | 20 generations, valid 128, test 300    | Does any of this hold outside math           |
| X     | Cross-evaluation grid           | Best agent per cell by every worker    | Q5, and the cheapest strong result available |

Stage 0.5 doubles as a sanity check on the axis. If the solo CoT confidence intervals of two adjacent worker rungs overlap, those rungs are not distinct capability tiers, and you should swap a rung before spending on Stage 1.

After Stage 1, review and redirect rather than pass or fail. The question is where the budget buys the most information, and the answer depends on what Stage 1 shows.

| Stage 1 outcome                              | Redirect                                                        |
|----------------------------------------------|------------------------------------------------------------------|
| Nothing beats the bare model or best seed     | Pivot to Q1, Q6, and Q8. That is a finding about modern scaffolding. |
| Most cells degenerate                         | Raise the bottom rungs, then spend the budget on Stage 2r        |
| Cells differ but noisily                      | Add seeds to the two most separated cells rather than all of them |
| Cells look similar on accuracy                | Keep all three meta tiers, cut worker levels, lean into Q8        |

When trimming, cut worker levels before meta levels. Q8 needs all three designer tiers to say anything about how design character varies with size, and a two-point ladder cannot distinguish a trend from a gap. The worker axis degrades more gracefully, because Q1 and Q2 remain answerable from the extremes plus the reasoning-effort contrast.

Note also that accuracy converging across cells does not mean the designs converged. That situation is the most favorable one for Q8, since it isolates differences in design character from differences in outcome.

Run the cross-evaluation grid regardless of what happens upstream. Take the best agent from each cell and re-run it under every worker. It is cheap relative to the searches, it answers Q5 directly, and it produces an interpretable figure even if every search underwhelms.

Stage 0 exits when both servers serve valid JSON, the sandbox blocks a deliberate network call, the reasoning effort control is confirmed working, and every new logging field is populated in the output archive.

## Analysis and reporting

Organize the write-up around the seven questions rather than around the factors.

The headline artifact is the accuracy-versus-cost Pareto front, with every cell, every reference condition, and the bare model plotted together, following the AFlow presentation. That single figure addresses Q1 and Q2 at once and is legible to a practitioner deciding what to deploy. Report cost in tokens rather than dollars, since your models are self-hosted.

For Q3 and Q4, plot architecture complexity against worker capability with one line per meta tier, then overlay the reasoning-effort pair. If the reasoning-effort line slopes the same way as the capability line, internal and external deliberation are substitutes, which is the more interesting outcome.

For Q5, present the cross-evaluation grid as a heatmap of best agent by worker. A strong diagonal means designs specialize to their worker. Asymmetry above versus below the diagonal tells you which transfer direction is safe.

For Q6, report the failure taxonomy as a stacked distribution per meta model rather than as a single rate.

For Q7, include two or three annotated agents in full, with commentary on what the meta agent was apparently trying to do, and use the stepping-stone lineage to show where each idea came from.

For Q8, report the designer signature table with one row per meta tier. This is the table that answers whether small and large models design differently, as opposed to whether one scores higher.

| Signal                  | Reported as                                          |
|-------------------------|------------------------------------------------------|
| Thought-code alignment  | Judge score distribution, per tier                    |
| Hallucinated API rate   | Calls to non-existent methods per generation          |
| Archive utilization     | Distance to nearest predecessor, plus reference rate  |
| Motif repertoire        | Size and composition, as a stacked bar per tier       |
| Trajectory shape        | Within-run diversity against generation index         |
| Debug attempts          | Distribution, kept separate from the design signals   |

Pair that table with side-by-side examples: the same generation index from the smallest and largest meta tier, shown in full. A reader learns more from two contrasting designs than from any of the aggregate numbers.

For Q9, lead with the complexity-performance curves, one per worker tier, raw and cost-matched side by side. Follow with the component attribution table placing all three methods in adjacent columns so disagreement between them is visible rather than hidden.

Report the per-cell resource table following the MaAS format, so cost stays visible rather than buried.

| Column                | Scope                                      |
|-----------------------|---------------------------------------------|
| Prompt tokens         | Search and inference, reported separately   |
| Completion tokens     | Search and inference, reported separately   |
| Reasoning tokens      | Where the server exposes the split          |
| LLM calls per query   | Runtime count, not static                   |
| Wall-clock time       | Search and inference, reported separately   |
| GPU hours and SBU     | Actual consumption per cell                 |

Also report the generalization gap per agent.

Follow the ADAS statistical convention where it applies: median with a 95% bootstrap confidence interval, treating as tied any entry whose median falls inside the interval of the best. With two seeds this is descriptive. Show per-seed points, avoid significance language, and resist the urge to rank cells.

## Risk register

In an exploratory study most surprises are results rather than risks. These four are genuine risks because they would leave you unable to interpret what you see.

The most likely one is that weak meta agents fail on JSON and Python syntax rather than on design quality, leaving you measuring format compliance while believing you are measuring design ability. Watch the ratio of parse failures to semantic failures in the debug log. If parse failures dominate for a rung, say so explicitly in the write-up rather than reporting that rung as a weak designer.

Wall-clock is the second risk. A 120b meta agent alongside a 27B worker on a single node is slow, and the original concurrency setting assumed a hosted API. Retune `max_workers` to measured vLLM throughput or you will construct an elaborate request queue.

The gpt-oss response format is the third risk, and it is concentrated. Both ends of the meta ladder use it, and the reasoning-effort factor depends on it too. Validate JSON mode and the effort control together during Stage 0, before any budget is committed.

The fourth is over-interpreting two seeds. ADAS search is high variance, and with two replicates a difference between adjacent cells is frequently noise. The mitigation is presentational rather than statistical: show per-seed points everywhere, and describe patterns without ranking.

## Execution checklist

1. Confirm SBU allocation and record it here.
2. Verify the Qwen3 small-model identifier on Hugging Face.
3. Confirm your vLLM build exposes the gpt-oss reasoning effort control.
4. Create the file tree under `experiments/`.
5. Fix the operational definitions and record the chosen thresholds.
6. Apply repository changes to the MGSM and MMLU search and prompt modules.
7. Download the small worker model from a login node.
8. Run Stage 0 and confirm all four exit conditions.
9. Run Stage 0.5 for the capability axis and the bare-model anchors.
10. Run Stage 1 plus the informed-versus-blind ablation.
11. Review Stage 1 against the redirect table and choose where the remaining budget goes.
12. Run the cross-evaluation grid regardless of what Stage 1 showed.
