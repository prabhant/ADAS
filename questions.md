# Research questions

The questions this project answers, extracted from `followinstructions.md`. Everything
else in that document — the grid, the staging, the measurement layers — exists to serve
these nine.

ADAS separates two model roles, and this study varies them independently:

| Role       | Function                                | Ladder                                     |
|------------|-----------------------------------------|--------------------------------------------|
| Meta agent | Writes agent code, reflects, debugs      | gpt-oss-20b, Qwen3.5-27B, gpt-oss-120b     |
| Worker     | Executes inside the discovered agent     | Mistral-7B, Phi-4, Qwen3.5-27B             |

The original ADAS work was built on `gpt-3.5-turbo-0125`, a non-reasoning model from
2023. Nearly every technique in its seed archive — self-consistency, debate, reflexion —
exists to compensate for weak single-pass reasoning. Today's models are different in
kind, which is what makes these questions live rather than settled.

One framing point worth keeping in view: if ADAS turns out not to help much with modern
models, that is a useful observation about the field, not a failed experiment. Several
questions below are interesting precisely because the answer could be "scaffolding no
longer matters much."

---

## Q1. Does scaffolding still pay when the model reasons internally?

The question with the most immediate practical weight. Teams are still wrapping
elaborate multi-agent pipelines around models that already deliberate internally. If
external scaffolding is largely redundant against a reasoning model, a great deal of
production complexity is unnecessary. Watch whether the discovered architectures thin
out as internal reasoning increases.

## Q2. Where is the cost crossover?

The real deployment decision is *small model plus expensive scaffold* versus *large
model plain*, at matched spend. AFlow reports workflows letting smaller models match
GPT-4o at 4.55% of its cost; if that generalizes it is the most economically
consequential result in this literature. Build the Pareto front and locate the crossover.

## Q3. Does the search know when to stop?

Given a strong worker, does the meta agent converge on simpler designs, or does it keep
adding machinery because nothing penalizes it? Automated design that cannot recognize
sufficiency is a different tool from one that can.

## Q4. Is external scaffolding a substitute or a complement for internal reasoning?

Test-time compute can be spent inside the model as reasoning tokens or outside it as
extra calls. Which buys more accuracy per token is genuinely open, and the answer likely
depends on the task. This is the cleanest version of Q1 and it is directly measurable.

## Q5. Which direction does a design transfer?

Designing under a cheap worker and deploying under an expensive one would be an
attractive workflow if it worked. Check both directions, since they need not be
symmetric.

## Q6. What actually breaks?

Which models cannot function as meta agents, and why. The failure taxonomy is a primary
result, not an appendix — knowing the floor is practically useful to anyone choosing a
model for this role.

## Q7. What do the discovered agents actually look like?

Read them. The most valuable output of ADAS-style work is usually a specific surprising
design that someone noticed, and no metric substitutes for looking.

## Q8. Does designer size change the *character* of the designs, not just their quality?

Every question above varies the worker. This one varies the designer and asks what
changes about the artifacts themselves when a small model rather than a large one is
doing the designing.

A primary question, not an afterthought, and nearly free because the meta ladder is
already a factor in the matrix. It needs its own metrics, because "scores higher" and
"designs differently" are separate claims and accuracy only addresses the first.

The interesting possibilities are qualitative rather than scalar. A small designer might
recombine archive entries while a large one invents. It might write an impressive design
description and then implement something much simpler. It might reach for one building
block repeatedly while a larger model routes, decomposes and verifies. It might ignore
the archive entirely and wander rather than building on what came before. None of these
show up in a fitness number, and two designs with identical accuracy can differ on all
of them.

## Q9. What predicts whether an agent performs well?

Two halves: how complexity relates to performance, and which components recur in high
performers as against low performers.

This needs no additional runs — by the end of the campaign the archive holds several
hundred design/fitness pairs spanning multiple workers, designer tiers and domains,
which is a dataset in its own right.

Expect a conditional rather than absolute answer. The complexity that suits a 7B worker
is probably not the complexity that suits a 27B one, which is why the curves are drawn
per worker tier and never pooled. A single global answer to "how much scaffolding is
right" would be the less interesting outcome.

---

## What answers each

| Q  | Primary evidence                                        | Headline artifact                          |
|----|---------------------------------------------------------|--------------------------------------------|
| Q1 | Bare-model and CoT baselines vs. searched cells; Stage 2r | Accuracy-vs-cost Pareto front              |
| Q2 | Same Pareto front, cost in tokens                        | Crossover point on the front               |
| Q3 | Stage 2d cost-penalized vs. unpenalized corner cells     | Complexity vs. worker capability, per meta |
| Q4 | Stage 2r, reasoning effort low vs. high, worker fixed    | Reasoning-effort line overlaid on Q3 plot  |
| Q5 | Cross-evaluation grid, best agent per cell by every worker | Transfer heatmap; diagonal strength       |
| Q6 | Seven-category failure taxonomy per meta model           | Stacked failure distribution               |
| Q7 | Direct reading of archives plus stepping-stone lineage   | Two or three annotated agents in full      |
| Q8 | Designer signature layer, one row per meta tier          | Designer signature table                   |
| Q9 | Pooled campaign; complexity curves and component attribution | Complexity-performance curves per worker |

Two of these are worth protecting when the budget tightens. Q8 needs all three designer
tiers — a two-point ladder cannot distinguish a trend from a gap — so cut worker levels
before meta levels. And the cross-evaluation grid should run regardless of what happens
upstream: it is cheap relative to the searches, answers Q5 directly, and yields an
interpretable figure even if every search underwhelms.

Note that accuracy converging across cells does not mean the designs converged. That
situation is the most favourable one for Q8, since it isolates differences in design
character from differences in outcome.
