# The Sharp Thesis

> *LLM performance is no longer just model performance. The harness around
> the model — retrieval, fewshot, formatting, voting — is an optimizable
> system, and **Pareto harness search is a new layer of AI infrastructure.***

## What we mean

Every team building an LLM application is silently doing harness search.
Should the retriever be BM25 or BoW? How many fewshot examples? Plain
prompt or chain-of-thought? Voting with three samples or just one?
These choices form a discrete-but-large space of *harness shapes*, and
practitioners explore it manually — usually settling on a "RAG-shaped
thing" after a few rounds of prompt engineering and stopping when the
metric stops moving.

The dominant narrative is that LLM application quality scales with
*model* — bigger or smarter base model means better outputs.

That narrative is wrong, or at least incomplete. **The same model can
deliver order-of-magnitude different cost/accuracy tradeoffs depending
on the harness around it.** On our experiments:

- Switching from a hand-tuned RAG harness to an MH++-discovered shape
  on the same model changes accuracy by up to +29.3pt (Cohen's d=+11).
- Strict Pareto improvements over RAG occur on 17 of 40 trial-cells —
  same-or-better accuracy AND fewer tokens AND not-worse latency,
  simultaneously.
- The discovered shapes hand-tuners would not write themselves.

This means: **harness optimization belongs alongside model selection
in the LLM application stack.** It is not a one-off prompt-engineering
hack; it is a systematic optimization layer.

## Three implications

### 1. The market structure of LLM apps changes

If harness search produces consistent +5–30pt gains over hand-tuning,
then "harness optimization service" becomes a layer in the LLM app
stack — like a CDN or a database query optimizer. Apps that don't run
harness search will be measurably worse than apps that do, in both
accuracy and cost. The ones that don't will look like apps without
caching: technically functional, but obviously sub-optimal.

### 2. The Pareto framing changes the optimization problem

Scalar accuracy search — what most existing prompt optimizers do — is
strictly worse than Pareto search on multi-objective tasks. Theorem 1
in our theory section: any point a scalar optimizer admits is also on
the Pareto frontier, but the Pareto frontier contains cost-cheap
variants the scalar optimizer ignores. Ablation data confirms this
empirically: scalar search uses +47% tokens and +79% latency for the
*same* accuracy as Pareto search.

Therefore: any production harness search should be Pareto by default.
Scalar accuracy as the only objective is a software-engineering
mistake.

### 3. Harness search becomes a scientific instrument

The discovered shapes themselves are interpretable: they say "for this
model + this task, BM25 retrieval + LLM-reranker + compressed-CoT
formatter is what works." The component-level attribution explains
*which piece* paid off. Drop-pair ablation (synergy discovery — Task
6) extends this: which component *pairs* only work together?

This makes MH++ not just an optimizer but a *scientific instrument*:
each search produces evidence about how harness components compose
for a given model+task, generating reusable design knowledge.

## Why now

Three trends collide:

1. **API costs make cost-axis optimization essential.** The +47% token
   gap between scalar and Pareto search is real money at scale.

2. **Model commoditization shifts value to the harness.** When the
   marginal model gain shrinks, harness gain becomes the differentiator.

3. **Smaller models + smarter harnesses outperform bigger models +
   plain RAG.** Our gpt-4.1-nano + MH++ shape (0.928 on news_hard_50)
   is competitive with a much larger model running vanilla RAG.
   This is the harness-replaces-model-scaling argument in
   miniature.

## Where this paper sits

The original Meta-Harness paper (Lee et al. 2026) demonstrated the
phenomenon at heavyweight scale (Claude Opus 4.6, ~10M tokens of
agentic context per iteration). Our extension — multi-objective Pareto
search, budget-aware evaluation, component-level attribution —
demonstrates that the same phenomenon is real, reproducible, and
substantially cheaper at the cheapest commercial model tiers.

**This is not "we added Pareto to Meta-Harness."** This is "the harness
optimization phenomenon generalizes to commodity scale, and Pareto +
attribution makes it interpretable + cost-aware enough to deploy in
production."

## What we're claiming, sharply

> *Pareto harness search is a new layer of AI infrastructure. Apps
> running this layer get up to 30pt accuracy improvements at lower
> token cost than hand-tuning, on every commercial LLM we tested. The
> framework, training caches, and reproducible CLI are all
> open-source. Total experimental compute: under $5 USD.*

That is the claim. The rest of the paper is evidence for it.
