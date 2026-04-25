# Roadmap: From Prototype to "Shock the World"

**Where we are (2026-04-25):** A working zero-dep Python framework implementing
multi-objective Pareto search over LLM harnesses with successive halving,
drop-one attribution, exploration-gap surfacing, reproducibility layer
(median-over-repeats), and a richer component library (reranker slot, CoT
formatter). 111 passing tests. Bakeoff results on two hand-curated tasks
× 2 local Ollama models showing MH++ adaptively picks different shapes per
model architecture, with the +0.13pt "win" partially evaporating under the
50-item eval (selection bias in 15-item set).

**The honest gap:** This is a strong prototype, not published research.
Below is every concrete item that would close that gap, then surpass it.

Items are roughly ordered by leverage within each tier. **Tier 1-3 are
necessary; tier 4+ are differentiating.**

---

## TIER 1 — Methodological credibility (table stakes for any claim)

### 1.1 Public benchmarks (replace hand-curated)
- [ ] **LawBench** (Chinese legal classification, 215 classes) — the original
  Meta-Harness paper's main bed. Gives direct comparability.
- [ ] **USPTO-50k** (English patent classification, 50 classes) — the paper's
  second bed.
- [ ] **MASSIVE** (Amazon-curated, 60 intents, 50 languages) — broader generality.
- [ ] **Symptom2Disease** (medical, ~1000 items) — same domain as our toy
  task but actually published.
- [ ] **20 Newsgroups** (classic 20-class) — easy public benchmark.
- [ ] **AG News** (4-class news, 120k train) — direct comparison to our
  hand-curated `news_hard`.

### 1.2 Statistical rigor
- [ ] **5+ random seeds per harness**, report mean ± std, not point estimates.
- [ ] **Held-out test set** distinct from search-eval set. Currently we
  effectively overfit search to the eval items.
- [ ] **Confidence intervals** on every reported number (bootstrap).
- [ ] **Statistical significance tests** (paired t-test or Wilcoxon) on
  MH++ vs RAG accuracy differences.
- [ ] **Multiple-comparison correction** (Bonferroni / Benjamini-Hochberg)
  when reporting per-component attribution across N ablations.
- [ ] **Power analysis** before each experiment: how many eval items do we
  need to detect a 3pt accuracy gap with α=0.05, β=0.2?

### 1.3 Strong baselines (not just RAG)
- [ ] **Hand-tuned chain-of-thought** prompt baseline.
- [ ] **DSPy-optimized RAG pipeline** as a stronger baseline than vanilla RAG.
- [ ] **OPRO** / **TextGrad** / **ProTeGi** as scalar-optimizer baselines.
- [ ] **The original Meta-Harness** itself, run on the same tasks. Until we
  beat the published method on its own benchmarks, we're not surpassing it.
- [ ] **Random search** baseline at matched compute budget (kills the
  "anything beats random" criticism).

### 1.4 Ablation isolating each contribution
- [ ] **C1 ablation** (Pareto vs scalar): turn off Pareto, score on accuracy
  alone. Does C1 actually find harnesses scalar search misses?
- [ ] **C2 ablation** (halving vs full eval): turn off successive halving,
  evaluate every candidate on full set. Does halving lose accuracy or just
  budget?
- [ ] **C3 ablation** (attribution vs uniform mutation): turn off
  attribution-weighted proposer mutations. How much sample-efficiency does
  C3 actually buy?
- [ ] **Exploration-gap surfacing ablation**: remove the "components never
  on the frontier" prompt section. Does the LLM proposer still discover
  unexplored shapes?
- [ ] **Reranker ablation**: re-run all bakeoffs without the reranker slot
  in the action space. Quantify lift.
- [ ] **eval_repeats ablation**: 1 vs 3 vs 5 repeats. Plot frontier
  stability as a function.

---

## TIER 2 — Statistical / experimental scale

### 2.1 Eval scale
- [ ] **500-1000 item eval** per benchmark (not 50).
- [ ] **Per-class breakdown** — accuracy, attribution, frontier shape per class.
- [ ] **Class imbalance robustness** — synthetic shifts of class proportions.
- [ ] **Adversarial item curation by external annotators**, not the same
  person who designed the baselines.

### 2.2 Model coverage
- [ ] **Frontier proprietary**: Claude Opus 4.7, GPT-5, Gemini 3.
- [ ] **Frontier OSS**: Llama 4 405B, Qwen3 large variants, DeepSeek-V4.
- [ ] **Mid-tier**: Mistral medium, Gemma 4 family, Llama 4 70B.
- [ ] **Small-tier**: Phi-4, Llama 4 8B, Qwen3 7B.
- [ ] Heterogeneity panel: 10+ models spanning size + reasoning vs not +
  open vs closed.
- [ ] **Held-out model generalization**: search on model A, evaluate on
  model B without re-search.

### 2.3 Domain breadth
- [ ] **Classification** (where we are now).
- [ ] **Math reasoning** (GSM8k, MATH, AIME) — original MH paper's second axis.
- [ ] **Code generation** (HumanEval, MBPP, SWE-bench).
- [ ] **Agentic** (TerminalBench-2 — original paper's third axis).
- [ ] **Multi-hop QA** (HotpotQA, 2WikiMultiHopQA).
- [ ] **Long-context retrieval** (LongBench, RULER).
- [ ] **Tool use** (ToolBench, BFCL).

### 2.4 Cost reporting in real units
- [ ] **Dollars per eval** (per-token API cost × tokens).
- [ ] **Wall-clock seconds per eval**.
- [ ] **Pareto frontier in $-vs-accuracy space**, not abstract token counts.
- [ ] **Search cost amortization** — how many production queries before
  search investment pays back?

---

## TIER 3 — Novel theoretical contributions

### 3.1 Sample complexity / regret
- [ ] **Regret bound** for attribution-guided harness search vs uniform mutation.
- [ ] **PAC-style sample complexity** — given N harness shapes, how many
  evals to find ε-optimal Pareto frontier with probability ≥ 1−δ?
- [ ] **Halving correctness proof** under non-deterministic eval (Ollama
  temp=0 case): probability of dropping the true optimum at round r.

### 3.2 Information-theoretic framing
- [ ] **Mutual information** between attribution snapshots and true component value.
- [ ] **Diagnostic context budget vs accuracy** scaling law (the original
  paper's filesystem-as-context idea formalized).
- [ ] **Pareto vs scalar information gain** — formal claim that vector
  feedback strictly contains more decision-relevant information.

### 3.3 Search-space geometry
- [ ] **Component graph metric** — when are two harnesses "close" in
  shape? Define a meaningful distance function over Pareto frontier.
- [ ] **Smoothness of accuracy surface** — local search behavior depends
  on how lipschitz accuracy is in component-space.
- [ ] **Connection to AutoML / NAS** — harness search as a structured-arms
  bandit; relate to existing NAS bounds.

### 3.4 Causal vs correlational attribution
- [ ] **Counterfactual attribution** instead of drop-one ablation.
  Drop-one mixes confounded effects; do-calculus would clean them up.
- [ ] **Shapley-value attribution** for components — fair credit assignment
  in coalitional sense.
- [ ] **Synergy scores** — pairs of components whose joint value exceeds
  their marginal (e.g., we observed CoT works with reranker but not alone;
  formalize that).

---

## TIER 4 — Framework contributions beyond the original

### 4.1 Search-strategy upgrades
- [ ] **Multi-proposer ensemble** with explicit diversity pressure
  (counters single-proposer mode collapse — limit MH itself acknowledges).
- [ ] **Bayesian optimization** over harness shape with Gaussian-process
  surrogate.
- [ ] **Population-based training**-style harness search — N parallel
  searches with periodic shape exchange.
- [ ] **Curriculum** — start search on small eval subset, expand as
  Pareto frontier stabilizes.
- [ ] **Online / streaming search** — harness updates from production
  traffic, not batch eval.
- [ ] **Hyperband** scheduler instead of plain successive halving.

### 4.2 Pareto-frontier mechanisms
- [ ] **Variance-gated admission**: reject candidates with
  `accuracy_spread > ε` regardless of point accuracy.
- [ ] **Hypervolume-driven exploration**: proposer rewarded by hypervolume
  gain, not just accuracy.
- [ ] **Reference-point Pareto** (R-NSGA-II) for steering toward a region.
- [ ] **Multi-objective ≥ 4 axes**: add robustness, fairness, calibration
  as additional Pareto dimensions.

### 4.3 New action-space components
- [ ] **Reranker variants**: TF-IDF, BM25, cross-encoder reranking.
- [ ] **Self-correction** components (LLM critiques + revises its own answer).
- [ ] **Routing** components (different model for different query types).
- [ ] **Tool-use** components (calculator, retrieval API, code-exec).
- [ ] **Verification** components (LLM-as-judge before final answer).
- [ ] **Caching layer** — answer-cache, embedding-cache as components.
- [ ] **Compression** component — summarize retrieved docs before few-shot.

### 4.4 Robustness / safety as Pareto axes
- [ ] **Adversarial accuracy** — how does harness perform on perturbed inputs?
- [ ] **Calibration** — does predicted-confidence track actual accuracy?
- [ ] **Fairness** — accuracy across demographic slices.
- [ ] **Refusal-rate** — for harms-related queries, does harness refuse
  appropriately?
- [ ] **Robustness to prompt injection** — adversarial query suffix tests.

---

## TIER 5 — Surpassing the original Meta-Harness

### 5.1 Direct head-to-head wins
- [ ] **Beat MH on TerminalBench-2** (their #1/Haiku, #2/Opus). MH++
  with richer components + Pareto + attribution should be capable; needs
  the actual run.
- [ ] **Beat MH on IMO math** (their +4.7pt result).
- [ ] **Beat MH on label-intensive classification** (their +7.7pt result
  with 4× fewer tokens).

### 5.2 Disprove or refine an assumption from the original
- [ ] **Filesystem-as-memory limit**: show our curated diagnostic prompt
  (frontier + attribution + exploration-gap) outperforms raw-grep access
  to the run dir at matched proposer-model size.
- [ ] **Single-proposer mode collapse**: empirically show ensemble
  proposers find regions single-proposer misses on a benchmark MH itself
  ran.
- [ ] **Scalar feedback ceiling**: construct a benchmark where scalar
  search provably can't reach a Pareto point our search can, then run
  both.

### 5.3 Reach where MH didn't
- [ ] **Cross-domain transfer**: search on math, deploy on code without
  re-search. Show transfer beats cold-start search at matched budget.
- [ ] **Continual harness improvement** in a deployed system, with
  attribution-driven updates.
- [ ] **Multi-task harness optimization** — single harness Pareto-optimal
  across N tasks simultaneously.

---

## TIER 6 — Adoption / infrastructure (multiplier on impact)

### 6.1 Open-source release
- [ ] **GitHub release** with documented onboarding.
- [ ] **Demo notebook** that runs in 5 minutes against a public model.
- [ ] **Hugging Face Spaces demo** for the bakeoff dashboard.
- [ ] **PyPI package** with stable API.

### 6.2 Reproducibility kit
- [ ] **Pinned model versions** (e.g., `gpt-oss:20b@digest`) in every result.
- [ ] **Tar artifact** per result containing harness JSON + run log + scoring code.
- [ ] **Re-run script** that takes an artifact + API key and reproduces
  the reported number ±ε.
- [ ] **Public leaderboard** with submission format spec.

### 6.3 Adapters for major proposer agents
- [ ] **Anthropic Claude Code agent** (matches original paper's default).
- [ ] **OpenAI Codex agent**.
- [ ] **Cursor Background agent**.
- [ ] **Aider** integration.
- [ ] **OpenHands** integration.

### 6.4 Hooks for real production use
- [ ] **Async streaming runner** for non-trivial deployments.
- [ ] **Distributed evaluator** with Ray / Modal / Coiled backends.
- [ ] **Cost dashboard** — running tally of search budget, projected payback.
- [ ] **Alerting** when production accuracy drifts below search-eval baseline.

---

## TIER 7 — Publication path

### 7.1 Writing
- [ ] **arXiv preprint**, ~10-12 pages plus appendix.
- [ ] **Reproducibility appendix** with full hyperparameters + seeds.
- [ ] **Negative results section** — be the rare paper that admits when
  the method didn't help.

### 7.2 Venue targeting (in order of bar-raising)
- [ ] **Workshop** at top venue first (NeurIPS / ICML / ICLR Foundation
  Models or AutoML workshop) — fast feedback.
- [ ] **EMNLP** main track for the classification benchmarks specifically.
- [ ] **NeurIPS** for the framework + theoretical results.
- [ ] **ICML** if the regret bounds are tight.

### 7.3 Outreach
- [ ] **Twitter / blog post** with replicable demo at submission time.
- [ ] **Talk at AutoML conference**.
- [ ] **Collaboration with industry lab** (Stanford IRIS itself? Anthropic?
  Google DeepMind?) — co-author on a follow-on paper.

---

## TIER 8 — Genuinely surpassing (the "shock" tier)

These are the items that would make this work *not* a small extension to MH
but a distinct contribution worth attention from outside the autoML niche.

### 8.1 Theoretical breakthroughs
- [ ] **Closed-form solution** for optimal halving schedule under Pareto
  ranking (vs the empirical eta=2 we use).
- [ ] **Proof** that attribution-guided proposers strictly outperform any
  scalar-blind proposer in the worst case.
- [ ] **Tight regret bound** matching empirical performance — settling the
  search-strategy literature.

### 8.2 Empirical breakthroughs
- [ ] **MH++ becomes the new SOTA** on TerminalBench, replacing the
  original Meta-Harness on the leaderboard.
- [ ] **Cross-model transfer law**: show search on cheap model A predicts
  Pareto frontier on expensive model B with ρ > 0.9.
- [ ] **One-shot harness discovery**: show 1-iteration search with the
  right proposer matches 100-iteration baselines.

### 8.3 Conceptual / framing shifts
- [ ] **Reframe LLM applications as harness-search problems** — argue
  every prompt-engineering effort is implicitly doing manual MH; provide
  an automated upgrade.
- [ ] **Connect to mechanistic interpretability** — show component
  attribution surfaces interpretable model-internal causes, not just
  black-box deltas.
- [ ] **Self-improving search**: search the search itself (search the
  proposer, search the attribution method) to a fixed point.

### 8.4 Beyond MH territory entirely
- [ ] **Automatic task generation** — search proposes the next task that
  best discriminates between competing harnesses.
- [ ] **Multimodal harness search** (vision + audio + code as components).
- [ ] **Harness search with safety constraints** as hard frontier
  conditions, not soft Pareto axes.
- [ ] **RL-based harness search** with credit assignment via on-policy
  policy gradients (vs our drop-one off-policy attribution).

### 8.5 Industrial / societal
- [ ] **Adoption by 3+ frontier labs** for internal model-evaluation work.
- [ ] **Citation in a frontier lab's model card** as the evaluation method.
- [ ] **Standardized benchmark** that other autoML methods compete against
  ("the Meta-Harness Olympic").

---

## Honest realism: what's achievable in tiers, with effort

| Tier | Months of focused work | Outcome |
|---|---|---|
| 1 | 1-2 | A defensible workshop paper |
| 1 + 2 | 3-4 | A solid main-track conference paper |
| 1-3 | 6-9 | A NeurIPS / ICML acceptance |
| 1-4 | 9-12 | A paper that gets picked up beyond the autoML niche |
| 1-5 | 12-18 | A paper that *might* be considered the new SOTA |
| 1-6, partial 7 | 18-24 | A widely-adopted framework with industry citations |
| 1-7 + parts of 8 | 24+ | "Shock the world" territory |

**Most-impactful single next step** if forced to pick one: **Tier 1.1 (LawBench)
+ Tier 1.2 (5 seeds + held-out test)**. Run our framework on a public benchmark
with proper statistical hygiene. That alone moves us from "interesting
prototype" to "publishable workshop result." Everything else builds on
having that foundation.

**Cheapest impressive item**: **Tier 5.2 (disprove single-proposer mode
collapse)**. We already have the infrastructure for ensemble proposers in
mind; running one experiment on a benchmark MH itself ran with a matched
budget would be a sharp single-result paper.

**Highest-leverage long-bet**: **Tier 8.3 (self-improving search)**. If the
search can search itself, the framework becomes recursively useful and
qualitatively new. But it's also the longest path.
