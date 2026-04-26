# Paper Section 3 — Method (Draft)

This is the prose draft of section 3. Length target: ~2 pages.

---

## 3. Method

### 3.1 Search-space formulation

A *harness* `h` is an ordered pipeline of components `[c_1, ..., c_n]`.
Each component has a `kind` (retriever, reranker, fewshot, formatter,
predictor, voter) and a config dict. The full harness shape space `H`
is the cartesian product across kinds. With our default registry,
`|H|` ≈ 5 retrievers × 3 rerankers × 2 fewshots × 3 formatters × 3
voters × continuous-config ≈ 270 discrete shapes plus a continuous
config tail.

Each harness scores into a 3-vector
`s(h) = (accuracy, tokens, latency_ms)`, all measured by running the
harness on a held-out eval set. Accuracy is maximized; tokens and
latency are minimized. The Pareto frontier `F ⊂ H` is the set of
non-dominated points under componentwise partial order.

### 3.2 Components (Section 4.1 lists the registered set)

Each component-kind has a *baseline counterpart* — a no-op variant
(NullRetriever, NullReranker, etc.) — used by drop-one attribution
to estimate the kind's contribution.

### 3.3 Search loop

Each search iteration:

1. **Propose** `N` candidate harnesses via the LLMProposer
   (default) or RandomProposer (no-c3 ablation). The LLMProposer
   reads the filesystem run log, the current frontier, attribution
   stats, and per-class accuracy of the best frontier point, and
   emits up to `N` candidate JSON specs.
2. **Screen** each candidate on `screen_size` items of the eval set;
   apply successive halving to keep the top `halving_final_keep`
   for full evaluation.
3. **Full-eval** the survivors on `full_eval_size` items with
   `eval_repeats` median aggregation to defeat LLM nondeterminism.
4. **Admit** non-dominated survivors to the Pareto frontier
   (with variance-gated rejection of unstable candidates).
5. **Drop-one** ablate each surviving frontier addition to update
   per-kind attribution stats.
6. **Early-stop** if the hypervolume improvement window is below
   threshold.

Pseudocode is in Appendix A.

### 3.4 LLMProposer prompt

The LLMProposer prompt has six sections:

- **Context.** Search task description + classes.
- **Available components.** Registry contents with config fields.
- **Current frontier.** Top-k frontier entries with full describe()
  output.
- **Attribution stats.** EWMA mean Δ per kind across frontier history,
  surfaced as "kind X has helped/hurt by Y on average across recent
  candidates."
- **Token-budget hint.** "Strict-Pareto-dominate-RAG target token
  count = …" — biases the proposer toward cheaper shapes when
  RAG is already accurate enough.
- **Per-class accuracy** of the best frontier point — surfaces which
  classes the search hasn't yet cracked.

The prompt is curated rather than raw filesystem grep — simpler
proposer reasoning, faster, comparable empirical performance to
giving the proposer raw filesystem access.

### 3.5 Frontier-factory injection (for the ablations)

Our `SearchRunner` accepts a `frontier_factory` callable that returns
a fresh ParetoFrontier on each search start. The default returns
`ParetoFrontier(max_accuracy_spread=...)`. The C1 ablation injects
`ScalarAccuracyFrontier()` which keeps only the best-accuracy point.
The C2 (no halving) ablation is achieved by setting
`halving_k0 = full_eval_size` and `halving_final_keep = N`. The C3
(random proposer) ablation swaps the proposer at construction.
This makes all four ablation conditions runnable from the same CLI
binary with a `--ablation` flag.

### 3.6 Statistical hygiene

For every cell of the experiment grid:

- 5 (or 10, future work) random seeds varying both the LLM proposer
  temperature seed and the search's screen-subset seed.
- Stratified eval-set holdout.
- `eval_repeats=2` median aggregation per harness.
- Paired bootstrap CI on per-seed `(MH++ peak − RAG)` accuracy
  differences (n=2000 resamples).
- Paired t-test for parametric significance.
- Strict-Pareto-dominance counts as exact integers per seed.

### 3.7 Implementation notes

The framework is a single Python package (`meta_harness_plus/`) with
no torch / numpy / scipy dependencies — it uses only the standard
library plus `urllib` for HTTP calls. This keeps the install
footprint minimal and makes the framework run on any Python 3.10+
host without environment management.

Per-prompt SHA-256-keyed JSONL append-only caching makes multi-seed
runs cheap: cache hit rate runs 80–93% on repeat seeds, since most
harness shapes share predictor calls.

LLM HTTP calls are issued via a single `HTTPClient` that auto-detects
provider (Anthropic / OpenAI / Gemini / Ollama) from the URL. Tokens
and latency are measured per-call by the client. Parallel scoring
uses Python `ThreadPoolExecutor` (8 threads default), which releases
the GIL during the urllib I/O wait.

The full framework is 267 unit tests (all passing). Test classes
include: components, scorer, runner, attribution, statistics, cache,
ablations, hyperband, baselines, predictor, proposer, registry,
multi-seed runner, CLI integration.
