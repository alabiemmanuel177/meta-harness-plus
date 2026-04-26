# Paper Section 2 — Related Work (Draft)

This is the prose draft of section 2. Length target: ~1 page.

---

## 2. Related Work

### Automated harness search

**Meta-Harness** (Lee et al. 2026; arXiv 2603.28052) is the most
direct prior work. They frame harness search as an agentic
optimization problem: an LLM proposer with filesystem access to all
prior candidates' run logs proposes the next harness shape; the
search yields +7.7pt accuracy at 4× lower token cost on
label-intensive Chinese legal classification (LawBench). We
extend their framework along three orthogonal axes: multi-objective
Pareto search instead of scalar accuracy, successive halving for
budget-aware evaluation, and component-level drop-one ablation as
proposer guidance. Section 4.4 directly replicates their LawBench
2-2 task and reports both positive and negative results.

**DSPy** (Khattab et al. 2023, 2024) and its bootstrap-fewshot
optimizer pioneered automated prompt-program optimization. Their
optimizer searches over fewshot examples and prompt templates with
gradient-free signal, but does not search over component-level
structural choices (retriever, reranker, voter) at a single-objective
level rather than Pareto. We treat component selection as the search
space and add cost objectives.

**OPRO** (Yang et al. 2023) and **APE** (Zhou et al. 2023) optimize
prompt strings via LLM-driven search over text. Their search space
is the prompt; ours is the harness shape. Complementary work; the
two approaches could compose.

### Multi-objective optimization

**NSGA-II** (Deb, Pratap, Agarwal, Meyarivan 2002) is the canonical
genetic-algorithm Pareto sorter. We use a simpler explicit-frontier
approach since our search space is discrete-and-small enough that
non-dominated sorting on the full frontier is cheap.

**Hypervolume Improvement** acquisition functions
(Hernandez-Lobato et al. 2016, Belakaria, Deshwal, Doppa 2019)
quantify Pareto frontier gain in Bayesian optimization. We use the
hypervolume metric for early-stopping but do not yet implement a GP
surrogate over the harness space; that would be a natural follow-on
(see THEORY.md §5).

### Budget-aware evaluation

**Successive Halving** (Jamieson & Talwalkar 2016) and **Hyperband**
(Li et al. 2017) are the standard budget-aware
hyperparameter-search algorithms. Our `SearchRunner` implements
single-bracket SH; Hyperband is implemented in
`meta_harness_plus/hyperband.py` but not yet wired into the default
search loop (deferred to future work — single-bracket SH was
sufficient to demonstrate the C2 ablation).

### Attribution methods

**Drop-one ablation** is the simplest model-agnostic attribution
method. Equivalent to grouped-feature Shapley value approximation
when components contribute additively to score (Lundberg & Lee 2017
for SHAP background). Our `AttributionTracker` computes EWMA-aggregated
per-kind drop-one deltas across the surviving frontier — cheaper
than full Shapley but adequate for the proposer's guidance signal.

### Retrieval-augmented generation foundations

**RAG** (Lewis et al. 2020), **Self-Consistency** (Wang et al.
2022), and **CoT** (Wei et al. 2022) provide the component vocabulary
of our search space. We treat these as composable building blocks.
**LLM-as-judge reranking** (Sun et al. 2023) is implemented as
`LLMReranker`; appears in the discovered alpha shape (Section 4.4).

### Statistical methodology

We use **paired bootstrap CIs** (Efron & Tibshirani 1993),
**paired t-tests**, and **Cohen's d** as standard tools. The
multi-seed protocol with stratified holdout and median-aggregated
repeats is meant to defeat LLM nondeterminism; nothing novel
methodologically, but rare in this class of paper to actually
report.
