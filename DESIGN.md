# Meta-Harness++: A Multi-Objective, Budget-Aware, Attribution-Guided Framework for Automated Harness Search

**Status:** research prototype
**Started:** 2026-04-22
**Positions against:** Lee, Nair, Zhang, Lee, Khattab, Finn. *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052, March 2026.

## 1. What the original does

Meta-Harness (MH) automates the search over *harnesses* — the code around a fixed base model that handles prompts, memory, retrieval, tool definitions, and execution. An agentic proposer (Claude Code by default) gets filesystem access to every prior candidate's source code, execution traces, and scores, then proposes new candidates. This gives the proposer ~10M tokens of diagnostic context per iteration vs ~25K for OPRO/TextGrad-style scalar-guided optimizers — roughly a 400× increase. Reported gains: +7.7pt with 4× fewer tokens on label-intensive text classification; +4.7pt average on IMO-level math across five held-out models; SOTA on TerminalBench-2 with Haiku 4.5 and #2 with Opus 4.6.

## 2. Limitations we address

Three weaknesses follow from the framing. MH's paper does not enumerate them, but they fall out of the design:

| # | Limitation | Why it matters |
|---|---|---|
| L1 | **Scalar objective.** The search optimizes a single score. | Real harnesses trade accuracy, tokens, and latency. A +2pt harness that costs 3× more tokens may be unshippable — but MH's search can't see the cost axis. |
| L2 | **Full eval per proposal.** Every candidate runs against the complete eval set. | Expensive. If you want 100 candidates and each eval costs $10 of API, that's $1000 per iteration — and most candidates are bad. Wasted compute. |
| L3 | **No component attribution.** When a candidate wins, MH doesn't know *which* component (retriever? prompt? voter?) caused the gain. The proposer has to re-infer it from traces next round. | Lost signal. Future proposals can't preferentially mutate the high-value components, so search is less sample-efficient than it could be. |

We do not claim these are fatal to MH — they're clearly solvable. Our contribution is to solve them in a clean, composable framework and show, on a reproducible toy benchmark, that each fix carries its weight.

## 3. Our contributions

### C1 — Multi-objective Pareto search (addresses L1)

Replace scalar score with a vector `(accuracy, –tokens, –latency)`. Candidates are ranked by non-dominated sorting, and we maintain a Pareto frontier rather than a "best." The proposer sees the frontier and is encouraged to target under-represented regions (e.g., "we have high-accuracy expensive harnesses and cheap low-accuracy harnesses — propose something cheap-and-accurate").

Why it matters: MH's single-axis ranking can reward tokens-bloating harnesses that won't ship. Pareto search surfaces the *curve*.

### C2 — Successive-halving, budget-aware evaluation (addresses L2)

Each search iteration uses a two-stage eval:

1. **Screening eval** on a small fixed subset (e.g., 10 items from the eval set). Fast, cheap.
2. **Full eval** only for candidates that survive the screen — we keep the top `k/η` survivors per round (Hyperband-style successive halving).

Given a fixed compute budget `B`, we quantify the number of candidates we can evaluate under naive vs halving strategies. The framework tracks actual token/latency cost spent and refuses to exceed `B`.

### C3 — Component-level credit assignment via targeted ablation (addresses L3)

A harness is a composition of components (Retriever, MemoryStore, FewShotSelector, Formatter, Voter, …). When a candidate improves on the Pareto frontier, we run *targeted ablations*: for each component, swap it with a minimal baseline and re-score on the screening set. The delta is that component's **attributed value**. These values are stored per-component and:

- Surfaced to the proposer (via the filesystem interface, staying faithful to MH's design ethos).
- Used to weight mutation probabilities: components with high historical attribution get mutated more aggressively, low-attribution components get mutated less (or removed — negative attribution triggers a prune proposal).

This is credit assignment through the search trajectory, not just gradient-style feedback — novel for harness search, to our knowledge.

## 4. Framework architecture

```
meta_harness_plus/
├── harness.py          # Harness = ordered composition of Components
├── components.py       # Retriever, MemoryStore, FewShotSelector, Formatter, Voter (+ baselines)
├── task.py             # Task interface: inputs, references, eval()
├── tasks/
│   └── toy_classification.py   # deterministic built-in task
├── scorer.py           # returns ScoreVector(accuracy, tokens, latency)
├── pareto.py           # Pareto frontier maintenance (non-dominated sort)
├── search/
│   ├── proposer.py         # Proposer ABC
│   ├── mock_proposer.py    # attribution-guided mutation proposer (offline, deterministic)
│   └── llm_proposer.py     # Claude/OpenAI hook (stub; documented interface)
├── halving.py          # successive halving budget allocator
├── attribution.py      # drop-one-component ablation + running stats
├── runner.py           # main loop: propose → screen → halve → full-eval → attribute → update frontier
└── logging_utils.py    # filesystem-laid-out run log (parity with MH's proposer-facing FS)
```

## 5. The toy benchmark

Running a real LLM in unit tests is flaky, slow, and costly. So we include a **deterministic mock LLM + toy classification task** where:

- The "LLM" is a scoring function that rewards harnesses with specific good properties (having a retriever, using few-shot examples, voting across multiple predictions), *and* penalizes harnesses that spend more tokens/latency, such that a clear Pareto curve exists.
- The task is small (50 train / 20 eval items over 5 classes), fully deterministic given a seed.
- Good harnesses beat the baseline by a predictable amount, so tests can assert monotonic improvement.

This is not the real experiment — it's the **unit test for the framework.** Real experiments plug in a real LLM via the `llm_proposer` stub.

## 6. How we test the contributions

| Claim | Test |
|---|---|
| C1 — Pareto frontier is non-dominated | `test_pareto.py`: random candidates, verify no frontier point dominates another; any dropped point is dominated. |
| C2 — Successive halving respects budget | `test_halving.py`: simulate N candidates with known scores, assert total evals ≤ budget, top candidate survives. |
| C2 — Halving finds the true top-k most of the time | Run 100 seeded trials; assert screening identifies the true top candidate in ≥90% of runs on the toy task. |
| C3 — Attribution recovers known-good components | Construct a harness where Retriever contributes +X and Voter contributes +Y by construction in the mock LLM; assert attribution recovers the ordering X vs Y. |
| End-to-end | `test_end_to_end.py`: run the full loop on the toy task; assert the final Pareto frontier dominates a single-point baseline harness. |

## 7. What this is not

- **Not a reimplementation of MH.** We don't re-run TerminalBench or IMO. We build a *clean framework* that addresses three real limitations and test it on a deterministic toy task — the real-LLM bakeoff against MH is a follow-on.
- **Not claiming SOTA.** The claim is methodological: multi-objective + budget-aware + attribution-guided is a better default than scalar + full-eval + opaque, and we show each piece works in isolation.
- **Not a replacement for the filesystem-as-memory interface.** We keep it — our logging lays out run state on disk the same way, so an LLM proposer can be dropped in.

## 8. Open questions / future work

- Multi-proposer ensemble with diversity pressure (addresses single-proposer mode collapse; out of scope here).
- Cross-domain warm start (transfer the Component library across tasks).
- Robustness-aware scoring (adversarial eval as a 4th axis).
- Real-LLM bakeoff on a label-intensive classification subset (we document the hook but don't run it in this prototype).
