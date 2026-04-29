# Meta-Harness++ / V10 SWE-bench Verified harness

This repository hosts two related lines of work:

1. **V10 — `harness/`** — a clean-rebuild SWE-bench Verified harness targeting 70%+ Pass@1, oracle-free, leaderboard-eligible. **This is the active project and the artifact under review.** See **[docs/V10_DESIGN.md](docs/V10_DESIGN.md)** for the full design.
2. **Meta-Harness++ — `meta_harness_plus/`** — the original research line (multi-objective harness search over LLM tooling), plus the V7 and V8 SWE-bench attempts. **V7/V8 are documented contaminated baselines.** They are retained here for paper comparison, reproducibility of prior runs, and so the published "documented contamination → clean rebuild" lift can be cited directly. See **[DESIGN.md](DESIGN.md)** for the original Meta-Harness++ research framing.

## V10 — what to look at for leaderboard review

If you're evaluating this work for SWE-bench Verified leaderboard eligibility or reading the arXiv preprint:

- **Read first:** [docs/V10_DESIGN.md](docs/V10_DESIGN.md) — full design including the contamination model (§2, §12), the phase-by-phase architecture, the cost model, and the asset inventory.
- **Look at:** `harness/`, `tests/test_no_oracle_leak.py`, `tests/test_dataset_and_conventions.py`, `tests/test_sandbox_and_cache.py`, `splits/dev_50.json`, `docs/audits/v7_missing_evals.md`, and the V10 PR.
- **Do NOT look at `meta_harness_plus/swebench_v7.py`, `swebench_adapter.py`, `agent_swebench_loop.py` as the V10 submission.** Those modules contain V7's actor with `FAIL_TO_PASS` pasted into the prompt at line 146, and they back V7's 63.6% number. They are kept tracked in this repo as the **explicit contaminated baseline**; the V10 firewall test (`tests/test_no_oracle_leak.py`) statically forbids any V10 module from importing them, and the runtime `LLMCallInspector` would raise `OracleLeakError` if any of their fields ever reached an LLM call.

The V10 contamination paths and the redesign that closes them are documented in **[V10_DESIGN.md §12](docs/V10_DESIGN.md)**.

### V10 quickstart

```bash
make verify-images          # 498/500 Verified images present locally
make test-v10               # 34 V10 unit tests, all green
make smoke                  # collect-only smoke (≤1 s)
make smoke-exec             # real pytest execution smoke (≤30 s)
make smoke-negative         # confirm firewall raises on real call paths
```

Phase 0 (this state) sets up the contamination firewall, the leak-free sandbox, and the dev-50 split. Phase 1+ (localization, repro oracle, hybrid generation, validation, selection, refinement, and an optional fine-tuned localizer) lands in subsequent PRs against `main`.

## Why both lines live in one public repo

Most leaderboard submissions ship a clean number and ask reviewers to trust it. By keeping V7's contaminated source side-by-side with V10's clean rebuild — and the V7 baseline data alongside the V10 firewall — this repository documents:

- What oracle leakage actually looks like in a working harness (V7's `swebench_v7.py:146` pastes `FAIL_TO_PASS` into the actor prompt).
- How to detect leakage statically and at runtime (V10's `tests/test_no_oracle_leak.py`).
- How much it inflates scores (V7 = 63.6% with leakage; V10 target = 70%+ oracle-free; the lift is part of the paper's contribution).

V7's run artifacts (`runs/swebench_500_v7/`) and the audit of why 33 V7 instances have no eval report (`docs/audits/v7_missing_evals.md`) are the calibration data that drives V10's difficulty priors and Phase 1 risk surfaces. They are deliberately public.

---

# Meta-Harness++ (the original research line)

A multi-objective, budget-aware, attribution-guided framework for automated search over LLM harnesses. Extends [Meta-Harness](https://arxiv.org/abs/2603.28052) (Lee et al. 2026, Stanford IRIS) with three fixes to limitations of the original.

See **[DESIGN.md](DESIGN.md)** for the research framing.

## What problem this solves

The original Meta-Harness automates search over the code around a fixed base model — prompts, retrieval, memory, tools, voting. It uses an agentic proposer with filesystem access to every prior candidate's source, traces, and scores (~10M tokens of diagnostic context per iteration). Reported gains: +7.7pt on label-intensive classification with 4× fewer tokens, +4.7pt on IMO math, SOTA on TerminalBench-2 with Haiku 4.5.

Three limitations fall out of its design that this project addresses:

| # | Limitation | This project's fix |
|---|---|---|
| L1 | Scalar objective — hides accuracy-vs-cost tradeoffs | **C1**: Pareto frontier over (accuracy, tokens, latency) |
| L2 | Full eval per proposal — wasted compute on bad candidates | **C2**: Successive halving — cheap screen, full eval only on survivors |
| L3 | No component attribution — can't tell which component caused a gain | **C3**: Drop-one ablation tracker → per-component mean delta → proposer bias |

## Install + run

No external deps. Python 3.10+. Run the bundled toy experiment:

```bash
python3 examples/run_toy_classification.py
```

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

## What the toy experiment shows

Starting from a deliberately weak baseline harness (no retrieval, no few-shot, one sample, no voting), 10 iterations × 8 proposals × successive halving discovers a Pareto curve:

```
acc   tokens   latency_ms   harness
0.85   43.0    23.1         fewshot(k=3) + predictor(n=7) + majority_voter
0.75   33.0    17.1         fewshot(k=3) + predictor(n=5) + majority_voter
0.45   13.0     5.0         baseline (no retrieval, no fewshot, n=1, null voter)
```

Accuracy rises from 0.45 to 0.85 (+0.40 absolute, ~89% relative) and the search surfaces the cost curve rather than collapsing to a single "best."

Attribution ranks components by their drop-one accuracy delta:

```
voter        mean_delta=+0.198   n=21   ← dominant accuracy lever
retriever    mean_delta=+0.000   n=21   ← didn't help because...
fewshot      mean_delta=+0.000   n=21   ← ...no winning candidate used retrieval
```

This is a real research signal: on this task the search's greedy Pareto discovery converged on a voting-heavy path before it explored the retrieval+few-shot path. A real LLM-backed proposer reading this attribution report would propose "try retrieval + few-shot harnesses — they're unexplored" — exactly the kind of reasoning Meta-Harness's filesystem interface is meant to enable.

## Wiring a real LLM

The mock proposer in `meta_harness_plus/search/mock_proposer.py` implements the `Proposer` protocol from `search/proposer.py`. Swap it out for an LLM-backed implementation that reads the filesystem-laid-out run log (`runs/<name>/candidates/`, `frontier.json`, `attribution_stats.json`) and proposes new harnesses. The disk layout is deliberately agent-friendly — `grep`, `cat`, and `find` are sufficient to surface every prior candidate's description, score, and attribution snapshot.

Likewise, `MockLLMPredictor` can be replaced with a real predictor that calls an API; the `Component` interface doesn't care what runs under it.

## Package layout

```
meta_harness_plus/
├── harness.py         Harness + Component abstractions
├── components.py      Retriever, FewShot, Formatter, Predictor, Voter (+ baselines)
├── task.py            Task + TaskExample
├── tasks/
│   └── toy_classification.py    deterministic offline task + mock LLM
├── scorer.py          ScoreVector + Scorer
├── pareto.py          Non-dominated sort + ParetoFrontier
├── halving.py         Successive halving search (Pareto-aware ranking)
├── attribution.py     Drop-one ablation tracker
├── search/
│   ├── proposer.py              Proposer protocol
│   └── mock_proposer.py         attribution-guided mutation proposer
├── logging_utils.py   filesystem run log (MH-compatible)
└── runner.py          end-to-end search loop
```

## Tests (22 of them, all pass, zero deps)

- **`test_pareto.py`** — dominance, admission/eviction, dedup, non-domination fuzz, knee point
- **`test_halving.py`** — budget accounting, monotone-signal recovery, Pareto-aware ranking
- **`test_attribution.py`** — delta recovery, running stats, softmax weights, zero-delta baseline
- **`test_toy_task.py`** — determinism, reward-surface ordering
- **`test_end_to_end.py`** — search improves over baseline; attribution correctly identifies high-value components

## What this isn't

- Not a reimplementation of the original Meta-Harness experiments (no TerminalBench, no IMO bakeoff here).
- Not a claim of SOTA — the claim is methodological: Pareto + budget-awareness + attribution is a better default than scalar + full-eval + opaque, and we show each piece carries its weight on a reproducible benchmark.
- Not a replacement for the filesystem-as-memory interface — we keep it, so the original paper's LLM proposer drops in.

## Branches

- **`main`** — the zero-dep prototype + merged LLM integration + merged reproducibility + merged real-dataset. Everything documented on this page is on main.
- **`reproducibility`** (merged) — adds `n_repeats` median-over-repeats aggregation to `Scorer`, `SearchConfig`, and `AttributionTracker`, plus a separate larger attribution screen. Defeats Ollama `temperature=0` GPU non-determinism. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- **`real-dataset`** (merged) — adds a JSONL loader + a hand-curated medical symptom classification dataset (5 specialties, 30 train / 20 eval). Ships `build_symptom_task()` and `symptom_mock_llm()` so the search can be exercised end-to-end on realistic data without a real LLM. See [DATASETS.md](DATASETS.md).
- **`llm-proposer`** (merged) — real LLM integration (`meta_harness_plus/llm/`):
  - `LLMClient` protocol, `ScriptedClient` fake, `HTTPClient` for Anthropic / Ollama / any OpenAI-compatible endpoint (vLLM, LM Studio, etc.) — all via stdlib `urllib`.
  - `ComponentRegistry` — structured JSON specs → validated Harness instances (no arbitrary Python exec).
  - `LLMPredictor` — real-LLM classifier Component with per-call token/latency accounting.
  - `LLMProposer` — reads the filesystem run log, assembles a curated diagnostic prompt (frontier + attribution stats + **exploration-gap surfacing**: components never seen on the frontier), asks the LLM for JSON proposals, validates them through the registry.
  - `examples/bakeoff.py` — compare mock vs LLM proposer on the same config.

### Running the bakeoff locally with Ollama

```bash
# Assumes Ollama is running on localhost:11434 and the model is pulled.
python3 examples/bakeoff.py --proposer llm \
    --ollama-url http://localhost:11434/api/chat \
    --model gpt-oss:20b
```

### With Anthropic

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 examples/bakeoff.py --proposer llm \
    --anthropic --model claude-opus-4-7
```

### With any OpenAI-compatible endpoint

```bash
OPENAI_API_KEY=sk-... python3 examples/bakeoff.py --proposer llm \
    --openai-url https://api.openai.com/v1/chat/completions \
    --model gpt-4o-mini
```

## Future work (not implemented)

- Multi-proposer ensemble with explicit diversity pressure (counter single-proposer mode collapse).
- Cross-domain warm start: transfer a Component library across tasks.
- Robustness-aware scoring (adversarial eval as a 4th Pareto axis).
- Real bakeoff at scale on a public label-intensive classification dataset (LawBench, USPTO-50k).

## Credit

Original: Lee Y., Nair R., Zhang Q., Lee K., Khattab O., Finn C. *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052, March 2026. [Stanford IRIS Lab](https://github.com/stanford-iris-lab/meta-harness).

License: MIT.
