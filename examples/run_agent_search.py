"""Run MH++-style search over agent-policy harnesses on the local
TerminalBench-style fixture.

The fixture is the deterministic 6-task suite in
``meta_harness_plus.tasks.terminalbench_fixture``. The "search space"
in this minimal end-to-end runner is:

- two candidate policies: a rule-based oracle (cheap baseline) and a
  reflective oracle that reads the last observation and only writes
  to the answer file (more turns but higher recovery on errors)

For real LLM-driven agent search:
    OPENAI_API_KEY=... python3 examples/run_agent_search.py --api openai

For the offline smoke run (no API key needed):
    python3 examples/run_agent_search.py --offline

The output JSON has the same shape as ``rag_vs_mh_bakeoff.py`` so the
existing aggregators work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running via `python3 examples/run_agent_search.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_harness_plus.agent import AgentHarness, AgentScorer, trajectory_to_dict
from meta_harness_plus.agent_policy import RuleBasedPolicy, llm_agent_policy
from meta_harness_plus.tasks.terminalbench_fixture import build_terminalbench_fixture


# Oracle rules — same as the test, written here for reproducibility.
ORACLE_SCRIPTS: list[tuple[str, list[str]]] = [
    ("Count how many times the word 'alpha'", [
        "grep -o 'alpha' doc.txt | wc -l | tr -d ' ' > answer.txt",
    ]),
    ("Read users.json", [
        ("python3 -c 'import json;"
         "d=json.load(open(\"users.json\"));"
         "print(max(d[\"users\"],key=lambda u:u[\"age\"])[\"name\"])'"
         " > oldest.txt"),
    ]),
    ("Rename every *.log file", [
        "for f in *.log; do mv \"$f\" \"${f%.log}.txt\"; done",
    ]),
    ("Read spec.txt and produce solution.py", [
        "printf 'def add(a, b):\\n    return a + b\\n' > solution.py",
    ]),
    ("Extract every line containing 'ERROR'", [
        "grep ERROR log.txt > errors.txt",
    ]),
    ("Compute the sum of all integers", [
        "awk '{s+=$1} END {print s}' nums.txt > sum.txt",
    ]),
]


def _build_chat_fn_openai(model: str):
    """Return a chat_fn(system, user) -> (text, tokens) using OpenAI."""
    try:
        from meta_harness_plus.llm.client import OpenAIClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Need meta_harness_plus.llm.client.OpenAIClient: {e}")
    client = OpenAIClient(model=model)

    def chat(system: str, user: str) -> tuple[str, int]:
        resp = client.chat(system=system, user=user, max_tokens=128, temperature=0.0)
        return (resp.text, resp.tokens or len(resp.text) // 4)

    return chat


def _build_chat_fn_gemini(model: str):
    try:
        from meta_harness_plus.llm.client import GeminiClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Need meta_harness_plus.llm.client.GeminiClient: {e}")
    client = GeminiClient(model=model)

    def chat(system: str, user: str) -> tuple[str, int]:
        resp = client.chat(system=system, user=user, max_tokens=128, temperature=0.0)
        return (resp.text, resp.tokens or len(resp.text) // 4)

    return chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="Use the offline rule-based oracle policy (no API).")
    ap.add_argument("--api", choices=["openai", "gemini"], default=None)
    ap.add_argument("--model", default="gpt-4.1-nano")
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--max-turns-override", type=int, default=0)
    ap.add_argument("--out", default="runs/agent_local_fixture/result.json")
    args = ap.parse_args()

    task = build_terminalbench_fixture()

    candidates: list[tuple[str, AgentHarness]] = []

    # Always include the deterministic oracle as the baseline.
    candidates.append((
        "rule_based_oracle",
        AgentHarness(
            policy=RuleBasedPolicy(ORACLE_SCRIPTS),
            name="rule_based_oracle",
            max_turns_override=args.max_turns_override,
        ),
    ))

    if not args.offline and args.api:
        if args.api == "openai":
            chat_fn = _build_chat_fn_openai(args.model)
        else:
            chat_fn = _build_chat_fn_gemini(args.model)
        candidates.append((
            f"{args.api}_{args.model}",
            AgentHarness(
                policy=llm_agent_policy(chat_fn),
                name=f"{args.api}_{args.model}",
                max_turns_override=args.max_turns_override,
            ),
        ))

    scorer = AgentScorer(task=task)
    results = []
    for label, h in candidates:
        t0 = time.perf_counter()
        sv = scorer.score(h, n_repeats=args.n_repeats)
        elapsed = time.perf_counter() - t0
        trajectories = [trajectory_to_dict(t) for t in scorer.last_trajectories]
        results.append({
            "label": label,
            "accuracy": sv.accuracy,
            "tokens": sv.tokens,
            "latency_ms": sv.latency_ms,
            "n_evaluated": sv.n_evaluated,
            "n_repeats": sv.n_repeats,
            "wall_seconds": elapsed,
            "trajectories": trajectories,
        })
        print(f"{label}: acc={sv.accuracy:.3f} "
              f"tokens={sv.tokens:.1f} latency_ms={sv.latency_ms:.1f} "
              f"wall={elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"task": task.name, "n_examples": len(task.examples),
         "candidates": results},
        indent=2,
    ))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
