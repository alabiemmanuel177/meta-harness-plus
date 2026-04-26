"""Simplified ProTeGi baseline (Pryzant et al. 2023, "Automatic Prompt
Optimization with Gradient Descent and Beam Search").

ProTeGi: beam search of LLM-mutated prompts. At each step:
1. Score current prompts on a small validation set.
2. Generate textual "gradients" (LLM critiques of failures).
3. Apply each gradient to produce mutated prompts.
4. Re-score mutations, keep top-`beam_size` for the next step.

This is a simplified instruction-only version (no decompositional
output structure) — closer in spirit to the canonical paper's
beam-search-over-prompts than to TextGrad's purely-gradient approach.

Usage:
    python3 examples/protegi_baseline.py \\
        --api openai --model gpt-4.1-nano \\
        --task news_hard_50 \\
        --rounds 3 --beam-size 3 \\
        --output runs/protegi_baseline_openai_news_hard_50.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.components import (
    NullFewShot, NullRetriever, NullVoter, SimpleFormatter,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.cache import CachedLLMClient, PromptCache
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import (
    build_news_hard_50_task, build_symptom_hard_task,
    build_agnews_task, build_emotion_task,
)
from meta_harness_plus.tasks.lawbench import build_lawbench_task

TASK_FACTORIES = {
    "news_hard_50": build_news_hard_50_task,
    "symptom_hard": build_symptom_hard_task,
    "agnews": build_agnews_task,
    "emotion": build_emotion_task,
    "lawbench_2_2": lambda: build_lawbench_task("2-2"),
}


def _build_client(api: str, model: str) -> HTTPClient:
    if api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required")
        return HTTPClient(api_url="https://api.openai.com/v1/chat/completions",
                          api_key=key, model=model, timeout_s=300.0)
    if api == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY required")
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{model}:generateContent")
        return HTTPClient(api_url=url, api_key=key, model=model, timeout_s=300.0)
    raise SystemExit(f"unknown api: {api}")


GRADIENT_PROMPT = (
    "You are critiquing a classification prompt.\n\n"
    "Current prompt:\n"
    "<<<{prompt}>>>\n\n"
    "It was applied to these training examples and got these results:\n"
    "{failures}\n\n"
    "Identify the SINGLE most useful change to the prompt that would\n"
    "fix the most failures. Output ONLY the new prompt text (replacing\n"
    "the original entirely). Be concise — under 200 words. No quotes,\n"
    "no labels, just the new prompt."
)


def make_harness(task, predictor, instruction: str) -> Harness:
    return Harness(components=[
        NullRetriever(), NullFewShot(),
        SimpleFormatter(system_hint=instruction),
        predictor, NullVoter(),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=["openai", "gemini"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_FACTORIES.keys()))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--n-failures", type=int, default=4,
                    help="Number of failure examples shown to the gradient LLM")
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    task = TASK_FACTORIES[args.task]()
    raw_client = _build_client(args.api, args.model)
    if args.cache_path:
        cache = PromptCache(path=args.cache_path)
        client = CachedLLMClient(raw_client, cache, model_id=args.model)
    else:
        client = raw_client

    predictor = LLMPredictor(client=client, classes=task.classes,
                             n_samples=1, temperature=0.0, max_tokens=512)
    scorer = Scorer(task)
    eval_examples = task.eval_set[:50]
    train_examples = list(task.train)[:30]  # Use subset for failure mining

    initial_prompt = "Classify the input."
    print(f"=== ProTeGi baseline on {args.api}/{args.model} × {args.task} ===")
    print(f"  rounds: {args.rounds}, beam: {args.beam_size}")
    print(f"  initial: '{initial_prompt}'")

    # Beam: list of (prompt, accuracy) tuples
    beam: list[tuple[str, float]] = []
    seed_h = make_harness(task, predictor, initial_prompt)
    t0 = time.time()
    s = scorer.score(seed_h, eval_examples, n_repeats=1, max_workers=8)
    beam.append((initial_prompt, s.accuracy))
    print(f"  seed acc: {s.accuracy:.3f}")

    rng = random.Random(0)
    for rnd in range(args.rounds):
        # Mine failures from training set using top-of-beam prompt
        top_prompt, _ = max(beam, key=lambda x: x[1])
        # Score top prompt on training examples to find failures
        top_h = make_harness(task, predictor, top_prompt)
        # Run individually to know which failed
        failures: list[tuple[str, str, str]] = []  # (input, gold, pred)
        # Sample 8 random training examples for failure mining
        sample = rng.sample(train_examples, min(8, len(train_examples)))
        # Score each individually
        for ex in sample:
            try:
                ctx = top_h.run(ex)
                pred = (ctx.prediction or "").lower().strip()
                if pred != ex.label.lower().strip():
                    failures.append((ex.input, ex.label, pred or "(empty)"))
            except Exception:
                pass

        if not failures:
            print(f"  round {rnd}: no failures found, stopping")
            break

        failures = failures[:args.n_failures]
        failures_str = "\n".join(
            f"  - input: {f[0][:120]}\n    gold: {f[1]}, predicted: {f[2]}"
            for f in failures
        )

        # Generate new candidates: for each beam entry, ask the LLM for one mutation
        new_candidates = []
        for prompt, acc in beam:
            try:
                grad_prompt = GRADIENT_PROMPT.format(prompt=prompt, failures=failures_str)
                resp = client.complete(
                    system="You are a prompt-engineering optimizer.",
                    user=grad_prompt,
                    temperature=0.7,
                    max_tokens=400,
                )
                new_prompt = resp.text.strip().strip('"').strip("'")
                # Take first paragraph if multi-paragraph
                if new_prompt and new_prompt not in [p[0] for p in beam]:
                    new_candidates.append(new_prompt)
            except Exception as e:
                print(f"  gradient gen failed: {e}")

        # Score all new candidates
        for new_p in new_candidates:
            h = make_harness(task, predictor, new_p)
            try:
                s = scorer.score(h, eval_examples, n_repeats=1, max_workers=8)
                beam.append((new_p, s.accuracy))
                print(f"  rnd{rnd}: new acc={s.accuracy:.3f} '{new_p[:80]}...'")
            except Exception as e:
                print(f"  scoring failed: {e}")

        # Trim beam
        beam = sorted(beam, key=lambda x: -x[1])[:args.beam_size]

    elapsed = time.time() - t0
    best_p, best_acc = max(beam, key=lambda x: x[1])
    print(f"\n  Best ProTeGi prompt (acc={best_acc:.3f}):")
    print(f"    '{best_p[:300]}'")

    out = {
        "task": args.task, "model": args.model, "api": args.api,
        "rounds": args.rounds, "beam_size": args.beam_size,
        "seed_acc": beam[0][1] if False else (
            beam[-1][1] if any(p == initial_prompt for p, _ in beam) else None
        ),
        "best_acc": best_acc,
        "best_prompt": best_p,
        "final_beam": beam,
        "elapsed_s": round(elapsed, 1),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
