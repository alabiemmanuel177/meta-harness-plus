"""Simplified OPRO baseline (Yang et al. 2023, "Optimization by PROmpting").

OPRO uses an LLM as the optimizer over textual instructions. We
implement the canonical instruction-tuning version: starting from a
seed instruction, ask the LLM to propose `N` better instructions
given the eval-set accuracy of prior instructions, then keep the best.

Run for `T` iterations × `N` proposals per iteration. Score the best
instruction found. Compare to MH++.

Note: pure OPRO doesn't search over harness *shape* (retriever, voter,
formatter), only over the instruction *string*. So it's a weaker
baseline than MH++ by construction — that's the point.

Usage:
    python3 examples/opro_baseline.py \\
        --api openai --model gpt-4.1-nano \\
        --task news_hard_50 \\
        --iterations 4 --proposals 4 \\
        --output runs/opro_baseline_openai_news_hard_50.json
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


def _build_client(args, model: str) -> HTTPClient:
    if args.api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required")
        return HTTPClient(api_url="https://api.openai.com/v1/chat/completions",
                          api_key=key, model=model, timeout_s=300.0)
    if args.api == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY required")
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{model}:generateContent")
        return HTTPClient(api_url=url, api_key=key, model=model, timeout_s=300.0)
    raise SystemExit(f"unknown api: {args.api}")


def make_harness(task, predictor, instruction: str) -> Harness:
    """Build a minimal harness with the given instruction as system_hint."""
    return Harness(components=[
        NullRetriever(), NullFewShot(),
        SimpleFormatter(system_hint=instruction),
        predictor, NullVoter(),
    ])


SEED_INSTRUCTION = "Classify the input."

OPRO_PROMPT = """You are optimizing an instruction for a text classifier.
The classifier outputs one of these classes: {classes}.

Below are instructions tried so far, with their accuracy on a small eval set.
Generate ONE new instruction that you predict will score higher.

Past instructions and their accuracies:
{history}

Output ONLY the new instruction text, nothing else. Do not number, quote, or label it."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=["openai", "gemini"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_FACTORIES.keys()))
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    task = TASK_FACTORIES[args.task]()
    raw_client = _build_client(args, args.model)
    if args.cache_path:
        cache = PromptCache(path=args.cache_path)
        client = CachedLLMClient(raw_client, cache, model_id=args.model)
    else:
        client = raw_client

    predictor = LLMPredictor(client=client, classes=task.classes,
                             n_samples=1, temperature=0.0, max_tokens=512)
    scorer = Scorer(task)
    eval_examples = task.eval_set[:50]

    print(f"=== OPRO baseline on {args.api}/{args.model} × {args.task} ===")
    print(f"  budget: {args.iterations} iterations × {args.proposals} proposals")

    # History: list of (instruction, accuracy) pairs.
    history: list[tuple[str, float]] = []

    # Score the seed.
    seed_h = make_harness(task, predictor, SEED_INSTRUCTION)
    t0 = time.time()
    score = scorer.score(seed_h, eval_examples, n_repeats=1, max_workers=8)
    history.append((SEED_INSTRUCTION, score.accuracy))
    print(f"  seed instruction: '{SEED_INSTRUCTION}' → acc={score.accuracy:.3f}")

    rng = random.Random(0)
    for it in range(args.iterations):
        # Build OPRO prompt with sorted-by-accuracy history.
        sorted_hist = sorted(history, key=lambda x: x[1])
        hist_str = "\n".join(f"  acc={a:.3f}: {instr[:120]}" for instr, a in sorted_hist[-8:])
        prompt = OPRO_PROMPT.format(classes=", ".join(task.classes), history=hist_str)

        new_instructions: list[str] = []
        for j in range(args.proposals):
            try:
                resp = client.complete(
                    system="You are a prompt-instruction optimizer.",
                    user=prompt,
                    temperature=0.7,
                    max_tokens=400,
                )
                new_instr = resp.text.strip().strip('"').strip("'").splitlines()[0].strip()
                if new_instr and new_instr not in [h[0] for h in history]:
                    new_instructions.append(new_instr)
            except Exception as e:
                print(f"  iter {it} prop {j}: gen failed: {e}")

        # Score each new instruction.
        for instr in new_instructions:
            h = make_harness(task, predictor, instr)
            try:
                s = scorer.score(h, eval_examples, n_repeats=1, max_workers=8)
                history.append((instr, s.accuracy))
                print(f"  it{it}: acc={s.accuracy:.3f}  '{instr[:100]}'")
            except Exception as e:
                print(f"  scoring failed: {e}")

    elapsed = time.time() - t0

    # Best-found
    best_instr, best_acc = max(history, key=lambda x: x[1])
    print(f"\n  Best instruction (acc={best_acc:.3f}):")
    print(f"    '{best_instr}'")
    print(f"  Total OPRO compute: {len(history)} candidates, {elapsed:.0f}s")

    out = {
        "task": args.task, "model": args.model, "api": args.api,
        "iterations": args.iterations, "proposals": args.proposals,
        "n_candidates": len(history),
        "seed_acc": history[0][1],
        "best_acc": best_acc,
        "best_instruction": best_instr,
        "all_history": history,
        "elapsed_s": round(elapsed, 1),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
