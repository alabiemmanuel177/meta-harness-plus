"""TextGrad baseline (Yuksekgonul et al. 2024).

TextGrad uses an LLM to compute "textual gradients" — natural-language
critiques of why a candidate prompt failed on training examples — and
applies them to update the prompt. The output is an optimized
classification instruction.

We score the optimized instruction with our standard scorer (same
LLMPredictor as the rest of the framework) for an apples-to-apples
comparison with MH++.

Usage:
    python3 examples/textgrad_baseline.py \\
        --api openai --model gpt-4.1-nano \\
        --task news_hard_50 --epochs 3 \\
        --output runs/textgrad_baseline_openai_news_hard_50.json
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

import textgrad as tg

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=["openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_FACTORIES.keys()))
    ap.add_argument("--epochs", type=int, default=3,
                    help="Number of TextGrad gradient-descent epochs")
    ap.add_argument("--n-train-batch", type=int, default=8,
                    help="Train items per batch (textgrad processes one at a time)")
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    task = TASK_FACTORIES[args.task]()

    # TextGrad uses litellm-style strings.
    if args.api == "openai":
        engine_str = f"{args.model}"
        os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    else:
        raise SystemExit(f"unknown api: {args.api}")

    # Configure backward (gradient) engine.
    backward_engine = tg.get_engine(engine_str)
    tg.set_backward_engine(backward_engine, override=True)

    # Forward engine (the model the prompt is actually executed against).
    forward_engine = tg.get_engine(engine_str)
    tg.set_backward_engine

    # The optimizable prompt.
    initial_prompt = "Classify the input."
    prompt_var = tg.Variable(
        initial_prompt,
        requires_grad=True,
        role_description="System prompt for a text classifier",
    )

    # Build a model that uses the optimizable prompt.
    classify_model = tg.BlackboxLLM(forward_engine, system_prompt=prompt_var)

    classes_str = ", ".join(task.classes)

    # Loss function: compares model output to gold label.
    def make_loss(text: str, gold_label: str) -> tg.TextLoss:
        # Run the model on the input.
        inp = tg.Variable(
            f"Input to classify: {text}\n\nClasses: {classes_str}\n\nOutput ONLY the class name, lowercase.",
            requires_grad=False,
            role_description="Input text and classification request",
        )
        prediction = classify_model(inp)
        # Loss: how the prediction compares to gold.
        loss_fn = tg.TextLoss(
            f"Evaluate this classification. The correct answer is '{gold_label}'. "
            f"Critique whether the prediction is right or wrong, and what aspect of the "
            f"system prompt could be improved to get this right next time.",
            engine=backward_engine,
        )
        return loss_fn(prediction)

    # Optimizer.
    optimizer = tg.TGD([prompt_var])

    train_examples = list(task.train)
    rng = random.Random(0)
    rng.shuffle(train_examples)
    train_examples = train_examples[:args.n_train_batch * args.epochs]

    print(f"=== TextGrad baseline on {args.api}/{args.model} × {args.task} ===")
    print(f"  initial prompt: '{initial_prompt}'")
    print(f"  epochs: {args.epochs}, batch: {args.n_train_batch}")

    t0 = time.time()
    history = []
    for epoch in range(args.epochs):
        batch = train_examples[epoch * args.n_train_batch : (epoch + 1) * args.n_train_batch]
        for ex in batch:
            try:
                optimizer.zero_grad()
                loss = make_loss(ex.input, ex.label)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"  step failed on '{ex.input[:50]}...': {e}")
        # Snapshot the prompt after each epoch.
        snap = str(prompt_var.value)[:200]
        history.append((epoch, snap))
        print(f"  epoch {epoch}: prompt='{snap[:120]}...'")

    optimized_prompt = str(prompt_var.value)
    elapsed = time.time() - t0
    print(f"  TextGrad optimization time: {elapsed:.0f}s")
    print(f"  Final prompt:\n    '{optimized_prompt[:400]}'")

    # Now score the optimized prompt with our standard scorer.
    raw_client = HTTPClient(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        timeout_s=300.0,
    )
    if args.cache_path:
        cache = PromptCache(path=args.cache_path)
        client = CachedLLMClient(raw_client, cache, model_id=args.model)
    else:
        client = raw_client

    predictor = LLMPredictor(client=client, classes=task.classes,
                             n_samples=1, temperature=0.0, max_tokens=512)
    h = Harness(components=[
        NullRetriever(), NullFewShot(),
        SimpleFormatter(system_hint=optimized_prompt),
        predictor, NullVoter(),
    ])
    scorer = Scorer(task)
    score = scorer.score(h, task.eval_set[:50], n_repeats=2, max_workers=8)

    print(f"\n  Final score (median over 2 repeats):")
    print(f"    accuracy: {score.accuracy:.3f}")
    print(f"    tokens: {score.tokens:.1f}")

    out = {
        "task": args.task, "model": args.model, "api": args.api,
        "epochs": args.epochs, "n_train_batch": args.n_train_batch,
        "initial_prompt": initial_prompt,
        "final_prompt": optimized_prompt,
        "score": {
            "accuracy": score.accuracy,
            "tokens": score.tokens,
            "latency_ms": score.latency_ms,
        },
        "elapsed_s": round(elapsed, 1),
        "history": history,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
