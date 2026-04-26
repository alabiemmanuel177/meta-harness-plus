"""DSPy BootstrapFewShot baseline — the "brutal baseline" comparison.

Runs DSPy's BootstrapFewShot optimizer on a classification task with
the same LLM + same train/eval split that MH++ used, then compares
the resulting program's accuracy/tokens/latency to MH++'s peak.

DSPy is the closest existing automated harness optimizer to MH++:
both search over component pipelines + few-shot examples. If MH++
wins this comparison, the case for harness search becomes much harder
to dismiss.

We use DSPy's stable BootstrapFewShot teleprompter rather than the
newer MIPRO since BootstrapFewShot is the canonical reference and
has the same compute budget knob (max_bootstrapped_demos). The
metric is exact-match accuracy, same as MH++.

Usage:
    python3 examples/dspy_baseline.py \\
        --api openai --model gpt-4.1-nano \\
        --task news_hard_50 \\
        --max-demos 4 \\
        --output runs/dspy_baseline_openai_news_hard_50.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# DSPy is heavy; import lazily inside main if possible.
import dspy

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
    ap.add_argument("--api", required=True, choices=["openai", "gemini"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_FACTORIES.keys()))
    ap.add_argument("--max-demos", type=int, default=4,
                    help="DSPy bootstrap demo budget (analogous to fewshot k)")
    ap.add_argument("--num-candidate-programs", type=int, default=4,
                    help="Number of candidate programs DSPy searches over")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    task = TASK_FACTORIES[args.task]()

    # Configure DSPy LM. Prefix model with provider/ for litellm.
    if args.api == "openai":
        os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        lm = dspy.LM(model=f"openai/{args.model}", max_tokens=512, temperature=0.0)
    elif args.api == "gemini":
        os.environ.setdefault(
            "GEMINI_API_KEY",
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))
        lm = dspy.LM(model=f"gemini/{args.model}", max_tokens=512, temperature=0.0)
    else:
        raise SystemExit(f"unknown api: {args.api}")
    dspy.settings.configure(lm=lm)

    # Build a Classify signature with the task's classes.
    class_str = ", ".join(task.classes)

    class Classify(dspy.Signature):
        """Classify the input into one of the given classes. Output ONLY the class name, lowercase."""
        text: str = dspy.InputField(desc="The text to classify")
        label: str = dspy.OutputField(desc=f"One of: {class_str}")

    classify = dspy.Predict(Classify)

    # Convert task to DSPy examples.
    train_examples = [
        dspy.Example(text=e.input, label=e.label.lower()).with_inputs("text")
        for e in task.train
    ]
    eval_examples = [
        dspy.Example(text=e.input, label=e.label.lower()).with_inputs("text")
        for e in task.eval_set[:50]
    ]

    # Define a metric: exact-match on lowercased class string.
    def exact_match(gold, pred, trace=None):
        return gold.label.lower().strip() == str(getattr(pred, "label", "")).lower().strip()

    # Pre-compile baseline (no DSPy optimization, just the bare predictor)
    print("=== DSPy bare baseline (no optimization) ===")
    bare_correct = 0
    for ex in eval_examples:
        try:
            r = classify(text=ex.text)
            if exact_match(ex, r):
                bare_correct += 1
        except Exception as e:
            print(f"  failed on '{ex.text[:50]}...': {e}")
    bare_acc = bare_correct / len(eval_examples)
    print(f"  bare DSPy acc: {bare_acc:.3f} ({bare_correct}/{len(eval_examples)})")

    # Compile with BootstrapFewShot
    print(f"\n=== DSPy BootstrapFewShot (max_demos={args.max_demos}) ===")
    from dspy.teleprompt import BootstrapFewShot
    teleprompter = BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
    )
    t0 = time.time()
    try:
        optimized = teleprompter.compile(classify, trainset=train_examples)
    except Exception as e:
        print(f"  DSPy compile failed: {e}")
        out = {
            "task": args.task, "model": args.model, "api": args.api,
            "error": str(e), "bare_dspy_acc": bare_acc,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        return
    compile_time = time.time() - t0
    print(f"  compile time: {compile_time:.0f}s")

    # Score the optimized program
    optimized_correct = 0
    for ex in eval_examples:
        try:
            r = optimized(text=ex.text)
            if exact_match(ex, r):
                optimized_correct += 1
        except Exception as e:
            print(f"  failed: {e}")
    opt_acc = optimized_correct / len(eval_examples)
    print(f"  optimized DSPy acc: {opt_acc:.3f} ({optimized_correct}/{len(eval_examples)})")

    out = {
        "task": args.task,
        "model": args.model,
        "api": args.api,
        "n_eval": len(eval_examples),
        "bare_dspy_acc": bare_acc,
        "optimized_dspy_acc": opt_acc,
        "compile_time_s": round(compile_time, 1),
        "max_demos": args.max_demos,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
