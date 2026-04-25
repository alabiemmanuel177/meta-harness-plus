"""Download LawBench subtask 2-2 (Chinese legal dispute-focus classification)
from open-compass/LawBench's GitHub raw, parse the labels, and write
balanced train/eval JSONL files compatible with our build_task_from_jsonl
loader.

Subtask 2-2: 16-class single-label classification of Chinese legal-case
sentences. Classes (Chinese):
  诉讼主体、租金情况、利息、本金争议、责任认定、责任划分、损失认定及处理、
  原审判决是否适当、合同效力、财产分割、责任承担、鉴定结论采信问题、
  诉讼时效、违约、合同解除、肇事逃逸

Usage:
    python3 scripts/download_lawbench_2_2.py [--n-eval 50] [--n-train 80]

Writes:
    meta_harness_plus/tasks/data/lawbench/lawbench_2_2_train.jsonl
    meta_harness_plus/tasks/data/lawbench/lawbench_2_2_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

URL = "https://raw.githubusercontent.com/open-compass/LawBench/main/data/zero_shot/2-2.json"


def parse_label(answer: str) -> str | None:
    """LawBench 2-2 answers are like '争议焦点类别：责任认定。'.

    Extract the class name. Returns None if the format doesn't match
    (which happens for a few malformed records — we drop those).
    """
    if not isinstance(answer, str):
        return None
    # Strategy 1: split on '：' and trim.
    parts = re.split(r"[：:]", answer.strip(), maxsplit=1)
    if len(parts) == 2:
        candidate = parts[1].strip().rstrip("。.").strip()
        if candidate:
            return candidate
    # Strategy 2: bracket form '[争议焦点]责任认定<eoa>'.
    m = re.search(r"\[争议焦点\](.+?)<eoa>", answer)
    if m:
        return m.group(1).strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=50,
                    help="number of eval items to sample (balanced per class)")
    ap.add_argument("--n-train", type=int, default=80,
                    help="number of train items to sample (balanced per class)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-classes", type=int, default=8,
                    help="cap to top-N most common classes for tractable bakeoff")
    args = ap.parse_args()

    print(f"Fetching {URL}...", flush=True)
    with urllib.request.urlopen(URL, timeout=30) as r:
        raw = json.loads(r.read().decode("utf-8"))
    print(f"  got {len(raw)} items", flush=True)

    # Parse labels and build a label-bucket dict.
    by_class: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for item in raw:
        label = parse_label(item.get("answer", ""))
        if not label:
            skipped += 1
            continue
        by_class[label].append({
            "input": item.get("question", "").strip(),
            "label": label,
        })
    print(f"  parsed: {sum(len(v) for v in by_class.values())} items "
          f"in {len(by_class)} classes; skipped {skipped} malformed", flush=True)

    # Stats.
    class_counts = sorted(((k, len(v)) for k, v in by_class.items()),
                          key=lambda kv: -kv[1])
    print("  per-class counts (top 20):")
    for klass, cnt in class_counts[:20]:
        print(f"    {klass:30s}  {cnt}")

    # Cap to the most common classes for a tractable bakeoff.
    chosen_classes = [k for k, _ in class_counts[:args.max_classes]]
    print(f"\n  using top-{args.max_classes} classes: {chosen_classes}", flush=True)

    import random
    rng = random.Random(args.seed)

    # Balanced sampling: per-class quota, shuffled.
    eval_per = max(1, args.n_eval // len(chosen_classes))
    train_per = max(1, args.n_train // len(chosen_classes))
    train_items: list[dict] = []
    eval_items: list[dict] = []
    for klass in chosen_classes:
        items = by_class[klass]
        rng.shuffle(items)
        if len(items) < eval_per + train_per:
            print(f"  WARN class {klass} has {len(items)} items, "
                  f"need {eval_per + train_per}; using what's available")
        eval_items.extend(items[:eval_per])
        train_items.extend(items[eval_per : eval_per + train_per])

    rng.shuffle(eval_items)
    rng.shuffle(train_items)

    # Write JSONL.
    out_dir = Path(__file__).parent.parent / "meta_harness_plus" / "tasks" / "data" / "lawbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "lawbench_2_2_train.jsonl"
    test_path = out_dir / "lawbench_2_2_test.jsonl"
    classes_path = out_dir / "lawbench_2_2_classes.json"

    with train_path.open("w") as f:
        for r in train_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with test_path.open("w") as f:
        for r in eval_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    classes_path.write_text(json.dumps(chosen_classes, ensure_ascii=False, indent=2))

    print(f"\nWrote:")
    print(f"  {train_path}  ({len(train_items)} train items)")
    print(f"  {test_path}  ({len(eval_items)} eval items)")
    print(f"  {classes_path}")
    print(f"\nFinal class balance:")
    for split_name, split in [("train", train_items), ("eval", eval_items)]:
        counts = Counter(it["label"] for it in split)
        for klass in chosen_classes:
            print(f"  {split_name:>5}  {klass:30s}  {counts.get(klass, 0)}")


if __name__ == "__main__":
    main()
