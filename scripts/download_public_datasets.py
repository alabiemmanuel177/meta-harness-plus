"""Download AG News + dair-ai/emotion public classification datasets, write
balanced train/eval JSONL files for Meta-Harness++ task loaders.

Picks the same shape as our hand-curated tasks: small balanced eval set
(50 items) for cost-controlled multi-seed search, larger train pool
(80 items) for retriever corpus.

Usage:
    python3 scripts/download_public_datasets.py [--n-train 80] [--n-eval 50]

Writes:
    meta_harness_plus/tasks/data/agnews/agnews_{train,test}.jsonl
    meta_harness_plus/tasks/data/emotion/emotion_{train,test}.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def balance_subset(items: list, label_key: str, classes: list[str],
                   n_per_class: int, seed: int) -> list:
    """Return up to n_per_class items per class, deterministic given seed."""
    by_class: dict[str, list] = {c: [] for c in classes}
    for it in items:
        cls = classes[it[label_key]] if isinstance(it[label_key], int) else it[label_key]
        if cls in by_class:
            by_class[cls].append(it)
    rng = random.Random(seed)
    out = []
    for cls in classes:
        rng.shuffle(by_class[cls])
        out.extend(by_class[cls][:n_per_class])
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, items: list, text_key: str, label_key: str,
                classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            label_val = it[label_key]
            label_str = classes[label_val] if isinstance(label_val, int) else label_val
            text = str(it[text_key]).strip().replace("\n", " ")
            f.write(json.dumps({"input": text, "label": label_str}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=80,
                    help="Train items per class (capped per-class via balance_subset).")
    ap.add_argument("--n-eval", type=int, default=50,
                    help="Total eval items (split equally across classes).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    # === AG News (4 classes: World, Sports, Business, Sci/Tech) ===
    print("Downloading AG News...")
    ag_train = load_dataset("fancyzhx/ag_news", split="train")
    ag_test = load_dataset("fancyzhx/ag_news", split="test")
    ag_classes = ag_train.features["label"].names

    n_per_class_train = max(1, args.n_train // len(ag_classes))
    n_per_class_eval = max(1, args.n_eval // len(ag_classes))

    train_items = balance_subset(list(ag_train), "label", ag_classes,
                                 n_per_class_train, seed=args.seed)
    test_items = balance_subset(list(ag_test), "label", ag_classes,
                                n_per_class_eval, seed=args.seed + 1)

    out_dir = Path("meta_harness_plus/tasks/data/agnews")
    write_jsonl(out_dir / "agnews_train.jsonl", train_items, "text", "label", ag_classes)
    write_jsonl(out_dir / "agnews_test.jsonl", test_items, "text", "label", ag_classes)
    print(f"  wrote AG News: {len(train_items)} train, {len(test_items)} eval, "
          f"classes={ag_classes}")

    # === dair-ai/emotion (6 classes) ===
    print("Downloading dair-ai/emotion...")
    em_train = load_dataset("dair-ai/emotion", split="train")
    em_test = load_dataset("dair-ai/emotion", split="test")
    em_classes = em_train.features["label"].names

    n_per_class_train = max(1, args.n_train // len(em_classes))
    n_per_class_eval = max(1, args.n_eval // len(em_classes))

    train_items = balance_subset(list(em_train), "label", em_classes,
                                 n_per_class_train, seed=args.seed)
    test_items = balance_subset(list(em_test), "label", em_classes,
                                n_per_class_eval, seed=args.seed + 1)

    out_dir = Path("meta_harness_plus/tasks/data/emotion")
    write_jsonl(out_dir / "emotion_train.jsonl", train_items, "text", "label", em_classes)
    write_jsonl(out_dir / "emotion_test.jsonl", test_items, "text", "label", em_classes)
    print(f"  wrote emotion: {len(train_items)} train, {len(test_items)} eval, "
          f"classes={em_classes}")


if __name__ == "__main__":
    main()
