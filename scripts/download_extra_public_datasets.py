"""Download USPTO-50k, MASSIVE, and additional LawBench subtasks.

These are the public benchmarks the audit flagged as missing. Each is
a separate ``--<flag>`` so users can pick what they need without
requiring all dataset hub round trips.

Usage:
    pip install datasets
    python3 scripts/download_extra_public_datasets.py --uspto50k
    python3 scripts/download_extra_public_datasets.py --massive --locale en-US
    python3 scripts/download_extra_public_datasets.py --lawbench-extra
    python3 scripts/download_extra_public_datasets.py --all   # everything

Each downloader writes balanced JSONL files into
``meta_harness_plus/tasks/data/<dataset>/`` so the corresponding
``build_*_task`` function picks them up.

If the network or the dataset hub is unavailable, the script prints
a clear error and exits non-zero. None of the loaders fall back to
fabricated data — for that, use the ``build_*_fixture_task`` helpers
in each loader module.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def balance_top_k(items, label_key: str, top_k: int, n_per_class: int, seed: int):
    """Return up to n_per_class items per class, restricted to the top-k
    most-common classes in ``items``."""
    counts = Counter(it[label_key] for it in items)
    top_classes = [c for c, _ in counts.most_common(top_k)]
    by_class: dict[str, list] = {c: [] for c in top_classes}
    for it in items:
        if it[label_key] in by_class:
            by_class[it[label_key]].append(it)
    rng = random.Random(seed)
    out = []
    for c in top_classes:
        rng.shuffle(by_class[c])
        out.extend(by_class[c][:n_per_class])
    rng.shuffle(out)
    return out, top_classes


def download_uspto50k(n_train: int, n_eval: int, seed: int) -> bool:
    """Download USPTO-50k reaction-class prediction (Schneider taxonomy).

    Tries `yairschiff/uspto-50k`, falling back to other mirrors. Each
    record gets ``input`` = "<reactants> >> <product>" and ``label`` =
    one of the 10 Schneider reaction superclasses.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets is required.", file=sys.stderr)
        return False

    candidates = [
        ("yairschiff/uspto-50k", None),
        ("ScalableFM/USPTO-50K", None),
    ]
    ds = None
    for name, cfg in candidates:
        try:
            ds = load_dataset(name) if cfg is None else load_dataset(name, cfg)
            print(f"  loaded {name}")
            break
        except Exception as e:
            print(f"  {name} failed: {e}")
            continue
    if ds is None:
        print("ERROR: no USPTO-50k mirror reachable.", file=sys.stderr)
        return False

    out_dir = Path("meta_harness_plus/tasks/data/uspto")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Field names vary; try the common ones.
    def to_record(ex) -> dict | None:
        # Possible keys: "reactants", "products", "rxn_class", "class"
        rxn = ex.get("rxn_smiles") or ex.get("reactions") or ex.get("rxn")
        cls = ex.get("class") or ex.get("rxn_class") or ex.get("label")
        if not rxn or cls is None:
            r = ex.get("reactants_mol") or ex.get("reactants")
            p = ex.get("products_mol") or ex.get("products")
            if r and p:
                rxn = f"{r} >> {p}"
        if rxn is None or cls is None:
            return None
        return {"input": str(rxn), "label": str(cls)}

    train_records = []
    test_records = []
    for split_name in ds.keys():
        recs = []
        for ex in ds[split_name]:
            r = to_record(ex)
            if r:
                recs.append(r)
        if "train" in split_name:
            train_records = recs
        elif "test" in split_name or "valid" in split_name:
            test_records = recs

    if not train_records or not test_records:
        # Fall back: split a single split.
        first = next(iter(ds.values()))
        all_recs = [to_record(e) for e in first if to_record(e)]
        n_test = max(50, len(all_recs) // 10)
        train_records = all_recs[n_test:]
        test_records = all_recs[:n_test]

    train_bal, classes = balance_top_k(train_records, "label", top_k=10,
                                       n_per_class=max(1, n_train // 10), seed=seed)
    test_bal, _ = balance_top_k(test_records, "label", top_k=10,
                                n_per_class=max(1, n_eval // 10), seed=seed + 1)

    write_jsonl(out_dir / "uspto50k_train.jsonl", train_bal)
    write_jsonl(out_dir / "uspto50k_test.jsonl", test_bal)
    print(f"  wrote USPTO-50k: {len(train_bal)} train, {len(test_bal)} eval, "
          f"classes={classes}")
    return True


def download_massive(n_train: int, n_eval: int, seed: int, locale: str) -> bool:
    """Download MASSIVE intent classification (top-8 classes per locale)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets required.", file=sys.stderr)
        return False
    try:
        ds = load_dataset("AmazonScience/massive", locale)
    except Exception as e:
        print(f"ERROR: load MASSIVE/{locale} failed: {e}", file=sys.stderr)
        return False

    out_dir = Path("meta_harness_plus/tasks/data/massive")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs = [{"input": ex["utt"], "label": ex["intent_str"]} for ex in ds["train"]]
    test_recs = [{"input": ex["utt"], "label": ex["intent_str"]} for ex in ds["test"]]

    train_bal, classes = balance_top_k(train_recs, "label", top_k=8,
                                       n_per_class=max(1, n_train // 8), seed=seed)
    test_bal, _ = balance_top_k(test_recs, "label", top_k=8,
                                n_per_class=max(1, n_eval // 8), seed=seed + 1)
    # Filter test to the train top-8 (otherwise OOV labels)
    train_classes = set(classes)
    test_bal = [r for r in test_bal if r["label"] in train_classes]

    write_jsonl(out_dir / f"massive_{locale}_train.jsonl", train_bal)
    write_jsonl(out_dir / f"massive_{locale}_test.jsonl", test_bal)
    print(f"  wrote MASSIVE/{locale}: {len(train_bal)} train, "
          f"{len(test_bal)} eval, classes={classes}")
    return True


def download_lawbench_extra(subtasks: list[str], n_train: int, n_eval: int,
                            seed: int) -> bool:
    """Download additional LawBench subtasks beyond 2-2."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets required.", file=sys.stderr)
        return False

    out_dir = Path("meta_harness_plus/tasks/data/lawbench")
    out_dir.mkdir(parents=True, exist_ok=True)
    any_ok = False
    for sub in subtasks:
        try:
            ds = load_dataset("open-compass/LawBench", sub, split="test")
        except Exception as e:
            print(f"  subtask {sub}: {e}")
            continue
        recs = [{"input": ex["instruction"], "label": str(ex["answer"]).strip()}
                for ex in ds]
        # Keep classes balanced; many LawBench subtasks have many classes,
        # so cap at top-8 the same way.
        recs_bal, classes = balance_top_k(recs, "label", top_k=8,
                                          n_per_class=max(1, n_eval // 8), seed=seed + 1)
        # Use first slice as train (no separate train split in many subtasks).
        n_train_actual = min(n_train, len(recs))
        train_recs = recs[:n_train_actual]
        train_recs = [r for r in train_recs if r["label"] in set(classes)]

        write_jsonl(out_dir / f"lawbench_{sub}_train.jsonl", train_recs)
        write_jsonl(out_dir / f"lawbench_{sub}_test.jsonl", recs_bal)
        print(f"  wrote LawBench/{sub}: {len(train_recs)} train, "
              f"{len(recs_bal)} eval, classes={classes}")
        any_ok = True
    return any_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uspto50k", action="store_true")
    ap.add_argument("--massive", action="store_true")
    ap.add_argument("--lawbench-extra", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--locale", default="en-US",
                    help="MASSIVE locale, e.g. en-US, fr-FR, ja-JP")
    ap.add_argument("--lawbench-subtasks", default="1-1,2-1,2-4,2-5,3-1,3-2",
                    help="Comma-separated extra LawBench subtask IDs.")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.all:
        args.uspto50k = args.massive = args.lawbench_extra = True

    if not (args.uspto50k or args.massive or args.lawbench_extra):
        ap.error("pick at least one of --uspto50k --massive --lawbench-extra --all")

    ok = True
    if args.uspto50k:
        print("Downloading USPTO-50k...")
        ok &= download_uspto50k(args.n_train, args.n_eval, args.seed)
    if args.massive:
        print(f"Downloading MASSIVE ({args.locale})...")
        ok &= download_massive(args.n_train, args.n_eval, args.seed, args.locale)
    if args.lawbench_extra:
        print(f"Downloading LawBench extra subtasks: {args.lawbench_subtasks}")
        subs = [s.strip() for s in args.lawbench_subtasks.split(",") if s.strip()]
        ok &= download_lawbench_extra(subs, args.n_train, args.n_eval, args.seed)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
