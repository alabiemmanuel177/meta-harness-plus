# Datasets

## Bundled

### `symptom_classification`

Hand-curated medical symptom → clinical specialty classification.

- **5 classes:** cardiology, dermatology, gastroenterology, neurology, orthopedics
- **30 train / 20 eval** items, perfectly balanced at 6/4 per class
- Each item is a single realistic patient-complaint sentence (10–15 words)
- Class-specific terminology dominates, but some inter-class vocabulary overlap (e.g. "pain", "worse", "after") so keyword lookup alone doesn't trivially solve it
- All items are unique across train + eval (no leakage)

Files: `meta_harness_plus/tasks/data/symptom_{train,eval}.jsonl`

Load it:
```python
from meta_harness_plus.tasks import build_symptom_task
task = build_symptom_task()
```

Ships with `symptom_mock_llm()` — a deterministic offline scorer keyed to medical vocabulary, tuned so a bare harness lands around 0.65 accuracy and a retrieval+few-shot+voting harness climbs to ~0.85–0.90. Lets tests and demos exercise the full search pipeline without hitting a real LLM backend.

### `toy_classification`

5-class synthetic keyword task (sports / cooking / finance / health / tech), generated deterministically from a seed. Intended for unit tests, not realistic evaluation — see `meta_harness_plus/tasks/toy_classification.py`.

## Plugging in your own

Write any JSONL file where each line is `{"input": "...", "label": "..."}`, then:

```python
from meta_harness_plus.tasks import build_task_from_jsonl
task = build_task_from_jsonl(
    name="my_task",
    train_path="path/to/train.jsonl",
    eval_path="path/to/eval.jsonl",
    # classes=["a", "b", "c"]  # optional: strict mode rejects surprise labels
)
```

`classes` defaults to the sorted union of labels found in both files. Pass it explicitly if you want a canonical ordering (prompts list classes in this order) or to enable strict-mode validation.

## Design notes

- **No `datasets` / HuggingFace dependency.** Zero external deps is a deliberate choice — makes the framework trivially portable and review-friendly.
- **JSONL, not CSV.** Parsing is simpler, labels that contain commas aren't a foot-gun, and per-line record validity means a bad line fails fast with a specific line number.
- **Classes passed explicitly are validated.** If a record carries a label not in `classes`, the loader raises at load time rather than silently letting it through.

## Why a hand-curated dataset instead of downloading a public one

Three pragmatic reasons:

1. **Zero setup.** Clone the repo → `python3 examples/run_symptom_classification.py` works. No `datasets.load_dataset("lawbench/...")` and no network.
2. **Small enough to eyeball.** 50 items total means you can read every example in 5 minutes. Helps when debugging the search or the mock LLM — you can trace "why did the harness get *this* example wrong?" straight from RUN_LOG.
3. **Calibrated noise.** The reward curve was hand-tuned to have meaningful headroom (0.65 → ~0.90) on a 20-item eval. Public datasets would either be too easy at 20 items (signal dominates) or too noisy (nothing is learnable with that few eval points).

A future branch could wire in a public dataset adapter (LawBench, USPTO-50k, 20 Newsgroups) — the `build_task_from_jsonl` loader is already the integration point. For this branch, the bundled dataset is the minimum viable real-data example.
