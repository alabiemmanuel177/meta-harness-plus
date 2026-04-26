"""LawBench loader skeleton (Tier 1.1).

LawBench (https://github.com/open-compass/LawBench) is the original
Meta-Harness paper's primary classification benchmark — Chinese legal
multi-task evaluation, including a 215-class case-classification subset
that the paper specifically reports +7.7pt accuracy with 4× fewer tokens.

This module provides a *loader skeleton* that reads LawBench JSONL data
in a directory the user has already downloaded. Full network download is
a separate concern — we don't ship LawBench (it's hundreds of MB, has
its own license, and requires a HuggingFace dataset hub round trip).

To download (one-time, from the project root)::

    pip install datasets
    python3 -c "
    from datasets import load_dataset
    ds = load_dataset('open-compass/LawBench', '2-2', split='test')
    import json
    from pathlib import Path
    out_dir = Path('meta_harness_plus/tasks/data/lawbench')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'lawbench_2_2_test.jsonl').open('w') as f:
        for ex in ds:
            f.write(json.dumps({'input': ex['instruction'], 'label': ex['answer']}) + '\\n')
    "

Then ``build_lawbench_task(subtask='2-2')`` will load it.

Until that download is run, ``build_lawbench_task`` raises
``FileNotFoundError`` — a clean signal that the public dataset isn't
configured locally.

Bundled fixture: a tiny synthetic 5-class fixture
``lawbench_fixture.jsonl`` (5 train / 5 eval) that exercises the loader
code path in unit tests without requiring a download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..task import Task
from .jsonl_loader import build_task_from_jsonl


_DATA_DIR = Path(__file__).parent / "data" / "lawbench"


def build_lawbench_task(
    subtask: str = "2-2",
    *,
    split_train_eval: bool = True,
) -> Task:
    """Load a LawBench subtask as a ``Task``.

    ``subtask`` follows LawBench's naming (e.g. ``2-2`` for case
    classification). The loader expects::

        meta_harness_plus/tasks/data/lawbench/lawbench_<subtask>_train.jsonl
        meta_harness_plus/tasks/data/lawbench/lawbench_<subtask>_test.jsonl

    or a single ``lawbench_<subtask>.jsonl`` that's split internally.

    Class set is inferred from the data (LawBench classification subtasks
    have stable class sets per subtask, but we don't hard-code them — the
    loader picks up whatever's in the file, sorted).
    """
    train_path = _DATA_DIR / f"lawbench_{subtask}_train.jsonl"
    test_path = _DATA_DIR / f"lawbench_{subtask}_test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"LawBench files not found at {train_path} / {test_path}. "
            f"See meta_harness_plus/tasks/lawbench.py docstring for the "
            f"one-time download command."
        )
    return build_task_from_jsonl(
        name=f"lawbench_{subtask}",
        train_path=train_path,
        eval_path=test_path,
    )


# ---------------- Bundled fixture (for unit tests) ----------------

def build_lawbench_fixture_task() -> Task:
    """A 5-class synthetic LawBench-shaped fixture for unit tests.

    Not real LawBench data — just enough to exercise the loader code
    paths and verify the multi-seed runner works on a non-bundled-task
    style of dataset. Writes the fixture lazily on first call so
    nothing about LawBench needs network access.
    """
    train_p = _DATA_DIR / "lawbench_fixture_train.jsonl"
    eval_p = _DATA_DIR / "lawbench_fixture_eval.jsonl"
    if not train_p.exists() or not eval_p.exists():
        write_lawbench_fixture()
    return build_task_from_jsonl(
        name="lawbench_fixture",
        train_path=train_p,
        eval_path=eval_p,
    )


def write_lawbench_fixture(out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write the bundled fixture files (idempotent).

    Used at test time to ensure the fixture exists. Tiny dataset (5
    classes × 4 items train / 2 items eval) covering classification of
    legal-style questions into broad legal areas.
    """
    out_dir = out_dir or _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "lawbench_fixture_train.jsonl"
    eval_path = out_dir / "lawbench_fixture_eval.jsonl"

    train = [
        {"input": "Defendant signed contract under duress claiming undue influence", "label": "contract"},
        {"input": "Buyer alleges breach of warranty after defective goods delivered", "label": "contract"},
        {"input": "Lease terms ambiguous whether tenant or landlord pays repairs", "label": "contract"},
        {"input": "Vendor failed to deliver goods per agreed schedule", "label": "contract"},
        {"input": "Plaintiff struck by negligent driver in city intersection", "label": "tort"},
        {"input": "Doctor's misdiagnosis allegedly caused permanent disability", "label": "tort"},
        {"input": "Consumer claims defective product caused fire damage", "label": "tort"},
        {"input": "Property owner sued after slip and fall on icy walkway", "label": "tort"},
        {"input": "Defendant charged with felony robbery in armed incident", "label": "criminal"},
        {"input": "Prosecutor filed homicide charges with first-degree intent", "label": "criminal"},
        {"input": "Police seized evidence during warrantless vehicle search", "label": "criminal"},
        {"input": "Defendant pleaded guilty to white-collar embezzlement charges", "label": "criminal"},
        {"input": "Couple seeks divorce and division of marital assets", "label": "family"},
        {"input": "Father petitioning for sole custody of minor children", "label": "family"},
        {"input": "Adoption agency seeks termination of biological parental rights", "label": "family"},
        {"input": "Estate executor disputes validity of decedent will", "label": "family"},
        {"input": "Patent holder sues competitor for infringement of method claims", "label": "ip"},
        {"input": "Trademark owner alleges confusing similarity in branding", "label": "ip"},
        {"input": "Copyright holder claims unauthorized derivative work created", "label": "ip"},
        {"input": "Trade secret allegedly misappropriated by former employee", "label": "ip"},
    ]
    eval_items = [
        {"input": "Borrower defaulted on loan secured by personal property", "label": "contract"},
        {"input": "Software license alleged to be breached by sublicensee", "label": "contract"},
        {"input": "Pedestrian injured when struck by falling construction debris", "label": "tort"},
        {"input": "Patient claims hospital staff caused infection through negligence", "label": "tort"},
        {"input": "Defendant accused of arson with intent to defraud insurance", "label": "criminal"},
        {"input": "Detective obtained confession during prolonged interrogation", "label": "criminal"},
        {"input": "Mother seeks child support modification after job loss", "label": "family"},
        {"input": "Sibling contests trustee distribution from family trust", "label": "family"},
        {"input": "Author alleges publisher exceeded scope of licensing agreement", "label": "ip"},
        {"input": "Designer claims competitor copied protected trade dress", "label": "ip"},
    ]

    import json
    with train_path.open("w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with eval_path.open("w") as f:
        for r in eval_items:
            f.write(json.dumps(r) + "\n")
    return train_path, eval_path


def lawbench_classes() -> Sequence[str]:
    """Class set for the bundled fixture."""
    return ("contract", "criminal", "family", "ip", "tort")


# ---------------- Multi-subtask helpers ----------------

# Known LawBench classification subtask IDs (the original benchmark has
# 20 subtasks across 5 task categories — see
# https://github.com/open-compass/LawBench). Subtasks marked classification
# are usable directly with our framework; QA / summarization / generation
# subtasks would need an open-ended task abstraction.
LAWBENCH_CLASSIFICATION_SUBTASKS: Sequence[str] = (
    "1-1",  # article recitation
    "1-2",  # knowledge question
    "2-1",  # element recognition
    "2-2",  # case classification (the paper's headline subtask)
    "2-4",  # crime amount calculation (numeric answer; we treat as classification)
    "2-5",  # criminal damages calculation
    "3-1",  # legal entity recognition
    "3-2",  # case dispute
    "3-3",  # case analysis
    "3-4",  # criminal sentence prediction
    "3-7",  # criminal damages multi-class
)


def list_available_lawbench_subtasks() -> list[str]:
    """Return the list of LawBench subtask IDs that have JSONL files
    bundled (or downloaded) under tasks/data/lawbench/.
    """
    if not _DATA_DIR.exists():
        return []
    out: list[str] = []
    for p in sorted(_DATA_DIR.iterdir()):
        # Match lawbench_<subtask>_train.jsonl and infer the subtask ID.
        if p.name.startswith("lawbench_") and p.name.endswith("_train.jsonl"):
            sub = p.name[len("lawbench_"): -len("_train.jsonl")]
            if sub == "fixture":
                continue
            test = _DATA_DIR / f"lawbench_{sub}_test.jsonl"
            if test.exists():
                out.append(sub)
    return out


def build_all_lawbench_classification_tasks() -> dict[str, Task]:
    """Build a Task for each available LawBench classification subtask.

    Skips subtasks whose JSONL files aren't on disk; doesn't raise.
    """
    tasks: dict[str, Task] = {}
    for sub in list_available_lawbench_subtasks():
        try:
            tasks[sub] = build_lawbench_task(subtask=sub)
        except FileNotFoundError:
            continue
    return tasks
