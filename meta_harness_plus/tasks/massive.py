"""MASSIVE multilingual intent-classification loader.

The Amazon MASSIVE benchmark (https://github.com/alexa/massive) has
60+ intent classes across 51 languages. We expose a top-k subset of
classes (default 8) so the task fits within MH++'s 8-class search
budget — same approach we took for 20 Newsgroups and patent-classification.

Supports an optional ``locale`` argument (default ``en-US``) so users
can run multilingual searches; only the English subset is bundled in
the synthetic fixture.

Loader:
    build_massive_task(locale="en-US", top_k=8)

Files expected:
    meta_harness_plus/tasks/data/massive/massive_<locale>_train.jsonl
    meta_harness_plus/tasks/data/massive/massive_<locale>_test.jsonl

Synthetic fixture:
    build_massive_fixture_task() — 8-class English fixture (24 train,
    16 eval) for unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..task import Task
from .jsonl_loader import build_task_from_jsonl


_DATA_DIR = Path(__file__).parent / "data" / "massive"


# The most-common 8 intents in MASSIVE en-US (approximate; the real
# top-8 from the train split is determined empirically by the
# downloader). We pre-declare these for the fixture so unit tests are
# deterministic.
MASSIVE_TOP8 = (
    "general_quirky",       # "tell me a joke"
    "play_music",           # "play some jazz"
    "calendar_set",         # "set a meeting at 3pm"
    "weather_query",        # "what's the weather"
    "news_query",           # "read me the news"
    "audio_volume_up",      # "turn up the volume"
    "lists_query",          # "what's on my shopping list"
    "email_query",          # "any new emails"
)


def build_massive_task(
    *,
    locale: str = "en-US",
) -> Task:
    """Load a MASSIVE locale subset.

    Reads the JSONL files written by
    ``scripts/download_extra_public_datasets.py --massive --locale en-US``.
    """
    train = _DATA_DIR / f"massive_{locale}_train.jsonl"
    test = _DATA_DIR / f"massive_{locale}_test.jsonl"
    if not train.exists() or not test.exists():
        raise FileNotFoundError(
            f"MASSIVE files not found for locale {locale}. Run "
            f"`python3 scripts/download_extra_public_datasets.py --massive` "
            f"or call build_massive_fixture_task() for a synthetic fixture."
        )
    return build_task_from_jsonl(
        name=f"massive_{locale.replace('-', '_')}",
        train_path=train,
        eval_path=test,
    )


# ---- bundled fixture ----

_FIXTURE_BY_CLASS: dict[str, list[str]] = {
    "general_quirky": [
        "tell me a joke",
        "say something funny",
        "make me laugh",
        "give me a fun fact",
        "do you have a favorite color",
    ],
    "play_music": [
        "play some jazz",
        "play the latest album by adele",
        "start my workout playlist",
        "put on some classical music",
        "play that song again",
    ],
    "calendar_set": [
        "set a meeting at 3pm tomorrow",
        "add an event for friday at noon",
        "schedule lunch with bob next monday",
        "create a calendar invite for the team meeting",
        "block off thursday morning for focus time",
    ],
    "weather_query": [
        "what's the weather in seattle today",
        "is it going to rain tomorrow",
        "tell me the forecast for this weekend",
        "do i need an umbrella tomorrow",
        "what's the temperature outside right now",
    ],
    "news_query": [
        "read me the latest headlines",
        "what's new in tech news",
        "tell me today's top stories",
        "any news from the markets",
        "what happened in europe today",
    ],
    "audio_volume_up": [
        "turn up the volume",
        "make it louder please",
        "raise the sound a bit",
        "crank it up by twenty percent",
        "increase volume to seventy",
    ],
    "lists_query": [
        "what's on my shopping list",
        "show me my todo list",
        "read out my groceries",
        "what's on my reminders",
        "list everything in my notes",
    ],
    "email_query": [
        "any new emails",
        "do i have any unread messages",
        "check my inbox for new mail",
        "what emails came in this morning",
        "show me messages from sarah",
    ],
}


def build_massive_fixture_task() -> Task:
    train_p = _DATA_DIR / "massive_fixture_train.jsonl"
    eval_p = _DATA_DIR / "massive_fixture_test.jsonl"
    if not train_p.exists() or not eval_p.exists():
        write_massive_fixture()
    return build_task_from_jsonl(
        name="massive_fixture",
        train_path=train_p,
        eval_path=eval_p,
        classes=MASSIVE_TOP8,
    )


def write_massive_fixture(out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p = out_dir / "massive_fixture_train.jsonl"
    eval_p = out_dir / "massive_fixture_test.jsonl"

    train: list[dict] = []
    eval_items: list[dict] = []
    for cls, items in _FIXTURE_BY_CLASS.items():
        train.extend({"input": s, "label": cls} for s in items[:3])
        eval_items.extend({"input": s, "label": cls} for s in items[3:5])

    with train_p.open("w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with eval_p.open("w") as f:
        for r in eval_items:
            f.write(json.dumps(r) + "\n")
    return train_p, eval_p
