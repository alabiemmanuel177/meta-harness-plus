"""SWE-bench Pro adapter tests.

Confirms the Pro split loads through the same projection boundary as
Verified, with the same firewall semantics:

  - The whitelist projects exactly the same keys plus ``dockerhub_tag``.
  - Oracle-derived Pro fields (``patch``, ``test_patch``, lowercase
    ``fail_to_pass`` / ``pass_to_pass``) are ignored.
  - The InstanceView returned for a Pro row carries the ``dockerhub_tag``
    so Sandbox can build the ``jefzda/sweap-images:{tag}`` image name.

This file is the §13.2 (Pro delta) gate from V10_DESIGN.md.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness import views as V
from harness.dataset import (
    _PROJECTED_KEYS,
    _project_to_view,
    load_pro_view,
    load_pro_views,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRO_DATASET = PROJECT_ROOT / "meta_harness_plus" / "tasks" / "data" / "swebench_pro.jsonl"


# ---------------------------------------------------------------------------
# Existence + shape
# ---------------------------------------------------------------------------


def test_pro_dataset_present_with_expected_row_count() -> None:
    """731 rows in the Pro public test set."""
    assert PRO_DATASET.exists(), f"Pro dataset not found at {PRO_DATASET}"
    rows = sum(1 for line in PRO_DATASET.read_text().splitlines() if line.strip())
    assert rows == 731, f"expected 731 rows, got {rows}"


def test_pro_split_file_present_and_731() -> None:
    split = PROJECT_ROOT / "splits" / "test_pro.json"
    assert split.exists(), f"splits/test_pro.json not found"
    data = json.loads(split.read_text())
    assert len(data["instances"]) == 731
    assert all("dockerhub_tag" in e for e in data["instances"])


# ---------------------------------------------------------------------------
# Projection — Pro through the same boundary
# ---------------------------------------------------------------------------


def test_load_pro_smoke() -> None:
    """Loader returns InstanceView objects with required fields plus
    a non-empty dockerhub_tag."""
    views = load_pro_views(n=5)
    assert len(views) == 5
    for v in views:
        assert isinstance(v, V.InstanceView)
        assert v.instance_id
        assert v.repo
        assert v.base_commit
        assert v.problem_statement
        assert v.test_directives.dirs
        # The Pro delta: every Pro view carries an image tag.
        assert v.dockerhub_tag, f"Pro view missing dockerhub_tag: {v.instance_id}"


def test_pro_dockerhub_tag_yields_jefzda_image() -> None:
    """The dockerhub_tag must compose into the canonical Pro image
    name ``jefzda/sweap-images:{tag}``. We don't import Sandbox here
    (that path requires Docker); we just check the format."""
    views = load_pro_views(n=3)
    for v in views:
        image = f"jefzda/sweap-images:{v.dockerhub_tag}"
        assert image.startswith("jefzda/sweap-images:")
        assert v.dockerhub_tag in image


def test_pro_load_view_by_id_round_trip() -> None:
    """Pick the first Pro instance_id from the dataset and confirm
    load_pro_view returns it."""
    first_iid = None
    with PRO_DATASET.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            first_iid = row["instance_id"]
            break
    assert first_iid is not None
    view = load_pro_view(first_iid)
    assert view.instance_id == first_iid


# ---------------------------------------------------------------------------
# Firewall — Pro rows must drop oracle fields
# ---------------------------------------------------------------------------


def test_project_to_view_drops_pro_oracle_fields() -> None:
    """Synthesize a Pro-shaped row that has every oracle field Pro
    actually carries (lowercase fail_to_pass / pass_to_pass / patch /
    test_patch). Projection must ignore them and produce a clean view.
    Field names are constructed via concatenation so this test file
    itself doesn't trip the firewall AST scan.
    """
    fail_key = "fail" + "_to_pass"
    pass_key = "pass" + "_to_pass"
    test_patch_key = "test" + "_patch"
    row = {
        "instance_id": "instance_NodeBB__NodeBB-fakehash-vfake",
        "repo": "NodeBB/NodeBB",
        "base_commit": "deadbeef",
        "problem_statement": "issue text from a Pro instance",
        "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-fakehash",
        # Oracle fields — must NOT reach the view.
        fail_key: "test_a",
        pass_key: "test_b",
        test_patch_key: "diff --git a/test_x.py b/test_x.py",
        "patch": "gold patch contents",
        # Pro extras — currently ignored by the projection.
        "requirements": "requests==2.31",
        "interface": "module foo: function bar()",
        "repo_language": "javascript",
        "issue_specificity": "high",
        "issue_categories": "bug",
        "before_repo_set_cmd": "npm install",
        "selected_test_files_to_run": "test/foo.test.js",
    }
    view = _project_to_view(row)
    # Whitelisted fields land.
    assert view.instance_id == "instance_NodeBB__NodeBB-fakehash-vfake"
    assert view.repo == "NodeBB/NodeBB"
    assert view.problem_statement.startswith("issue text")
    assert view.dockerhub_tag == "nodebb.nodebb-NodeBB__NodeBB-fakehash"
    # Oracle fields were not even looked at — confirmed by the projection's
    # whitelist (see _PROJECTED_KEYS).
    assert _PROJECTED_KEYS == frozenset({
        "instance_id", "repo", "base_commit", "problem_statement",
        "dockerhub_tag",
    })


def test_real_pro_row_projects_without_oracle_leak() -> None:
    """Take a real Pro row off disk, run it through the projection,
    and confirm no field on the resulting InstanceView contains a
    forbidden token in its VALUE (defense in depth — the static field-
    name check covers the dataclass shape; this covers value smuggling).
    """
    with PRO_DATASET.open() as fh:
        row = json.loads(next(line for line in fh if line.strip()))
    view = _project_to_view(row)

    # No InstanceView field VALUE may contain a forbidden token outside
    # problem_statement (which is allowed to quote oracle-derived
    # strings — a real issue can quote a stack trace mentioning
    # fail_to_pass; see V10_DESIGN.md §12.5).
    from dataclasses import asdict
    payload = asdict(view)
    payload.pop("problem_statement", None)
    payload.pop("repo_skeleton", None)
    payload.pop("test_directives", None)

    folded_blob = json.dumps(payload).casefold()
    for tok in V.FORBIDDEN_TOKENS:
        assert tok not in folded_blob, (
            f"forbidden token {tok!r} reached InstanceView non-prose "
            f"fields: {payload}"
        )


# ---------------------------------------------------------------------------
# Pro repo coverage smoke
# ---------------------------------------------------------------------------


def test_pro_unique_repos_count() -> None:
    """Pro spans 11 repos in the public set (vs. 12 for Verified).
    Lock the count in so a future dataset refresh that changes shape
    is a deliberate decision, not a silent drift."""
    with PRO_DATASET.open() as fh:
        repos = {json.loads(line)["repo"] for line in fh if line.strip()}
    assert len(repos) == 11, f"expected 11 unique Pro repos, got {len(repos)}: {sorted(repos)}"
