"""Stage 1a — repo skeleton builder with ``(repo, base_commit)`` cache.

Per V10_DESIGN.md §3.2 the skeleton is the foundation of Phase 1
localization: a compact tree of files / classes / functions with one-line
summaries, fed into BM25 / embedding / rerank stages downstream. Verified
has high per-repo concentration (~20 instances per repo on average), so
the cache is a load-bearing optimization — building once per
``(repo, base_commit)`` and reusing across all instances at that commit.

Cache layout:

    repo_cache/v10_skeletons/{repo_slug}/{sha[:12]}/skeleton.json

The ``v10_`` prefix on ``v10_skeletons`` keeps the cache compatible with
``harness.cache``'s namespace check on ``V10_NAMESPACED_PARENTS``.

This module is leak-free by construction:
  - It reads only public source via ``Sandbox`` primitives at base_commit.
  - It does NOT import from ``meta_harness_plus`` (the firewall blocks it).
  - It produces ``RepoSkeleton`` (a typed view from ``harness.views``)
    whose field names cannot match forbidden tokens.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import asdict
from typing import Optional

from harness.views import (
    ClassSummary,
    FileSummary,
    FunctionSummary,
    InstanceView,
    RepoSkeleton,
)


CACHE_ROOT = pathlib.Path("repo_cache") / "v10_skeletons"


# In-container probe: walks /testbed for .py files, AST-parses each,
# emits per-file dict with classes (name + line range + method names)
# and top-level functions (name + line range). Bounded depth via rglob;
# excludes hidden dirs and common build/test caches.
_SKELETON_PROBE_SCRIPT = r"""
import ast, json, pathlib, sys

root = pathlib.Path('/testbed')
out = []

EXCLUDED = {'.git', '__pycache__', '.tox', '.eggs', 'build', 'dist', '.venv'}

def is_excluded(path):
    parts = set(path.parts)
    if any(p in parts for p in EXCLUDED):
        return True
    return any(p.startswith('.') and p not in {'.', '..'} for p in parts)

for path in root.rglob('*.py'):
    if is_excluded(path):
        continue
    try:
        text = path.read_text(errors='replace')
        tree = ast.parse(text)
    except Exception:
        continue
    rel = str(path.relative_to(root))

    classes = []
    functions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        'name': child.name,
                        'line_start': child.lineno,
                        'line_end': getattr(child, 'end_lineno', child.lineno),
                    })
            classes.append({
                'name': node.name,
                'line_start': node.lineno,
                'line_end': getattr(node, 'end_lineno', node.lineno),
                'methods': methods,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                'name': node.name,
                'line_start': node.lineno,
                'line_end': getattr(node, 'end_lineno', node.lineno),
            })

    out.append({
        'path': rel,
        'classes': classes,
        'functions': functions,
    })

print(json.dumps(out))
"""


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


_REPO_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(repo: str) -> str:
    """Convert a 'org/repo' string to a filesystem-safe slug."""
    return _REPO_SLUG_SAFE.sub("_", repo)


def _cache_path(repo: str, base_commit: str) -> pathlib.Path:
    return CACHE_ROOT / _slug(repo) / base_commit[:12] / "skeleton.json"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_skeleton(skel: RepoSkeleton, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_v10_skeleton_version": 1,
        "repo": skel.repo,
        "base_commit": skel.base_commit,
        "files": [_file_summary_dict(f) for f in skel.files],
        "n_files": len(skel.files),
        "n_classes": sum(len(f.classes) for f in skel.files),
        "n_functions": sum(
            len(f.functions) + sum(len(c.methods) for c in f.classes)
            for f in skel.files
        ),
        "built_at": time.time(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(path)


def _file_summary_dict(f: FileSummary) -> dict:
    return {
        "path": f.path,
        "classes": [
            {
                "name": c.name,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "methods": [
                    {"name": m.name, "line_start": m.line_start, "line_end": m.line_end}
                    for m in c.methods
                ],
            }
            for c in f.classes
        ],
        "functions": [
            {"name": fn.name, "line_start": fn.line_start, "line_end": fn.line_end}
            for fn in f.functions
        ],
    }


def _file_summary_from_dict(d: dict) -> FileSummary:
    return FileSummary(
        path=d["path"],
        classes=tuple(
            ClassSummary(
                name=c["name"],
                line_start=c["line_start"],
                line_end=c["line_end"],
                methods=tuple(
                    FunctionSummary(
                        name=m["name"],
                        line_start=m["line_start"],
                        line_end=m["line_end"],
                    )
                    for m in c.get("methods", [])
                ),
            )
            for c in d.get("classes", [])
        ),
        functions=tuple(
            FunctionSummary(
                name=fn["name"],
                line_start=fn["line_start"],
                line_end=fn["line_end"],
            )
            for fn in d.get("functions", [])
        ),
    )


def _deserialize_skeleton(path: pathlib.Path) -> RepoSkeleton:
    payload = json.loads(path.read_text())
    if payload.get("_v10_skeleton_version") != 1:
        raise ValueError(
            f"skeleton cache schema mismatch at {path}: "
            f"expected version 1, got {payload.get('_v10_skeleton_version')!r}"
        )
    return RepoSkeleton(
        repo=payload["repo"],
        base_commit=payload["base_commit"],
        files=tuple(_file_summary_from_dict(d) for d in payload["files"]),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_skeleton(view: InstanceView, sandbox) -> RepoSkeleton:
    """Walk the repo at base_commit inside ``sandbox``, AST-parse every
    .py file, and return a typed ``RepoSkeleton``.

    Caller's responsibility: pass an already-started ``Sandbox``
    constructed with a sufficiently large ``max_observation_chars``
    (default 32 KB is too small for django-sized repos; 8 MB is the
    audit script's working value).
    """
    res = sandbox.run_shell(
        "python3 -c " + _shell_quote(_SKELETON_PROBE_SCRIPT),
        timeout_s=180,
    )
    if res.exit_code != 0:
        raise RuntimeError(
            f"skeleton probe failed for {view.instance_id}: "
            f"exit={res.exit_code} stderr={(res.stderr or '')[:400]}"
        )
    files_raw = json.loads(res.stdout)

    files: list[FileSummary] = []
    for entry in files_raw:
        files.append(_file_summary_from_dict({
            "path": entry["path"],
            "classes": [
                {
                    "name": c["name"],
                    "line_start": c["line_start"],
                    "line_end": c["line_end"],
                    "methods": c["methods"],
                }
                for c in entry["classes"]
            ],
            "functions": entry["functions"],
        }))
    # Stable order for reproducibility.
    files.sort(key=lambda f: f.path)

    return RepoSkeleton(
        repo=view.repo,
        base_commit=view.base_commit,
        files=tuple(files),
    )


def load_or_build_skeleton(
    view: InstanceView,
    sandbox=None,
    *,
    cache_root: Optional[pathlib.Path] = None,
    force_rebuild: bool = False,
) -> RepoSkeleton:
    """Return the skeleton for ``view``, building from sandbox if not
    cached. Cache key: ``(view.repo, view.base_commit)``.

    If ``cache_root`` is given, it overrides the default (CACHE_ROOT) —
    used by tests to redirect to ``tmp_path``.
    """
    base = cache_root or CACHE_ROOT
    rel = _cache_path(view.repo, view.base_commit)
    cache_path = base / rel.relative_to(CACHE_ROOT) if cache_root else rel

    if not force_rebuild and cache_path.exists():
        return _deserialize_skeleton(cache_path)

    if sandbox is None:
        raise RuntimeError(
            f"skeleton not cached for {view.repo}@{view.base_commit[:12]} "
            f"and no sandbox provided; pass sandbox= to build"
        )
    skel = build_skeleton(view, sandbox)
    _serialize_skeleton(skel, cache_path)
    return skel


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\\''") + "'"


def cache_path_for(view: InstanceView) -> pathlib.Path:
    """Public helper for test/debug: where would this view's skeleton
    cache file live?"""
    return _cache_path(view.repo, view.base_commit)


__all__ = [
    "CACHE_ROOT",
    "build_skeleton",
    "load_or_build_skeleton",
    "cache_path_for",
]
