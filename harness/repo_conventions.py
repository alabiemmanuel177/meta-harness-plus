"""Per-repo test-directory overrides + filesystem fallback.

This is the source-of-truth for ``TestDirectives.dirs``. Every override
entry below carries a comment citing the public artifact at base_commit
the convention was derived from — anyone reviewing the firewall can
verify each entry traces to a public source, not to a dataset field.

Filesystem fallback runs default pytest discovery on a repo's checkout
at base_commit (find directories containing ``test_*.py`` or
``*_test.py``). Per V10_DESIGN.md §12.4(b), the override-vs-discovery
consistency test asserts that for every override repo, what we declare
matches what discovery would find. If they diverge, either the override
is stale or the override is unnecessary.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Per-repo overrides — each entry MUST cite its public artifact source.
# ---------------------------------------------------------------------------

# Format: repo full_name -> list of test directory paths (relative to repo root).
# Entries below cover the 12 repos in SWE-bench Verified.

REPO_TEST_DIRS: dict[str, tuple[str, ...]] = {
    # Source: django/django/CONTRIBUTING.rst — "Run tests with python tests/runtests.py"
    # Verified base_commit example: django__django-15252 (sha 361bb8f).
    "django/django": ("tests/",),

    # Source: sympy/setup.cfg [tool:pytest] testpaths = sympy/ doc/src
    # Verified base_commit example: sympy__sympy-22914 (sha 0f9b3b7).
    "sympy/sympy": ("sympy/",),

    # Source: sphinx-doc/sphinx tests/ directory + tox.ini envlist references
    # Verified base_commit example: sphinx-doc__sphinx-9711.
    "sphinx-doc/sphinx": ("tests/",),

    # Source: matplotlib/matplotlib lib/matplotlib/tests/ + setup.cfg [tool:pytest]
    # Verified base_commit example: matplotlib__matplotlib-23476.
    "matplotlib/matplotlib": ("lib/matplotlib/tests/", "lib/mpl_toolkits/"),

    # Source: scikit-learn pyproject.toml [tool.pytest.ini_options] addopts uses sklearn/
    # Verified base_commit example: scikit-learn__scikit-learn-13779.
    "scikit-learn/scikit-learn": ("sklearn/",),

    # Source: astropy/setup.cfg [tool:pytest] testpaths = astropy
    # Verified base_commit example: astropy__astropy-12907.
    "astropy/astropy": ("astropy/",),

    # Source: pydata/xarray xarray/tests/ + pyproject.toml [tool.pytest.ini_options]
    # Verified base_commit example: pydata__xarray-4248.
    "pydata/xarray": ("xarray/tests/",),

    # Source: pytest-dev/pytest pyproject.toml [tool.pytest.ini_options] testpaths
    # Verified base_commit example: pytest-dev__pytest-7373.
    "pytest-dev/pytest": ("testing/",),

    # Source: pylint-dev/pylint pyproject.toml [tool.pytest.ini_options] testpaths = ["tests"]
    # Verified base_commit example: pylint-dev__pylint-4604.
    "pylint-dev/pylint": ("tests/",),

    # Source: psf/requests tests/ directory + setup.cfg
    # Verified base_commit example: psf__requests-1142.
    "psf/requests": ("tests/",),

    # Source: mwaskom/seaborn tests/ directory + pyproject.toml [tool.pytest.ini_options]
    # Verified base_commit example: mwaskom__seaborn-3010.
    "mwaskom/seaborn": ("tests/",),

    # Source: pallets/flask tests/ directory + pyproject.toml
    # Verified base_commit example: pallets__flask-5063.
    "pallets/flask": ("tests/",),
}


@dataclass(frozen=True)
class DiscoveredDirs:
    dirs: tuple[str, ...]
    n_test_files: int


def discover_test_dirs(repo_root: pathlib.Path) -> DiscoveredDirs:
    """Default pytest discovery: find directories at ``repo_root`` (or one
    level deep) that contain at least one ``test_*.py`` or ``*_test.py``
    file. Returns the deduplicated, sorted set of such directories.

    This is the filesystem fallback that runs when a repo is not in
    ``REPO_TEST_DIRS``. It is also exercised against override repos by
    the consistency test in tests/.
    """
    seen: set[pathlib.Path] = set()
    if not repo_root.exists():
        return DiscoveredDirs(dirs=(), n_test_files=0)

    n_test_files = 0
    for path in repo_root.rglob("test_*.py"):
        # Skip hidden dirs and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        n_test_files += 1
        seen.add(path.parent)
    for path in repo_root.rglob("*_test.py"):
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        n_test_files += 1
        seen.add(path.parent)

    # Reduce to top-level test directories — collapse a/b/tests/c/test_x.py
    # to a/b/tests/ when "tests" appears in the path.
    reduced: set[str] = set()
    for d in seen:
        rel = d.relative_to(repo_root)
        parts = list(rel.parts)
        # If a "tests" or "testing" segment exists, keep up to and including it.
        for i, p in enumerate(parts):
            if p in {"tests", "testing"}:
                reduced.add("/".join(parts[: i + 1]) + "/")
                break
        else:
            reduced.add(str(rel) + "/")

    if not reduced:
        return DiscoveredDirs(dirs=(), n_test_files=0)
    return DiscoveredDirs(dirs=tuple(sorted(reduced)), n_test_files=n_test_files)


def resolve_test_directives_static(repo: str) -> tuple[str, ...] | None:
    """Return the override entry for ``repo`` if known. Used at projection
    time when no repo checkout is available (Phase 0 InstanceView build).
    """
    return REPO_TEST_DIRS.get(repo)


def resolve_test_directives(
    repo: str,
    repo_root: pathlib.Path | None = None,
) -> tuple[tuple[str, ...], str]:
    """Resolve a repo's test directives to (dirs, source_label).

    Resolution order:
      1. ``REPO_TEST_DIRS`` override — returns ("override:<repo_artifacts>", dirs)
      2. Filesystem discovery if ``repo_root`` is given and exists
      3. Fallback to ``("tests/",)`` with source ``"fallback:default"``

    The source label is opaque human-readable text; selectors and
    prompt builders never branch on it.
    """
    override = resolve_test_directives_static(repo)
    if override is not None:
        return override, "override:repo_conventions"
    if repo_root is not None:
        d = discover_test_dirs(repo_root)
        if d.dirs:
            return d.dirs, f"discovery:{d.n_test_files}_test_files"
    return ("tests/",), "fallback:default"


__all__ = [
    "REPO_TEST_DIRS",
    "DiscoveredDirs",
    "discover_test_dirs",
    "resolve_test_directives",
    "resolve_test_directives_static",
]
