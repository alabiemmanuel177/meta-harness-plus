"""Phase 4 — cross-phase firewall tests.

Per docs/V10_DESIGN_PHASE4.md §2.1 hard rules:

  - No harness.validation.* module imports harness.eval.
  - No harness.validation.* module imports harness.repro.
  - No harness.validation.* module shells out to pytest directly via
    Sandbox.run_shell — all test invocations go through
    Sandbox.run_public_suite.
  - No harness.validation.* module's source contains "-k" or
    "--collect-only" flag literals (catches refactors trying to
    whittle the public suite to a smaller set, which is the V7
    leak shape).

Mirrors tests/test_repro_firewall.py pattern.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VALIDATION_PKG = PROJECT_ROOT / "harness" / "validation"


def _validation_modules() -> list[pathlib.Path]:
    if not VALIDATION_PKG.is_dir():
        return []
    return [
        p for p in VALIDATION_PKG.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _imports_module(py_path: pathlib.Path, module_name: str) -> bool:
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name or alias.name.startswith(module_name + "."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == module_name or mod.startswith(module_name + "."):
                return True
    return False


def test_no_phase4_module_imports_harness_eval():
    """harness.eval is the post-submission grader. Phase 4 must not
    read its outputs (V8-class leak)."""
    violators: list[str] = []
    for p in _validation_modules():
        if _imports_module(p, "harness.eval"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "harness.validation modules importing harness.eval (V8 leak): "
        + ", ".join(violators)
    )


def test_no_phase4_module_imports_harness_repro():
    """Repro test bodies are firewalled. The orchestrator threads
    ReproSignal in as an argument; the validator never imports
    harness.repro."""
    violators: list[str] = []
    for p in _validation_modules():
        if _imports_module(p, "harness.repro"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "harness.validation modules importing harness.repro: "
        + ", ".join(violators)
    )


def test_no_phase4_module_imports_harness_memory():
    violators: list[str] = []
    for p in _validation_modules():
        if _imports_module(p, "harness.memory"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "harness.validation modules importing harness.memory: "
        + ", ".join(violators)
    )


def test_phase4_no_test_target_arguments():
    """No Phase 4 module's source contains a literal `-k` or
    `--collect-only` pytest flag — those are the shapes V7's leak
    used to whittle the suite to a smaller set."""
    violators: list[tuple[str, str]] = []
    for p in _validation_modules():
        try:
            src = p.read_text()
        except OSError:
            continue
        # Look for "-k" or "--collect-only" inside string literals.
        # Skip docstrings/comments by looking only at .py content with
        # quote characters around it. Simple regex heuristic.
        for pattern in (r'(?<!\w)-k\s', r'--collect-only'):
            for m in re.finditer(pattern, src):
                # Look at the surrounding context — only flag if inside
                # what looks like a quoted string.
                window = src[max(0, m.start() - 60): m.end() + 5]
                if any(q in window for q in ("'", '"', '"""', "'''")):
                    violators.append((str(p.relative_to(PROJECT_ROOT)), m.group(0)))
    # ALLOW: pytest internals like `-xvs` are fine. The firewall is
    # about explicit test-list arguments. -k accepts a name expression
    # and --collect-only narrows the discovery.
    assert not violators, (
        "Phase 4 modules with forbidden pytest flags (test-list narrowing):\n  "
        + "\n  ".join(f"{p}: {flag!r}" for p, flag in violators)
    )


def test_phase4_modules_loadable_via_package():
    """Sanity: the package imports cleanly. If a refactor breaks
    the import surface, the firewall test fires."""
    if not VALIDATION_PKG.is_dir():
        pytest.skip("harness.validation not present yet")
    import harness.validation  # noqa: F401
