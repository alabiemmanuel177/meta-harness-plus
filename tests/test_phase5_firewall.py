"""Phase 5 — selection package firewall.

Per docs/V10_DESIGN_PHASE5.md §2.2:

  - No harness.selection.* module imports harness.eval.
  - No harness.selection.* module imports harness.repro.
  - No harness.selection.* module imports harness.memory.

Mirrors tests/test_repro_firewall.py + tests/test_phase4_firewall.py.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SELECTION_PKG = PROJECT_ROOT / "harness" / "selection"


def _selection_modules() -> list[pathlib.Path]:
    if not SELECTION_PKG.is_dir():
        return []
    return [
        p for p in SELECTION_PKG.rglob("*.py")
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


def test_no_phase5_module_imports_harness_eval():
    violators: list[str] = []
    for p in _selection_modules():
        if _imports_module(p, "harness.eval"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "harness.selection modules importing harness.eval (V8 leak): "
        + ", ".join(violators)
    )


def test_no_phase5_module_imports_harness_repro():
    violators: list[str] = []
    for p in _selection_modules():
        if _imports_module(p, "harness.repro"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "harness.selection modules importing harness.repro: "
        + ", ".join(violators)
    )


def test_no_phase5_module_imports_harness_memory():
    violators: list[str] = []
    for p in _selection_modules():
        if _imports_module(p, "harness.memory"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "harness.selection modules importing harness.memory: "
        + ", ".join(violators)
    )


def test_phase5_modules_loadable_via_package():
    if not SELECTION_PKG.is_dir():
        pytest.skip("harness.selection not present yet")
    import harness.selection  # noqa: F401
