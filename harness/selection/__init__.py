"""Phase 5 — candidate selection.

Public API:

    from harness.selection import select, SelectionResult

The combiner (combiner.py) is the only entrypoint the eval script
calls. Sub-modules expose individual selector primitives but are
not invoked directly outside ``harness.selection``.

Per docs/V10_DESIGN_PHASE5.md §2 the package must NEVER import
harness.repro or harness.eval — the firewall test in
tests/test_phase5_firewall.py enforces this at AST level.
"""

from harness.selection.combiner import (
    DEFAULT_WEIGHTS,
    SelectionResult,
    select,
)
from harness.selection.selector_b_cluster import score_by_cluster
from harness.selection.selector_c_heuristic import score_by_heuristic


__all__ = [
    "DEFAULT_WEIGHTS",
    "SelectionResult",
    "score_by_cluster",
    "score_by_heuristic",
    "select",
]
