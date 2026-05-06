"""Phase 4 — patch validation.

Public API:

    from harness.validation import (
        validate_candidate,
        ValidationResult,
        apply_clean,
        compute_diff_stats,
        run_public_suite_diff,
        run_static,
    )

The orchestrator (validator.py) is the only entrypoint the eval
script calls. Sub-modules are not invoked directly by code outside
``harness.validation``. Per docs/V10_DESIGN_PHASE4.md §2 the package
must NEVER import harness.repro or harness.eval — the firewall test
in tests/test_phase4_firewall.py enforces this at AST level.
"""

from harness.validation.apply import (
    ApplyResult,
    apply_clean,
)
from harness.validation.diff_stats import (
    compute_diff_stats,
    compute_ast_normalized_hash,
)
from harness.validation.public_suite import (
    BaselineSnapshot,
    PublicSuiteRunResult,
    diff_against_baseline,
    load_baseline_snapshot,
    run_baseline,
    run_patched,
    save_baseline_snapshot,
)
from harness.validation.static import (
    run_static,
)
from harness.validation.validator import (
    ValidationResult,
    validate_candidate,
)


__all__ = [
    "ApplyResult",
    "BaselineSnapshot",
    "PublicSuiteRunResult",
    "ValidationResult",
    "apply_clean",
    "compute_ast_normalized_hash",
    "compute_diff_stats",
    "diff_against_baseline",
    "load_baseline_snapshot",
    "run_baseline",
    "run_patched",
    "run_static",
    "save_baseline_snapshot",
    "validate_candidate",
]
