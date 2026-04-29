"""Negative smoke — confirm the firewall fires on real call paths.

The firewall test in ``tests/test_no_oracle_leak.py`` covers synthetic
construction. The positive smoke (``scripts/smoke_phase0.py``) covers a
real instance through the leak-free path. This script is the missing
third leg: confirm the firewall RAISES on a real call path when
something is tampered.

Two assertions, both must trip ``OracleLeakError``:

  1. Tamper an InstanceView via ``object.__setattr__`` (bypasses the
     frozen check) to add a ``fail_to_pass`` attribute. Pass it through
     an ``LLMCallInspector``-wrapped fake LLM call. Inspector must trip.

  2. Pass a forbidden-token ``state_label`` to ``Sandbox.run_public_suite``.
     The runtime guard added in commit 9 must trip before any docker
     exec runs.

If either path does NOT raise, the firewall has a hole — the script
exits non-zero so the Makefile target fails CI.
"""

from __future__ import annotations

import sys
import traceback

# We deliberately avoid importing OracleLeakError from a single place;
# both sandbox and conftest define their own. We catch by class hierarchy.
from harness.sandbox import OracleLeakError as SandboxLeak
from harness.dataset import load_verified_view


def assert_raises(label: str, fn, *exc_types):
    print(f"[neg] {label} … ", end="", flush=True)
    try:
        fn()
    except exc_types as exc:
        print(f"OK (raised {type(exc).__name__})")
        return True
    except Exception as exc:
        print(f"FAIL (raised wrong type: {type(exc).__name__}: {exc})")
        traceback.print_exc()
        return False
    print("FAIL (did not raise)")
    return False


def negative_assertion_1_tampered_instance_view() -> bool:
    """Construct a real InstanceView via the asserted boundary, then
    tamper it via object.__setattr__ to add a forbidden attribute.
    Confirm the LLMCallInspector walker catches the smuggled field
    when the view is passed as an LLM kwarg.
    """
    # Local import so this script is independent of pytest's conftest.
    from tests.conftest import LLMCallInspector, OracleLeakError as InspectorLeak

    view = load_verified_view("psf__requests-2317")
    # Smuggle: bypass the frozen-dataclass setattr by going through object.
    object.__setattr__(view, "fail_to_pass", ["test_a", "test_b"])

    def fake_llm_call(*, ctx, prompt: str) -> str:
        return "ok"

    wrapped = LLMCallInspector("neg_smoke.assertion_1", fake_llm_call)

    return assert_raises(
        "tampered InstanceView via object.__setattr__",
        lambda: wrapped(ctx=view, prompt="please fix this issue"),
        InspectorLeak,
    )


def negative_assertion_2_forbidden_state_label() -> bool:
    """Pass a forbidden-token state_label to Sandbox.run_public_suite.
    The runtime guard must trip BEFORE any docker exec — we never even
    start the container in this path.
    """
    from harness.sandbox import Sandbox

    view = load_verified_view("psf__requests-2317")
    # Don't actually start docker — just construct the Sandbox; the guard
    # check on state_label fires at the start of run_public_suite, well
    # before any container interaction.
    sb = Sandbox(view)

    def call_with_leaky_label():
        # Build forbidden token via concatenation so this script's own
        # source doesn't trip the firewall AST scan.
        leaky = "fail" + "_to_pass=1 base"
        sb.run_public_suite(state_label=leaky, collect_only=True)

    try:
        return assert_raises(
            "forbidden token in state_label",
            call_with_leaky_label,
            SandboxLeak,
        )
    finally:
        sb.cleanup()


def main() -> int:
    print("=" * 64)
    print("V10 negative smoke — confirm the firewall raises on real paths")
    print("=" * 64)
    results = [
        negative_assertion_1_tampered_instance_view(),
        negative_assertion_2_forbidden_state_label(),
    ]
    n_pass = sum(results)
    print("=" * 64)
    print(f"{n_pass}/{len(results)} negative assertions passed")
    if n_pass == len(results):
        print("OK: firewall fires at both real call paths")
        return 0
    print("FAIL: firewall has a hole — see traceback above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
