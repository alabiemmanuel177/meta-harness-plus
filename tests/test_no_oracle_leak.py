"""V10 contamination firewall — type, AST, and runtime layers.

This is the test that V7 and V8 would have failed. Specifically:

- V7's leak (FAIL_TO_PASS in actor prompt strings) trips the AST string-
  literal scan and the type-layer view audit — `SWEBenchInstance`
  carries `fail_to_pass` as a field, and any harness module touching
  it would fail the AST import check.

- V8's leak (`resolved_lookup` consults the eval verdict in the
  selector) trips the runtime layer because the `resolved` field would
  reach an LLM call argument or a CandidateView, and both are scanned.

The firewall is enforced at three layers:

1. **Type** — every dataclass in ``harness.views.VIEW_CLASSES`` is scanned
   for forbidden field names. ``ForbiddenFieldError`` raised in
   ``__post_init__`` if violated; this test verifies that mechanism is
   wired correctly for every view.
2. **AST** — every ``.py`` module under ``harness/`` is parsed and walked.
   Imports from forbidden legacy modules, ``getattr(x, "fail_to_pass")``,
   ``x["FAIL_TO_PASS"]``, string literals containing forbidden tokens,
   and reads under ``eval_outputs/`` all fail the test.
3. **Runtime** — a pytest fixture wraps an LLM-call shape, inspects
   every kwarg payload, and raises ``OracleLeakError`` if a forbidden
   token reaches the call site. Verified here with a positive (clean
   call) and a negative (leak detected) test.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass

import pytest

from harness import views as V


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS_ROOT = PROJECT_ROOT / "harness"
TESTS_ROOT = PROJECT_ROOT / "tests"

# Modules that legitimately need to mention forbidden tokens because they
# ARE the firewall infrastructure or the post-submission grader (which the
# rest of the package is forbidden from importing).
FIREWALL_INFRA_FILES = {
    HARNESS_ROOT / "views.py",      # defines FORBIDDEN_TOKENS
    HARNESS_ROOT / "sandbox.py",    # defines _FORBIDDEN_TOKENS for runtime guard
    HARNESS_ROOT / "eval.py",       # post-submission grader; reads run_evaluation
}
# Test files may name forbidden tokens (they're the firewall test).
TEST_FILES_ALLOWED = {
    TESTS_ROOT / "test_no_oracle_leak.py",
    TESTS_ROOT / "conftest.py",
}

# Forbidden imports — exact module paths or module-prefix patterns.
# These are the leak surfaces audited in §12 of V10_DESIGN.md.
FORBIDDEN_IMPORT_MODULES = {
    "meta_harness_plus.swebench_v7",
    "meta_harness_plus.agent_swebench_loop",
}
FORBIDDEN_IMPORT_NAMES_FROM = {
    # (module, name) — `from <module> import <name>` is forbidden.
    ("meta_harness_plus.swebench_adapter", "SWEBenchInstance"),
    ("meta_harness_plus.swebench_adapter", "build_user_prompt"),
    ("meta_harness_plus.swebench_adapter", "load_swebench_verified"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _harness_py_files() -> list[pathlib.Path]:
    return sorted(p for p in HARNESS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _contains_forbidden_token(s: str) -> str | None:
    folded = s.casefold()
    for tok in V.FORBIDDEN_TOKENS:
        if tok in folded:
            return tok
    return None


# ---------------------------------------------------------------------------
# Layer 1 — type-level
# ---------------------------------------------------------------------------


def test_view_classes_have_no_forbidden_field_names() -> None:
    """Every VIEW_CLASSES dataclass field name must be free of forbidden
    tokens. This is the static side of the type firewall."""
    from dataclasses import fields

    bad: list[str] = []
    for cls in V.VIEW_CLASSES:
        for f in fields(cls):
            tok = _contains_forbidden_token(f.name)
            if tok is not None:
                bad.append(f"{cls.__name__}.{f.name} (token={tok!r})")
    assert not bad, "view dataclass fields with forbidden tokens: " + ", ".join(bad)


def test_attempting_to_construct_view_with_forbidden_field_raises() -> None:
    """Adding a forbidden field at class definition time raises at first
    construction. We synthesize such a class and confirm the validator
    fires — proving the mechanism is wired for any future addition."""
    from dataclasses import dataclass as dc

    @dc(frozen=True)
    class _BadView:
        gold_patch: str

        def __post_init__(self) -> None:
            V._assert_no_forbidden_field_names(type(self))

    with pytest.raises(V.ForbiddenFieldError):
        _BadView(gold_patch="anything")


# ---------------------------------------------------------------------------
# Layer 2 — AST-level
# ---------------------------------------------------------------------------


def test_no_harness_module_imports_forbidden_legacy_modules() -> None:
    violations: list[str] = []
    for path in _harness_py_files():
        if path in FIREWALL_INFRA_FILES:
            continue  # firewall infra may import nothing from legacy anyway
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORT_MODULES:
                        violations.append(f"{path.relative_to(PROJECT_ROOT)}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in FORBIDDEN_IMPORT_MODULES:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}: from {mod} import …"
                    )
                for alias in node.names:
                    if (mod, alias.name) in FORBIDDEN_IMPORT_NAMES_FROM:
                        violations.append(
                            f"{path.relative_to(PROJECT_ROOT)}: from {mod} import {alias.name}"
                        )
    assert not violations, "forbidden legacy imports: " + "; ".join(violations)


def test_no_harness_module_uses_getattr_with_forbidden_string_literal() -> None:
    """`getattr(x, "fail_to_pass")` and similar are forbidden anywhere in
    harness/ outside firewall infra files."""
    violations: list[str] = []
    for path in _harness_py_files():
        if path in FIREWALL_INFRA_FILES:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (
                fn.id if isinstance(fn, ast.Name)
                else fn.attr if isinstance(fn, ast.Attribute)
                else ""
            )
            if name not in {"getattr", "hasattr", "setattr", "delattr"}:
                continue
            if len(node.args) < 2:
                continue
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str):
                tok = _contains_forbidden_token(second.value)
                if tok is not None:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"{name}(_, {second.value!r}) -> token {tok!r}"
                    )
    assert not violations, "getattr/setattr with forbidden literal: " + "; ".join(violations)


def test_no_harness_module_uses_subscript_with_forbidden_string_literal() -> None:
    """`x["FAIL_TO_PASS"]`, `x['fail_to_pass']`, etc. are forbidden."""
    violations: list[str] = []
    for path in _harness_py_files():
        if path in FIREWALL_INFRA_FILES:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                tok = _contains_forbidden_token(sl.value)
                if tok is not None:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"_[{sl.value!r}] -> token {tok!r}"
                    )
    assert not violations, "subscript with forbidden literal: " + "; ".join(violations)


def test_no_harness_module_has_forbidden_string_literal_in_source() -> None:
    """String literals containing forbidden tokens are forbidden outside
    firewall infra files. This catches the V7-class leak: putting
    'FAIL_TO_PASS' verbatim in a prompt template."""
    violations: list[str] = []
    for path in _harness_py_files():
        if path in FIREWALL_INFRA_FILES:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                tok = _contains_forbidden_token(node.value)
                if tok is not None:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"string literal contains {tok!r}: {node.value[:120]!r}"
                    )
    assert not violations, "forbidden string literals: " + "; ".join(violations)


def test_no_harness_module_outside_eval_reads_patch_field() -> None:
    """The gold ``patch`` field is read only at evaluation time
    (harness/eval.py) — never in the inference pipeline. AST scan
    rejects any other harness/ module that:

      - subscripts a value with the literal string ``"patch"``
        (``row["patch"]``)
      - accesses a ``.patch`` attribute (``row.patch``)
      - calls ``getattr(_, "patch")`` / ``hasattr`` / ``setattr``
        / ``delattr`` with the literal string ``"patch"``

    Phase 1 stage 1b's retrieval recall eval reads the gold patch in
    ``harness/eval.py:_load_gold_touched_files``. That's the only
    legitimate use site.
    """
    patch_eval_only = {HARNESS_ROOT / "eval.py"}
    violations: list[str] = []
    for path in _harness_py_files():
        if path in patch_eval_only:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # Subscript: x["patch"]
            if isinstance(node, ast.Subscript):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    if sl.value == "patch":
                        violations.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                            f"_['patch'] subscript"
                        )
            # Attribute: x.patch (excluding method names like .apply_patch)
            if isinstance(node, ast.Attribute) and node.attr == "patch":
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: _.patch attribute"
                )
            # getattr/setattr/hasattr/delattr with "patch"
            if isinstance(node, ast.Call):
                fn = node.func
                name = (
                    fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute)
                    else ""
                )
                if name not in {"getattr", "hasattr", "setattr", "delattr"}:
                    continue
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if node.args[1].value == "patch":
                        violations.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                            f"{name}(_, 'patch')"
                        )
    assert not violations, (
        "harness modules outside eval.py read the gold patch field: "
        + "; ".join(violations)
    )


def test_no_harness_module_reads_eval_outputs() -> None:
    """`eval_outputs/` is the post-submission grader's exclusive write
    target. No harness module may reference that path in any string
    literal except:
      - harness/eval.py (the grader itself)
      - harness/cache.py (lists eval_outputs as a V10_EXCLUSIVE_DIR
        for the import-time hygiene check; never reads from it)
    """
    eval_outputs_allowlist = {
        HARNESS_ROOT / "eval.py",
        HARNESS_ROOT / "cache.py",
    }
    violations: list[str] = []
    for path in _harness_py_files():
        if path in eval_outputs_allowlist:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "eval_outputs" in node.value.casefold():
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"references eval_outputs in {node.value[:120]!r}"
                    )
    assert not violations, (
        "harness modules outside the allowlist reference eval_outputs: "
        + "; ".join(violations)
    )


# ---------------------------------------------------------------------------
# Layer 3 — runtime
# ---------------------------------------------------------------------------


def test_runtime_inspector_blocks_forbidden_token_in_kwargs(oracle_leak_guard: dict) -> None:
    """The LLMCallInspector wraps a callable and inspects every kwarg
    payload at runtime. Confirm a forbidden-token kwarg raises."""
    Inspector = oracle_leak_guard["Inspector"]
    OracleLeakError = oracle_leak_guard["OracleLeakError"]

    def fake_llm_call(*, model: str, messages: list[dict]) -> str:
        return "ok"

    wrapped = Inspector("test.fake_llm_call", fake_llm_call)

    # Positive: clean call passes.
    out = wrapped(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Fix the bug in foo.py"}],
    )
    assert out == "ok"
    assert wrapped.call_count == 1

    # Negative: passing a message with FAIL_TO_PASS in the content raises.
    leaky_content = "Run the F" + "AIL_TO_PASS test selectors first"
    with pytest.raises(OracleLeakError):
        wrapped(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": leaky_content}],
        )


def test_runtime_inspector_blocks_dataclass_field_with_forbidden_value(oracle_leak_guard: dict) -> None:
    """A dataclass kwarg whose field VALUE contains a forbidden token
    must also trip the inspector."""
    Inspector = oracle_leak_guard["Inspector"]
    OracleLeakError = oracle_leak_guard["OracleLeakError"]

    @dataclass
    class FakeContext:
        repo: str
        notes: str

    def fake_llm_call(*, ctx: FakeContext, prompt: str) -> str:
        return "ok"

    wrapped = Inspector("test.fake_llm_call_dc", fake_llm_call)

    # Clean field value passes.
    wrapped(
        ctx=FakeContext(repo="astropy/astropy", notes="just a note"),
        prompt="hello",
    )

    # Field value smuggling a forbidden token raises.
    with pytest.raises(OracleLeakError):
        wrapped(
            ctx=FakeContext(
                repo="astropy/astropy",
                notes="hint" + "s_text observed in issue",  # avoid literal
            ),
            prompt="hello",
        )


def test_runtime_inspector_blocks_dataclass_field_NAME_being_forbidden(oracle_leak_guard: dict) -> None:
    """If someone constructs a dataclass with a field NAMED a forbidden
    token at runtime (e.g., dynamically), the inspector also catches it
    via field-name walking."""
    Inspector = oracle_leak_guard["Inspector"]
    OracleLeakError = oracle_leak_guard["OracleLeakError"]

    @dataclass
    class _Smuggle:
        # Field name itself contains a forbidden token. (We can construct
        # this here even though VIEW_CLASSES validates against it, because
        # _Smuggle is not a VIEW_CLASS.) The inspector should still trip.
        gold_patch: str

    def fake_llm_call(*, payload) -> None:
        return None

    wrapped = Inspector("test.smuggle", fake_llm_call)
    with pytest.raises(OracleLeakError):
        wrapped(payload=_Smuggle(gold_patch="x"))


def test_walker_handles_nested_collections(oracle_leak_guard: dict) -> None:
    walk = oracle_leak_guard["walk"]
    scan = oracle_leak_guard["scan"]
    payload = {
        "a": ["clean", {"nested": ["clean", "fail" + "_to_pass smuggled"]}],
    }
    strings = walk(payload)
    hit = scan(strings)
    assert hit is not None
    assert hit[0] == "fail_to_pass"
