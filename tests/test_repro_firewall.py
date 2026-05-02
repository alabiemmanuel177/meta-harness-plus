"""Phase 2 commit-17c — cross-phase firewall + calibration-respect tests.

Per V10_DESIGN_PHASE2.md §7 acceptance criteria, three structural
checks land here:

  (a) No Phase 3 module (when it exists) imports ``harness.repro``.
      Phase 3 is the patch generator; the contamination model says
      the patch generator MUST NOT see the repro test code, the
      generator's rationale, or any internal repro state. The check
      is forward-looking — Phase 3 modules don't exist yet — but the
      test is wired now so it'll fire the moment one lands.

  (b) The repro generator's prompt-build path refuses synthetic
      InstanceView with planted forbidden tokens (per §8.4 input-
      layer enforcement). Each FORBIDDEN_TOKENS entry is planted in
      each firewalled field; we assert ReproGeneratorError fires
      with a message naming the field.

  (c) The substring-rule calibration from 17a is respected. Real
      patterns from the 12 Verified repos that legitimately match
      (the 2 Django test_patch cases) are flagged at the OUTPUT
      layer (informational only). Patterns we deliberately
      eliminated (test_patches_distortion, with_hints, unresolved)
      are NOT flagged.

Plus a cross-module structural test (the matcher must not be
re-implemented elsewhere) and a runtime-tamper test (bypassing
__post_init__ via ``object.__setattr__`` doesn't escape the input
firewall, because the firewall reads field VALUES at call time).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from harness.localization_signals import RankedFile
from harness.repro import (
    ReproGeneratorError,
    _detect_forbidden_tokens,
    _validate_input_firewall,
    check_output_substring_hits,
)
from harness.views import (
    FORBIDDEN_TOKENS,
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS = PROJECT_ROOT / "harness"


def _walk_harness_py() -> list[pathlib.Path]:
    return [p for p in HARNESS.rglob("*.py") if "__pycache__" not in p.parts]


# ---------------------------------------------------------------------------
# (a) No Phase 3 module imports harness.repro
# ---------------------------------------------------------------------------


# Forward-looking allowlist of paths that COULD become Phase 3 modules.
# Two shapes (per the V10_DESIGN_PHASE3.md §3.5 module layout):
#   - Flat: harness/<name>.py for legacy compatibility (Phase 4/5 may
#     land flat).
#   - Subdir: harness/patch_gen/*.py (the Phase 3 layout).
# When any of these lands, the test starts enforcing.
PHASE_3_MODULE_BASENAMES = (
    "patch_generator.py",
    "patch_gen.py",
    "patch.py",
    "patch_agent.py",
    "patch_pipeline.py",
    "patch_minimizer.py",
    "agent.py",
    "patch_validation.py",  # Phase 4
    "selection.py",         # Phase 5
    "selector.py",
)
PHASE_3_PACKAGE_DIRS = (
    "patch_gen",
)
# Backward-compat alias used by older callers.
PHASE_3_MODULE_CANDIDATES = PHASE_3_MODULE_BASENAMES


def _phase3_module_paths() -> list[pathlib.Path]:
    """Return every existing Phase 3 module path under harness/, both
    flat (harness/<name>.py) and subdir (harness/patch_gen/*.py)."""
    found: list[pathlib.Path] = []
    for name in PHASE_3_MODULE_BASENAMES:
        p = HARNESS / name
        if p.exists():
            found.append(p)
    for d in PHASE_3_PACKAGE_DIRS:
        pkg = HARNESS / d
        if pkg.is_dir():
            for p in pkg.rglob("*.py"):
                if "__pycache__" in p.parts:
                    continue
                found.append(p)
    return found


def _imports_module(py_path: pathlib.Path, module_name: str) -> bool:
    """True iff py_path has any import statement that resolves to
    ``module_name`` or any submodule under ``module_name.*``."""
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


def _imports_harness_repro(py_path: pathlib.Path) -> bool:
    return _imports_module(py_path, "harness.repro")


def test_no_phase3_module_imports_harness_repro():
    """Forward-looking gate (still forward-looking for Phase 4/5; live
    for the Phase 3 patch_gen package per V10_DESIGN_PHASE3.md §2.1
    rule 1). Walks every Phase 3 module path under harness/ and asserts
    none import harness.repro."""
    found = _phase3_module_paths()
    violators: list[str] = []
    for p in found:
        if _imports_harness_repro(p):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    print(
        f"[firewall-test:phase3] scanned {len(found)} Phase 3 module paths: "
        f"{sorted(str(p.relative_to(PROJECT_ROOT)) for p in found)}"
    )
    assert not violators, (
        "These Phase 3 modules import harness.repro — that's a "
        "contamination-rule violation per V10_DESIGN_PHASE3.md §2.1 rule 1:\n  "
        + "\n  ".join(violators)
    )


def test_no_phase3_module_imports_harness_eval():
    """Per V10_DESIGN_PHASE3.md §2.1 rule 2: NO module under harness.patch_gen
    (or future Phase 4/5 modules) imports harness.eval. Eval verdicts
    are post-submission grader output; selection NEVER reads them."""
    found = _phase3_module_paths()
    violators: list[str] = []
    for p in found:
        if _imports_module(p, "harness.eval"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "These Phase 3 modules import harness.eval — that's the V8-style "
        "selection-side leak class per V10_DESIGN.md §2 + V10_DESIGN_PHASE3.md "
        "§2.1 rule 2:\n  " + "\n  ".join(violators)
    )


def test_no_phase3_module_imports_harness_memory():
    """Per V10_DESIGN_PHASE3.md §2.1 rule 5: cross-instance memory is
    OFF in V0. No Phase 3 module imports anything from a hypothetical
    harness.memory.* submodule (placeholder for Bet #6 future work).
    The directory may not exist yet — the test is forward-looking."""
    found = _phase3_module_paths()
    violators: list[str] = []
    for p in found:
        if _imports_module(p, "harness.memory"):
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "These Phase 3 modules import harness.memory — cross-instance "
        "memory is forbidden in V0 per V10_DESIGN_PHASE3.md §2.1 rule 5:\n  "
        + "\n  ".join(violators)
    )


def test_phase3_context_superset_assertion_present():
    """Per V10_DESIGN_PHASE3.md §2.1 rule 3 + §2.2: every prompt-build
    path under harness.patch_gen must funnel through
    ``harness.patch_gen.context.build_patch_gen_context_with_superset_check``.

    AST scan: every harness.patch_gen module that calls
    ``complete_chat`` (the LLM call surface) must also reference
    ``build_patch_gen_context_with_superset_check`` somewhere. This
    catches refactors that silently drop the assertion.
    """
    pkg = HARNESS / "patch_gen"
    if not pkg.is_dir():
        # Phase 3 hasn't landed yet — test is forward-looking.
        return
    violators: list[str] = []
    for p in pkg.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            src = p.read_text()
        except Exception:
            continue
        if "complete_chat" not in src:
            continue
        if "build_patch_gen_context_with_superset_check" not in src:
            violators.append(str(p.relative_to(PROJECT_ROOT)))
    assert not violators, (
        "These harness.patch_gen modules call complete_chat() but do not "
        "reference build_patch_gen_context_with_superset_check — they "
        "may bypass the patch-generator-context-superset assertion per "
        "V10_DESIGN_PHASE3.md §2.1 rule 3:\n  " + "\n  ".join(violators)
    )


# ---------------------------------------------------------------------------
# (b) Input-layer firewall fires for every forbidden token in every field
# ---------------------------------------------------------------------------


def _make_view(
    problem_statement: str = "Bug: foo() returns wrong value",
    test_dirs: tuple[str, ...] = ("tests",),
) -> InstanceView:
    return InstanceView(
        instance_id="example__example-1",
        repo="example/example",
        base_commit="abcdef0123",
        problem_statement=problem_statement,
        repo_skeleton=RepoSkeleton(
            repo="example/example",
            base_commit="abcdef0123",
        ),
        test_directives=TestDirectives(dirs=test_dirs, source="discovery:1"),
    )


def _make_ranked_files(paths: list[str]) -> list[RankedFile]:
    return [
        RankedFile(
            file_path=p,
            final_score=1.0 - i * 0.05,
            rationale="r",
            upstream_signals=("bm25",),
            upstream_best_rank={"bm25": i + 1},
        )
        for i, p in enumerate(paths)
    ]


# All FORBIDDEN_TOKENS that match standalone via word-boundary. Note
# that 'hints' and 'resolved' are bare words and only fire under
# word-boundary contexts (e.g., '"hints":' or '"resolved":'). For the
# (b) check we plant each token in a context that creates explicit
# word boundaries.
@pytest.mark.parametrize("tok", FORBIDDEN_TOKENS)
def test_input_firewall_blocks_token_in_problem_statement(tok):
    """Each forbidden token, planted in problem_statement, fires the
    input-layer firewall with a message naming the field."""
    poisoned = f'(see "{tok}" field for details)'
    view = _make_view(problem_statement=poisoned)
    rfs = _make_ranked_files(["src/foo.py"])
    with pytest.raises(ReproGeneratorError, match=r"problem_statement"):
        _validate_input_firewall(view, rfs)


@pytest.mark.parametrize("tok", FORBIDDEN_TOKENS)
def test_input_firewall_blocks_token_in_test_directive(tok):
    """Each forbidden token, planted as a test-dir path component,
    fires the firewall with a message naming the test_directives
    field."""
    poisoned_dir = f"tests/{tok}"
    view = _make_view(test_dirs=(poisoned_dir,))
    rfs = _make_ranked_files(["src/foo.py"])
    with pytest.raises(ReproGeneratorError, match=r"test_directives"):
        _validate_input_firewall(view, rfs)


@pytest.mark.parametrize("tok", FORBIDDEN_TOKENS)
def test_input_firewall_blocks_token_in_ranked_file_path(tok):
    """Each forbidden token, planted as a ranked-file path component,
    fires the firewall with a message naming ranked_files."""
    poisoned_path = f"src/{tok}.py"
    view = _make_view()
    rfs = _make_ranked_files([poisoned_path])
    with pytest.raises(ReproGeneratorError, match=r"ranked_files"):
        _validate_input_firewall(view, rfs)


# ---------------------------------------------------------------------------
# (c) Calibration is respected — real test names trip iff they should
# ---------------------------------------------------------------------------


# The 2 known trippers from the §8.4 calibration audit, both Django
# HTTP-PATCH method tests literally named "test_patch".
KNOWN_TRIPPING_TEST_NAMES = (
    "test_patch",
    "RequestMethodTests.test_patch",
)

# Patterns the substring-on-normalized rule used to over-flag (commit
# 17a calibration found 69 such cases); word-boundary now correctly
# leaves them alone.
KNOWN_NOT_TRIPPING_TEST_NAMES = (
    "test_patches_distortion",         # matplotlib
    "test_patch_vary_headers",          # django
    "test_patch_cache_control",         # django
    "test_patches_alpha_coloring",      # matplotlib
    "with_hints",
    "without_hints",
    "test_compile_unresolved",          # django
    "test_main_module_is_resolved",     # django
)


@pytest.mark.parametrize("name", KNOWN_TRIPPING_TEST_NAMES)
def test_output_guard_flags_known_trippers(name):
    """The 2 Django test_patch methods literally collide with the
    SWE-bench dataset field name. Per §8.4 the OUTPUT-layer scan
    flags them as informational; production logs WARNING, doesn't
    block."""
    hits = check_output_substring_hits(
        test_code=f"def {name.split('.')[-1]}(): pass",
        target_test_id=f"tests/x.py::{name.split('.')[-1]}",
    )
    assert hits, (
        f"{name!r} should trip the output guard — it literally "
        f"matches 'test_patch' on word boundary"
    )


@pytest.mark.parametrize("name", KNOWN_NOT_TRIPPING_TEST_NAMES)
def test_output_guard_does_not_flag_known_false_positives(name):
    """Patterns that the substring-on-normalized rule used to over-
    flag must remain clean under word-boundary matching. Per §8.4
    calibration."""
    hits = check_output_substring_hits(
        test_code=f"def {name}(): pass",
        target_test_id=f"tests/x.py::{name}",
    )
    assert hits == [], (
        f"{name!r} is a real test name in the 12 Verified repos; "
        f"the rule should NOT flag it (got hits={hits})"
    )


def test_input_firewall_does_not_block_legitimate_phrases():
    """Issue text legitimately uses 'hints' or 'resolved' as
    natural language. These must not block the firewall."""
    legitimate_phrasings = [
        "Any hints would be appreciated",
        "Once this is resolved we can move on",
        "myFailToPassFlag is set elsewhere",
        "test_patches_distortion fails on this branch",
    ]
    for text in legitimate_phrasings:
        view = _make_view(problem_statement=text)
        rfs = _make_ranked_files(["src/foo.py"])
        # Must not raise.
        _validate_input_firewall(view, rfs)


# ---------------------------------------------------------------------------
# Cross-module structural test — matcher must not be re-implemented
# ---------------------------------------------------------------------------


# Files that legitimately DEFINE the matchers. Right now only repro.py.
MATCHER_DEFINITION_ALLOWLIST = {
    HARNESS / "repro.py",
}

MATCHER_FUNCTION_NAMES = {
    "_detect_forbidden_tokens",
    "_validate_input_firewall",
}


def _function_definitions(py_path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
    return out


def test_matcher_is_not_re_implemented_elsewhere():
    """Catches a future regression where someone copy-pastes the
    forbidden-token matcher into a new module instead of importing
    it. AST-scans every harness/*.py; flags any non-allowlist file
    that DEFINES _detect_forbidden_tokens or _validate_input_firewall.
    Importing the function from harness.repro is fine; redefining
    is not."""
    violators: list[str] = []
    allowlist_resolved = {p.resolve() for p in MATCHER_DEFINITION_ALLOWLIST}
    for py in _walk_harness_py():
        if py.resolve() in allowlist_resolved:
            continue
        defs = _function_definitions(py)
        clash = defs & MATCHER_FUNCTION_NAMES
        if clash:
            violators.append(
                f"{py.relative_to(PROJECT_ROOT)}: defines {sorted(clash)}"
            )
    assert not violators, (
        "These modules redefine the forbidden-token matcher. Import "
        "from harness.repro instead of re-implementing:\n  "
        + "\n  ".join(violators)
    )


# ---------------------------------------------------------------------------
# Runtime-tamper test — frozen-dataclass bypass still gets caught
# ---------------------------------------------------------------------------


def test_input_firewall_catches_post_construction_tamper():
    """InstanceView is a frozen dataclass; ``object.__setattr__``
    bypasses both the frozen guard and __post_init__'s field-name
    validator. The repro firewall reads field VALUES at call time,
    so a tampered InstanceView with a forbidden token in
    problem_statement is still caught.

    This is the structural argument for why field-VALUE validation
    (here) and field-NAME validation (in views.py) are both
    necessary — neither alone catches all leak vectors."""
    view = _make_view(problem_statement="benign issue text")
    rfs = _make_ranked_files(["src/foo.py"])
    # Sanity: clean view passes.
    _validate_input_firewall(view, rfs)

    # Tamper with the frozen view by going around __setattr__.
    object.__setattr__(view, "problem_statement", '"fail_to_pass" leak')

    with pytest.raises(ReproGeneratorError, match=r"problem_statement"):
        _validate_input_firewall(view, rfs)


def test_input_firewall_catches_tampered_test_directive():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    object.__setattr__(view.test_directives, "dirs", ("tests/test_patch_x.py",))
    # Note: the dirs path "tests/test_patch_x.py" — `test_patch` is
    # followed by `_x` (word char), so word-boundary doesn't fire.
    # Tamper with a real boundary instead:
    object.__setattr__(view.test_directives, "dirs", ("tests/test_patch",))
    with pytest.raises(ReproGeneratorError, match=r"test_directives"):
        _validate_input_firewall(view, rfs)


def test_input_firewall_catches_tampered_ranked_file():
    """Mutating a frozen RankedFile via object.__setattr__ doesn't
    escape the firewall — it reads the field value at call time."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    object.__setattr__(rfs[0], "file_path", "src/test_patch.py")
    with pytest.raises(ReproGeneratorError, match=r"ranked_files"):
        _validate_input_firewall(view, rfs)
