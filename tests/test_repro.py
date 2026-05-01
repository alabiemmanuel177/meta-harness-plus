"""Phase 2 commit-17a tests for harness.repro and Sandbox.run_repro_test.

Covers:
  - ReproTestCase schema validation (frozen dataclass + FORBIDDEN_TOKENS).
  - Input-layer firewall (§8.4): planted forbidden tokens in
    problem_statement / test_directives / ranked_files block at
    prompt-build time.
  - AST snippet extraction: function signatures + class headers
    only, no bodies.
  - Output-layer guard is INFORMATIONAL (returns hits but doesn't
    raise; sandbox method logs WARNING and continues).
  - End-to-end generate_repro_attempt with mocked complete_chat +
    sandbox.

Doesn't actually call the LLM or start docker — all external
boundaries are mocked. The 17b retry loop and 17d coverage tests
are separate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from harness.localization_signals import RankedFile
from harness.repro import (
    ReproGeneratorError,
    ReproStatus,
    ReproTestCase,
    _ast_snippet_for_source,
    _build_user_prompt,
    _parse_response,
    _validate_input_firewall,
    check_output_substring_hits,
    generate_repro_attempt,
)
from harness.views import (
    FileSummary,
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_view(problem_statement: str = "Bug: foo() returns wrong value when bar is None") -> InstanceView:
    return InstanceView(
        instance_id="example__example-1",
        repo="example/example",
        base_commit="abcdef0123",
        problem_statement=problem_statement,
        repo_skeleton=RepoSkeleton(
            repo="example/example",
            base_commit="abcdef0123",
            files=(),
        ),
        test_directives=TestDirectives(dirs=("tests",), source="discovery:1"),
    )


def _make_ranked_files(paths: list[str]) -> list[RankedFile]:
    return [
        RankedFile(
            file_path=p,
            final_score=1.0 - i * 0.05,
            rationale="test",
            upstream_signals=("bm25",),
            upstream_best_rank={"bm25": i + 1},
        )
        for i, p in enumerate(paths)
    ]


@dataclass
class _FakeReadResult:
    exit_code: int
    stdout: str


class _FakeSandbox:
    """Minimal sandbox stand-in for repro tests.

    Maps file_path -> source string. read_file returns ExecResult-shaped
    objects with .exit_code and .stdout.
    """

    def __init__(self, file_contents: dict[str, str]):
        self._files = dict(file_contents)
        self._view = _make_view()
        self.run_calls: list[tuple] = []

    def read_file(self, rel_path: str, max_chars: int | None = None):
        if rel_path not in self._files:
            return _FakeReadResult(exit_code=2, stdout="")
        text = self._files[rel_path]
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        return _FakeReadResult(exit_code=0, stdout=text)


# ---------------------------------------------------------------------------
# 1. ReproTestCase schema
# ---------------------------------------------------------------------------


def test_repro_test_case_required_fields():
    rtc = ReproTestCase(
        instance_id="x",
        test_filename="tests/test_x.py",
        test_code="def test_x(): pass",
        target_test_id="tests/test_x.py::test_x",
        rationale="r",
        generator_model="deepseek-chat",
        generator_input_tokens=10,
        generator_output_tokens=5,
        attempt_index=0,
    )
    assert rtc.instance_id == "x"
    assert rtc.attempt_index == 0


def test_repro_test_case_rejects_empty_required():
    with pytest.raises(ValueError):
        ReproTestCase(
            instance_id="",
            test_filename="tests/x.py",
            test_code="pass",
            target_test_id="tests/x.py::test",
            rationale="r",
            generator_model="m",
            generator_input_tokens=0,
            generator_output_tokens=0,
            attempt_index=0,
        )


def test_repro_test_case_rejects_negative_attempt():
    with pytest.raises(ValueError):
        ReproTestCase(
            instance_id="x",
            test_filename="t.py",
            test_code="pass",
            target_test_id="t::n",
            rationale="r",
            generator_model="m",
            generator_input_tokens=0,
            generator_output_tokens=0,
            attempt_index=-1,
        )


def test_repro_status_constants():
    assert ReproStatus.USABLE == "usable"
    assert ReproStatus.NO_REPRO == "no_repro"
    assert ReproStatus.NO_REPRO_BUDGET == "no_repro_budget"


# ---------------------------------------------------------------------------
# 2. Input-layer firewall (§8.4)
# ---------------------------------------------------------------------------


def test_input_firewall_blocks_problem_statement_token():
    """Planted FAIL_TO_PASS in problem_statement → ReproGeneratorError."""
    view = _make_view(problem_statement="The FAIL_TO_PASS test is failing.")
    rfs = _make_ranked_files(["src/foo.py"])
    with pytest.raises(ReproGeneratorError, match=r"input-firewall.*problem_statement"):
        _validate_input_firewall(view, rfs)


def test_input_firewall_blocks_dataset_field_in_problem_statement():
    """The realistic leak: an issue text that quotes the dataset's
    fail_to_pass field. The word-boundary matcher catches this."""
    view = _make_view(
        problem_statement="The 'fail_to_pass' tests in the dataset are: ..."
    )
    rfs = _make_ranked_files(["src/foo.py"])
    with pytest.raises(ReproGeneratorError, match="fail_to_pass"):
        _validate_input_firewall(view, rfs)


def test_input_firewall_does_not_block_substring_in_word():
    """Word-boundary: 'failtopass' embedded in a longer identifier
    (no surrounding word boundary) is NOT a leak signal — likely a
    coincidental variable name. Per §8.4 calibration."""
    view = _make_view(problem_statement="myFailToPassFlag was set somewhere")
    rfs = _make_ranked_files(["src/foo.py"])
    # No raise — word boundary check passes.
    _validate_input_firewall(view, rfs)


def test_input_firewall_blocks_ranked_file_path_token():
    """Forbidden token as a discrete path component is blocked.
    Word boundaries treat `/` and `.` as non-word, so a path like
    ``src/test_patch.py`` matches `\\btest_patch\\b`."""
    view = _make_view()
    rfs = _make_ranked_files(["src/test_patch.py"])
    with pytest.raises(ReproGeneratorError, match=r"ranked_files"):
        _validate_input_firewall(view, rfs)


def test_input_firewall_passes_clean_inputs():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py", "src/bar.py"])
    _validate_input_firewall(view, rfs)  # no raise


def test_input_firewall_blocks_test_dir_token():
    bad_view = InstanceView(
        instance_id="i",
        repo="r/r",
        base_commit="c",
        problem_statement="ok",
        repo_skeleton=RepoSkeleton(repo="r/r", base_commit="c"),
        test_directives=TestDirectives(dirs=("tests/pass_to_pass",), source="discovery:1"),
    )
    with pytest.raises(ReproGeneratorError, match=r"test_directives"):
        _validate_input_firewall(bad_view, _make_ranked_files(["src/foo.py"]))


# ---------------------------------------------------------------------------
# 3. AST snippet extraction
# ---------------------------------------------------------------------------


def test_ast_snippet_extracts_function_signatures():
    src = """
def add(a: int, b: int = 0) -> int:
    return a + b

def helper():
    pass
"""
    snippet = _ast_snippet_for_source(src)
    assert "def add(a: int, b: int=0) -> int: ..." in snippet
    assert "def helper(): ..." in snippet
    # Bodies are NOT in the snippet
    assert "return a + b" not in snippet


def test_ast_snippet_extracts_class_with_methods():
    src = """
class Foo(Bar):
    def __init__(self, x: int):
        self.x = x

    def method(self) -> str:
        return str(self.x)
"""
    snippet = _ast_snippet_for_source(src)
    assert "class Foo(Bar):" in snippet
    assert "def __init__(self, x: int): ..." in snippet
    assert "def method(self) -> str: ..." in snippet
    # Method body absent
    assert "return str" not in snippet
    assert "self.x = x" not in snippet


def test_ast_snippet_handles_async_functions():
    src = """
async def fetch(url: str) -> str:
    return ""
"""
    snippet = _ast_snippet_for_source(src)
    assert "async def fetch(url: str) -> str: ..." in snippet


def test_ast_snippet_truncation():
    huge = "\n".join(f"def fn_{i}(x): return x" for i in range(2000))
    snippet = _ast_snippet_for_source(huge, max_chars=1000)
    assert len(snippet) <= 1100  # cap + some padding for the truncation marker
    assert "(snippet truncated" in snippet


def test_ast_snippet_handles_unparseable():
    snippet = _ast_snippet_for_source("def broken(:")
    assert "could not parse" in snippet


def test_ast_snippet_empty_module():
    snippet = _ast_snippet_for_source("# only comments\n")
    assert "no top-level" in snippet


# ---------------------------------------------------------------------------
# 4. Output-layer guard is INFORMATIONAL
# ---------------------------------------------------------------------------


def test_output_guard_returns_hits_does_not_raise():
    """Word-boundary catches a real leak pattern in test code: the
    generator wrote the dataset field name in a comment / docstring /
    string literal. (A function NAME like ``test_pass_to_pass`` would
    NOT trip — that's an internal underscored identifier — but a
    string ``"pass_to_pass"`` does.)"""
    hits = check_output_substring_hits(
        test_code='# checks the "pass_to_pass" field of the row',
        target_test_id="tests/test_x.py::test_x",
    )
    assert "pass_to_pass" in hits


def test_output_guard_clean_inputs():
    hits = check_output_substring_hits(
        test_code="def test_resolves(): pass",
        target_test_id="tests/test_x.py::test_resolves",
    )
    # 'resolved' is in FORBIDDEN_TOKENS — make sure 'resolves' doesn't trip
    # (the substring rule normalizes underscores; 'resolves' != 'resolved'
    # in normalized form so it should pass).
    assert "resolved" not in hits


def test_output_guard_word_boundary_match_on_field_name():
    """Word-boundary match catches direct dataset field-name leaks."""
    hits = check_output_substring_hits(
        test_code='data["fail_to_pass"] = True',
        target_test_id="x::y",
    )
    assert "fail_to_pass" in hits


def test_output_guard_no_false_positive_on_test_patches():
    """Per §8.4 calibration: 'test_patches_distortion' is a real
    matplotlib test name, not a leak. Word-boundary should NOT match."""
    hits = check_output_substring_hits(
        test_code="def test_patches_distortion(): pass",
        target_test_id="t.py::test_patches_distortion",
    )
    assert "test_patch" not in hits


# ---------------------------------------------------------------------------
# 5. Prompt build
# ---------------------------------------------------------------------------


def test_user_prompt_contains_issue_and_files():
    view = _make_view(problem_statement="The bug is here.")
    rfs = _make_ranked_files(["src/foo.py"])
    snippets = [("src/foo.py", "def foo(): ...")]
    prompt = _build_user_prompt(view, rfs, snippets, widen=False)
    assert "The bug is here." in prompt
    assert "src/foo.py" in prompt
    assert "def foo(): ..." in prompt
    assert "Widened" not in prompt


def test_user_prompt_widen_marker():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    snippets = [("src/foo.py", "def foo(): ...")]
    prompt = _build_user_prompt(view, rfs, snippets, widen=True)
    assert "Widened from top-10 to top-30" in prompt


# ---------------------------------------------------------------------------
# 6. Response parsing
# ---------------------------------------------------------------------------


def test_parse_response_clean_json():
    text = '{"test_filename":"t.py","test_code":"x","target_test_id":"t::n","rationale":"r"}'
    data = _parse_response(text)
    assert data["test_filename"] == "t.py"


def test_parse_response_strips_fence():
    text = '```json\n{"test_filename":"t.py","test_code":"x","target_test_id":"t::n","rationale":"r"}\n```'
    data = _parse_response(text)
    assert data["test_filename"] == "t.py"


def test_parse_response_missing_keys_raises():
    text = '{"test_filename":"t.py"}'
    with pytest.raises(ReproGeneratorError, match="missing keys"):
        _parse_response(text)


def test_parse_response_invalid_json_raises():
    with pytest.raises(ReproGeneratorError, match="not valid JSON"):
        _parse_response("not json at all")


def test_parse_response_empty_field_raises():
    text = '{"test_filename":"","test_code":"x","target_test_id":"t::n","rationale":"r"}'
    with pytest.raises(ReproGeneratorError, match="non-empty string"):
        _parse_response(text)


# ---------------------------------------------------------------------------
# 7. End-to-end generate_repro_attempt (mocked LLM + sandbox)
# ---------------------------------------------------------------------------


def test_generate_repro_attempt_happy_path():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py", "src/bar.py"])
    sandbox = _FakeSandbox({
        "src/foo.py": "def foo(): pass",
        "src/bar.py": "class Bar: pass",
    })

    fake_chat = MagicMock(
        text='{"test_filename":"tests/test_repro_v10_foo.py",'
             '"test_code":"def test_foo_bar(): assert foo() is None",'
             '"target_test_id":"tests/test_repro_v10_foo.py::test_foo_bar",'
             '"rationale":"reproduces"}',
        model="deepseek-chat",
        input_tokens=100,
        output_tokens=50,
    )

    with patch("harness.llm.clients.complete_chat", return_value=fake_chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        rtc = generate_repro_attempt(
            view=view,
            ranked_files=rfs,
            sandbox=sandbox,
            attempt_index=0,
            widen=False,
        )

    assert rtc.instance_id == view.instance_id
    assert rtc.test_filename.startswith("tests/")
    assert rtc.attempt_index == 0
    assert rtc.generator_model == "deepseek-chat"
    assert rtc.generator_input_tokens == 100


def test_generate_repro_attempt_blocks_planted_token():
    """If the orchestrator somehow gets a poisoned InstanceView through,
    the input-layer firewall STOPS the generator from sending the
    prompt. Real protection per §8.4."""
    view = _make_view(problem_statement="see FAIL_TO_PASS list")
    rfs = _make_ranked_files(["src/foo.py"])
    sandbox = _FakeSandbox({"src/foo.py": "pass"})

    # No mock needed — we never reach complete_chat.
    with pytest.raises(ReproGeneratorError, match="input-firewall"):
        generate_repro_attempt(
            view=view, ranked_files=rfs, sandbox=sandbox,
        )


def test_generate_repro_attempt_logs_output_warning(caplog):
    """When the generator output hits the substring scan, a WARNING is
    logged (not raised, not retried). Per §8.4."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sandbox = _FakeSandbox({"src/foo.py": "def foo(): pass"})

    fake_chat = MagicMock(
        text='{"test_filename":"tests/test_repro_v10.py",'
             '"test_code":"# refers to \\"fail_to_pass\\" in docstring",'
             '"target_test_id":"tests/test_repro_v10.py::test_x",'
             '"rationale":"r"}',
        model="deepseek-chat",
        input_tokens=10, output_tokens=5,
    )
    with patch("harness.llm.clients.complete_chat", return_value=fake_chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"), \
         caplog.at_level(logging.WARNING, logger="harness.repro"):
        rtc = generate_repro_attempt(view=view, ranked_files=rfs, sandbox=sandbox)

    assert rtc is not None  # NOT raised
    assert any("informational substring match" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 8. Sandbox.run_repro_test — informational substring scan
# ---------------------------------------------------------------------------


class _MockExec:
    def __init__(self):
        self.run_calls: list[tuple] = []

    def run(self, cmd: str, timeout_s: float):
        self.run_calls.append((cmd, timeout_s))
        from meta_harness_plus.agent_docker import ExecResult
        return ExecResult(stdout="passed", stderr="", exit_code=0, elapsed_s=0.1, truncated=False)


def test_sandbox_run_repro_test_logs_warning_does_not_block(caplog):
    from harness.sandbox import Sandbox
    sb = Sandbox.__new__(Sandbox)  # bypass __init__ (no docker)
    sb._view = _make_view()
    sb._exec = _MockExec()
    sb._dirs = ("tests",)

    with caplog.at_level(logging.WARNING, logger="harness.sandbox"):
        res = sb.run_repro_test(
            test_id="tests/test_x.py::test_main_method",
            test_code='# generator wrote "fail_to_pass" in a comment',
            timeout_s=30.0,
        )
    assert res.exit_code == 0
    assert any("informational substring match" in r.message for r in caplog.records)
    assert sb._exec.run_calls  # the run actually fired despite the warning


def test_sandbox_run_repro_test_clean_inputs(caplog):
    from harness.sandbox import Sandbox
    sb = Sandbox.__new__(Sandbox)
    sb._view = _make_view()
    sb._exec = _MockExec()
    sb._dirs = ("tests",)

    with caplog.at_level(logging.WARNING, logger="harness.sandbox"):
        sb.run_repro_test(
            test_id="tests/test_x.py::test_user_creates_account",
            test_code="def test_x(): pass",
        )
    # No warning for clean inputs
    assert not any("informational substring match" in r.message for r in caplog.records)
