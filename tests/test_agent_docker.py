"""Tests for DockerShellExecutor — uses a fake subprocess.run so the
real test suite stays Docker-independent.

The integration test (which actually starts a container) lives in
``tests/test_agent_docker_integration.py`` and only runs when an env
var ``MH_RUN_DOCKER_INTEGRATION_TESTS=1`` is set.
"""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness_plus.agent_docker import (
    DockerShellExecutor,
    ExecResult,
    _swebench_hub_image,
)


class TestSWEBenchHubImage(unittest.TestCase):
    def test_double_underscore_to_1776(self):
        self.assertEqual(
            _swebench_hub_image("sympy__sympy-22914"),
            "swebench/sweb.eval.x86_64.sympy_1776_sympy-22914:latest",
        )

    def test_django_naming(self):
        self.assertEqual(
            _swebench_hub_image("django__django-11451"),
            "swebench/sweb.eval.x86_64.django_1776_django-11451:latest",
        )


def _fake_run(stdout="", stderr="", returncode=0):
    """Build a fake subprocess.run side-effect for unit tests."""
    def side(cmd, **kwargs):
        m = mock.MagicMock()
        m.stdout = stdout
        m.stderr = stderr
        m.returncode = returncode
        return m
    return side


class TestDockerShellExecutorLifecycle(unittest.TestCase):
    def test_start_records_container_id(self):
        sh = DockerShellExecutor("sympy__sympy-22914")
        with mock.patch("subprocess.run") as run:
            run.return_value.stdout = "abc123\n"
            run.return_value.returncode = 0
            sh.start()
            self.assertEqual(sh.container_id, "abc123")
            # Subsequent start() is idempotent.
            sh.start()
            self.assertEqual(run.call_count, 1)

    def test_run_truncates_long_stdout(self):
        sh = DockerShellExecutor("x__x-1", max_observation_chars=10)
        sh.container_id = "fake"
        with mock.patch("subprocess.run") as run:
            run.return_value.stdout = "a" * 100
            run.return_value.stderr = ""
            run.return_value.returncode = 0
            r = sh.run("echo hi")
            self.assertTrue(r.truncated)
            self.assertLess(len(r.stdout), 30)
            self.assertIn("…[truncated]", r.stdout)

    def test_run_handles_timeout(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch("subprocess.run") as run:
            import subprocess
            run.side_effect = subprocess.TimeoutExpired("docker", 1.0)
            r = sh.run("sleep 60", timeout_s=1.0)
            self.assertEqual(r.exit_code, 124)
            self.assertIn("timeout", r.stderr.lower())

    def test_cleanup_stops_container(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch("subprocess.run") as run:
            run.return_value.returncode = 0
            sh.cleanup()
            self.assertIsNone(sh.container_id)

    def test_context_manager_cleans_on_exception(self):
        sh = DockerShellExecutor("x__x-1")
        with mock.patch("subprocess.run") as run:
            run.return_value.stdout = "cid\n"
            run.return_value.returncode = 0
            try:
                with sh:
                    self.assertEqual(sh.container_id, "cid")
                    raise ValueError("boom")
            except ValueError:
                pass
            # cleanup() was called.
            self.assertIsNone(sh.container_id)

    def test_run_command_contains_workdir_and_image(self):
        # Verify the docker run invocation matches expected args.
        sh = DockerShellExecutor("x__x-1", memory_gb=2.0, cpus=1.5)
        with mock.patch("subprocess.run") as run:
            run.return_value.stdout = "cid\n"
            run.return_value.returncode = 0
            sh.start()
            args = run.call_args.args[0]
            self.assertIn("docker", args[0])
            self.assertIn("run", args)
            self.assertIn("--memory", args)
            self.assertIn("2.0g", args)
            self.assertIn("--cpus", args)
            self.assertIn("1.5", args)
            # network none by default
            self.assertIn("--network", args)
            self.assertIn("none", args)


class TestRunTests(unittest.TestCase):
    def test_run_tests_no_targets_runs_full_suite(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("ok", "", 0, 0.1)
            sh.run_tests([])
            cmd_arg = r.call_args.args[0]
            self.assertIn("pytest", cmd_arg)
            # No specific targets passed.
            self.assertNotIn("::", cmd_arg)

    def test_run_tests_with_targets_quotes_them(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("ok", "", 0, 0.1)
            sh.run_tests(["tests/foo.py::test_a", "tests/bar.py"])
            cmd_arg = r.call_args.args[0]
            self.assertIn("tests/foo.py::test_a", cmd_arg)
            self.assertIn("tests/bar.py", cmd_arg)


class TestFileTools(unittest.TestCase):
    def test_read_file_range_clips_and_numbers_lines(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("    5\tline", "", 0, 0.1)
            out = sh.read_file_range("pkg/foo.py", 5, 500, max_lines=2)
            cmd_arg = r.call_args.args[0]
            self.assertIn("awk", cmd_arg)
            self.assertIn("NR>=5", cmd_arg)
            self.assertIn("NR<=6", cmd_arg)
            self.assertIn("pkg/foo.py", cmd_arg)
            self.assertIn("range clipped", out.stdout)

    def test_search_text_uses_fixed_recursive_grep(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("pkg/foo.py:1:needle", "", 0, 0.1)
            sh.search_text("needle", "pkg", max_matches=3)
            cmd_arg = r.call_args.args[0]
            self.assertIn("grep -RInIF", cmd_arg)
            self.assertIn("--exclude-dir=.git", cmd_arg)
            self.assertIn("head -n 3", cmd_arg)

    def test_replace_text_runs_encoded_python_rewrite(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("replaced 1 occurrence(s)", "", 0, 0.1)
            sh.replace_text("pkg/foo.py", "old text", "new text")
            cmd_arg = r.call_args.args[0]
            self.assertIn("python -c", cmd_arg)
            self.assertIn("/testbed/pkg/foo.py", cmd_arg)
            self.assertIn("b2xkIHRleHQ=", cmd_arg)
            self.assertIn("bmV3IHRleHQ=", cmd_arg)

    def test_get_diff_can_exclude_tests(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("diff", "", 0, 0.1)
            out = sh.get_diff(exclude_tests=True)
            cmd_arg = r.call_args.args[0]
            self.assertEqual(out, "diff")
            self.assertIn("git diff -- .", cmd_arg)
            self.assertIn(":(exclude)*/tests/*", cmd_arg)

    def test_run_python_uses_encoded_payload(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("ok", "", 0, 0.1)
            sh.run_python("print('secret')")
            cmd_arg = r.call_args.args[0]
            self.assertIn("python -c", cmd_arg)
            self.assertNotIn("print('secret')", cmd_arg)
            self.assertEqual(r.call_args.kwargs["timeout_s"], 60.0)

    def test_run_reproduction_persists_encoded_script(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("ok", "", 0, 0.1)
            out = sh.run_reproduction("print('secret')", timeout_s=7)
            cmd_arg = r.call_args.args[0]
            self.assertIn("/tmp/mh_reproductions/repro_", cmd_arg)
            self.assertNotIn("print('secret')", cmd_arg)
            self.assertIn("[reproduction:", out.stdout)
            self.assertEqual(r.call_args.kwargs["timeout_s"], 7)

    def test_lsp_definition_tool_runs_ast_index(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("pkg/foo.py:1:1: function def foo()", "", 0, 0.1)
            sh.go_to_definition("foo", "pkg", max_results=2)
            cmd_arg = r.call_args.args[0]
            self.assertIn("ast", cmd_arg)
            self.assertIn("/testbed", cmd_arg)
            self.assertIn("foo", cmd_arg)
            self.assertIn(" 2", cmd_arg)

    def test_lsp_references_tool_runs_token_index(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("pkg/foo.py:3:5: foo()", "", 0, 0.1)
            sh.find_references("foo", "pkg", max_matches=4)
            cmd_arg = r.call_args.args[0]
            self.assertIn("tokenize", cmd_arg)
            self.assertIn("/testbed", cmd_arg)
            self.assertIn("foo", cmd_arg)
            self.assertIn(" 4", cmd_arg)

    def test_checkpoint_and_restore_commands(self):
        sh = DockerShellExecutor("x__x-1")
        sh.container_id = "fake"
        with mock.patch.object(sh, "run") as r:
            r.return_value = ExecResult("/tmp/mh_checkpoint_a.patch", "", 0, 0.1)
            sh.create_checkpoint("a")
            create_cmd = r.call_args.args[0]
            self.assertIn("git diff --binary", create_cmd)
            self.assertIn("mh_checkpoint_a.patch", create_cmd)

            r.return_value = ExecResult("", "", 0, 0.1)
            sh.restore_checkpoint("/tmp/mh_checkpoint_a.patch")
            restore_cmd = r.call_args.args[0]
            self.assertIn("git reset --hard HEAD", restore_cmd)
            self.assertIn("git apply", restore_cmd)

            sh.reset_to_clean()
            reset_cmd = r.call_args.args[0]
            self.assertIn("git clean -fd", reset_cmd)


if __name__ == "__main__":
    unittest.main()
