"""DockerShellExecutor — per-task Docker container backing the agent's
shell. Built on the SWE-bench pre-built instance images (already
pulled to local Docker) so each task starts with the repo cloned at
``/testbed`` and dependencies installed.

Design choices:

- One container per (instance_id, task run). Started lazily on first
  ``run`` call; cleaned up by the caller (typically via the
  ``with DockerShellExecutor(...) as sh:`` context manager).
- ``--rm`` on the container so a forgotten cleanup just leaves a
  stopped+removed container after the host process exits.
- ``--network none`` so the agent can't pip install / hit the
  internet — keeps tasks deterministic and prevents unrelated bugs.
- Hard ``--memory`` and ``--cpus`` caps so a single runaway task
  can't eat the host.
- Each ``run`` honours a per-call ``timeout_s`` via subprocess.

The agent loop (next file) calls into this executor via four
specific tools: ``read_file``, ``write_file``, ``run_tests``, and
``list_files``. Raw shell exec is exposed for the agent's `bash`
escape hatch but shouldn't be the primary mode.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def _swebench_hub_image(instance_id: str) -> str:
    """Convert ``sympy__sympy-22914`` to the Docker Hub image name
    ``swebench/sweb.eval.x86_64.sympy_1776_sympy-22914:latest``."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    elapsed_s: float
    truncated: bool = False


class DockerShellExecutor:
    """Per-instance long-lived Docker container.

    Usage::

        with DockerShellExecutor("sympy__sympy-22914") as sh:
            sh.start()                              # explicit; or implicit on first run
            r = sh.run("python -c 'print(1+1)'", timeout_s=10)
            sh.write_file("foo.txt", "hi")
            sh.read_file("foo.txt")
            sh.run_tests(["sympy/printing/tests/test_pycode.py"])
    """

    DEFAULT_WORKDIR = "/testbed"

    def __init__(
        self,
        instance_id: str,
        *,
        image: str | None = None,
        memory_gb: float = 4.0,
        cpus: float = 2.0,
        workdir: str = DEFAULT_WORKDIR,
        max_observation_chars: int = 32_768,
        no_network: bool = True,
        ensure_pytest: bool = False,
        entrypoint_override: str | None = None,
    ):
        self.instance_id = instance_id
        self.image = image or _swebench_hub_image(instance_id)
        self.memory_gb = memory_gb
        self.cpus = cpus
        self.workdir = workdir
        self.max_observation_chars = max_observation_chars
        self.no_network = no_network
        self.ensure_pytest = ensure_pytest
        # Some image families (notably SWE-bench Pro at jefzda/sweap-images)
        # ship a non-empty ENTRYPOINT (e.g. ``[/bin/bash]``) which would
        # consume our ``sleep infinity`` cmd as a script name and exit.
        # Setting an explicit entrypoint here clears that.
        self.entrypoint_override = entrypoint_override
        self.container_id: str | None = None
        # Tally for cost / budget guards.
        self.total_exec_calls = 0
        self.total_wall_s = 0.0

    # ----- lifecycle -----

    def start(self) -> None:
        if self.container_id:
            return
        cmd = [
            "docker", "run", "-d", "--rm",
            "--memory", f"{self.memory_gb}g",
            "--cpus", str(self.cpus),
            "--workdir", self.workdir,
        ]
        if self.entrypoint_override is not None:
            cmd.extend(["--entrypoint", self.entrypoint_override])
        if self.no_network:
            cmd.extend(["--network", "none"])
        cmd.append(self.image)
        if self.entrypoint_override is None:
            # Default: entrypoint is empty → cmd ``sleep infinity`` runs sleep.
            cmd.extend(["sleep", "infinity"])
        else:
            # Entrypoint explicitly set (e.g. to ``sleep`` for Pro images);
            # cmd args are appended to it. ``sleep infinity`` would become
            # ``sleep sleep infinity`` and exit immediately, so we pass
            # only the duration here.
            cmd.append("infinity")
        try:
            out = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"docker run failed for {self.instance_id} ({self.image}): "
                f"{e.stderr.strip()[:300]}"
            ) from e
        self.container_id = out.stdout.strip()
        if self.ensure_pytest:
            self._ensure_pytest()

    def _ensure_pytest(self) -> None:
        """SWE-bench testbed envs often lack pytest. Install it once at
        startup so ``run_tests`` can actually report failures back to
        the agent. Silent if it's already there."""
        check = self.run("python -m pytest --version", timeout_s=10)
        if check.exit_code == 0:
            return
        # Install quietly. If --network none is set, this will fail —
        # but we keep going; tests just won't work.
        self.run("python -m pip install -q pytest 2>&1 | tail -3", timeout_s=120)

    def cleanup(self) -> None:
        if self.container_id:
            try:
                subprocess.run(
                    ["docker", "stop", "-t", "1", self.container_id],
                    capture_output=True, timeout=30,
                )
            except Exception:
                pass
            self.container_id = None

    def __enter__(self) -> "DockerShellExecutor":
        self.start()
        return self

    def __exit__(self, *_a) -> None:
        self.cleanup()

    # ----- core exec -----

    def run(self, command: str, timeout_s: float = 30.0) -> ExecResult:
        """Execute a shell command inside the container.

        Returns ``ExecResult``; truncates stdout to
        ``self.max_observation_chars`` so the agent can't OOM on huge
        outputs.
        """
        if not self.container_id:
            self.start()
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["docker", "exec", self.container_id, "/bin/bash", "-lc", command],
                capture_output=True, text=True, timeout=timeout_s,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="replace")
            stderr = ((e.stderr or "") if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="replace"))
            stderr = f"timeout after {timeout_s}s\n{stderr}"
            code = 124
        except Exception as exc:
            stdout, stderr, code = "", f"exec error: {type(exc).__name__}: {exc}", 1
        elapsed = time.perf_counter() - t0
        self.total_exec_calls += 1
        self.total_wall_s += elapsed
        truncated = False
        if len(stdout) > self.max_observation_chars:
            stdout = stdout[: self.max_observation_chars] + "\n…[truncated]"
            truncated = True
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=code,
                          elapsed_s=elapsed, truncated=truncated)

    # ----- file ops -----

    def write_file(self, rel_path: str, content: str) -> ExecResult:
        """Write ``content`` to ``<workdir>/<rel_path>`` via ``docker cp``.

        Why ``docker cp`` over ``echo > file``? Heredocs corrupt on
        embedded newlines / shell metacharacters; ``docker cp`` is
        binary-safe.
        """
        if not self.container_id:
            self.start()
        # Ensure parent dir exists inside the container.
        rel_path = rel_path.lstrip("/")
        parent = "/".join(rel_path.split("/")[:-1])
        if parent:
            mk = self.run(f"mkdir -p {shlex.quote(self.workdir + '/' + parent)}",
                          timeout_s=5)
            if mk.exit_code != 0:
                return mk

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            data = content.encode("utf-8")
            ti = tarfile.TarInfo(name=rel_path)
            ti.size = len(data)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
        buf.seek(0)
        t0 = time.perf_counter()
        try:
            subprocess.run(
                ["docker", "cp", "-", f"{self.container_id}:{self.workdir}"],
                input=buf.getvalue(), check=True, timeout=60,
            )
            return ExecResult("", "", 0, time.perf_counter() - t0)
        except subprocess.CalledProcessError as e:
            return ExecResult("", f"docker cp failed: {e}", 1,
                              time.perf_counter() - t0)

    def read_file(self, rel_path: str, max_chars: int | None = None) -> ExecResult:
        """Read ``<workdir>/<rel_path>`` via ``cat``. stdout = file content."""
        if not self.container_id:
            self.start()
        cap = max_chars or self.max_observation_chars
        # Use head -c to limit upfront — saves time on huge files.
        # tail -c handles the remainder if we want a smarter slice later.
        rel = shlex.quote(rel_path.lstrip("/"))
        cmd = f"head -c {cap} {self.workdir}/{rel}"
        return self.run(cmd, timeout_s=15)

    def read_file_range(
        self,
        rel_path: str,
        start_line: int,
        end_line: int,
        *,
        max_lines: int = 240,
    ) -> ExecResult:
        """Read a bounded line range with stable line numbers."""
        if not self.container_id:
            self.start()
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        clipped = False
        if end - start + 1 > max_lines:
            end = start + max_lines - 1
            clipped = True
        rel = shlex.quote(rel_path.lstrip("/"))
        cmd = (
            "awk "
            + shlex.quote(
                f'NR>={start} && NR<={end} {{printf "%6d\\t%s\\n", NR, $0}}'
            )
            + f" {self.workdir}/{rel}"
        )
        result = self.run(cmd, timeout_s=15)
        if clipped and result.exit_code == 0:
            result.stdout += (
                f"\n...[range clipped to {max_lines} lines; request a later "
                "range if needed]"
            )
        return result

    def search_text(
        self,
        query: str,
        rel_path: str = ".",
        *,
        max_matches: int = 80,
    ) -> ExecResult:
        """Fixed-string recursive search with line numbers."""
        if not self.container_id:
            self.start()
        rel = shlex.quote(rel_path.lstrip("/"))
        target = f"{self.workdir}/{rel}" if rel != "." else self.workdir
        limit = max(1, min(int(max_matches), 200))
        cmd = (
            "grep -RInIF --exclude-dir=.git -- "
            f"{shlex.quote(query)} {target} | head -n {limit}"
        )
        return self.run(cmd, timeout_s=20)

    def replace_text(
        self,
        rel_path: str,
        old: str,
        new: str,
        *,
        expected_count: int = 1,
    ) -> ExecResult:
        """Replace exact text in a file without requiring a full-file rewrite."""
        if not self.container_id:
            self.start()
        rel_path = rel_path.lstrip("/")
        path = f"{self.workdir}/{rel_path}"
        old_b64 = base64.b64encode(old.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new.encode("utf-8")).decode("ascii")
        expected = max(0, int(expected_count))
        script = (
            "import base64, pathlib, sys\n"
            "path = pathlib.Path(sys.argv[1])\n"
            "old = base64.b64decode(sys.argv[2]).decode('utf-8')\n"
            "new = base64.b64decode(sys.argv[3]).decode('utf-8')\n"
            "expected = int(sys.argv[4])\n"
            "text = path.read_text()\n"
            "count = text.count(old)\n"
            "if count == 0:\n"
            "    print('old text not found', file=sys.stderr)\n"
            "    sys.exit(2)\n"
            "if expected and count != expected:\n"
            "    print(f'expected {expected} occurrence(s), found {count}', file=sys.stderr)\n"
            "    sys.exit(3)\n"
            "n = expected or count\n"
            "path.write_text(text.replace(old, new, n))\n"
            "print(f'replaced {n} occurrence(s)')\n"
        )
        cmd = (
            f"python -c {shlex.quote(script)} "
            f"{shlex.quote(path)} {shlex.quote(old_b64)} "
            f"{shlex.quote(new_b64)} {expected}"
        )
        return self.run(cmd, timeout_s=20)

    def run_reproduction(
        self,
        code: str,
        *,
        timeout_s: float = 60.0,
    ) -> ExecResult:
        """Persist and run a focused reproduction script inside the container.

        The script is written outside the git checkout under
        ``/tmp/mh_reproductions`` so it can be audited from trajectory logs
        without polluting the submitted patch.
        """
        digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
        repro_path = f"/tmp/mh_reproductions/repro_{digest}.py"
        payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        writer = (
            "import base64, pathlib, sys\n"
            "path = pathlib.Path(sys.argv[1])\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "code = base64.b64decode(sys.argv[2]).decode('utf-8')\n"
            "path.write_text(code)\n"
        )
        cmd = (
            f"python -c {shlex.quote(writer)} "
            f"{shlex.quote(repro_path)} {shlex.quote(payload)} && "
            f"python {shlex.quote(repro_path)}"
        )
        result = self.run(cmd, timeout_s=timeout_s)
        result.stdout = f"[reproduction: {repro_path}]\n{result.stdout}"
        return result

    def run_python(self, code: str, *, timeout_s: float = 60.0) -> ExecResult:
        """Run a short Python reproduction in the testbed environment."""
        payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        script = (
            "import base64, sys\n"
            "code = base64.b64decode(sys.argv[1]).decode('utf-8')\n"
            "exec(compile(code, '<agent-repro>', 'exec'))\n"
        )
        cmd = f"python -c {shlex.quote(script)} {shlex.quote(payload)}"
        return self.run(cmd, timeout_s=timeout_s)

    def go_to_definition(
        self,
        symbol: str,
        rel_path: str = ".",
        *,
        max_results: int = 40,
    ) -> ExecResult:
        """Find Python definitions for a symbol using an AST-backed index.

        This is intentionally LSP-shaped while remaining dependency-free for
        SWE-bench images that do not ship a language server.
        """
        script = r"""
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
rel = sys.argv[2].lstrip("/")
symbol = sys.argv[3].split(".")[-1]
max_results = int(sys.argv[4])
start = (root / rel).resolve() if rel and rel != "." else root

def iter_files(start_path):
    if start_path.is_file():
        if start_path.suffix == ".py":
            yield start_path
        return
    for path in start_path.rglob("*.py"):
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        yield path

def target_names(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from target_names(elt)
    elif isinstance(node, ast.Attribute):
        yield node.attr

results = []
for path in iter_files(start):
    if len(results) >= max_results:
        break
    try:
        text = path.read_text(errors="replace")
        tree = ast.parse(text)
    except Exception:
        continue
    lines = text.splitlines()
    for node in ast.walk(tree):
        kind = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                kind = "function"
        elif isinstance(node, ast.ClassDef):
            if node.name == symbol:
                kind = "class"
        elif isinstance(node, ast.Assign):
            if any(name == symbol for target in node.targets for name in target_names(target)):
                kind = "assignment"
        elif isinstance(node, ast.AnnAssign):
            if any(name == symbol for name in target_names(node.target)):
                kind = "assignment"
        if not kind:
            continue
        line = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            rel_path = path
        results.append(f"{rel_path}:{node.lineno}:{node.col_offset + 1}: {kind} {line[:220]}")
        if len(results) >= max_results:
            break

if results:
    print("\n".join(results))
else:
    print(f"No definitions found for {symbol!r}.")
"""
        limit = max(1, min(int(max_results), 200))
        cmd = (
            f"python -c {shlex.quote(script)} "
            f"{shlex.quote(self.workdir)} {shlex.quote(rel_path)} "
            f"{shlex.quote(symbol)} {limit}"
        )
        return self.run(cmd, timeout_s=30)

    def find_references(
        self,
        symbol: str,
        rel_path: str = ".",
        *,
        max_matches: int = 80,
    ) -> ExecResult:
        """Find token-level Python references for a symbol."""
        script = r"""
import io
import pathlib
import sys
import tokenize

root = pathlib.Path(sys.argv[1]).resolve()
rel = sys.argv[2].lstrip("/")
symbol = sys.argv[3].split(".")[-1]
max_matches = int(sys.argv[4])
start = (root / rel).resolve() if rel and rel != "." else root

def iter_files(start_path):
    if start_path.is_file():
        if start_path.suffix == ".py":
            yield start_path
        return
    for path in start_path.rglob("*.py"):
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        yield path

matches = []
seen_lines = set()
for path in iter_files(start):
    if len(matches) >= max_matches:
        break
    try:
        text = path.read_text(errors="replace")
    except Exception:
        continue
    lines = text.splitlines()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in tokens:
            if tok.type != tokenize.NAME or tok.string != symbol:
                continue
            key = (path, tok.start[0])
            if key in seen_lines:
                continue
            seen_lines.add(key)
            line = lines[tok.start[0] - 1].strip() if tok.start[0] <= len(lines) else ""
            try:
                rel_path = path.relative_to(root)
            except ValueError:
                rel_path = path
            matches.append(f"{rel_path}:{tok.start[0]}:{tok.start[1] + 1}: {line[:220]}")
            if len(matches) >= max_matches:
                break
    except tokenize.TokenError:
        continue

if matches:
    print("\n".join(matches))
else:
    print(f"No token references found for {symbol!r}.")
"""
        limit = max(1, min(int(max_matches), 200))
        cmd = (
            f"python -c {shlex.quote(script)} "
            f"{shlex.quote(self.workdir)} {shlex.quote(rel_path)} "
            f"{shlex.quote(symbol)} {limit}"
        )
        return self.run(cmd, timeout_s=30)

    def list_files(self, rel_path: str = ".", *,
                   pattern: str | None = None) -> ExecResult:
        """Listing under ``<workdir>/<rel_path>`` (recursive).

        With ``pattern`` set (a glob), use ``find -name``; without, use
        ``ls -laR``.
        """
        rel = shlex.quote(rel_path.lstrip("/"))
        target = f"{self.workdir}/{rel}" if rel != "." else self.workdir
        if pattern:
            cmd = f"find {target} -type f -name {shlex.quote(pattern)}"
        else:
            cmd = f"ls -la {target}"
        return self.run(cmd, timeout_s=15)

    # ----- test runner -----

    def run_tests(self, test_targets: list[str], *, timeout_s: float = 300) -> ExecResult:
        """Run pytest on the given test targets. Returns ExecResult.

        ``test_targets`` is a list of pytest selectors (file paths or
        ``module::Class::test_name`` style). When empty, runs the full
        suite (rarely what you want — typically pass FAIL_TO_PASS).
        """
        if not test_targets:
            cmd = "python -m pytest --tb=line -x"
        else:
            quoted = " ".join(shlex.quote(t) for t in test_targets)
            cmd = f"python -m pytest -p no:cacheprovider --tb=line -x {quoted}"
        return self.run(cmd, timeout_s=timeout_s)

    # ----- patch extraction -----

    def get_diff(self, *, exclude_tests: bool = False) -> str:
        """Return ``git diff`` of the working tree — the patch the
        agent has produced."""
        cmd = "cd " + shlex.quote(self.workdir) + " && git diff"
        if exclude_tests:
            # SWE-bench applies the hidden test patch during evaluation.
            # Submitting model-written tests only adds noise and can conflict.
            cmd += " -- . ':(exclude)*/tests/*' ':(exclude)tests/*'"
        out = self.run(cmd, timeout_s=30)
        return out.stdout if out.exit_code == 0 else ""

    # ----- state recovery -----

    def create_checkpoint(self, label: str = "stable") -> ExecResult:
        """Write the current worktree diff to a container-local patch file."""
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        if not safe:
            safe = "stable"
        path = f"/tmp/mh_checkpoint_{safe}.patch"
        cmd = (
            "cd " + shlex.quote(self.workdir) + " && "
            "git ls-files --others --exclude-standard -z | "
            "xargs -0 -r git add -N -- >/dev/null 2>&1 || true; "
            f"git diff --binary > {shlex.quote(path)}; "
            "git reset -q >/dev/null 2>&1 || true; "
            f"printf {shlex.quote(path)}"
        )
        return self.run(cmd, timeout_s=45)

    def reset_to_clean(self) -> ExecResult:
        """Reset the worktree to repository HEAD inside the container."""
        cmd = (
            "cd " + shlex.quote(self.workdir)
            + " && git reset --hard HEAD >/dev/null && git clean -fd >/dev/null"
        )
        return self.run(cmd, timeout_s=60)

    def restore_checkpoint(self, checkpoint_path: str) -> ExecResult:
        """Reset to HEAD and apply a previously-created checkpoint patch."""
        quoted_checkpoint = shlex.quote(checkpoint_path)
        cmd = (
            "cd " + shlex.quote(self.workdir)
            + " && git reset --hard HEAD >/dev/null"
            + " && git clean -fd >/dev/null"
            + f" && if [ -s {quoted_checkpoint} ]; then "
            + f"git apply --whitespace=nowarn {quoted_checkpoint}; "
            + "fi"
        )
        return self.run(cmd, timeout_s=90)

    def changed_files(self) -> ExecResult:
        """Return tracked and untracked files changed in the worktree."""
        cmd = (
            "cd " + shlex.quote(self.workdir)
            + " && { git diff --name-only; git ls-files --others --exclude-standard; }"
        )
        return self.run(cmd, timeout_s=20)


@contextmanager
def docker_shell(instance_id: str, **kwargs) -> Iterator[DockerShellExecutor]:
    """Convenience context manager — guarantees cleanup even on exceptions."""
    sh = DockerShellExecutor(instance_id, **kwargs)
    try:
        sh.start()
        yield sh
    finally:
        sh.cleanup()
