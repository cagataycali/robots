"""In-process execution helpers for the training backends (no ``subprocess``).

The trainers used to drive their upstream pipelines by assembling a string
``argv`` (partly from caller-controlled ``TrainSpec.extra``) and handing it to
``subprocess.run`` / the ``torchrun`` binary. Spawning interpreters on command
lines built from external input is a needless injection / arbitrary-flag
surface. This module replaces that with two primitives that keep the work in
THIS Python process (or torch-managed worker processes), never a shell:

* :func:`run_python_module` / :func:`run_python_path` - execute a module
  (``python -m pkg.mod``) or a script file (``python script.py``) IN-PROCESS via
  :mod:`runpy`, with a controlled ``argv`` **list** and captured output. This is
  the exact semantics of the old single-process subprocess call, minus the
  child interpreter and the shell.

* :func:`elastic_launch_callable` - multi-GPU single-node launch via torch's
  programmatic :class:`torch.distributed.launcher.api.elastic_launch`, NOT the
  ``torchrun`` binary. Workers are spawned by torch's elastic agent (Python
  multiprocessing) and each calls a Python callable with arguments passed as
  Python objects - there is no command line to inject into.

Argument hygiene
----------------
:func:`safe_flag_key` rejects passthrough keys that aren't a conservative
``[A-Za-z0-9_.-]+`` so a stray ``extra`` entry can never smuggle extra tokens
(spaces, shell metacharacters, leading dashes) into the argv list.
"""

from __future__ import annotations

import io
import logging
import os
import re
import runpy
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

logger = logging.getLogger(__name__)

# Conservative allowlist for any key that becomes a CLI-style flag token.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def safe_flag_key(key: str) -> bool:
    """True if ``key`` is safe to turn into a ``--key=...`` / ``key=...`` token.

    Rejects empty strings, leading dashes, whitespace, and shell
    metacharacters - anything that could split into extra argv tokens or be
    misread as a separate flag.
    """
    return bool(_SAFE_KEY.match(key))


def filter_safe_extra(
    extra: dict[str, Any], consumed: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Split ``extra`` into (safe passthrough items, rejected keys).

    ``consumed`` keys are dropped silently (handled explicitly by the caller).
    Unsafe keys are returned separately so the caller can warn and ignore them.
    """
    safe: dict[str, Any] = {}
    rejected: list[str] = []
    for key, value in extra.items():
        if key in consumed:
            continue
        if safe_flag_key(key):
            safe[key] = value
        else:
            rejected.append(key)
    return safe, rejected


class _Tee(io.TextIOBase):
    """Write-through tee: forwards writes to both a live console and a log file."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:  # type: ignore[override]
        for stream in (self._primary, self._secondary):
            try:
                stream.write(s)
            except Exception:  # noqa: BLE001 - never let logging break training
                pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        for stream in (self._primary, self._secondary):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass


class capture_to_file:
    """Context manager: tee stdout/stderr + root-logger output into ``log_path``.

    The upstream pipelines log progress to stdout / the root logger; capturing
    them to a file preserves the per-run log the trainers parse for their
    "RUNNING != learning" verdict, exactly as the old subprocess log did.
    """

    def __init__(self, log_path: str | None) -> None:
        self.log_path = log_path
        self._stream: io.TextIOBase | None = None
        self._fh: logging.FileHandler | None = None
        self._r_out = None
        self._r_err = None

    def __enter__(self) -> capture_to_file:
        if not self.log_path:
            return self
        self._stream = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._fh = logging.FileHandler(self.log_path)
        self._fh.setLevel(logging.INFO)
        self._fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self._fh)
        self._r_out = redirect_stdout(_Tee(sys.stdout, self._stream))
        self._r_err = redirect_stderr(_Tee(sys.stderr, self._stream))
        self._r_out.__enter__()
        self._r_err.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._r_err is not None:
                self._r_err.__exit__(*exc)
            if self._r_out is not None:
                self._r_out.__exit__(*exc)
        finally:
            if self._fh is not None:
                logging.getLogger().removeHandler(self._fh)
                self._fh.close()
            if self._stream is not None:
                self._stream.close()


def _with_process_context(
    argv: list[str],
    cwd: str | None,
    env: dict[str, str] | None,
    body: Callable[[], None],
) -> None:
    """Run ``body`` with a temporarily-patched ``sys.argv`` / cwd / env, restored after.

    This reproduces a child interpreter's view (its own argv, working dir, and
    a few extra env vars) WITHOUT spawning one, then puts the process back
    exactly as it was - so the upstream ``__main__`` argv parsers (tyro / hydra /
    draccus) see the controlled argv list and nothing leaks into our process.
    """
    old_argv = sys.argv
    old_cwd = os.getcwd()
    set_env_keys: list[str] = []
    prev_env: dict[str, str] = {}
    try:
        sys.argv = list(argv)
        if env:
            for k, v in env.items():
                if k in os.environ:
                    prev_env[k] = os.environ[k]
                else:
                    set_env_keys.append(k)
                os.environ[k] = v
        if cwd:
            os.chdir(cwd)
        body()
    finally:
        sys.argv = old_argv
        try:
            os.chdir(old_cwd)
        except OSError:
            pass
        for k, v in prev_env.items():
            os.environ[k] = v
        for k in set_env_keys:
            os.environ.pop(k, None)


def run_python_module(
    module: str,
    args: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    log_path: str | None = None,
) -> None:
    """Equivalent of ``python -m <module> <args...>`` run IN-PROCESS via runpy.

    Args:
        module: Dotted module path (e.g. ``"cosmos_framework.scripts.train"``).
        args: Argument tokens (a LIST - never a shell string). ``sys.argv``
            inside the module becomes ``[module, *args]``.
        cwd: Working directory for the run (restored afterwards).
        env: Extra environment variables to set for the run (restored afterwards).
        log_path: If given, stdout/stderr + root-logger output are tee'd here.

    Raises:
        Whatever the module raises. ``SystemExit(0)`` is swallowed (a clean
        ``sys.exit()`` from the module's ``__main__`` is success); a non-zero
        ``SystemExit`` is re-raised so callers can detect failure.
    """
    argv = [module, *args]

    def _body() -> None:
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit as se:  # a script calling sys.exit()
            code = se.code if se.code is not None else 0
            if code not in (0, None):
                raise

    with capture_to_file(log_path):
        _with_process_context(argv, cwd, env, _body)


def run_python_path(
    script_path: str,
    args: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    log_path: str | None = None,
) -> None:
    """Equivalent of ``python <script_path> <args...>`` run IN-PROCESS via runpy.

    Same contract as :func:`run_python_module` but for a script FILE (used by
    GR00T's ``gr00t/experiment/launch_finetune.py``, which lives in a checkout
    and is run by path rather than as an installed module).
    """
    argv = [script_path, *args]

    def _body() -> None:
        try:
            runpy.run_path(script_path, run_name="__main__")
        except SystemExit as se:
            code = se.code if se.code is not None else 0
            if code not in (0, None):
                raise

    with capture_to_file(log_path):
        _with_process_context(argv, cwd, env, _body)


def elastic_launch_callable(
    fn: Callable[..., Any],
    *,
    nproc_per_node: int,
    nnodes: int = 1,
    rdzv_endpoint: str = "",
    rdzv_backend: str = "c10d",
    run_id: str = "",
    fn_args: tuple[Any, ...] = (),
) -> Any:
    """Multi-process launch via torch's programmatic elastic launcher (no torchrun).

    Spawns ``nproc_per_node`` workers using torch's elastic agent (Python
    multiprocessing) and calls ``fn(*fn_args)`` in each. Arguments are passed as
    Python objects, so there is no command line to assemble or inject into - the
    direct, shell-free replacement for ``torchrun --nproc_per_node=N``.

    For ``nnodes > 1`` a shared ``rdzv_endpoint`` (host:port reachable by every
    node) is required; run this once per node with the same ``run_id`` and
    endpoint. With the default single node, a local c10d rendezvous is used.
    """
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    config = LaunchConfig(
        min_nodes=nnodes,
        max_nodes=nnodes,
        nproc_per_node=nproc_per_node,
        run_id=run_id or f"strands-{os.getpid()}",
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint or "localhost:0",
        max_restarts=0,
        start_method="spawn",
    )
    return elastic_launch(config, fn)(*fn_args)
