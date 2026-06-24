"""In-process execution helpers for the training backends (no ``subprocess``).

The trainers drive their upstream pipelines by **importing the package and
calling it** - never by spawning an interpreter on a command line built from
caller input. This module provides the small, shared plumbing for that:

* :func:`call_callable` - run a Python callable (e.g. GR00T's
  ``experiment.run`` or a per-worker entry) with output captured to a log file.
  This is the purest form: build the upstream's own config object and hand it
  to its own function. No argv at all.

* :func:`elastic_launch_callable` - multi-GPU single-node launch via torch's
  programmatic :class:`torch.distributed.launcher.api.elastic_launch` (the API
  behind ``torchrun``). Workers are spawned by torch's elastic agent and each
  calls a Python callable with arguments passed as Python objects - there is no
  command line to inject into.

Argument hygiene
----------------
:func:`safe_flag_key` rejects passthrough keys that aren't a conservative
``[A-Za-z0-9][A-Za-z0-9_.-]*`` so a stray ``extra`` entry can never smuggle
extra tokens (spaces, shell metacharacters, leading dashes) into an argv list.
"""

from __future__ import annotations

import io
import logging
import os
import re
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
    "RUNNING != learning" verdict, exactly as the old subprocess log did. With
    ``log_path=None`` it is a no-op (used by non-rank-0 workers).
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


def call_callable(
    fn: Callable[..., Any],
    *args: Any,
    log_path: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call a Python callable in-process, with output captured to ``log_path``.

    The purest "import the package and use it" path: the caller has already
    built the upstream's own config object (e.g. GR00T's ``Config``) and just
    needs its function (``experiment.run``) invoked here. No argv, no shell.
    """
    with capture_to_file(log_path):
        return fn(*args, **kwargs)


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
