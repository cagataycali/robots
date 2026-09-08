"""The bound a ``docker`` query states must be one a healthy daemon can meet.

:data:`~strands_robots.policies.vera.server_runner._DOCKER_QUERY_TIMEOUT` is what
stops an unresponsive daemon from leaving ``DockerServerRunner``'s readiness wait
unable to reach its own deadline. That the queries *state* a bound, and that a
daemon which does not answer inside it is reported rather than answered for, is
graded by ``test_docker_daemon_queries_are_bounded``. Two properties of the bound
itself were not, and both were measured by mutating ``server_runner.py`` and
running that file alone -- nine mutations, control 0, ``collected`` stable at 12:

| mutation | cells it fired there |
| --- | --- |
| probe unbounded again | 5 |
| timeout answered ``False`` instead of reported | 3 |
| over-reach: bound the launch too | 1 |
| raw ``TimeoutExpired`` escapes | 3 |
| drop the launched-container teardown | 1 |
| the bound is ``None`` | 3 |
| ``_tail_logs`` back on its own literal ``10`` | **0** |
| the bound is ``0.0`` | **0** |

**A bound of ``0.0``** is the one worth a cell, because it is an over-correction
rather than a return to the old state: ``subprocess.run(timeout=0.0)`` raises
``TimeoutExpired`` before the child can answer, so every query fails against a
perfectly healthy daemon and ``server_mode="docker"`` stops working altogether.
It goes unseen where the daemon is a callable standing in for ``subprocess.run``,
because no real bound is ever applied to a real child; the control below runs a
real executable that answers promptly, which is what makes the difference between
"bounded" and "refuses everything" observable. ``None`` is the other end of the
same axis and is caught by the same assertion.

**``_tail_logs`` back on a literal** passes there because ``10 == 10.0``, so a log
read that stops reading the constant still satisfies a cell comparing its value.
The constant's own comment describes the two reads as sharing one bound; this
holds them to it.

The launch keeps no bound, and the SIGTERM grace period ``stop`` waits out keeps
its own larger one -- neither is a query, and neither is in scope here.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera import server_runner as sr


def _docker_that_answers(tmp_path: Path, stdout: str) -> Path:
    """A real executable that answers ``docker ps`` promptly on stdout.

    A real child is the point: the bound is handed to ``subprocess.run``, so only
    a real process being waited on can tell a usable bound from one that refuses
    every answer before it arrives.
    """
    exe = tmp_path / "docker"
    exe.write_text(f"#!/bin/sh\nprintf '{stdout}\\n'\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def _runner(exe: Path) -> sr.DockerServerRunner:
    runner = sr.DockerServerRunner(VeraConfig(embodiment="pusht", server_mode="docker"))
    runner._docker = lambda: str(exe)  # type: ignore[method-assign]
    return runner


class TestTheBoundIsOneAHealthyDaemonCanMeet:
    """A bound that no answer can arrive inside refuses the daemon, not the wedge."""

    def test_the_bound_is_a_positive_finite_span_of_seconds(self) -> None:
        """``None`` is no bound at all and ``0`` refuses every query."""
        bound = sr._DOCKER_QUERY_TIMEOUT
        assert isinstance(bound, float), f"the shared query bound is {bound!r}, which states no span"
        assert bound > 0, (
            f"a query bound of {bound!r} raises TimeoutExpired before any answer can arrive, so "
            "every docker query fails against a healthy daemon"
        )

    @pytest.mark.parametrize("running", [True, False])
    def test_a_daemon_that_answers_promptly_is_still_read(self, running: bool, tmp_path: Path) -> None:
        """The control, driven through a real child so the bound is really applied.

        Both verdicts matter: an unusable bound refuses the answer whatever it
        would have said, so a cell that only checked the running case would read
        an inverted verdict as success.
        """
        listed = "vera-server-pusht" if running else "some-other-container"
        assert _runner(_docker_that_answers(tmp_path, listed)).is_running() is running


class TestBothDaemonReadsShareTheOneBound:
    """The probe and the log read are one bound, so neither drifts from the other."""

    def test_the_log_read_reads_the_constant(self) -> None:
        body = inspect.getsource(sr.DockerServerRunner._tail_logs)
        assert "_DOCKER_QUERY_TIMEOUT" in body, (
            "the log read carries its own literal, so raising or lowering the shared bound silently leaves it behind"
        )
