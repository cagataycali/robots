"""A ``docker`` query the readiness wait makes is bounded, like the port probe beside it.

:meth:`~strands_robots.policies.vera.server_runner.DockerServerRunner._wait_until_ready`
polls ``while time.monotonic() < deadline``, and its loop body makes two
blocking calls one line apart: ``_port_open``, which carries ``timeout=1.0``,
and ``_container_running``, which ran ``docker ps`` with no bound at all. A
docker client whose daemon has stopped answering blocks in that read, so the
deadline was never re-evaluated: the wait that documents "or raise on timeout"
could not reach its own timeout, and because the raise is what calls
:meth:`stop`, the container it had just launched was never torn down either.

That is the same harm ``test_vera_server_ready_timeout_domain`` closed for
``inf`` and ``nan`` budgets, reachable here with a budget that passes every
check :class:`~strands_robots.policies.vera.VeraConfig` makes. Measured on
``977019449`` with ``server_ready_timeout=3.0`` (legal, resolved, checked) and a
``docker`` executable that never returns:

| tree | outcome | wall time | port probes |
| --- | --- | --- | --- |
| before | **still blocked** | stopped observing at 5x the budget | 0 |
| after | ``RuntimeError`` "docker did not answer 'ps' ... within 10s" | 40.0 s | 0 |

Zero probes both ways because the daemon query is the *first* statement in the
loop body: the readiness check the wait exists to perform never ran once. The
40 s is one query bound (10 s) plus the teardown's own pre-existing
``docker stop`` bound (30 s, which logs a warning and does not extend); both
are finite, which is the whole difference.

The two remaining docker calls in the class were already bounded (``docker
logs`` at 10 s, ``docker stop`` at 30 s), so this is the one query on the
budgeted path that was not. ``docker run`` stays unbounded on purpose: it may
pull the image, which is legitimately long, and it happens before any readiness
budget starts.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import subprocess
import time
from typing import Any

import pytest

from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera import server_runner as sr

# A budget every check on the config accepts, short enough that a wait which
# honours it cannot be confused with one that hangs.
LEGAL_BUDGET = 3.0


class _UnboundedQuery(AssertionError):
    """Raised in place of the block a real unbounded docker query performs.

    A test cannot wait out the real thing, so the fake stands in for it: a
    query that states no bound is one the daemon can keep forever, and the
    substitute says so instead of hanging the suite.
    """


def _daemon_that_never_answers(seen: list[dict[str, Any]]):
    """A ``subprocess.run`` whose child never answers, honouring any bound given."""

    def _run(cmd, **kwargs):
        seen.append({"cmd": list(cmd), **kwargs})
        timeout = kwargs.get("timeout")
        if timeout is None:
            raise _UnboundedQuery(
                f"{' '.join(str(c) for c in cmd)} states no timeout, so this call "
                "never returns while the daemon is unresponsive"
            )
        time.sleep(min(float(timeout), 0.01))
        raise subprocess.TimeoutExpired(cmd, timeout)

    return _run


def _recording_run(seen: list[dict[str, Any]]):
    """A ``subprocess.run`` that answers immediately and records its kwargs."""

    def _run(cmd, **kwargs):
        seen.append({"cmd": list(cmd), **kwargs})
        return _FakeCompleted()

    return _run


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _runner(**overrides) -> sr.DockerServerRunner:
    cfg = VeraConfig(embodiment="pusht", server_mode="docker", **overrides)
    runner = sr.DockerServerRunner(cfg)
    runner._docker = lambda: "/usr/bin/docker"  # type: ignore[method-assign]
    return runner


class TestEveryQueryStatesABound:
    """A call that asks the daemon a question states how long it will wait."""

    @pytest.mark.parametrize("verb", ["ps", "logs"])
    def test_query_passes_the_shared_bound(self, monkeypatch, verb):
        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(sr.subprocess, "run", _recording_run(seen))
        runner = _runner()
        if verb == "ps":
            runner.is_running()
        else:
            runner._tail_logs()
        (call,) = [c for c in seen if verb in c["cmd"]]
        assert call["timeout"] == sr._DOCKER_QUERY_TIMEOUT

    def test_the_launch_is_not_a_query_and_states_no_bound(self, monkeypatch):
        # `docker run` may pull the image, so it has no defensible bound and is
        # not on the budgeted loop. Pinned so the asymmetry is a decision.
        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        monkeypatch.setattr(sr.subprocess, "run", _recording_run(seen))
        runner = _runner(docker_image="vera:latest")
        monkeypatch.setattr(runner, "_container_running", lambda: False)
        monkeypatch.setattr(runner, "_wait_until_ready", lambda: None)
        runner.start()
        (launch,) = [c for c in seen if "run" in c["cmd"]]
        assert "timeout" not in launch

    def test_no_docker_query_in_the_module_omits_a_timeout(self):
        """Completeness: every ``docker <verb>`` argv built inline states a bound.

        A query names its verb as a literal in the argv it passes; the launch
        builds its argv in ``_build_run_command`` and so is not matched here.
        """
        source = pathlib.Path(inspect.getfile(sr)).read_text(encoding="utf-8")
        verbs = {"ps", "logs", "stop", "inspect", "kill", "rm"}
        unbounded, queries = [], []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "run"):
                continue
            if not node.args or not isinstance(node.args[0], ast.List):
                continue
            named = {e.value for e in node.args[0].elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if not named & verbs:
                continue
            queries.append(sorted(named & verbs))
            if "timeout" not in {kw.arg for kw in node.keywords}:
                unbounded.append((node.lineno, sorted(named & verbs)))
        assert len(queries) >= 3, f"matcher found only {queries}; the roster cannot be empty"
        assert unbounded == [], f"unbounded docker queries: {unbounded}"


class TestTheWaitEndsWhenTheDaemonStopsAnswering:
    def _wait_with_dead_daemon(self, monkeypatch) -> tuple[sr.DockerServerRunner, list[Any], list[Any]]:
        seen: list[dict[str, Any]] = []
        stopped: list[Any] = []
        probes: list[float] = []
        runner = _runner(server_ready_timeout=LEGAL_BUDGET)
        runner._started_container = True  # we launched it, so we own teardown
        monkeypatch.setattr(sr.subprocess, "run", _daemon_that_never_answers(seen))

        def _never_open(*_a: Any, **_k: Any) -> bool:
            probes.append(time.monotonic())
            return False

        monkeypatch.setattr(sr, "_port_open", _never_open)
        monkeypatch.setattr(runner, "stop", lambda: stopped.append(True))
        return runner, stopped, probes

    def test_it_raises_instead_of_polling_forever(self, monkeypatch):
        runner, _stopped, _probes = self._wait_with_dead_daemon(monkeypatch)
        with pytest.raises(RuntimeError, match=r"docker did not answer 'ps'"):
            runner._wait_until_ready()

    def test_an_answer_it_never_got_is_not_reported_as_a_container_that_exited(self, monkeypatch):
        runner, _stopped, _probes = self._wait_with_dead_daemon(monkeypatch)
        with pytest.raises(RuntimeError) as excinfo:
            runner._wait_until_ready()
        assert "exited before becoming ready" not in str(excinfo.value)
        assert "unresponsive" in str(excinfo.value)

    def test_giving_up_tears_down_the_container_it_launched(self, monkeypatch):
        runner, stopped, _probes = self._wait_with_dead_daemon(monkeypatch)
        with pytest.raises(RuntimeError):
            runner._wait_until_ready()
        assert stopped == [True], "the documented timeout path stops what it started; so must this one"

    def test_a_container_we_did_not_launch_is_never_queried(self, monkeypatch):
        runner, _stopped, probes = self._wait_with_dead_daemon(monkeypatch)
        runner._started_container = False
        runner.config.server_ready_timeout = 1.0  # nothing to query, so nothing to wait out
        with pytest.raises(TimeoutError, match="did not become ready"):
            runner._wait_until_ready()
        assert probes, "the port must still be probed for a pre-existing container"


class TestADaemonThatAnswersIsUnchanged:
    @pytest.mark.parametrize(
        ("stdout", "expected"),
        [("vera-server-pusht\n", True), ("other\n", False), ("", False)],
    )
    def test_running_verdict_reads_the_same_output(self, monkeypatch, stdout, expected):
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout=stdout))
        assert _runner().is_running() is expected

    def test_a_container_that_really_exited_still_reports_its_logs(self, monkeypatch):
        runner = _runner(server_ready_timeout=LEGAL_BUDGET)
        runner._started_container = True
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="other\n"))
        monkeypatch.setattr(runner, "_tail_logs", lambda lines=40: "boom: CUDA OOM")
        with pytest.raises(RuntimeError, match="exited before becoming ready"):
            runner._wait_until_ready()
