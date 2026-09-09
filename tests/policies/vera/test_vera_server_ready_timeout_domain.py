"""``VeraConfig.server_ready_timeout`` is a span of seconds, so it takes that domain.

:class:`~strands_robots.policies.vera.VeraConfig` checks four numeric fields on
the *effective* value - both ports, ``render_width`` and ``motion_plan_scale`` -
and its own comment says why: ``__post_init__`` is "the one funnel every caller
passes through". ``server_ready_timeout`` arrives through the same funnel and was
checked nowhere, and ``VERA_SERVER_READY_TIMEOUT`` - which both readiness
timeouts name in the message they raise - was read nowhere either. Those two
message strings were the variable's only appearances in the tree.

Measured on ``4ac86c5``, one config per row, then
``VeraServerRunner._wait_until_ready`` against a simulated clock (1 s per
``monotonic()`` call) and a port that opens at t = 3 s. No ``vera`` package, no
server, no socket:

| ``server_ready_timeout=`` | effective | server up at 3 s | server never up |
| --- | --- | --- | --- |
| omitted | ``600.0`` | ready | ``TimeoutError`` "within 600s" |
| ``inf`` | ``inf`` | ready | **never exits** (stopped at 100000 polls) |
| ``nan`` | ``nan`` | ``TimeoutError`` "within nans", **0 port probes** | same |
| ``0.0`` | ``0.0`` | ``TimeoutError`` "within 0s", 0 probes | same |
| ``-30.0`` | ``-30.0`` | ``TimeoutError`` "within -30s" | same |
| ``True`` | ``True`` | ``TimeoutError`` "within 1s" | same |
| ``"600"`` | ``'600'`` | ``TypeError: ... for +: 'float' and 'str'`` | same |
| ``10**400`` | as given | ``OverflowError`` | same |
| env ``=1800`` | ``600.0`` | ready | ``TimeoutError`` "within **600s**" |

The last row is what makes this worth refusing and reading at the config. The
wait raises ``did not become ready ... within 600s (WAN model load can be slow -
raise server_ready_timeout / VERA_SERVER_READY_TIMEOUT if needed)``, so an
operator whose WAN load needs longer exports the variable the message names,
re-runs, and gets the same failure after the same 600 seconds with the same
message pointing at the same variable.

``inf`` is the other end of it: ``deadline`` becomes infinite, so
``while time.monotonic() < deadline`` cannot end. The method documents "or raise
on timeout", and because the raise is what calls ``self.stop()``, the server
subprocess (or container) it had just launched is never torn down. ``nan`` is
below nothing, so the same loop cannot *begin* - the port is never probed once
and a server that was coming up fine is torn down and reported as unready.

``0`` is not an opt-out, for the reason it is not one for ``motion_plan_scale``:
``_ensure_started`` probes the port and reuses a listening server *before* it
launches anything, so "do not wait" needs no zero budget, and a zero budget only
ever launches a server and instantly tears it down.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera import server_runner as sr

ENV = "VERA_SERVER_READY_TIMEOUT"

# The budget a config that names none resolves to, stated here rather than
# imported: the documented default is the contract, and a test that read it back
# out of the module could not notice the module changing it.
DOCUMENTED_DEFAULT = 600.0

# Budgets no readiness wait can be bounded by. ``0``/``False`` because a zero
# budget tears down the server it just launched; ``inf`` because the loop cannot
# end; ``nan`` because it cannot begin; ``True`` because an ``int`` subclass
# silently meant one second; ``"600"`` because the environment's own shape
# raised ``TypeError`` past the documented failure channel; ``10**400`` because
# it raised ``OverflowError`` there.
UNUSABLE: list[Any] = [
    0,
    0.0,
    False,
    -30.0,
    True,
    float("nan"),
    float("inf"),
    float("-inf"),
    "600",
    [600.0],
    10**400,
]

# Environment spellings of the same unusable budgets. ``1e999`` and ``Infinity``
# are here because ``float()`` accepts both and yields ``inf``.
UNUSABLE_ENV: list[str] = ["0", "-30", "nan", "inf", "1e999", "Infinity"]

# Budgets a wait can be bounded by. The shared domain admits any real scalar, so
# a plain ``int`` and a NumPy float are usable and must survive normalization.
USABLE: list[Any] = [1, 30, 0.5, 1800.0, np.float64(45.0)]


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config, funnelling the deliberately off-type rows through ``Any``."""
    return VeraConfig(embodiment="pusht", **kwargs)


class _Clock:
    """A monotonic clock that advances a fixed step per read.

    Lets a readiness wait be measured in polls rather than in wall time, and
    makes a wait that cannot end observable instead of hanging the suite.
    """

    def __init__(self, step: float = 1.0, cap: int = 5000) -> None:
        self.t = 0.0
        self.reads = 0
        self._step = step
        self._cap = cap

    def monotonic(self) -> float:
        self.reads += 1
        if self.reads > self._cap:
            raise _Unbounded()
        now = self.t
        self.t += self._step
        return now


class _Unbounded(BaseException):
    """Raised when a readiness wait outlives its clock's read budget."""


def _wait(runner: Any, monkeypatch: pytest.MonkeyPatch, *, ready_at: float) -> _Clock:
    """Run ``runner._wait_until_ready()`` against a simulated clock.

    Args:
        runner: A subprocess or docker server runner.
        monkeypatch: Patches the module clock, sleep and port probe.
        ready_at: Simulated second the websocket opens; ``inf`` never opens.

    Returns:
        The clock, whose ``t`` is the simulated time the wait consumed.
    """
    clock = _Clock()
    monkeypatch.setattr(sr.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sr.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sr, "_port_open", lambda *a, **k: clock.t >= ready_at)
    runner._wait_until_ready()
    return clock


class TestTheConfigRefusesAnUnusableBudget:
    """A span no readiness wait can be bounded by is refused at construction."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_budget_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _config(server_ready_timeout=value)
        assert "server_ready_timeout" in str(excinfo.value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_message_names_the_field_and_the_surface(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _config(server_ready_timeout=value)
        assert str(excinfo.value).startswith("VeraConfig: server_ready_timeout "), str(excinfo.value)

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_budget_is_kept_as_a_float(self, value: Any) -> None:
        """The domain admits any real scalar; the consumers need a plain ``float``."""
        resolved = _config(server_ready_timeout=value).server_ready_timeout
        assert type(resolved) is float
        assert resolved == pytest.approx(float(value))


class TestTheEnvironmentOverrideIsReadAndTakesTheSameDomain:
    """``VERA_SERVER_READY_TIMEOUT`` is the remedy both timeouts name, so it is read."""

    def test_the_environment_budget_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV, "1800")
        assert _config().server_ready_timeout == pytest.approx(1800.0)

    @pytest.mark.parametrize("raw", UNUSABLE_ENV)
    def test_an_unusable_environment_budget_is_refused(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """One value, two spellings, one verdict - as the ports already are."""
        monkeypatch.setenv(ENV, raw)
        with pytest.raises(ValueError) as excinfo:
            _config()
        assert "server_ready_timeout" in str(excinfo.value)

    def test_an_explicit_budget_wins_over_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The precedence every other override on this dataclass uses."""
        monkeypatch.setenv(ENV, "1800")
        assert _config(server_ready_timeout=45.0).server_ready_timeout == pytest.approx(45.0)

    def test_a_non_numeric_environment_budget_still_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unparsable spelling takes the default, as it does for the ports."""
        monkeypatch.setenv(ENV, "abc")
        assert _config().server_ready_timeout == pytest.approx(DOCUMENTED_DEFAULT)

    def test_the_default_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A config that names no budget still resolves to the documented 600 s."""
        monkeypatch.delenv(ENV, raising=False)
        assert _config().server_ready_timeout == 600.0

    def test_following_the_advice_the_timeout_gives_changes_the_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The remedy a failure names has to be a remedy.

        Walked exactly as an operator does: let the readiness wait time out, read
        the variable name out of the message it raised, export that variable, and
        wait again. Pre-fix the second wait ended after the same 600 s and raised
        the same sentence, because the name in the message was read nowhere.
        """
        monkeypatch.delenv(ENV, raising=False)
        runner = sr.VeraServerRunner(_config())
        runner._proc = None
        with pytest.raises(TimeoutError) as first:
            _wait(runner, monkeypatch, ready_at=float("inf"))
        assert "within 600s" in str(first.value)

        advised = [word.strip(".,()") for word in str(first.value).split() if word.startswith("VERA_")]
        assert advised, f"the timeout should name the variable that widens it: {first.value}"

        monkeypatch.setenv(advised[0], "1200")
        retried = sr.VeraServerRunner(_config())
        retried._proc = None
        with pytest.raises(TimeoutError) as second:
            _wait(retried, monkeypatch, ready_at=float("inf"))
        assert "within 1200s" in str(second.value)


class TestTheResolvedBudgetIsWhatBoundsTheWait:
    """Both runners bound their loop with the budget the config resolved."""

    def _subprocess_runner(self, **overrides: Any) -> Any:
        runner = sr.VeraServerRunner(_config(**overrides))
        runner._proc = None
        return runner

    def _docker_runner(self, **overrides: Any) -> Any:
        return sr.DockerServerRunner(VeraConfig(embodiment="pusht", server_mode="docker", **overrides))

    @pytest.mark.parametrize("runner_kind", ["subprocess", "docker"])
    def test_a_server_that_never_opens_times_out_at_the_environment_budget(
        self, runner_kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the variable: a longer budget really waits longer.

        Pre-fix the environment was not read, so both runners timed out after
        600 simulated seconds and quoted 600 in a message naming this variable
        as the fix.
        """
        monkeypatch.setenv(ENV, "1200")
        runner = self._subprocess_runner() if runner_kind == "subprocess" else self._docker_runner()
        monkeypatch.setattr(runner, "_container_running", lambda: True, raising=False)
        with pytest.raises(TimeoutError) as excinfo:
            _wait(runner, monkeypatch, ready_at=float("inf"))
        assert "within 1200s" in str(excinfo.value)

    @pytest.mark.parametrize("runner_kind", ["subprocess", "docker"])
    def test_a_server_that_comes_up_inside_the_budget_is_reported_ready(
        self, runner_kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: a usable budget still lets a slow-but-fine server through."""
        runner = self._subprocess_runner(server_ready_timeout=60.0)
        if runner_kind == "docker":
            runner = self._docker_runner(server_ready_timeout=60.0)
        monkeypatch.setattr(runner, "_container_running", lambda: True, raising=False)
        clock = _wait(runner, monkeypatch, ready_at=3.0)
        assert clock.t <= 60.0

    @pytest.mark.parametrize("runner_kind", ["subprocess", "docker"])
    def test_the_wait_always_ends(self, runner_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """No config the constructor accepts can leave the wait unable to end.

        Pre-fix ``server_ready_timeout=inf`` was accepted and this loop polled
        past 100000 simulated seconds without ever reaching the raise that tears
        the launched server down.
        """
        runner = self._subprocess_runner() if runner_kind == "subprocess" else self._docker_runner()
        monkeypatch.setattr(runner, "_container_running", lambda: True, raising=False)
        with pytest.raises(TimeoutError):
            _wait(runner, monkeypatch, ready_at=float("inf"))

    @pytest.mark.parametrize("runner_kind", ["subprocess", "docker"])
    def test_the_wait_probes_the_port_at_least_once(self, runner_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A budget that never probes cannot notice a server that is up.

        Pre-fix ``nan`` and ``0`` made ``time.monotonic() < deadline`` false on
        the first test, so the loop body never ran: the port was probed zero
        times and a healthy server was torn down as unready.
        """
        probes = {"n": 0}
        runner = self._subprocess_runner() if runner_kind == "subprocess" else self._docker_runner()
        monkeypatch.setattr(runner, "_container_running", lambda: True, raising=False)
        clock = _Clock()
        monkeypatch.setattr(sr.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(sr.time, "sleep", lambda _s: None)

        def _probe(*_a: Any, **_k: Any) -> bool:
            probes["n"] += 1
            return False

        monkeypatch.setattr(sr, "_port_open", _probe)
        with pytest.raises(TimeoutError):
            runner._wait_until_ready()
        assert probes["n"] >= 1
