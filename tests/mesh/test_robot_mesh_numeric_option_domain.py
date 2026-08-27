"""Numeric-option domain for the ``robot_mesh`` dispatcher.

``robot_mesh`` exposes an agent four numeric options. Two of them travel inside
the command body that :func:`strands_robots.mesh.security.validate_command`
inspects, so that validator already bounds them: ``duration`` to
``[0, MAX_DURATION_S]`` and ``policy_port`` to ``[1, 65535]``. The other two
never enter a command body, so nothing on that path can see them:

* ``timeout`` is a wait budget. Every action that reads it hands it to a
  :class:`threading.Event` wait inside :meth:`strands_robots.mesh.core.Mesh.send`
  (or to a Device Connect ``invoke``) and reports ``{"status": "timeout"}`` when
  nothing arrived in time. A non-positive or ``nan`` budget makes that wait
  return immediately, so the tool reports a peer that did not answer without
  ever giving it the chance - and ``stop``'s ``min(timeout, 5.0)`` cap does not
  help, because ``min(nan, 5.0)`` is ``nan``.
* ``limit`` caps how many buffered messages ``inbox`` pulls into the agent's
  context, consumed directly as a slice index. A non-positive or ``nan`` limit
  used to select the whole buffer - the opposite of a cap - and a fractional one
  raised ``TypeError`` from the slice, out of a dispatcher documented never to
  raise.

These tests pin both options against the shared domains
(:func:`strands_robots.utils.positive_finite_number_error` for the continuous
budget, :func:`strands_robots.utils.positive_count_error` for the discrete cap),
pin the per-action scoping in both directions so an action is never refused for
an option it does not read, and pin that a refusal precedes the operator
approval gate, the rate-limit accounting and the transport.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.robot_mesh import (
    _ACTION_NUMERIC_OPTIONS,
    _reset_rate_limits,
    robot_mesh,
)
from strands_robots.utils import positive_count_error, positive_finite_number_error

#: Every action the tool advertises, from its own unknown-action message.
ALL_ACTIONS = (
    "peers",
    "status",
    "tell",
    "send",
    "rpc",
    "sim_call",
    "broadcast",
    "stop",
    "emergency_stop",
    "subscribe",
    "unsubscribe",
    "watch",
    "inbox",
)

#: Actions that hand ``timeout`` to a wait, so an unusable value is refused.
READS_TIMEOUT = ("tell", "send", "rpc", "sim_call", "broadcast", "stop")

#: Wait budgets that cannot be honored. ``0`` / ``-1`` / ``nan`` make the wait
#: return immediately (reported as a peer that did not answer), ``inf`` raises
#: ``OverflowError`` from the deadline arithmetic, ``True`` is a silent 1s, and
#: the non-numerics reach a bare comparison.
UNUSABLE_TIMEOUTS: tuple[Any, ...] = (0, -1.0, float("nan"), float("inf"), True, "30", None, [30.0])

#: Message caps that cannot be honored. ``0`` / ``-5`` / ``nan`` used to select
#: the whole buffer; ``2.7`` and the non-numerics reached the slice or a bare
#: comparison.
UNUSABLE_LIMITS: tuple[Any, ...] = (0, -5, float("nan"), True, 2.7, "50", None)


class RecordingMesh:
    """Mesh stand-in that records the budget each call was handed.

    Returns immediately rather than waiting, so these tests never sleep: the
    contract under test is which value reaches the transport, not how long the
    transport then blocks for.
    """

    def __init__(self) -> None:
        self.peer_id = "local-a"
        self.peer_type = "sim"
        self.inbox: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.calls: list[tuple[str, Any]] = []

    def send(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        self.calls.append(("send", timeout))
        return {"status": "ok"}

    def broadcast(self, cmd: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
        self.calls.append(("broadcast", timeout))
        return []

    def tell(self, target: str, instruction: str, **kw: Any) -> dict[str, Any]:
        self.calls.append(("tell", kw))
        return {"status": "ok"}

    def emergency_stop(self) -> list[dict[str, Any]]:
        self.calls.append(("emergency_stop", None))
        return []

    def unsubscribe(self, name: str) -> bool:
        self.calls.append(("unsubscribe", name))
        return True


@pytest.fixture(autouse=True)
def _isolate_rate_limits():
    """Each accepted call consumes a per-action rate-limit slot; reset so the
    cases stay independent of collection order."""
    _reset_rate_limits()
    yield
    _reset_rate_limits()


@pytest.fixture
def mesh():
    """A recording mesh installed as the only local peer."""
    fake = RecordingMesh()
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={"local-a": fake}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        yield fake


def _ctx() -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    return ctx


def _call(*, _ctx_obj: MagicMock | None = None, **kwargs: Any) -> dict[str, Any]:
    """Invoke the underlying tool fn (Strands ``@tool`` wraps it as ``.original``)."""
    fn = getattr(robot_mesh, "original", robot_mesh)
    return fn(tool_context=_ctx_obj if _ctx_obj is not None else _ctx(), **kwargs)


def _text(out: dict[str, Any]) -> str:
    return str(out["content"][0]["text"])


def _args_for(action: str) -> dict[str, Any]:
    """The minimum arguments that carry *action* past its own presence checks."""
    if action in ("tell",):
        return {"target": "peer-b", "instruction": "pick up the cube"}
    if action in ("send", "broadcast"):
        return {"target": "peer-b", "command": '{"action": "status"}'}
    if action == "rpc":
        return {"target": "peer-b", "function": "nod"}
    if action == "sim_call":
        return {"target": "peer-b", "function": "add_object"}
    if action == "stop":
        return {"target": "peer-b"}
    if action == "inbox":
        return {"name": "sub-x"}
    if action == "unsubscribe":
        return {"name": "sub-x"}
    return {}


class TestTimeoutIsAPositiveFiniteBudget:
    """Every action that waits on ``timeout`` refuses a budget it cannot honor."""

    @pytest.mark.parametrize("action", READS_TIMEOUT)
    @pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS, ids=repr)
    def test_an_unusable_budget_is_refused(self, mesh, action, value):
        out = _call(action=action, timeout=value, **_args_for(action))
        assert out["status"] == "error", f"{action} accepted timeout={value!r}"
        text = _text(out)
        assert "timeout" in text
        assert "robot_mesh" in text and action in text, f"message does not name the surface: {text}"

    @pytest.mark.parametrize("action", READS_TIMEOUT)
    def test_a_usable_budget_is_not_refused(self, mesh, action):
        """Asserted as the absence of a budget refusal rather than as success:
        ``rpc`` has no Zenoh-mesh equivalent and declines for that reason on a
        host without Device Connect, which is not a verdict about the budget."""
        out = _call(action=action, timeout=2.5, **_args_for(action))
        assert "timeout must be" not in _text(out), _text(out)

    @pytest.mark.parametrize("action", ["tell", "send", "broadcast", "stop"])
    def test_a_usable_budget_reaches_the_transport(self, mesh, action):
        out = _call(action=action, timeout=2.5, **_args_for(action))
        assert out["status"] == "success", _text(out)
        assert mesh.calls, f"{action} accepted the budget but made no transport call"

    def test_a_fractional_budget_is_usable(self, mesh):
        """The budget is a span of seconds, so 2.7s is a real request - the
        domain constrains only the sign and the finiteness."""
        out = _call(action="send", timeout=2.7, **_args_for("send"))
        assert out["status"] == "success"
        assert ("send", 2.7) in mesh.calls


class TestLimitIsAPositiveCount:
    """``inbox``'s cap is a count, consumed as a slice index."""

    @pytest.mark.parametrize("value", UNUSABLE_LIMITS, ids=repr)
    def test_an_unusable_cap_is_refused(self, mesh, value):
        mesh.inbox = {"sub-x": [("strands/peer/stream", {"step": i}) for i in range(120)]}
        out = _call(action="inbox", name="sub-x", limit=value)
        assert out["status"] == "error", f"inbox accepted limit={value!r}"
        text = _text(out)
        assert "limit" in text and "robot_mesh inbox" in text

    def test_the_cap_bounds_what_reaches_the_agent_context(self, mesh):
        """A usable cap returns exactly that many of the buffered messages.

        Pinned as the count actually rendered rather than only the status,
        because a non-positive cap used to report success while returning the
        WHOLE buffer - the opposite of a cap, on the action that pulls another
        peer's stream into the agent's context.
        """
        mesh.inbox = {"sub-x": [("strands/peer/stream", {"step": i}) for i in range(120)]}
        out = _call(action="inbox", name="sub-x", limit=5)
        assert out["status"] == "success"
        text = _text(out)
        assert "120 total, showing last 5" in text
        assert text.count("strands/peer/stream") == 5


class TestOnlyTheOptionsAnActionReadsAreRefused:
    """A caller is never refused for a value the requested action ignores."""

    @pytest.mark.parametrize("action", [a for a in ALL_ACTIONS if a not in READS_TIMEOUT])
    def test_an_action_that_never_waits_ignores_the_budget(self, mesh, action):
        out = _call(action=action, timeout=float("nan"), **_args_for(action))
        assert "timeout must be" not in _text(out), f"{action} was refused for a budget it never reads"

    def test_an_action_that_never_reads_the_cap_ignores_it(self, mesh):
        out = _call(action="send", limit=0, **_args_for("send"))
        assert out["status"] == "success", _text(out)

    def test_emergency_stop_fans_out_on_its_own_budget(self, mesh):
        """``emergency_stop`` uses a fixed internal budget, so the caller's
        ``timeout`` is not effective and must not be refused."""
        out = _call(action="emergency_stop", timeout=-1.0)
        assert out["status"] == "success", _text(out)
        assert ("emergency_stop", None) in mesh.calls


class TestARefusedOptionReachesNothing:
    """The refusal precedes the approval gate, the rate limit and the transport."""

    def test_no_transport_call_is_made(self, mesh):
        out = _call(action="stop", target="peer-b", timeout=float("nan"))
        assert out["status"] == "error"
        assert mesh.calls == [], f"a refused stop still reached the transport: {mesh.calls}"

    def test_the_operator_is_not_asked_to_approve_it(self, mesh):
        """``broadcast`` is human-in-the-loop gated. An option the action cannot
        honor must not burn an operator approval on a call that never runs."""
        ctx = _ctx()
        out = _call(action="broadcast", command='{"action": "status"}', timeout=0, _ctx_obj=ctx)
        assert out["status"] == "error"
        ctx.interrupt.assert_not_called()

    def test_it_does_not_consume_a_rate_limit_slot(self, mesh):
        """``broadcast`` is capped at 10 calls/min. Refused calls must not spend
        the budget, or a malformed argument would lock out a valid one."""
        for _ in range(12):
            out = _call(action="broadcast", command='{"action": "status"}', timeout=0)
            assert out["status"] == "error"
            assert "timeout must be" in _text(out), _text(out)
        good = _call(action="broadcast", command='{"action": "status"}', timeout=2.5)
        assert good["status"] == "success", _text(good)


class TestTheStopCapIsACapNotAGuard:
    """``stop`` caps the budget at 5s; the cap cannot stand in for the domain."""

    def test_a_long_budget_is_capped(self, mesh):
        out = _call(action="stop", target="peer-b", timeout=30.0)
        assert out["status"] == "success"
        assert ("send", 5.0) in mesh.calls, mesh.calls

    def test_a_budget_below_the_cap_is_passed_through(self, mesh):
        out = _call(action="stop", target="peer-b", timeout=2.5)
        assert out["status"] == "success"
        assert ("send", 2.5) in mesh.calls, mesh.calls

    def test_the_cap_does_not_bound_a_non_finite_budget(self, mesh):
        """``min(nan, 5.0)`` is ``nan``, so the cap passes it straight through -
        which is why ``stop`` needs the domain and not only the cap."""
        assert min(float("nan"), 5.0) != 5.0
        out = _call(action="stop", target="peer-b", timeout=float("nan"))
        assert out["status"] == "error"
        assert mesh.calls == []


class TestTheSharedDomainsAreTheOwner:
    """The tool's verdict is the shared domain's verdict, not a local rule."""

    @pytest.mark.parametrize("value", [*UNUSABLE_TIMEOUTS, 2.5, 30.0, 0.001, 1, 3600.0], ids=repr)
    def test_a_budget_is_refused_exactly_when_the_shared_domain_refuses_it(self, mesh, value):
        shared_refuses = positive_finite_number_error(value, "timeout", "robot_mesh send") is not None
        out = _call(action="send", timeout=value, **_args_for("send"))
        tool_refuses = out["status"] == "error"
        assert tool_refuses is shared_refuses, f"verdicts differ for timeout={value!r}: {_text(out)}"

    @pytest.mark.parametrize("value", [*UNUSABLE_LIMITS, 1, 5, 50, 1000], ids=repr)
    def test_a_cap_is_refused_exactly_when_the_shared_domain_refuses_it(self, mesh, value):
        mesh.inbox = {"sub-x": [("strands/peer/stream", {"step": i}) for i in range(120)]}
        shared_refuses = positive_count_error(value, "limit", "robot_mesh inbox") is not None
        out = _call(action="inbox", name="sub-x", limit=value)
        tool_refuses = out["status"] == "error"
        assert tool_refuses is shared_refuses, f"verdicts differ for limit={value!r}: {_text(out)}"


class TestNumericOptionScopingDoesNotDrift:
    """The per-action table must keep describing the actions the tool has."""

    def test_every_scoped_action_is_a_real_action(self):
        assert set(_ACTION_NUMERIC_OPTIONS) <= set(ALL_ACTIONS)

    def test_the_table_scopes_exactly_the_actions_that_read_an_option(self):
        assert {a for a, opts in _ACTION_NUMERIC_OPTIONS.items() if "timeout" in opts} == set(READS_TIMEOUT)
        assert {a for a, opts in _ACTION_NUMERIC_OPTIONS.items() if "limit" in opts} == {"inbox"}

    def test_only_the_two_options_the_wire_validator_cannot_see_are_scoped_here(self):
        """``duration`` and ``policy_port`` ride inside the command body, so
        ``validate_command`` owns them; duplicating them here would be a second
        owner for one rule."""
        scoped = {opt for opts in _ACTION_NUMERIC_OPTIONS.values() for opt in opts}
        assert scoped == {"timeout", "limit"}

    @pytest.mark.parametrize(
        ("param", "value", "reason"),
        [
            ("duration", -1.0, "out of bounds"),
            ("duration", float("nan"), "must be finite"),
            ("duration", True, "must be a number"),
            ("policy_port", 70000, "out of bounds"),
            ("policy_port", float("nan"), "must be finite"),
        ],
    )
    def test_the_wire_validator_really_bounds_the_other_two(self, mesh, param, value, reason):
        """Non-vacuity for the test above: the reason ``duration`` and
        ``policy_port`` are out of scope is that they are already refused."""
        out = _call(action="tell", target="peer-b", instruction="go", **{param: value})
        assert out["status"] == "error"
        text = _text(out)
        assert param in text and reason in text, text
