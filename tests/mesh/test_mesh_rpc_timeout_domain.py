"""Both mesh RPC publishers refuse a wait budget they cannot honor.

:meth:`~strands_robots.mesh.core.Mesh.send` and
:meth:`~strands_robots.mesh.core.Mesh.broadcast` hand ``timeout`` straight to a
:class:`threading.Event` wait. The ``robot_mesh`` tool already holds that same
parameter to :func:`~strands_robots.utils.positive_finite_number_error`, and the
docstring of the tool's domain helper names *these* methods as the consumer --
"every action that reads it hands it to a threading.Event wait
(:meth:`strands_robots.mesh.core.Mesh.send`) ... Only a positive finite number
can be honored". A caller reaching ``Mesh`` directly got no such check, which is
the same tool-vs-library gap ``send``'s own ``validate_command`` call closes for
``cmd`` -- left open for the budget.

Every unusable spelling still published the command and only then failed:

* ``nan`` and a negative make ``Event.wait`` return immediately, so the caller
  is handed ``{"status": "timeout"}`` about 0.01ms after a command that did go
  out -- the peer may be executing it.
* ``inf`` and a string raise ``OverflowError`` / ``TypeError`` out of methods
  whose contract is to return an envelope (``send``) or a list of responses
  (``broadcast``).
* ``True`` is silently a one-second budget.
* ``None`` waits forever, so the call never returns.

``broadcast`` carries the sharper consequence: the comment on its wait explains
that it deliberately spans the FULL window so an operator can distinguish "1 of
12 stopped" from "all stopped". An unusable budget returns immediately and
reports an empty fleet for a broadcast that was published.
"""

import ast
import inspect
import logging
import math
import textwrap
import threading
import time
from typing import Any

import pytest

import strands_robots.tools.robot_mesh as _tool
from strands_robots.mesh import core
from strands_robots.utils import positive_finite_number_error

#: Budgets no ``Event.wait`` can honor as a wait of that length.
_UNUSABLE: tuple[Any, ...] = (0, -1.0, math.nan, math.inf, -math.inf, True, False, "30", None, [0.5])

#: Budgets a caller may legitimately ask for. ``0.05`` is the value the shipped
#: mesh suites pass, so it is the over-reach control for this whole file.
_USABLE: tuple[float, ...] = (0.05, 0.2, 1.0, 30.0)


def _mesh() -> core.Mesh:
    m = core.Mesh(robot=object(), peer_id="op")
    m._running = True
    return m


def _drive(publisher: str, timeout: Any) -> tuple[Any, list[str]]:
    """Run *publisher* with *timeout*, returning ``(result, keys that reached the wire)``."""
    m = _mesh()
    reached: list[str] = []
    m.publish = lambda key, payload: reached.append(key)  # type: ignore[method-assign]
    if publisher == "send":
        return m.send("r1", {"action": "stop"}, timeout=timeout), reached
    return m.broadcast({"action": "stop"}, timeout=timeout), reached


class TestEventWaitCannotHonorTheseBudgets:
    """Premise: the reason a domain is needed lives in the stdlib, not here."""

    @pytest.mark.parametrize("budget", [0, -1.0, math.nan])
    def test_a_non_positive_or_nan_budget_returns_at_once(self, budget: Any) -> None:
        # These are the silent ones: the wait is a no-op, so the caller reads a
        # timeout for a command that went out microseconds earlier.
        started = time.monotonic()
        assert threading.Event().wait(timeout=budget) is False
        assert time.monotonic() - started < 0.01

    def test_an_infinite_budget_raises_rather_than_waiting(self) -> None:
        with pytest.raises(OverflowError):
            threading.Event().wait(timeout=math.inf)

    def test_a_string_budget_raises_rather_than_waiting(self) -> None:
        with pytest.raises(TypeError):
            threading.Event().wait(timeout="30")  # type: ignore[arg-type]

    def test_a_true_budget_is_silently_one_second(self) -> None:
        # ``bool`` is an ``int`` subclass, so this is accepted as a wait of 1.0s
        # rather than reported as a flag handed to a budget.
        started = time.monotonic()
        assert threading.Event().wait(timeout=True) is False
        assert 0.9 < time.monotonic() - started < 1.5


class TestTheToolAlreadyHeldThisParameterToTheSameDomain:
    """Premise: the domain is the tool's, adopted rather than invented here."""

    def test_the_tool_consults_the_shared_positive_finite_domain(self) -> None:
        source = textwrap.dedent(inspect.getsource(_tool._numeric_option_error))
        assert "positive_finite_number_error(timeout" in source

    def test_the_tools_domain_docstring_names_these_methods_as_the_consumer(self) -> None:
        doc = " ".join((_tool._numeric_option_error.__doc__ or "").split())
        assert "Mesh.send" in doc
        assert "Only a positive finite number can be honored" in doc

    def test_every_rpc_action_the_tool_exposes_reads_the_budget(self) -> None:
        # Non-vacuity: the tool really does route these actions' timeout, so the
        # library methods under them are the same quantity consumed the same way.
        for action in ("send", "broadcast"):
            assert "timeout" in _tool._ACTION_NUMERIC_OPTIONS[action]


class TestSendRefusesTheBudgetBeforeTheCommandGoesOut:
    """Regression: the refusal precedes ``publish``, not the wait."""

    @pytest.mark.parametrize("budget", _UNUSABLE)
    def test_an_unusable_budget_is_refused(self, budget: Any) -> None:
        result, reached = _drive("send", budget)
        assert result["status"] == "error"

    @pytest.mark.parametrize("budget", _UNUSABLE)
    def test_nothing_reaches_the_wire(self, budget: Any) -> None:
        # The whole point of refusing here: previously every one of these
        # published the command and only then failed.
        _result, reached = _drive("send", budget)
        assert reached == []

    @pytest.mark.parametrize("budget", _UNUSABLE)
    def test_the_refusal_names_the_method_and_the_parameter(self, budget: Any) -> None:
        result, _reached = _drive("send", budget)
        assert "Mesh.send" in result["error"]
        assert "timeout" in result["error"]

    def test_the_refusal_is_the_shared_domains_own_text(
        self,
    ) -> None:
        result, _reached = _drive("send", math.nan)
        expected = positive_finite_number_error(math.nan, "timeout", "Mesh.send")
        assert result["error"] == expected


class TestBroadcastRefusesTheSameBudget:
    """Regression: reported the way this method reports a client-side rejection."""

    @pytest.mark.parametrize("budget", _UNUSABLE)
    def test_an_unusable_budget_yields_no_responses(self, budget: Any) -> None:
        result, _reached = _drive("broadcast", budget)
        assert result == []

    @pytest.mark.parametrize("budget", _UNUSABLE)
    def test_nothing_reaches_the_wire(self, budget: Any) -> None:
        _result, reached = _drive("broadcast", budget)
        assert reached == []

    def test_the_reason_is_logged_since_the_return_type_has_no_error_slot(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # ``broadcast`` returns ``list[dict]``, so an empty list is also what a
        # broadcast nobody answered returns. The log is the only place the
        # difference can be recorded.
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            _drive("broadcast", math.nan)
        messages = [r.getMessage() for r in caplog.records]
        assert any("timeout" in m and "rejected client-side" in m for m in messages)


class TestAUsableBudgetIsUnchanged:
    """Over-reach control: every value a caller may legitimately ask for."""

    @pytest.mark.parametrize("budget", _USABLE)
    def test_the_domain_accepts_it(self, budget: float) -> None:
        assert positive_finite_number_error(budget, "timeout", "Mesh.send") is None

    def test_send_still_publishes_and_still_reports_a_timeout(self) -> None:
        result, reached = _drive("send", 0.05)
        assert reached == ["strands/r1/cmd"]
        assert result == {"status": "timeout"}

    def test_broadcast_still_publishes_and_still_returns_no_responses(self) -> None:
        result, reached = _drive("broadcast", 0.05)
        assert reached == ["strands/broadcast"]
        assert result == []

    def test_a_fractional_budget_is_honored_for_its_full_span(self) -> None:
        # The continuous domain rather than a count: a sub-second budget is the
        # value the shipped mesh suites pass.
        started = time.monotonic()
        _drive("send", 0.2)
        assert 0.15 < time.monotonic() - started < 1.0


class TestBothPublishersConsultTheSharedDomain:
    """Structural: one domain, so the two methods cannot come to disagree."""

    @pytest.mark.parametrize("method", ["send", "broadcast"])
    def test_the_method_calls_the_shared_domain(self, method: str) -> None:
        fn = getattr(core.Mesh, method)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert "positive_finite_number_error" in called

    @pytest.mark.parametrize("method", ["send", "broadcast"])
    def test_the_guard_precedes_the_wait_it_bounds(self, method: str) -> None:
        source = textwrap.dedent(inspect.getsource(getattr(core.Mesh, method)))
        assert "positive_finite_number_error" in source, f"{method} consults no shared budget domain"
        assert ".wait(timeout=timeout)" in source, f"{method} no longer waits on the budget"
        assert source.index("positive_finite_number_error") < source.index(".wait(timeout=timeout)")

    @pytest.mark.parametrize("method", ["send", "broadcast"])
    def test_the_guard_precedes_the_publish_it_protects(self, method: str) -> None:
        source = textwrap.dedent(inspect.getsource(getattr(core.Mesh, method)))
        assert "self.publish(" in source, f"{method} no longer publishes"
        assert source.index("positive_finite_number_error") < source.index("self.publish(")
