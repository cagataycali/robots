"""``IotMqttTransport`` refuses a connect timeout it cannot spend.

``connect_timeout`` is the third surface in the library to carry that exact
parameter name, and it was the one left without a domain. The other two are the
remote-inference clients settled by #1984 -
:class:`~strands_robots.inference.RemotePolicy` over WebSocket and
:class:`~strands_robots.policies.lerobot_async.LerobotAsyncPolicy` over gRPC -
whose contracts live in :mod:`tests.test_remote_client_timeout_domain`. That
module's own framing is why this one was missed: it settled "the knobs left
behind on the same two constructors", and the survey it describes was scoped to
remote-inference clients. A mesh transport spelling the same knob the same way
sat outside it.

What the transport does with the value makes an unusable one worse than a late
crash, because ``connect()``'s failure report cannot tell it apart from a broker
that is genuinely unreachable. It spends the budget on ``threading.Event.wait``,
and measured against a fake broker whose CONNACK arrives 50 ms after ``start()``
(a real one is a network round trip, never instant):

* ``0``, ``-1`` and ``nan`` make ``wait`` return ``False`` in ~0 ms, so
  ``connect()`` logs "IoT connection to ... timed out after 0.0s", stops the
  client that was connecting normally, and returns ``False``. The operator is
  pointed at the endpoint, the certs and the broker - the three things that were
  not wrong.
* ``inf`` and ``'15'`` raise ``OverflowError`` / ``TypeError`` from the ``wait``
  itself. That call sits *after* the ``try`` that contains client construction,
  so the exception leaves a method documented to return ``bool`` with the MQTT5
  client started and no ``stop()`` on any path.
* ``None`` is not a spelling for an unbounded wait. ``Event.wait(None)`` blocks
  forever, and ``connect()`` holds ``self._lock`` for its whole body, so
  :meth:`close` and every subscription call block behind it with no deadline.
* ``True`` is a silent one-second budget, arriving through ``bool`` being an
  ``int`` subclass.

The domain is therefore :func:`~strands_robots.utils.positive_finite_number_error`
unchanged - the same function, the same message shape, the same reasoning the two
clients already record. The guard sits in the constructor rather than beside the
wait so that it also precedes the ``awsiot`` import inside ``connect()``: the same
caller mistake then reports identically whether or not the ``[mesh-iot]`` extra is
installed, which is the property the gRPC client's own comment names.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from strands_robots.inference import RemotePolicy
from strands_robots.mesh.transport.iot_transport import IotMqttTransport
from strands_robots.policies.lerobot_async import LerobotAsyncPolicy
from strands_robots.utils import positive_finite_number_error

from .test_iot_reconnect_client_lifecycle import _FakeClient, _make_certs

#: Seconds the fake broker takes to report CONNACK. Non-zero on purpose: with an
#: instant callback ``Event.wait(0)`` returns ``True`` because the event is
#: already set, and the misdiagnosis this module is about would be invisible.
_CONNACK_DELAY_S = 0.05

#: Values that name no wait budget. Shared with the two remote-inference clients;
#: spelled out here rather than imported so this module stays readable on its own
#: and does not inherit another module's import surface.
UNUSABLE_TIMEOUTS: list[Any] = [
    0,
    0.0,
    -1,
    -0.5,
    math.nan,
    math.inf,
    -math.inf,
    True,
    False,
    "15",
    None,
    [15],
]

#: Accepted. ``np.float32`` is a real spelling for a budget read out of a config
#: array, and the shared domain documents it as usable, so this pins that the
#: transport needs no coercion of its own.
USABLE_TIMEOUTS: list[Any] = [0.001, 1, 15.0, 60.0, np.float32(0.5)]


def _transport(tmp_path: Any, **kwargs: Any) -> IotMqttTransport:
    """Build the transport, splatted so off-type values reach the guard.

    ``connect_timeout`` is annotated ``float``; several cases here hand it
    something else on purpose, to prove the runtime refuses it rather than the
    type checker.
    """
    kwargs.setdefault("thing_name", "thor-arm")
    kwargs.setdefault("endpoint", "x-ats.iot.us-west-2.amazonaws.com")
    kwargs.setdefault("cert_dir", str(_make_certs(tmp_path)))
    return IotMqttTransport(**kwargs)


class _ConnectingClient(_FakeClient):
    """A broker whose CONNACK arrives after a delay, as a real one does."""

    def start(self) -> None:
        self.started = True
        success = self._kwargs["on_lifecycle_connection_success"]
        threading.Timer(_CONNACK_DELAY_S, lambda: success(object())).start()


class _SilentClient(_FakeClient):
    """A broker that never reports CONNACK, so the wait runs to its deadline."""

    def start(self) -> None:
        self.started = True


def _install_broker(monkeypatch: pytest.MonkeyPatch, client_cls: type[_FakeClient]) -> list[_FakeClient]:
    """Point ``mtls_from_path`` at ``client_cls`` and collect what it built."""
    import awsiot.mqtt5_client_builder as builder

    built: list[_FakeClient] = []

    def fake_mtls_from_path(**kwargs: Any) -> _FakeClient:
        client = client_cls(**kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(builder, "mtls_from_path", fake_mtls_from_path)
    return built


def _refuses_the_timeout(build: Callable[[], object]) -> bool:
    """Did ``build`` refuse specifically because of ``connect_timeout``?

    Only a ``ValueError`` naming the parameter counts. The gRPC client also
    raises ``ValueError`` for an absent ``policy_type``, so the verdict has to
    read the message rather than the exception type; anything else propagates.
    """
    try:
        build()
    except ValueError as exc:
        return "connect_timeout" in str(exc)
    return False


class TestTheTransportRefusesAConnectTimeoutThatNamesNoBudget:
    """The refusal is a ``ValueError`` naming the class, the parameter and the domain."""

    @pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS)
    def test_it_is_refused_at_construction(self, tmp_path: Any, value: Any) -> None:
        with pytest.raises(ValueError) as exc:
            _transport(tmp_path, connect_timeout=value)
        text = str(exc.value)
        assert "IotMqttTransport" in text, f"the refusal must name the class, got {text!r}"
        assert "connect_timeout" in text, f"the refusal must name the parameter, got {text!r}"
        assert "must be > 0" in text, f"the refusal must state the domain, got {text!r}"

    @pytest.mark.parametrize("value", USABLE_TIMEOUTS)
    def test_a_usable_budget_is_stored_unchanged(self, tmp_path: Any, value: Any) -> None:
        """No coercion: the shared domain accepts these spellings as they are."""
        transport = _transport(tmp_path, connect_timeout=value)
        assert transport._connect_timeout == value

    def test_the_default_is_inside_the_domain(self, tmp_path: Any) -> None:
        """A caller who names no budget gets one the transport can spend."""
        transport = _transport(tmp_path)
        assert positive_finite_number_error(transport._connect_timeout, "connect_timeout", "x") is None


class TestAConnectingBrokerIsNoLongerReportedAsUnreachable:
    """The misdiagnosis: ``connect()`` blamed the broker for the caller's budget."""

    def test_a_usable_budget_reaches_a_connecting_broker(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control: the same fake broker connects, so the refusals below are about the value."""
        built = _install_broker(monkeypatch, _ConnectingClient)
        transport = _transport(tmp_path, connect_timeout=5.0)
        assert transport.connect() is True
        assert len(built) == 1
        assert built[0].stopped is False

    @pytest.mark.parametrize("value", [0, -1, math.nan])
    def test_a_budget_that_expires_at_once_never_reaches_the_broker(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, value: Any
    ) -> None:
        """Each of these used to return ``False`` and stop a healthy client."""
        built = _install_broker(monkeypatch, _ConnectingClient)
        with pytest.raises(ValueError, match="connect_timeout"):
            _transport(tmp_path, connect_timeout=value)
        assert built == [], "the refusal must precede client construction"

    @pytest.mark.parametrize("value", [math.inf, "15"])
    def test_a_budget_the_wait_cannot_read_no_longer_escapes_connect(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, value: Any
    ) -> None:
        """These raised out of ``connect()`` leaving a started client with no ``stop()``."""
        built = _install_broker(monkeypatch, _ConnectingClient)
        with pytest.raises(ValueError, match="connect_timeout"):
            _transport(tmp_path, connect_timeout=value)
        assert built == [], "no MQTT5 client may be started for a refused budget"

    def test_a_usable_budget_is_still_bounded_against_a_silent_broker(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The genuine timeout path is untouched: report ``False`` and stop the client."""
        built = _install_broker(monkeypatch, _SilentClient)
        transport = _transport(tmp_path, connect_timeout=0.05)
        assert transport.connect() is False
        assert built[0].stopped is True
        assert transport._client is None


class TestAnUnboundedWaitIsNotExpressibleThroughThisKnob:
    """``None`` blocks forever *and* holds the lock, so it is refused rather than read."""

    def test_none_is_refused(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="connect_timeout"):
            _transport(tmp_path, connect_timeout=None)

    def test_the_wait_would_never_return_and_would_hold_the_lock(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Premise for the refusal, measured on the real ``connect()`` body.

        ``connect_timeout`` is smuggled past the constructor guard by writing the
        attribute directly, which is the only way to reach a wait the guard now
        makes unreachable.
        """
        _install_broker(monkeypatch, _SilentClient)
        transport = _transport(tmp_path, connect_timeout=1.0)
        transport._connect_timeout = None  # type: ignore[assignment]

        returned: list[object] = []
        worker = threading.Thread(target=lambda: returned.append(transport.connect()), daemon=True)
        worker.start()
        worker.join(timeout=0.5)

        assert worker.is_alive(), "Event.wait(None) is expected to block indefinitely"
        assert returned == []
        assert transport._lock.acquire(blocking=False) is False, "connect() holds the lock while it waits"

        # Release the blocked worker so the test leaves no thread parked forever.
        transport._connected.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()

    @pytest.mark.parametrize(("value", "expected"), [(0, False), (-1, False), (math.nan, False)])
    def test_an_expired_budget_returns_at_once_rather_than_erring(self, value: Any, expected: bool) -> None:
        """Premise: these are silent on ``Event.wait``, which is why they misdiagnose."""
        assert threading.Event().wait(value) is expected

    def test_infinity_is_not_an_unbounded_wait_here_either(self) -> None:
        """Premise: ``inf`` is refused by the primitive, matching the sibling transports."""
        with pytest.raises(OverflowError):
            threading.Event().wait(math.inf)


class TestTheRefusalPrecedesTheOptionalDependency:
    """One report for one mistake, with or without the ``[mesh-iot]`` extra."""

    def test_the_value_is_refused_with_awsiotsdk_absent(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "awsiot", None)
        with pytest.raises(ValueError, match="connect_timeout"):
            _transport(tmp_path, connect_timeout=0)

    def test_a_usable_value_still_defers_to_the_missing_extra(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must not shadow the install hint for an accepted budget."""
        monkeypatch.setitem(sys.modules, "awsiot", None)
        transport = _transport(tmp_path, connect_timeout=5.0)
        assert transport.connect() is False


class TestTheThreeSurfacesShareOneDomain:
    """A budget one surface refuses cannot be accepted by another spelling the same knob."""

    @pytest.mark.parametrize("value", [*UNUSABLE_TIMEOUTS, *USABLE_TIMEOUTS])
    def test_every_surface_agrees_with_the_shared_verdict(self, tmp_path: Any, value: Any) -> None:
        shared = positive_finite_number_error(value, "connect_timeout", "Ctx") is not None
        builders: dict[str, Callable[[], object]] = {
            "IotMqttTransport": lambda: _transport(tmp_path, connect_timeout=value),
            "RemotePolicy": lambda: RemotePolicy(connect_timeout=value),
            "LerobotAsyncPolicy": lambda: LerobotAsyncPolicy(
                policy_type="act", pretrained_name_or_path="org/model", connect_timeout=value
            ),
        }
        for name, build in builders.items():
            assert _refuses_the_timeout(build) is shared, f"{name} disagrees for {value!r}"


class TestEverySurfaceTakingAConnectTimeoutRoutesThroughTheDomain:
    """Structural: a fourth surface cannot ship this knob without the shared rule.

    Scoped by the parameter name rather than a module list, so it is the
    definition of the surface set rather than a copy of it.
    """

    @staticmethod
    def _surfaces(source_root: pathlib.Path) -> dict[str, bool]:
        """Map ``file::Class.func`` to whether its body calls the shared domain."""
        found: dict[str, bool] = {}
        for path in sorted(source_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - the package parses
                continue

            def walk(node: ast.AST, cls: str | None = None) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.ClassDef):
                        walk(child, child.name)
                        continue
                    if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        continue
                    args = [a.arg for a in child.args.args + child.args.kwonlyargs]
                    if "connect_timeout" in args:
                        calls = {
                            call.func.id
                            for call in ast.walk(child)
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        }
                        label = f"{path.relative_to(source_root)}::{cls + '.' if cls else ''}{child.name}"
                        found[label] = "positive_finite_number_error" in calls
                    walk(child, cls)

            walk(tree)
        return found

    #: Every shipped surface taking the parameter, as of this change.
    EXPECTED = {
        "inference/client.py::RemotePolicy.__init__",
        "mesh/transport/iot_transport.py::IotMqttTransport.__init__",
        "policies/lerobot_async/policy.py::LerobotAsyncPolicy.__init__",
    }

    @staticmethod
    def _source_root() -> pathlib.Path:
        """Derive the package root from a symbol, never from a path literal."""
        return pathlib.Path(inspect.getfile(IotMqttTransport)).parents[2]

    def test_the_scan_finds_exactly_the_known_surfaces(self) -> None:
        """Non-vacuity: a scan resolving elsewhere reports a clean sweep over nothing."""
        assert set(self._surfaces(self._source_root())) == self.EXPECTED

    def test_no_surface_takes_the_parameter_without_the_shared_domain(self) -> None:
        adrift = sorted(name for name, guarded in self._surfaces(self._source_root()).items() if not guarded)
        assert adrift == [], f"connect_timeout is accepted without the shared domain by: {adrift}"

    def test_the_scan_detects_a_surface_that_skips_the_domain(self, tmp_path: Any) -> None:
        """Meta: an empty result must mean clean sources, not a scanner matching nothing."""
        planted = tmp_path / "pkg"
        planted.mkdir()
        (planted / "leaky.py").write_text(
            "class Rogue:\n    def __init__(self, connect_timeout: float = 1.0) -> None:\n"
            "        self._t = connect_timeout\n",
            encoding="utf-8",
        )
        found = self._surfaces(planted)
        assert found == {"leaky.py::Rogue.__init__": False}


def test_the_module_under_test_is_the_one_the_scan_root_points_at() -> None:
    """Guard the symbol-derived root: it must be the shipped package directory."""
    root = TestEverySurfaceTakingAConnectTimeoutRoutesThroughTheDomain._source_root()
    assert root.name == "strands_robots"
    assert (root / "mesh" / "transport" / "iot_transport.py").exists()
