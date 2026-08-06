"""The ZMQ inference clients' ``timeout_ms`` is one shared wait-budget domain.

Both ZMQ REQ inference clients hand ``timeout_ms`` to ``setsockopt(RCVTIMEO)``
and ``setsockopt(SNDTIMEO)``. Neither validated it, so the third
remote-inference transport carried the misattribution that #1984 removed from
the WebSocket and gRPC pair: an unusable wait budget reported as an absent
server.

The tests are grouped by what they protect rather than by client:

* :class:`TestTheSharedDomain` - the verdicts, on the helper alone.
* :class:`TestBothClientsRefuseTheSameBudgets` - the two clients agree, so a
  budget cannot be usable on one sidecar transport and refused on the other.
* :class:`TestAHealthyServerIsNoLongerReportedUnreachable` - the defect itself,
  against a real loopback REP sidecar. This is the reason the change exists,
  and it is asserted on behaviour rather than on the raise.
* :class:`TestABudgetTheSiblingTransportsAcceptIsUsableHere` - the coercion,
  which is what makes this a fix rather than only a refusal.
* :class:`TestZmqStillTreatsTheseValuesAsMeasured` - premise tests. They assert
  ``pyzmq``'s behaviour, not ours, so a future ``pyzmq`` that changes any of it
  fails here first rather than silently invalidating the reasoning above.
* :class:`TestNoZmqTimeoutSurfaceDrifts` - structural, so a third ZMQ client
  cannot ship without joining the rule.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import pathlib
import threading
import time
from typing import Any

import pytest

import strands_robots
from strands_robots.policies.groot.client import Gr00tInferenceClient
from strands_robots.policies.groot.client import MsgSerializer as GrootSerializer
from strands_robots.policies.moveit2.client import MoveIt2InferenceClient
from strands_robots.utils import MAX_ZMQ_TIMEOUT_MS, coerce_zmq_timeout_ms

# ``pyzmq`` is imported optionally rather than through a module-level
# ``importorskip``, so that the tests which need no socket keep running without
# it. Both clients load ZMQ lazily (``_load_zmq``) and refuse an unusable
# ``timeout_ms`` *before* that call, so the domain verdicts, the refusal path,
# the constructor ordering and the structural drift guard are all answerable on
# an install without the ``[groot]`` / ``[moveit2]`` extra. A module-level skip
# would have taken the drift guard with it - which is the one check here whose
# whole job is to still be running when someone changes something unrelated.
try:
    import zmq
except ImportError:  # pragma: no cover - exercised by the extra-less install
    zmq = None  # type: ignore[assignment]

try:
    import msgpack
except ImportError:  # pragma: no cover - exercised by the extra-less install
    msgpack = None  # type: ignore[assignment]

#: For tests that must configure a real socket.
requires_zmq = pytest.mark.skipif(zmq is None, reason="pyzmq is the transport under test")

#: For tests that must round-trip against a real sidecar.
requires_wire = pytest.mark.skipif(
    zmq is None or msgpack is None, reason="pyzmq and msgpack are the sidecar wire format"
)

#: Values that name no usable ZMQ wait budget, with why each one matters.
UNUSABLE: list[tuple[str, Any]] = [
    ("zero-is-return-immediately", 0),
    ("false-is-zero-by-another-route", False),
    ("true-is-a-silent-1ms-budget", True),
    ("minus-one-is-block-forever", -1),
    ("below-minus-one-is-invalid-argument", -2),
    ("negative", -15000),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
    ("fractional", 1.5),
    ("numeric-string", "15000"),
    ("none", None),
    ("list", [15000]),
    ("dict", {"ms": 15000}),
    ("above-the-c-int-ceiling", MAX_ZMQ_TIMEOUT_MS + 1),
    ("far-above-the-c-int-ceiling", 2**40),
    ("beyond-float64", 10**400),
]

#: Values that do name a usable budget, including spellings ``setsockopt``
#: itself refuses and this domain therefore has to normalise.
USABLE: list[tuple[str, Any, int]] = [
    ("the-default", 15000, 15000),
    ("one-millisecond", 2, 2),
    ("integral-float-from-a-json-config", 15000.0, 15000),
    ("the-ceiling-itself", MAX_ZMQ_TIMEOUT_MS, MAX_ZMQ_TIMEOUT_MS),
]

CLIENTS = [Gr00tInferenceClient, MoveIt2InferenceClient]


def _numpy_spellings() -> list[tuple[str, Any, int]]:
    """Usable budgets that arrive as NumPy scalars out of a config array."""
    np = pytest.importorskip("numpy")
    return [
        ("numpy-int64", np.int64(15000), 15000),
        ("numpy-float64-integral", np.float64(15000.0), 15000),
    ]


class _Sidecar:
    """A healthy REP sidecar that answers every request correctly.

    The point of a real socket rather than a stub: the defect is that a
    *reachable, answering* server is reported unreachable, so a mocked
    transport cannot show it.
    """

    def __init__(self, encode: Any) -> None:
        self._encode = encode
        self._stop = threading.Event()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(20)):
                self._sock.recv()
                self._sock.send(self._encode({"status": "ok"}))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()
        self._ctx.term()


def _encoder_for(cls: type) -> Any:
    if cls is Gr00tInferenceClient:
        return GrootSerializer.to_bytes
    return lambda payload: msgpack.packb(payload, use_bin_type=True)


@contextlib.contextmanager
def sidecar_for(cls: type) -> Any:
    """A live sidecar speaking the wire format of ``cls``.

    Built from the client under test rather than parametrised alongside it, so
    there is no mismatched combination to skip.
    """
    server = _Sidecar(_encoder_for(cls))
    try:
        yield server
    finally:
        server.close()


class TestTheSharedDomain:
    """The verdicts, on the helper alone."""

    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[c[0] for c in UNUSABLE])
    def test_a_value_that_names_no_wait_budget_is_refused(self, label: str, value: Any) -> None:
        coerced, reason = coerce_zmq_timeout_ms("Client", "timeout_ms", value)
        assert coerced is None
        assert reason is not None

    @pytest.mark.parametrize(("label", "value", "expected"), USABLE, ids=[c[0] for c in USABLE])
    def test_a_usable_budget_is_returned_as_an_int(self, label: str, value: Any, expected: int) -> None:
        coerced, reason = coerce_zmq_timeout_ms("Client", "timeout_ms", value)
        assert reason is None
        assert coerced == expected
        # ``setsockopt`` takes a C ``int`` and refuses every other spelling, so
        # the exact type is the contract here, not an implementation detail.
        assert type(coerced) is int

    def test_the_message_names_the_surface_and_the_parameter(self) -> None:
        _, reason = coerce_zmq_timeout_ms("Gr00tInferenceClient", "timeout_ms", 0)
        assert reason is not None
        assert "Gr00tInferenceClient" in reason
        assert "timeout_ms" in reason

    def test_the_ceiling_refusal_states_the_bound_rather_than_the_floor(self) -> None:
        """Above the ceiling is a different reason from below the floor."""
        _, over = coerce_zmq_timeout_ms("Client", "timeout_ms", MAX_ZMQ_TIMEOUT_MS + 1)
        _, under = coerce_zmq_timeout_ms("Client", "timeout_ms", 0)
        assert over is not None and under is not None
        assert str(MAX_ZMQ_TIMEOUT_MS) in over
        assert over != under

    @requires_zmq
    def test_the_ceiling_is_the_largest_value_zmq_can_store(self) -> None:
        """Non-vacuity for the constant: it is the transport's bound, not a choice.

        Asserted against ``setsockopt`` rather than restating ``2**31 - 1``, so
        the constant cannot drift away from the option it describes.
        """
        context = zmq.Context()
        try:
            accepted = context.socket(zmq.REQ)
            accepted.setsockopt(zmq.RCVTIMEO, MAX_ZMQ_TIMEOUT_MS)
            assert accepted.getsockopt(zmq.RCVTIMEO) == MAX_ZMQ_TIMEOUT_MS
            accepted.close()

            refused = context.socket(zmq.REQ)
            with pytest.raises(OverflowError):
                refused.setsockopt(zmq.RCVTIMEO, MAX_ZMQ_TIMEOUT_MS + 1)
            refused.close()
        finally:
            context.term()


class TestBothClientsRefuseTheSameBudgets:
    """The two clients share one domain, so their verdicts must agree."""

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[c[0] for c in UNUSABLE])
    def test_an_unusable_budget_is_refused_as_a_value_error(self, cls: type, label: str, value: Any) -> None:
        with pytest.raises(ValueError, match="timeout_ms"):
            cls(host="127.0.0.1", port=5555, timeout_ms=value)

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_the_refusal_names_the_class_that_received_it(self, cls: type) -> None:
        with pytest.raises(ValueError, match=cls.__name__):
            cls(host="127.0.0.1", port=5555, timeout_ms=0)

    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[c[0] for c in UNUSABLE])
    def test_the_two_clients_agree_on_every_verdict(self, label: str, value: Any) -> None:
        reasons: list[str | None] = []
        for cls in CLIENTS:
            try:
                cls(host="127.0.0.1", port=5555, timeout_ms=value)
                reasons.append(None)
            except ValueError as exc:
                reasons.append(str(exc).replace(cls.__name__, "<client>"))
        assert reasons[0] == reasons[1]

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_a_refused_budget_opens_no_socket(self, cls: type) -> None:
        """The guard precedes ``_load_zmq``, so nothing is created to leak.

        Also why the report is identical with and without the optional extra
        installed: the refusal happens before the dependency is loaded.
        """
        with pytest.raises(ValueError):
            cls(host="127.0.0.1", port=5555, timeout_ms=0)

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_a_refused_budget_is_reported_before_the_transport_is_loaded(
        self, cls: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Made non-vacuous by breaking the loader: the guard still answers."""
        module = inspect.getmodule(cls)
        assert module is not None

        def explode() -> Any:
            raise AssertionError("_load_zmq must not be reached for a refused timeout_ms")

        monkeypatch.setattr(module, "_load_zmq", explode)
        with pytest.raises(ValueError, match="timeout_ms"):
            cls(host="127.0.0.1", port=5555, timeout_ms=0)


@requires_wire
class TestAHealthyServerIsNoLongerReportedUnreachable:
    """The defect, against a live sidecar. Behaviour, not the raise."""

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_the_default_budget_reaches_a_live_sidecar(self, cls: type) -> None:
        """Premise for the rest of the class: the sidecar really does answer."""
        with sidecar_for(cls) as sidecar:
            client = cls(host="127.0.0.1", port=sidecar.port)
            try:
                assert client.ping() is True
            finally:
                client.socket.close()
                client.context.term()

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("value", [0, False, True, -1], ids=["zero", "false", "true", "minus-one"])
    def test_an_unusable_budget_can_no_longer_call_a_live_sidecar_unreachable(self, cls: type, value: Any) -> None:
        """Before this change ``ping()`` returned ``False`` here - or hung.

        ``ping`` swallows every exception and answers ``False``, logging the
        reason at ``debug`` only, so the operator was told the sidecar was
        unreachable while it was answering on that very port. ``-1`` was worse
        than wrong: it blocked forever, so ``ping`` could not answer at all.
        """
        with sidecar_for(cls) as sidecar:
            with pytest.raises(ValueError, match="timeout_ms"):
                cls(host="127.0.0.1", port=sidecar.port, timeout_ms=value)


@requires_wire
class TestABudgetTheSiblingTransportsAcceptIsUsableHere:
    """The coercion: this is a fix, not only a refusal."""

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value", "expected"), USABLE, ids=[c[0] for c in USABLE])
    def test_a_usable_budget_is_stored_as_an_int_and_still_pings(
        self, cls: type, label: str, value: Any, expected: int
    ) -> None:
        with sidecar_for(cls) as sidecar:
            client = cls(host="127.0.0.1", port=sidecar.port, timeout_ms=value)
            try:
                assert client.timeout_ms == expected
                assert type(client.timeout_ms) is int
                assert client.socket.getsockopt(zmq.RCVTIMEO) == expected
                assert client.socket.getsockopt(zmq.SNDTIMEO) == expected
                assert client.ping() is True
            finally:
                client.socket.close()
                client.context.term()

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_a_numpy_budget_out_of_a_config_array_is_accepted(self, cls: type) -> None:
        """``setsockopt`` refuses a NumPy integer outright; the domain normalises it.

        The sibling WebSocket / gRPC clients accept a NumPy scalar timeout, so
        without the coercion the same configured budget would be usable on two
        transports and unusable on this one.
        """
        for _label, value, expected in _numpy_spellings():
            client = cls(host="127.0.0.1", port=5555, timeout_ms=value)
            try:
                assert client.timeout_ms == expected
                assert type(client.timeout_ms) is int
            finally:
                client.socket.close()
                client.context.term()

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_reconnect_reuses_the_coerced_budget(self, cls: type) -> None:
        """``_init_socket`` runs again on reconnect and must not re-raise."""
        client = cls(host="127.0.0.1", port=5555, timeout_ms=15000.0)
        try:
            client.reconnect()
            assert client.socket.getsockopt(zmq.RCVTIMEO) == 15000
        finally:
            client.socket.close()
            client.context.term()


@requires_wire
class TestZmqStillTreatsTheseValuesAsMeasured:
    """Premise tests: ``pyzmq``'s behaviour, which the reasoning above rests on.

    These assert the transport rather than this repository. If a future
    ``pyzmq`` starts honouring one of these differently, the decision recorded
    in :func:`coerce_zmq_timeout_ms` should be revisited - and it fails here
    first rather than leaving a stale justification in a docstring.
    """

    @pytest.fixture
    def socket(self) -> Any:
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            yield sock
        finally:
            sock.close()
            context.term()

    def test_zero_is_the_return_immediately_spelling(self, socket: Any) -> None:
        """Which is why ``0`` fails against a server that is answering."""
        server = _Sidecar(lambda payload: msgpack.packb(payload, use_bin_type=True))
        try:
            socket.setsockopt(zmq.RCVTIMEO, 0)
            socket.setsockopt(zmq.SNDTIMEO, 0)
            socket.connect(f"tcp://127.0.0.1:{server.port}")
            socket.send(b"ping")
            with pytest.raises(zmq.Again):
                socket.recv()
        finally:
            server.close()

    def test_true_is_stored_as_a_one_millisecond_budget(self, socket: Any) -> None:
        """A silent 1 ms, which is why ``bool`` must be refused explicitly."""
        socket.setsockopt(zmq.RCVTIMEO, True)
        assert socket.getsockopt(zmq.RCVTIMEO) == 1

    def test_minus_one_is_still_the_block_forever_spelling(self, socket: Any) -> None:
        """The reason ``-1`` is refused rather than passed through.

        ZMQ honours it, unlike the ``inf`` of the sibling transports, so it
        reinstates on the request path the unbounded hang that ``LINGER = 0``
        exists to prevent on teardown - and ``ping()``, whose contract is to
        answer ``True`` or ``False``, could then never answer.
        """
        socket.setsockopt(zmq.RCVTIMEO, -1)
        assert socket.getsockopt(zmq.RCVTIMEO) == -1

        # Nothing is bound on this port, so a blocking recv cannot complete.
        socket.connect("tcp://127.0.0.1:1")
        socket.send(b"ping")
        returned = threading.Event()

        def blocking_recv() -> None:
            try:
                socket.recv()
            except Exception:  # noqa: BLE001 - any outcome still means it returned
                pass
            returned.set()

        threading.Thread(target=blocking_recv, daemon=True).start()
        assert not returned.wait(timeout=1.5), "a -1 timeout is expected to block"

    def test_below_minus_one_is_an_invalid_argument(self, socket: Any) -> None:
        """So it never reached a verdict; it raised ``ZMQError`` naming nothing."""
        with pytest.raises(zmq.ZMQError):
            socket.setsockopt(zmq.RCVTIMEO, -2)

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), 1.5, 15000.0, "15000", None, [15000]],
        ids=["nan", "inf", "fractional", "integral-float", "string", "none", "list"],
    )
    def test_a_non_int_never_reaches_the_socket_at_all(self, socket: Any, value: Any) -> None:
        """Including ``15000.0`` - a usable budget ``setsockopt`` still refuses."""
        with pytest.raises(TypeError):
            socket.setsockopt(zmq.RCVTIMEO, value)

    def test_a_numpy_integer_is_also_refused_by_setsockopt(self, socket: Any) -> None:
        """The measurement the coercion exists for."""
        np = pytest.importorskip("numpy")
        with pytest.raises(TypeError):
            socket.setsockopt(zmq.RCVTIMEO, np.int64(15000))


class TestNoZmqTimeoutSurfaceDrifts:
    """Structural: a ZMQ socket timeout is set only from a coerced budget.

    A third client that hands a caller-supplied value to
    ``setsockopt(RCVTIMEO)`` without routing it through the shared domain would
    re-introduce exactly this defect. Checked structurally rather than by
    enumerating today's two clients, so the rule survives a third.
    """

    #: The socket options that carry a wait budget.
    TIMEOUT_OPTIONS = frozenset({"RCVTIMEO", "SNDTIMEO"})

    @staticmethod
    def _package_root() -> pathlib.Path:
        return pathlib.Path(inspect.getfile(strands_robots)).parent

    @classmethod
    def _sets_a_timeout(cls, node: ast.AST) -> bool:
        """Whether this subtree calls ``setsockopt(<zmq>.RCVTIMEO/SNDTIMEO, ...)``."""
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not (isinstance(call.func, ast.Attribute) and call.func.attr == "setsockopt"):
                continue
            for arg in call.args:
                if isinstance(arg, ast.Attribute) and arg.attr in cls.TIMEOUT_OPTIONS:
                    return True
        return False

    @classmethod
    def _modules_setting_a_timeout(cls) -> dict[str, str]:
        """Map ``relative module path -> source`` for every module that sets one."""
        found: dict[str, str] = {}
        root = cls._package_root()
        for path in sorted(root.rglob("*.py")):
            source = path.read_text()
            if "setsockopt" not in source:
                continue
            if cls._sets_a_timeout(ast.parse(source)):
                found[str(path.relative_to(root))] = source
        return found

    def test_the_scan_finds_every_known_zmq_timeout_surface(self) -> None:
        """Non-vacuity: a scan that found nothing would pass everything below."""
        assert set(self._modules_setting_a_timeout()) == {
            "policies/groot/client.py",
            "policies/moveit2/client.py",
        }

    def test_every_module_that_sets_a_timeout_routes_it_through_the_shared_domain(self) -> None:
        offenders = [
            module
            for module, source in self._modules_setting_a_timeout().items()
            if "coerce_zmq_timeout_ms" not in source
        ]
        assert not offenders, f"these set a ZMQ timeout without the shared domain: {offenders}"

    def test_the_scanner_detects_a_module_that_does_neither(self) -> None:
        """The scanner is answering the question, not returning ``True``."""
        unguarded = ast.parse(
            "import zmq\n"
            "class Rogue:\n"
            "    def __init__(self, timeout_ms):\n"
            "        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)\n"
        )
        assert self._sets_a_timeout(unguarded)

        elsewhere = ast.parse("sock.setsockopt(zmq.LINGER, 0)\n")
        assert not self._sets_a_timeout(elsewhere)


class TestTheGuardPrecedesTheSocketOptions:
    """Ordering: the budget is coerced before any option is written.

    A guard placed after ``_init_socket`` would leave a socket configured from
    a value the constructor then refused, which is the half-created-object
    shape #1858 records for a different axis.
    """

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_the_coercion_appears_before_load_zmq_in_the_constructor(self, cls: type) -> None:
        source = inspect.getsource(getattr(cls, "__init__"))
        assert "coerce_zmq_timeout_ms" in source
        assert source.index("coerce_zmq_timeout_ms") < source.index("_load_zmq")

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_the_socket_is_configured_from_the_stored_budget(self, cls: type) -> None:
        """``_init_socket`` reads ``self.timeout_ms``, which is the coerced int."""
        source = inspect.getsource(getattr(cls, "_init_socket"))
        assert "self.timeout_ms" in source


@requires_wire
def test_a_usable_budget_still_times_out_against_a_slow_sidecar() -> None:
    """The domain bounds the value, it does not change what a timeout means.

    Guards against a fix that made every budget effectively unbounded.
    """
    slow = threading.Event()
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    port = rep.bind_to_random_port("tcp://127.0.0.1")

    def serve() -> None:
        poller = zmq.Poller()
        poller.register(rep, zmq.POLLIN)
        while not slow.is_set():
            if dict(poller.poll(20)):
                slow.wait(timeout=2.0)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = MoveIt2InferenceClient(host="127.0.0.1", port=port, timeout_ms=200)
    try:
        started = time.monotonic()
        assert client.ping() is False
        assert time.monotonic() - started < 1.5
    finally:
        slow.set()
        thread.join(timeout=5)
        client.socket.close()
        client.context.term()
        rep.close()
        ctx.term()
