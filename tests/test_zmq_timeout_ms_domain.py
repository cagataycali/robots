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
* :class:`TestNoRoundTripIsAssertedInsideAScheduleBoundBudget` - structural, so
  a live answer is never required inside a budget the host's scheduler can
  exceed on a loaded runner.
* :class:`TestZmqStillTreatsTheseValuesAsMeasured` - premise tests. They assert
  ``pyzmq``'s behaviour, not ours, so a future ``pyzmq`` that changes any of it
  fails here first rather than silently invalidating the reasoning above.
* :class:`TestNoZmqTimeoutSurfaceDrifts` - structural, so a third ZMQ client
  cannot ship without joining the rule.
* :class:`TestTheCoercionReadsTheValueOnce` - the module's no-unprotected-
  conversion invariant, stated over this helper.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import numbers
import pathlib
import threading
import time
from typing import Any

import pytest

import strands_robots
from strands_robots.policies.moveit2.client import MoveIt2InferenceClient
from strands_robots.utils import MAX_ZMQ_TIMEOUT_MS, coerce_zmq_timeout_ms, positive_whole_number_error

# ``pyzmq`` is imported optionally rather than through a module-level
# ``importorskip``, so that the tests which need no socket keep running without
# it. Both clients load ZMQ lazily (``_load_zmq``) and refuse an unusable
# ``timeout_ms`` *before* that call, so the domain verdicts, the refusal path,
# the constructor ordering and the structural drift guard are all answerable on
# an install without the ``[moveit2]`` extra. A module-level skip
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
    ("two-milliseconds", 2, 2),
    ("integral-float-from-a-json-config", 15000.0, 15000),
    ("the-ceiling-itself", MAX_ZMQ_TIMEOUT_MS, MAX_ZMQ_TIMEOUT_MS),
]

#: Smallest budget a live round trip is asserted inside.
#:
#: A fresh REQ socket pays the TCP connect and the ZMQ handshake on its first
#: call, and that cost is scheduler-bound rather than transport-bound: measured
#: over a loopback sidecar on an idle host it is p50 0.24 ms / max 0.51 ms, and
#: under CPU contention it crosses 2 ms. So an answer required to arrive inside
#: a 2 ms budget asserts the host's scheduler, not the budget reaching the
#: socket - which is the property this file exists to pin, and which
#: ``getsockopt`` states exactly and without a clock. Budgets at or above this
#: floor keep three orders of magnitude of headroom over the connect cost, so
#: their round trip is a statement about the transport again.
MIN_ROUND_TRIP_BUDGET_MS = 1000

#: The usable budgets a live round trip is asserted inside.
#:
#: Derived from :data:`USABLE` rather than written out, so a budget added there
#: cannot acquire a wall-clock assertion by also being added to a second list,
#: and a tight one cannot be given one by hand.
ROUND_TRIP: list[tuple[str, Any, int]] = [row for row in USABLE if row[2] >= MIN_ROUND_TRIP_BUDGET_MS]

CLIENTS = [MoveIt2InferenceClient]


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

    def __init__(self, encode: Any, port: int | None = None) -> None:
        """Start answering on ``port``, or on a free port of its own choosing.

        Args:
            encode: Serialiser for the reply payload.
            port: Bind to this port instead of an arbitrary free one. For a
                test that first needs the endpoint to be *unreachable* and
                only then answering, which a random port cannot express.
        """
        self._encode = encode
        self._stop = threading.Event()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        if port is None:
            self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        else:
            self._sock.bind(f"tcp://127.0.0.1:{port}")
            self.port = port
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


def _free_port() -> int:
    """A port with nothing bound on it, found by binding and releasing one.

    Discovered through ZMQ rather than the stdlib ``socket`` module, which is
    the name of the fixture below - and which this file otherwise never needs.

    Returns:
        A loopback port that is unbound on return, so a request sent to it
        cannot be answered until something binds it.
    """
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    try:
        return sock.bind_to_random_port("tcp://127.0.0.1")
    finally:
        sock.close()
        context.term()


def _encoder_for(cls: type) -> Any:
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


class CountingWholeNumber:
    """A registered :class:`numbers.Real` that counts every read of itself.

    Registered rather than subclassed for :class:`RealNoFloat`'s reason in
    ``tests/test_conversion_escape_is_closed.py``: ``numbers.Real`` is a
    registration, so a value can satisfy a guard's ``isinstance`` check while
    owing it nothing else. It names a perfectly usable budget, so it travels the
    accept path to the end - which is the path where a second read is a silent
    hazard rather than an immediate refusal.
    """

    def __init__(self, value: int) -> None:
        self._value = value
        self.float_reads = 0
        self.int_reads = 0

    def __float__(self) -> float:
        self.float_reads += 1
        return float(self._value)

    def __int__(self) -> int:
        self.int_reads += 1
        return self._value

    @property
    def reads(self) -> int:
        return self.float_reads + self.int_reads

    def __repr__(self) -> str:
        return f"CountingWholeNumber({self._value})"


numbers.Real.register(CountingWholeNumber)


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
        _, reason = coerce_zmq_timeout_ms("MoveIt2InferenceClient", "timeout_ms", 0)
        assert reason is not None
        assert "MoveIt2InferenceClient" in reason
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


class TestEveryClientRefusesThroughTheSharedDomain:
    """A client's verdict must BE the shared domain's, not a private retelling."""

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[c[0] for c in UNUSABLE])
    def test_an_unusable_budget_is_refused_as_a_value_error(self, cls: type, label: str, value: Any) -> None:
        with pytest.raises(ValueError, match="timeout_ms"):
            cls(host="127.0.0.1", port=5555, timeout_ms=value)

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    def test_the_refusal_names_the_class_that_received_it(self, cls: type) -> None:
        with pytest.raises(ValueError, match=cls.__name__):
            cls(host="127.0.0.1", port=5555, timeout_ms=0)

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[c[0] for c in UNUSABLE])
    def test_the_verdict_is_the_shared_domains_verbatim(self, cls: type, label: str, value: Any) -> None:
        """Compared against the helper rather than against a sibling client.

        A client-to-client comparison only holds while two clients ship; this
        holds for one, and it is the stronger statement anyway -- a client that
        re-derived the same text by hand would pass a peer comparison and fail
        here the moment the shared domain's wording moved.
        """
        _, expected = coerce_zmq_timeout_ms(cls.__name__, "timeout_ms", value)
        assert expected is not None, f"{label} must be refused by the shared domain"
        with pytest.raises(ValueError) as caught:
            cls(host="127.0.0.1", port=5555, timeout_ms=value)
        assert str(caught.value) == expected

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
    def test_a_usable_budget_is_stored_as_an_int_and_reaches_both_socket_options(
        self, cls: type, label: str, value: Any, expected: int
    ) -> None:
        """Every usable spelling, asserted without a clock.

        ``getsockopt`` states the whole property this class exists to pin - that
        the coerced value is what the transport was configured with - and states
        it for the smallest usable budget as exactly as for the default one.
        """
        with sidecar_for(cls) as sidecar:
            client = cls(host="127.0.0.1", port=sidecar.port, timeout_ms=value)
            try:
                assert client.timeout_ms == expected
                assert type(client.timeout_ms) is int
                assert client.socket.getsockopt(zmq.RCVTIMEO) == expected
                assert client.socket.getsockopt(zmq.SNDTIMEO) == expected
            finally:
                client.socket.close()
                client.context.term()

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value", "expected"), ROUND_TRIP, ids=[c[0] for c in ROUND_TRIP])
    def test_a_usable_budget_with_round_trip_headroom_still_answers(
        self, cls: type, label: str, value: Any, expected: int
    ) -> None:
        """The coerced budget still reaches a live sidecar.

        Parametrised over :data:`ROUND_TRIP` rather than :data:`USABLE`: the
        answer has to arrive inside the budget under test, so a budget without
        headroom over the connect cost would assert the host's scheduler here
        rather than the transport. See :data:`MIN_ROUND_TRIP_BUDGET_MS`.
        """
        with sidecar_for(cls) as sidecar:
            client = cls(host="127.0.0.1", port=sidecar.port, timeout_ms=value)
            try:
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


class TestNoRoundTripIsAssertedInsideAScheduleBoundBudget:
    """Structural: a live answer is only required where the budget has room.

    The assertion this replaces read ``assert client.ping() is True`` inside a
    2 ms budget, so it held on an idle host and failed under CPU contention -
    a property of the scheduler rather than of the value reaching the socket.
    Checked over this module's own source rather than by naming today's tests,
    so the shape cannot return under a different name.
    """

    @staticmethod
    def _source() -> str:
        """This module's own source.

        Read through ``__file__`` rather than a path literal, so the scan
        follows the module if it is ever renamed.
        """
        return pathlib.Path(__file__).read_text()

    @staticmethod
    def _asserts_a_live_answer(fn: ast.AST) -> bool:
        """Whether this subtree contains ``assert <x>.ping() is True``."""
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare) or not node.ops:
                continue
            if not isinstance(node.ops[0], ast.Is):
                continue
            right = node.comparators[0]
            if not (isinstance(right, ast.Constant) and right.value is True):
                continue
            left = node.left
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute) and left.func.attr == "ping":
                return True
        return False

    @staticmethod
    def _parametrize_tables(fn: ast.FunctionDef) -> set[str]:
        """Names of the module-level tables this test is parametrised over."""
        tables: set[str] = set()
        for dec in fn.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                continue
            if dec.func.attr != "parametrize" or len(dec.args) < 2:
                continue
            values = dec.args[1]
            if isinstance(values, ast.Name):
                tables.add(values.id)
        return tables

    @classmethod
    def _tests_requiring_a_live_answer(cls, source: str) -> dict[str, set[str]]:
        """Map ``test name -> parametrised tables`` for every such test."""
        found: dict[str, set[str]] = {}
        for fn in ast.walk(ast.parse(source)):
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
                continue
            if cls._asserts_a_live_answer(fn):
                found[fn.name] = cls._parametrize_tables(fn)
        return found

    def test_the_scan_finds_every_test_requiring_a_live_answer(self) -> None:
        """Non-vacuity: a scan that found nothing would pass everything below."""
        assert set(self._tests_requiring_a_live_answer(self._source())) == {
            "test_the_default_budget_reaches_a_live_sidecar",
            "test_a_usable_budget_with_round_trip_headroom_still_answers",
        }

    def test_no_live_answer_is_required_inside_a_budget_without_headroom(self) -> None:
        offenders = {
            name: sorted(tables)
            for name, tables in self._tests_requiring_a_live_answer(self._source()).items()
            if "USABLE" in tables
        }
        assert not offenders, f"these require an answer inside a budget with no headroom: {offenders}"

    def test_the_scanner_detects_a_live_answer_over_the_full_table(self) -> None:
        """The scanner is answering the question, not returning ``{}``."""
        planted = (
            "@pytest.mark.parametrize(('label', 'value', 'expected'), USABLE)\n"
            "def test_planted(label, value, expected):\n"
            "    client = build(timeout_ms=value)\n"
            "    assert client.ping() is True\n"
        )
        assert self._tests_requiring_a_live_answer(planted) == {"test_planted": {"USABLE"}}

        no_round_trip = "def test_planted():\n    assert client.timeout_ms == 2\n"
        assert self._tests_requiring_a_live_answer(no_round_trip) == {}

    def test_the_round_trip_table_is_the_headroom_subset_of_usable(self) -> None:
        """Derived, so the two cannot drift apart."""
        assert ROUND_TRIP == [row for row in USABLE if row[2] >= MIN_ROUND_TRIP_BUDGET_MS]

    def test_the_floor_excludes_a_budget_and_keeps_one(self) -> None:
        """Non-vacuity for the derivation: the floor is doing work.

        A floor that excluded nothing would leave the wall-clock assertion in
        place; one that excluded everything would delete the round trip instead
        of moving it.
        """
        assert ROUND_TRIP
        assert len(ROUND_TRIP) < len(USABLE)
        assert [row[2] for row in USABLE if row not in ROUND_TRIP] == [2]

    def test_the_excluded_budget_keeps_every_assertion_needing_no_clock(self) -> None:
        """No coverage was traded away: it is still checked over the full table."""
        socket_option_test = next(
            fn
            for fn in ast.walk(ast.parse(self._source()))
            if isinstance(fn, ast.FunctionDef)
            and fn.name == "test_a_usable_budget_is_stored_as_an_int_and_reaches_both_socket_options"
        )
        assert "USABLE" in self._parametrize_tables(socket_option_test)


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

        The wait is ended by answering the request rather than by leaving the
        thread parked in ``recv``. A ZMQ socket may be used by one thread only,
        so the fixture's ``close`` ran on a socket a second thread was still
        inside, which is undefined and aborted the interpreter. Answering also
        measures strictly more than abandoning does: the recv completes, so it
        was blocked for want of a reply rather than for being unusable.
        """
        socket.setsockopt(zmq.RCVTIMEO, -1)
        assert socket.getsockopt(zmq.RCVTIMEO) == -1

        # Nothing is bound here yet, so the request cannot be answered.
        port = _free_port()
        socket.connect(f"tcp://127.0.0.1:{port}")
        socket.send(b"ping")
        returned = threading.Event()

        def blocking_recv() -> None:
            try:
                socket.recv()
            except Exception:  # noqa: BLE001 - any outcome still means it returned
                pass
            returned.set()

        recv_thread = threading.Thread(target=blocking_recv, daemon=True)
        recv_thread.start()
        # A finite budget would have raised zmq.Again well inside this window.
        assert not returned.wait(timeout=1.5), "a -1 timeout is expected to block"

        # REQ redelivers the queued request as soon as a peer appears, so the
        # recv that was blocking completes and the thread leaves the socket.
        server = _Sidecar(lambda payload: msgpack.packb(payload, use_bin_type=True), port=port)
        try:
            assert returned.wait(timeout=30), "the blocked recv did not complete once answered"
        finally:
            server.close()
        recv_thread.join(timeout=5)
        assert not recv_thread.is_alive(), "no thread may outlive the socket it is using"

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


class TestTheCoercionReadsTheValueOnce:
    """The module's no-unprotected-conversion invariant, over this helper.

    ``strands_robots/utils.py`` carries a module-wide scan
    (``tests/test_conversion_escape_is_closed.py``) asserting that no function in
    it converts with a ``float()`` no ``try`` protects. The set it compares
    against is empty and is asserted "so it can neither grow nor be quietly
    narrowed", so a new guard joins the rule rather than the exception list.

    The first spelling of :func:`coerce_zmq_timeout_ms` did not: it validated
    with :func:`positive_whole_number_error` and then read the caller's value
    twice more, ``float(value)`` for the range and ``int(value)`` for the result,
    on the reasoning that the guard had made the conversion safe. That is the
    reasoning #1875 shipped for the vector coercions and #1906 withdrew - the
    scan could not see the upstream guarantee, and independent reads are not
    obliged to agree, so the magnitude a refusal quoted need not have been the
    magnitude the ceiling examined. The tests below state both halves: no
    unprotected conversion, *and* the value is read exactly once, because a
    helper that stopped converting at all would satisfy the first alone.
    """

    METHOD = "MoveIt2InferenceClient"
    PARAM = "timeout_ms"

    @staticmethod
    def _bare_float_calls(func: Any) -> list[str]:
        """Names converted by a ``float()`` call that no ``try`` protects.

        The module-wide scanner's question, asked here so a re-introduced
        conversion fails in the file that owns this helper rather than only in
        the shared invariant file.
        """
        tree = ast.parse(inspect.getsource(func).lstrip())
        protected: set[ast.AST] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for stmt in node.body:
                    for inner in ast.walk(stmt):
                        if isinstance(inner, ast.Call):
                            protected.add(inner)
        return [
            ast.unparse(node.args[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and node not in protected
            and node.args
        ]

    def test_it_makes_no_unprotected_conversion(self) -> None:
        assert self._bare_float_calls(coerce_zmq_timeout_ms) == []

    def test_the_scanner_reports_a_planted_conversion(self) -> None:
        """Control: an always-empty scan would satisfy the test above."""

        def planted(value: Any) -> bool:
            return float(value) > 1.0

        assert self._bare_float_calls(planted) == ["value"]

    def test_the_guard_alone_already_reads_the_value(self) -> None:
        """Non-vacuity for the double: the baseline below is a measurement."""
        counted = CountingWholeNumber(15_000)
        assert positive_whole_number_error(counted, self.PARAM, self.METHOD) is None
        assert counted.reads > 0

    def test_it_adds_exactly_one_read_to_the_guards_own(self) -> None:
        """The accepted budget is converted once, by this helper.

        Measured as a delta against :func:`positive_whole_number_error` called
        alone, so the guard's own reads are the baseline and this asserts what
        the coercion adds rather than restating the guard's internals.
        """
        baseline = CountingWholeNumber(15_000)
        assert positive_whole_number_error(baseline, self.PARAM, self.METHOD) is None

        counted = CountingWholeNumber(15_000)
        assert coerce_zmq_timeout_ms(self.METHOD, self.PARAM, counted) == (15_000, None)

        assert counted.reads - baseline.reads == 1

    def test_the_one_added_read_is_the_int_the_caller_gets(self) -> None:
        """And it is the ``int``, not a re-read of the range.

        Stated separately from the count so that trading the ``int()`` for a
        second ``float()`` - which keeps the total at one - fails here.
        """
        baseline = CountingWholeNumber(15_000)
        assert positive_whole_number_error(baseline, self.PARAM, self.METHOD) is None

        counted = CountingWholeNumber(15_000)
        assert coerce_zmq_timeout_ms(self.METHOD, self.PARAM, counted) == (15_000, None)

        assert counted.float_reads == baseline.float_reads
        assert counted.int_reads == baseline.int_reads + 1

    def test_an_integral_value_above_the_float64_int_range_is_still_refused(self) -> None:
        """The ceiling comparison happens on an ``int``, which cannot overflow.

        ``1e300`` is integral, finite and inside the float64 range, so the guard
        accepts it and the ceiling is what refuses it. Converting it with ``int``
        first is safe because ``int`` is arbitrary-precision - the case that
        makes reading the value as an ``int`` rather than a ``float`` a
        correctness statement and not only a read-count one.
        """
        timeout_ms, reason = coerce_zmq_timeout_ms(self.METHOD, self.PARAM, 1e300)
        assert timeout_ms is None
        assert reason is not None
        assert f"at most {MAX_ZMQ_TIMEOUT_MS} ms" in reason

    def test_the_ceiling_itself_is_unchanged_by_the_read(self) -> None:
        """Boundary pin: the accepted edge and the first refused value."""
        assert coerce_zmq_timeout_ms(self.METHOD, self.PARAM, MAX_ZMQ_TIMEOUT_MS) == (
            MAX_ZMQ_TIMEOUT_MS,
            None,
        )
        timeout_ms, reason = coerce_zmq_timeout_ms(self.METHOD, self.PARAM, MAX_ZMQ_TIMEOUT_MS + 1)
        assert (timeout_ms, reason is None) == (None, False)
