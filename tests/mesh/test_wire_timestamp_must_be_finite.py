"""A wire timestamp is refused unless it is finite.

Four handlers in this package decide whether to act on an envelope by its age:
presence (``timestamp``), the teleop input stream, remote E-stop and remote
resume (``t``). Each computes ``now - t`` and compares the result against a
freshness window and a forward-skew allowance, and each gated the read on
``isinstance(value, (int, float))`` alone.

``nan`` is an instance of ``float`` and compares False against every bound, so
it satisfied the type gate and then satisfied neither age test: the pair of
comparisons that defines "fresh enough to act on" stopped bounding anything.
``json.loads`` accepts the bare ``NaN`` / ``Infinity`` / ``-Infinity`` tokens by
default, so the value arrives from the wire rather than only from a Python
caller.

The teleop path is the one that moves a robot. Its own comment states the threat
the gate exists to stop -- an eavesdropper storing frames and replaying them
"hours/days later" -- and a captured stream whose ``t`` is rewritten to ``NaN``
was applied in full, at any later date, by the guard written to refuse exactly
that.

The rule is not new here: ``mesh.core._parse_positive_float_env`` already
refuses a non-finite *window*, on the reasoning that a ``nan`` "does not widen
the bound but removes it" (see
``tests/mesh/test_env_float_knobs_reject_non_finite.py``). Both operands of one
comparison need it, and only the one supplied by the operator had it. So these
cells grade the wire operand at all four gates, and
:class:`TestEveryWireTimestampGateReadsTheOneRule` derives the population from
the package rather than listing it, so a fifth handler is graded when it lands.

Every class here carries controls -- a fresh timestamp still acts, a stale one
is still refused -- because a guard that refuses more things is not what was
asked for.
"""

from __future__ import annotations

import ast
import json
import math
import pathlib
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from strands_robots.mesh import core as core_mod
from strands_robots.mesh import session as session_mod
from strands_robots.mesh.security import as_wire_timestamp
from tests.mesh.test_input_stream_lifecycle import _make_receiver
from tests.mesh.test_resume_replay import _make_envelope, _make_mesh, _sample

#: The three non-finite values, each spelled as a Python float and as the JSON
#: token ``json.loads`` decodes to it - the wire spelling is the reachable one.
_NON_FINITE = (
    pytest.param(math.nan, "NaN", id="nan"),
    pytest.param(math.inf, "Infinity", id="inf"),
    pytest.param(-math.inf, "-Infinity", id="-inf"),
)

#: Older than any default freshness window, and the control for "still refused".
_STALE_AGE_S = 9999.0


def _decoded_from_the_wire(token: str) -> float:
    """The value a peer's JSON envelope actually delivers for *token*."""
    return json.loads(f'{{"t": {token}}}')["t"]


class TestTheHelperNamesTheDomain:
    """The rule itself: a finite real number of seconds, and nothing else."""

    @pytest.mark.parametrize(("value", "token"), _NON_FINITE)
    def test_a_non_finite_number_is_not_a_timestamp(self, value, token):
        assert as_wire_timestamp(value) is None
        assert as_wire_timestamp(_decoded_from_the_wire(token)) is None

    @pytest.mark.parametrize("value", [0, 1, 1.5, time.time(), -1.0, 10**9])
    def test_a_finite_number_is_returned_as_it_arrived(self, value):
        # Identity, not equality: ``_on_safety_resume`` verifies an HMAC whose
        # input binds ``t`` through ``json.dumps``, which writes ``1`` and
        # ``1.0`` differently, so widening an integer stamp here would refuse a
        # correctly signed envelope.
        returned = as_wire_timestamp(value)
        assert returned is value
        assert type(returned) is type(value)

    @pytest.mark.parametrize("value", [None, "1700000000", True, False, [1.0], {"t": 1.0}])
    def test_a_non_number_is_not_a_timestamp(self, value):
        # ``True`` is an ``int``, so it cleared the type gate and read as one
        # second past the epoch - refused, but as stale rather than malformed.
        assert as_wire_timestamp(value) is None


class TestTheTeleopStreamRefusesANonFiniteFrameTimestamp:
    """The path that moves a robot: no frame reaches ``send_action``."""

    @staticmethod
    def _frame(t, seq=1):
        return {"action": {"j0": 0.1}, "seq": seq, "t": t}

    @pytest.mark.parametrize(("value", "token"), _NON_FINITE)
    def test_the_frame_is_refused_and_counted_under_freshness(self, value, token, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # isolate from the rate gate
        for supplied in (value, _decoded_from_the_wire(token)):
            recv, applied = _make_receiver()
            recv._on_input(recv.topic, self._frame(supplied))
            assert applied == [], "a frame with a non-finite t must never reach the robot"
            assert recv.stats["rejected_freshness"] == 1
            assert recv.stats["rejected"] == 1

    def test_a_captured_stream_replayed_with_a_nan_stamp_moves_nothing(self, monkeypatch):
        """The replay this gate exists to refuse, with the stamp blanked out."""
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, applied = _make_receiver()
        for seq in range(20):
            recv._on_input(recv.topic, self._frame(math.nan, seq=seq))
        assert applied == []
        assert recv.stats["rejected_freshness"] == 20

    def test_a_fresh_frame_is_still_applied(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, self._frame(time.time()))
        assert len(applied) == 1
        assert recv.stats["rejected_freshness"] == 0

    def test_a_stale_frame_is_still_refused(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, self._frame(time.time() - _STALE_AGE_S))
        assert applied == []
        assert recv.stats["rejected_freshness"] == 1


@pytest.fixture
def clean_peer_registry():
    """The peer registry is process-global; a leaked peer would fake a pass."""
    session_mod.clear_peers()
    yield
    session_mod.clear_peers()


class TestPresenceRefusesANonFiniteHeartbeatTimestamp:
    """A phantom peer cannot be registered by blanking the heartbeat's clock."""

    @staticmethod
    def _receiver():
        mesh = core_mod.Mesh.__new__(core_mod.Mesh)
        mesh.peer_id = "receiver"
        return mesh

    @staticmethod
    def _heartbeat(peer_id, timestamp):
        body = {"robot_id": peer_id, "robot_type": "robot", "hostname": "h", "timestamp": timestamp}
        raw = json.dumps(body).encode()
        return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))

    def _registered(self):
        return {peer["peer_id"] for peer in session_mod.get_peers()}

    @pytest.mark.parametrize(("value", "token"), _NON_FINITE)
    def test_the_heartbeat_is_dropped(self, value, token, clean_peer_registry):
        for name, supplied in (("ghost-py", value), ("ghost-wire", _decoded_from_the_wire(token))):
            self._receiver()._on_presence(self._heartbeat(name, supplied))
            assert name not in self._registered()

    def test_a_fresh_heartbeat_still_registers_the_peer(self, clean_peer_registry):
        self._receiver()._on_presence(self._heartbeat("live", time.time()))
        assert "live" in self._registered()

    def test_a_stale_heartbeat_is_still_dropped(self, clean_peer_registry):
        self._receiver()._on_presence(self._heartbeat("old", time.time() - _STALE_AGE_S))
        assert "old" not in self._registered()


class TestRemoteEstopRefusesANonFiniteEnvelopeTimestamp:
    """A replayed E-stop cannot lock a fleet out by blanking its clock."""

    @staticmethod
    def _receiver():
        mesh = core_mod.Mesh.__new__(core_mod.Mesh)
        mesh.peer_id = "receiver"
        mesh._estop_replay_cache = {}
        mesh._estop_replay_lock = threading.Lock()
        mesh._estop_lockout = threading.Event()
        mesh._last_estop_ts = 0.0
        mesh._last_estop_mono = 0.0
        mesh._running = False
        mesh.robot = None
        mesh.publish_safety_event = lambda **kw: None
        return mesh

    @staticmethod
    def _envelope(t):
        raw = json.dumps({"peer_id": "issuer", "t": t}).encode()
        return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))

    @pytest.mark.parametrize(("value", "token"), _NON_FINITE)
    def test_the_lockout_is_not_engaged(self, value, token):
        for supplied in (value, _decoded_from_the_wire(token)):
            mesh = self._receiver()
            mesh._on_safety_estop(self._envelope(supplied))
            assert mesh._estop_lockout.is_set() is False

    def test_a_fresh_envelope_still_engages_the_lockout(self):
        mesh = self._receiver()
        mesh._on_safety_estop(self._envelope(time.time()))
        assert mesh._estop_lockout.is_set() is True

    def test_a_stale_envelope_is_still_refused(self):
        mesh = self._receiver()
        mesh._on_safety_estop(self._envelope(time.time() - _STALE_AGE_S))
        assert mesh._estop_lockout.is_set() is False


class TestRemoteResumeRefusesANonFiniteEnvelopeTimestamp:
    """The signed path too: a valid HMAC over a ``nan`` lifts no lockout.

    The MAC binds ``t``, and ``json.dumps`` writes ``NaN`` for it, so an issuer
    holding the override code can sign one - which is what makes the freshness
    gate rather than the signature the thing under test here.
    """

    _CODE = "test-override-code"

    def _locked_out_receiver(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", self._CODE)
        mesh = _make_mesh()
        mesh._estop_lockout.set()
        return mesh

    @pytest.mark.parametrize(("value", "token"), _NON_FINITE)
    def test_the_lockout_is_not_lifted(self, value, token, monkeypatch):
        for supplied in (value, _decoded_from_the_wire(token)):
            mesh = self._locked_out_receiver(monkeypatch)
            envelope = _make_envelope(self._CODE, t=supplied, proof_nonce=uuid.uuid4().hex)
            mesh._on_safety_resume(_sample(envelope))
            assert mesh._estop_lockout.is_set() is True

    def test_a_fresh_envelope_still_lifts_the_lockout(self, monkeypatch):
        mesh = self._locked_out_receiver(monkeypatch)
        mesh._on_safety_resume(_sample(_make_envelope(self._CODE, t=time.time())))
        assert mesh._estop_lockout.is_set() is False

    def test_a_stale_envelope_is_still_refused(self, monkeypatch):
        mesh = self._locked_out_receiver(monkeypatch)
        mesh._on_safety_resume(_sample(_make_envelope(self._CODE, t=time.time() - _STALE_AGE_S)))
        assert mesh._estop_lockout.is_set() is True


#: The handlers the four behavioural classes above cover, as
#: ``(module, function)``. Held as a floor on the derived population rather than
#: as its definition: a finder that stops matching reports an empty set, and an
#: empty set passes a rule quantified over it.
_KNOWN_GATES = frozenset(
    {
        ("core.py", "_on_presence"),
        ("core.py", "_on_safety_estop"),
        ("core.py", "_on_safety_resume"),
        ("input.py", "_on_input"),
    }
)

#: What a wire timestamp is read out of. Both keys, because presence names the
#: field differently from the three envelope handlers.
_TIMESTAMP_KEYS = frozenset({"t", "timestamp"})


def _reads_a_wire_timestamp(node: ast.AST) -> bool:
    """Whether *node* is a ``<something>.get("t"|"timestamp")`` call.

    The call rather than the assignment, because the read need not be bound to
    a name: three of the four handlers also read the same field straight into
    an audit record, and a handler that reads it at all is a handler that has
    to know whether the value is usable.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in _TIMESTAMP_KEYS
    )


def _calls_the_rule(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
        if name == as_wire_timestamp.__name__:
            return True
    return False


def _wire_timestamp_gates() -> dict[tuple[str, str], bool]:
    """Every function in this package that binds a wire timestamp read.

    The population is derived from the tree, not listed, so a handler added
    later is graded on arrival rather than when someone remembers this file.
    Derived from the imported package's own location so a move does not leave
    the walk grading an empty directory.
    """
    package = pathlib.Path(core_mod.__file__).parent
    gates: dict[tuple[str, str], bool] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_reads_a_wire_timestamp(node) for node in ast.walk(func)):
                gates[(path.name, func.name)] = _calls_the_rule(func)
    return gates


class TestEveryWireTimestampGateReadsTheOneRule:
    """One rule, one owner, and a population the tree supplies."""

    def test_the_finder_still_reaches_the_four_known_gates(self):
        found = set(_wire_timestamp_gates())
        assert _KNOWN_GATES <= found, f"the finder no longer reads: {sorted(_KNOWN_GATES - found)}"

    def test_every_gate_passes_the_value_through_the_rule(self):
        unguarded = sorted(gate for gate, guarded in _wire_timestamp_gates().items() if not guarded)
        assert unguarded == [], (
            "these functions read a wire timestamp without passing it through "
            f"security.as_wire_timestamp, which is what refuses a non-finite one: {unguarded}"
        )
