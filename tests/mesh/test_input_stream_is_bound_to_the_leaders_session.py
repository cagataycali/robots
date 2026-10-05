"""Regression tests: an approved teleop stream follows one publisher, for a stated time.

``InputReceiver`` subscribed to ``strands/<leader>/input/<device>`` and applied
every frame that arrived there. The key expression scopes the stream to a
leader's NAME, which is a body field any admitted peer can publish under, so
once an operator approved ``teleop_receive`` for one leader, any peer that could
reach the topic drove the follower for as long as the stream lived, and the
stream lived until someone stopped it.

Now the receiver binds at ``start()`` to the TLS-bound session the leader
announced its presence from (``Mesh.peer_wire_zid``), refuses to start when the
leader has none, refuses every frame whose sample came from another session,
and expires after ``STRANDS_MESH_INPUT_STREAM_TTL_S`` seconds. The opening, the
lifetime and the expiry are written to the safety log.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import input as mesh_input
from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.input import InputReceiver

LEADER_ZID = "a1b2c3d4e5f60718"
OTHER_ZID = "0f0e0d0c0b0a0908"


class _BoundMesh:
    """A mesh stand-in that knows which session the leader announced itself from."""

    peer_id = "follower-1"

    def __init__(self, leader_zid: str | None = LEADER_ZID) -> None:
        self._estop_lockout = None
        self._leader_zid = leader_zid
        self.subscriptions: list[dict[str, Any]] = []
        self.audits: list[tuple[str, dict[str, Any]]] = []

    def peer_wire_zid(self, peer_id: str) -> str | None:
        return self._leader_zid if peer_id == "leader-1" else None

    def subscribe(self, topic: str, callback: Any = None, name: str | None = None, **kw: Any) -> str:
        self.subscriptions.append({"topic": topic, "callback": callback, "name": name, **kw})
        return name or topic

    def unsubscribe(self, name: str) -> None:
        self.subscriptions = [s for s in self.subscriptions if s["name"] != name]

    def _audit_local(self, event: str, payload: dict[str, Any]) -> None:
        self.audits.append((event, payload))


def _frame(seq: int = 0) -> dict[str, Any]:
    return {"action": {"j0": 0.1}, "seq": seq, "t": time.time()}


def _receiver(mesh: _BoundMesh) -> tuple[InputReceiver, list[dict[str, float]]]:
    applied: list[dict[str, float]] = []
    recv = InputReceiver(mesh=mesh, robot=object(), source_peer_id="leader-1", apply_fn=lambda r, a: applied.append(a))
    return recv, applied


def _deliver(mesh: _BoundMesh, recv: InputReceiver, data: dict[str, Any], *, zid: str | None) -> None:
    """Hand a frame to the receiver the way ``Mesh.subscribe`` does, with the sample's session id."""
    sub = next(s for s in mesh.subscriptions if s["name"] == recv._sub_name)
    sub["on_sample"](recv.topic, data, zid)


class TestTheStreamIsBoundToTheLeadersSession:
    def test_start_refuses_a_leader_that_announced_no_session(self) -> None:
        mesh = _BoundMesh(leader_zid=None)
        recv, _ = _receiver(mesh)

        recv.start()

        assert recv.stats["running"] is False
        assert mesh.subscriptions == []
        assert "session" in (recv.start_refusal or "")
        assert [e for e, _ in mesh.audits] == ["input_stream_refused"]

    def test_start_subscribes_with_the_sample_source_and_records_the_lifetime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_INPUT_STREAM_TTL_S", "120")
        mesh = _BoundMesh()
        recv, _ = _receiver(mesh)

        recv.start()

        assert recv.stats["running"] is True
        assert len(mesh.subscriptions) == 1 and callable(mesh.subscriptions[0]["on_sample"])
        opened = [p for e, p in mesh.audits if e == "input_stream_opened"]
        assert len(opened) == 1
        assert opened[0]["source"] == "leader-1" and opened[0]["wire_zid"] == LEADER_ZID
        assert opened[0]["lifetime_s"] == 120.0

    def test_a_frame_from_the_leaders_session_is_applied(self) -> None:
        mesh = _BoundMesh()
        recv, applied = _receiver(mesh)
        recv.start()

        _deliver(mesh, recv, _frame(), zid=LEADER_ZID)

        assert applied == [{"j0": 0.1}]
        assert recv.stats["rejected"] == 0

    @pytest.mark.parametrize("zid", [OTHER_ZID, None], ids=["another-session", "no-session"])
    def test_a_frame_from_any_other_session_is_refused_and_counted(self, zid: str | None) -> None:
        mesh = _BoundMesh()
        recv, applied = _receiver(mesh)
        recv.start()

        _deliver(mesh, recv, _frame(), zid=zid)

        assert applied == []
        assert recv.stats["rejected"] == 1 and recv.stats["rejected_source"] == 1

    def test_the_source_check_is_a_declared_refusal_cause(self) -> None:
        assert "source" in mesh_input._REJECTION_CAUSES
        assert "expired" in mesh_input._REJECTION_CAUSES


class TestTheStreamHasAStatedLifetime:
    def test_an_expired_stream_refuses_the_frame_stops_and_is_audited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_INPUT_STREAM_TTL_S", "0.05")
        mesh = _BoundMesh()
        recv, applied = _receiver(mesh)
        recv.start()
        _deliver(mesh, recv, _frame(0), zid=LEADER_ZID)
        time.sleep(0.08)

        _deliver(mesh, recv, _frame(1), zid=LEADER_ZID)

        assert applied == [{"j0": 0.1}]
        assert recv.stats["rejected_expired"] == 1
        assert recv.stats["running"] is False
        expired = [p for e, p in mesh.audits if e == "input_stream_expired"]
        assert len(expired) == 1 and expired[0]["source"] == "leader-1"

    @pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "soon"])
    def test_an_unusable_lifetime_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("STRANDS_MESH_INPUT_STREAM_TTL_S", value)
        assert mesh_input._input_stream_ttl_s() == mesh_input._INPUT_STREAM_TTL_DEFAULT_S


class TestMeshSubscribeHandsTheSampleSourceToTheCallback:
    def test_on_sample_receives_the_publisher_session_id(self) -> None:
        sess = MagicMock()
        handlers: list[Any] = []
        sess.declare_subscriber.side_effect = lambda topic, handler: handlers.append(handler) or MagicMock()
        with (
            patch.object(mesh_session, "current_session", return_value=sess),
            patch.object(mesh_core, "current_session", return_value=sess),
        ):
            mesh = Mesh(None, peer_id="follower-1")
            mesh._running = True
            seen: list[tuple[str, dict[str, Any], str | None]] = []
            name = mesh.subscribe("strands/leader-1/input/leader", on_sample=lambda k, d, z: seen.append((k, d, z)))
            assert name == "strands/leader-1/input/leader" and len(handlers) == 1
            body = SimpleNamespace(to_bytes=lambda: json.dumps({"seq": 1}).encode())
            source_info = SimpleNamespace(
                source_id=SimpleNamespace(zid=type("Z", (), {"__str__": lambda s: LEADER_ZID})())
            )
            handlers[0](
                SimpleNamespace(key_expr="strands/leader-1/input/leader", payload=body, source_info=source_info)
            )
            handlers[0](SimpleNamespace(key_expr="strands/leader-1/input/leader", payload=body, source_info=None))

        assert seen == [
            ("strands/leader-1/input/leader", {"seq": 1}, LEADER_ZID),
            ("strands/leader-1/input/leader", {"seq": 1}, None),
        ]

    def test_callback_and_on_sample_are_exclusive(self) -> None:
        mesh = Mesh(None, peer_id="follower-1")
        with pytest.raises(TypeError):
            mesh.subscribe("strands/x", callback=lambda k, d: None, on_sample=lambda k, d, z: None)


class TestTheHostReportsARefusedStream:
    def test_start_teleop_receive_answers_an_error_when_the_leader_is_unbound(self) -> None:
        from strands_robots.teleop_mixin import TeleopMixin

        class _Host(TeleopMixin):
            def __init__(self) -> None:
                self.mesh = _BoundMesh(leader_zid=None)
                self.mesh.alive = True  # type: ignore[attr-defined]

            def _teleop_target_error(self, robot_name: str | None) -> str | None:
                return None

            def send_action(self, action: dict[str, float], **kw: Any) -> dict[str, Any]:
                return {"status": "success"}

        out = _Host().start_teleop_receive("leader-1", "leader")

        assert out["status"] == "error"
        assert "session" in out["content"][0]["text"]
