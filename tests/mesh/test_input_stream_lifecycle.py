"""Teleop input-stream lifecycle and shaping behavior.

Covers the dependency-free behavior of :mod:`strands_robots.mesh.input`
beyond frame validation (which lives in ``test_input_validation.py``):

* env-tunable knobs (``_input_max_hz`` / ``_input_audit_every``) and their
  bad-value fallbacks,
* ``InputPublisher._normalize_action`` coercion across teleoperator formats,
* publisher start/stop idempotency, stats, and ``__repr__`` state,
* receiver sequence-drop accounting, apply-rate ceiling, E-stop lockout
  gating, sampled positive audit, and the default ``send_action`` apply.

These exercise the high-rate teleop hot loop without any zenoh/torch deps.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from strands_robots.mesh import input as mesh_input
from strands_robots.mesh.input import (
    INPUT_AUDIT_EVERY_DEFAULT,
    INPUT_MAX_HZ_DEFAULT,
    InputPublisher,
    InputReceiver,
)

# --- env knob resolution -------------------------------------------------


class TestInputMaxHz:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_INPUT_MAX_HZ", raising=False)
        assert mesh_input._input_max_hz() == INPUT_MAX_HZ_DEFAULT

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "25")
        assert mesh_input._input_max_hz() == 25.0

    def test_zero_disables_cap(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        assert mesh_input._input_max_hz() == 0.0

    def test_bad_value_falls_back(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "not-a-number")
        assert mesh_input._input_max_hz() == INPUT_MAX_HZ_DEFAULT

    def test_negative_falls_back(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "-5")
        assert mesh_input._input_max_hz() == INPUT_MAX_HZ_DEFAULT


class TestInputAuditEvery:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_INPUT_AUDIT_EVERY", raising=False)
        assert mesh_input._input_audit_every() == INPUT_AUDIT_EVERY_DEFAULT

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_AUDIT_EVERY", "10")
        assert mesh_input._input_audit_every() == 10

    def test_non_positive_disables(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_AUDIT_EVERY", "0")
        assert mesh_input._input_audit_every() == 0

    def test_bad_value_falls_back(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_AUDIT_EVERY", "abc")
        assert mesh_input._input_audit_every() == INPUT_AUDIT_EVERY_DEFAULT


# --- _normalize_action coercion ------------------------------------------


class TestNormalizeAction:
    def test_dict_with_numpy_scalar_uses_item(self):
        out = InputPublisher._normalize_action({"j0": np.float32(1.5)})
        assert out == {"j0": 1.5}
        assert isinstance(out["j0"], float)

    def test_dict_with_python_numbers(self):
        out = InputPublisher._normalize_action({"a": 1, "b": 2.5})
        assert out == {"a": 1.0, "b": 2.5}
        assert all(isinstance(v, float) for v in out.values())

    def test_array_becomes_indexed_joints(self):
        out = InputPublisher._normalize_action(np.array([0.1, 0.2, 0.3]))
        assert out == {"j0": pytest.approx(0.1), "j1": pytest.approx(0.2), "j2": pytest.approx(0.3)}

    def test_bare_scalar_becomes_raw(self):
        assert InputPublisher._normalize_action(2) == {"raw": 2.0}


# --- publisher lifecycle -------------------------------------------------


class _FakeTeleop:
    def __init__(self):
        self.action = {"j0": np.float32(0.4)}

    def get_action(self):
        return self.action


class _PubMesh:
    peer_id = "leader-1"

    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, topic, data):
        self.published.append((topic, data))


class TestPublisherLifecycle:
    def test_repr_reflects_running_state(self):
        pub = InputPublisher(_PubMesh(), _FakeTeleop(), device_name="leader", method="arm")
        assert "stopped" in repr(pub)
        pub._running = True
        assert "running" in repr(pub)

    def test_stop_when_not_running_returns_stats_without_error(self):
        pub = InputPublisher(_PubMesh(), _FakeTeleop())
        stats = pub.stop()
        assert stats["running"] is False
        assert stats["frames"] == 0

    def test_start_publishes_frames_then_stop_returns_stats(self):
        mesh = _PubMesh()
        pub = InputPublisher(mesh, _FakeTeleop(), device_name="leader", method="arm", hz=200.0)
        pub.start()
        # second start is a no-op (idempotent guard)
        pub.start()
        deadline = time.time() + 2.0
        while not mesh.published and time.time() < deadline:
            time.sleep(0.01)
        stats = pub.stop()
        assert stats["running"] is False
        assert mesh.published, "publisher should route at least one frame through Mesh.publish()"
        topic, payload = mesh.published[0]
        assert topic == pub.topic
        assert payload["action"] == {"j0": pytest.approx(0.4)}
        assert payload["method"] == "arm"


# --- receiver behavior ---------------------------------------------------


class _RecvMesh:
    peer_id = "follower-1"

    def __init__(self):
        self._estop_lockout = None

    def subscribe(self, *a, **k):
        return "sub"

    def unsubscribe(self, *a, **k):
        pass


def _make_receiver(mesh=None):
    applied: list[dict] = []
    recv = InputReceiver(
        mesh=mesh or _RecvMesh(),
        robot=object(),
        source_peer_id="leader-1",
        apply_fn=lambda robot, action: applied.append(action),
    )
    recv._running = True
    return recv, applied


class TestReceiverBehavior:
    def test_repr_reflects_running_state(self):
        recv, _ = _make_receiver()
        assert "running" in repr(recv)
        recv._running = False
        assert "stopped" in repr(recv)

    def test_on_input_dropped_when_not_running(self):
        recv, applied = _make_receiver()
        recv._running = False
        recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time()})
        assert applied == []
        assert recv._frame_count == 0

    def test_sequence_gap_counted_as_drops(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # isolate seq accounting from rate cap
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time()})
        # jump from seq 0 to seq 5 => 4 missing frames
        recv._on_input(recv.topic, {"action": {"j0": 0.2}, "seq": 5, "t": time.time()})
        assert recv._drops == 4
        assert recv._frame_count == 2

    def test_apply_rate_ceiling_drops_burst(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "10")  # 100ms min interval
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time()})
        # immediate second frame is over the cap -> rate-dropped, not applied
        recv._on_input(recv.topic, {"action": {"j0": 0.2}, "seq": 1, "t": time.time()})
        assert len(applied) == 1
        assert recv._rate_dropped == 1

    def test_rate_cap_disabled_applies_burst(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # disabled
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time()})
        recv._on_input(recv.topic, {"action": {"j0": 0.2}, "seq": 1, "t": time.time()})
        assert len(applied) == 2
        assert recv._rate_dropped == 0

    def test_estop_lockout_rejects_frame(self):
        import threading

        mesh = _RecvMesh()
        mesh._estop_lockout = threading.Event()
        mesh._estop_lockout.set()
        recv, applied = _make_receiver(mesh)
        recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time()})
        assert applied == []
        assert recv._rejected == 1

    def test_none_action_is_ignored(self):
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, {"action": None, "seq": 0, "t": time.time()})
        assert applied == []
        assert recv._frame_count == 0

    def test_sampled_audit_emitted_at_interval(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_AUDIT_EVERY", "3")
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # no rate gate
        events: list[tuple] = []
        monkeypatch.setattr(mesh_input, "_log_safety_event", lambda *a, **k: events.append((a, k)), raising=False)
        recv, applied = _make_receiver()
        for i in range(3):
            recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": i, "t": time.time()})
        assert len(applied) == 3
        # one audit heartbeat at frame 3 (the configured interval)
        assert len(events) == 1
        assert events[0][0][0] == "input_stream_applied"

    def test_stop_when_not_running_returns_stats(self):
        recv, _ = _make_receiver()
        recv._running = False
        stats = recv.stop()
        assert stats["running"] is False

    def test_start_subscribes_and_stop_unsubscribes(self):
        unsub: list[str] = []

        class _SubMesh(_RecvMesh):
            def subscribe(self, *a, **k):
                return "sub-token"

            def unsubscribe(self, name):
                unsub.append(name)

        recv = InputReceiver(_SubMesh(), object(), source_peer_id="leader-1")
        recv.start()
        assert recv._running is True
        assert recv._sub_name == "sub-token"
        # second start is a no-op
        recv.start()
        stats = recv.stop()
        assert stats["running"] is False
        assert unsub == ["sub-token"]

    def test_start_failed_subscribe_marks_not_running(self):
        class _NoSubMesh(_RecvMesh):
            def subscribe(self, *a, **k):
                return None  # transport refused the subscription

        recv = InputReceiver(_NoSubMesh(), object(), source_peer_id="leader-1")
        recv.start()
        assert recv._running is False
        assert recv._sub_name is None

    def test_apply_error_counted_not_raised(self):
        def _boom(robot, action):
            raise RuntimeError("servo bus offline")

        recv = InputReceiver(_RecvMesh(), object(), source_peer_id="leader-1", apply_fn=_boom)
        recv._running = True
        # must not propagate; error is counted instead
        recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time()})
        assert recv._error_count == 1
        assert recv._frame_count == 0


# --- default apply -------------------------------------------------------


class TestDefaultApply:
    def test_calls_send_action_on_robot(self):
        captured = {}

        class _Robot:
            def send_action(self, action):
                captured["action"] = action

        InputReceiver._default_apply(_Robot(), {"j0": 0.5})
        assert captured["action"] == {"j0": 0.5}

    def test_calls_nested_robot_send_action(self):
        captured = {}

        class _Inner:
            def send_action(self, action):
                captured["action"] = action

        class _Wrapper:
            def __init__(self):
                self.robot = _Inner()

        InputReceiver._default_apply(_Wrapper(), {"j0": 0.5})
        assert captured["action"] == {"j0": 0.5}

    def test_no_send_action_is_noop(self):
        # robot without send_action must not raise
        InputReceiver._default_apply(object(), {"j0": 0.5})
