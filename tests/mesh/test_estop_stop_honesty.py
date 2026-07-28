"""An emergency stop must never report a peer as halted when it was not.

``Mesh._dispatch`` answered ``{"ok": True}`` for ``action="stop"`` when the
registered robot exposed no ``stop_task`` -- nothing was stopped, yet the peer
acknowledged the stop. ``emergency_stop`` then counted that response in
``responses_received``, so an operator watching a fleet E-STOP saw a clean
acknowledgement from a robot that was still executing. On a safety path an
affirmative lie is the worst available failure mode.

The dispatch now reports the failure, and ``emergency_stop`` counts such peers
separately: logged at CRITICAL and carried in the safety envelope / audit record
as ``peers_not_stopped``.
"""

from __future__ import annotations

import logging

from strands_robots.mesh.core import Mesh, _peers_that_did_not_stop


class _StoppableRobot:
    """A robot that can genuinely stop."""

    def stop_task(self) -> dict[str, object]:
        return {"status": "success", "content": [{"text": "stopped"}]}


class _StatusOnlyRobot:
    """A peer with no ``stop_task`` -- it cannot stop anything."""

    def get_task_status(self) -> dict[str, object]:
        return {"status": "success", "content": [{"text": "idle"}]}


class TestDispatchHonesty:
    def test_missing_stop_task_reports_failure(self):
        mesh = Mesh(_StatusOnlyRobot(), peer_id="p")

        out = mesh._dispatch({"action": "stop"})

        assert out["ok"] is False
        assert "nothing was stopped" in out["error"]

    def test_missing_stop_task_logs_at_error(self, caplog):
        """An unstoppable peer must be loud in the log, not only in the return."""
        mesh = Mesh(_StatusOnlyRobot(), peer_id="p")

        with caplog.at_level(logging.ERROR):
            mesh._dispatch({"action": "stop"})

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("no stop_task" in m for m in msgs), msgs

    def test_real_stop_task_result_is_passed_through(self):
        """A robot that can stop keeps its own result shape."""
        mesh = Mesh(_StoppableRobot(), peer_id="p")

        out = mesh._dispatch({"action": "stop"})

        assert out["status"] == "success"
        assert out.get("ok") is not False


class TestFailedStopAccounting:
    def test_ok_false_response_is_counted_as_not_stopped(self):
        responses = [
            {"responder_id": "arm-1", "result": {"status": "success"}},
            {"responder_id": "arm-2", "result": {"ok": False, "error": "no stop_task"}},
        ]
        assert _peers_that_did_not_stop(responses) == {"arm-2"}

    def test_error_status_response_is_counted_as_not_stopped(self):
        """A stop_task that itself failed is also a peer that did not stop."""
        responses = [{"responder_id": "arm-3", "result": {"status": "error", "error": "bus fault"}}]
        assert _peers_that_did_not_stop(responses) == {"arm-3"}

    def test_successful_peers_are_not_flagged(self):
        responses = [
            {"responder_id": "arm-1", "result": {"status": "success"}},
            {"responder_id": "arm-2", "result": {"ok": True}},
        ]
        assert _peers_that_did_not_stop(responses) == set()

    def test_unidentified_responder_still_counted(self):
        """A failure with no responder_id must not vanish from the count."""
        responses = [{"result": {"ok": False}}]
        flagged = _peers_that_did_not_stop(responses)
        assert len(flagged) == 1
        assert "unidentified-responder" in next(iter(flagged))

    def test_unrecognised_shapes_are_not_guessed_at(self):
        """Only affirmative failures count: a false alarm trains operators to ignore it."""
        responses = [
            {"responder_id": "a", "result": {}},
            {"responder_id": "b"},
            {"responder_id": "c", "result": "not-a-dict"},
            "not-a-dict",
        ]
        assert _peers_that_did_not_stop(responses) == set()

    def test_bare_result_dict_without_envelope_is_handled(self):
        """Some transports hand back the result directly, not wrapped."""
        assert _peers_that_did_not_stop([{"ok": False}])


class TestEmergencyStopSurfacesUnstoppablePeers:
    def _mesh_with_responses(self, monkeypatch, responses):
        mesh = Mesh(_StoppableRobot(), peer_id="operator")
        monkeypatch.setattr(mesh, "broadcast", lambda cmd, timeout=3.0: responses)
        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            mesh,
            "_publish_safety_envelope",
            lambda topic, envelope: published.append((topic, envelope)),
        )
        events: list[dict] = []
        monkeypatch.setattr(
            mesh,
            "publish_safety_event",
            lambda **kwargs: events.append(kwargs),
        )
        monkeypatch.setattr(mesh, "_local_session_zid", lambda: None)
        return mesh, published, events

    def test_envelope_and_audit_name_the_peers_that_did_not_stop(self, monkeypatch):
        responses = [
            {"responder_id": "arm-1", "result": {"status": "success"}},
            {"responder_id": "arm-2", "result": {"ok": False, "error": "no stop_task"}},
        ]
        mesh, published, events = self._mesh_with_responses(monkeypatch, responses)

        mesh.emergency_stop()

        assert published, "no safety envelope published"
        _topic, envelope = published[0]
        assert envelope["peers_not_stopped"] == ["arm-2"]
        # The raw ack count is preserved so the two numbers can be compared.
        assert envelope["responses_received"] == 2
        assert events[0]["payload"]["peers_not_stopped"] == ["arm-2"]

    def test_critical_log_when_a_peer_did_not_stop(self, monkeypatch, caplog):
        responses = [{"responder_id": "arm-2", "result": {"ok": False}}]
        mesh, _published, _events = self._mesh_with_responses(monkeypatch, responses)

        with caplog.at_level(logging.CRITICAL):
            mesh.emergency_stop()

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert any("did NOT stop" in m for m in msgs), msgs
        assert any("arm-2" in m for m in msgs), msgs

    def test_all_clear_fleet_reports_no_failures(self, monkeypatch):
        responses = [{"responder_id": "arm-1", "result": {"status": "success"}}]
        mesh, published, events = self._mesh_with_responses(monkeypatch, responses)

        mesh.emergency_stop()

        _topic, envelope = published[0]
        assert envelope["peers_not_stopped"] == []
        assert events[0]["payload"]["peers_not_stopped"] == []

    def test_lockout_still_engages_even_when_a_peer_cannot_stop(self, monkeypatch):
        """The local lockout is independent of remote acknowledgement."""
        responses = [{"responder_id": "arm-2", "result": {"ok": False}}]
        mesh, _published, _events = self._mesh_with_responses(monkeypatch, responses)

        mesh.emergency_stop()

        assert mesh._estop_lockout.is_set()
