"""Route-level tests for POST /api/safety/estop and /api/safety/resume.

The fleet's most safety-critical route had NO dashboard-level test - every
existing e-stop test lives at the fleet/mesh layer. These pin the route's
promises:

* BOTH rails fire: per-peer broadcast stop AND the signed lockout envelope,
  and a signed-rail failure must not degrade the broadcast's report.
* stale peers are SKIPPED and named - never counted as "stopped" (a stale
  peer counted stopped is exactly the lie an e-stop must never tell).
* ``all_stopped`` only on unanimity of LIVE peers, and never for an empty
  fleet (nothing was stopped, so nothing may claim it was).
* resume requires the override code (422 before any mesh contact).

Run with --no-cov.
"""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.server import create_app


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """This machine has an enrolled passkey + live settings token; point auth
    and settings at empty temp stores so the guard stays in open posture."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


OK_STOP = {"type": "response", "result": {"ok": True, "status": "stopped"}}
SIGNED_OK = {"signed": True, "issuer": "dash", "responses": {}, "lockout_engaged": True}


def _client(live=("arm-1", "arm-2"), all_peers=None, answers=None, signed=None):
    """App with a bridge whose mesh surface is scripted, not real."""
    app = create_app()
    bridge = app.state.bridge
    bridge.record_activity = mock.Mock()
    bridge.peers = {p: {"peer_id": p} for p in (all_peers or live)}
    bridge.live_peers = mock.Mock(return_value=list(live))
    answers = answers or {}

    async def fake_send(peer, cmd, timeout=5.0, source="api"):
        assert cmd == {"action": "stop"}, "estop must send exactly {action: stop}"
        return answers.get(peer, OK_STOP)

    bridge.send_cmd_async = fake_send
    bridge.signed_estop = mock.Mock(return_value=signed or dict(SIGNED_OK))
    bridge.signed_resume = mock.Mock(return_value={"signed": True, "status": "ok"})
    return TestClient(app), bridge


def test_both_rails_fire_and_report_side_by_side():
    client, bridge = _client()
    body = client.post("/api/safety/estop").json()
    assert body["all_stopped"] is True
    assert body["counts"] == {"stopped": 2, "not_stopped": 0, "no_answer": 0}
    assert sorted(body["targeted"]) == ["arm-1", "arm-2"]
    bridge.signed_estop.assert_called_once()
    assert body["lockout_engaged"] is True
    assert body["signed_rail"]["signed"] is True
    # raw per-peer responses stay out of the summary rail
    assert "responses" not in body["signed_rail"]


def test_stale_peer_is_skipped_and_named_never_stopped():
    client, _ = _client(live=("arm-1",), all_peers=("arm-1", "ghost"))
    body = client.post("/api/safety/estop").json()
    assert body["stale_skipped"] == ["ghost"]
    assert "ghost" not in body["stopped"]
    assert "ghost" not in body["targeted"]
    # the live peer's success must not be diluted by the ghost
    assert body["all_stopped"] is True


def test_all_stopped_requires_unanimity():
    answers = {"arm-2": {"type": "error", "error": "timeout waiting for peer"}}
    client, _ = _client(answers=answers)
    body = client.post("/api/safety/estop").json()
    assert body["all_stopped"] is False
    assert body["counts"]["no_answer"] == 1
    assert body["stopped"]["arm-2"]["state"] == "no_answer"
    assert body["stopped"]["arm-1"]["state"] == "stopped"


def test_refusal_is_not_stopped_not_no_answer():
    answers = {"arm-2": {"type": "response", "result": {"ok": False, "error": "no stop_task"}}}
    client, _ = _client(answers=answers)
    body = client.post("/api/safety/estop").json()
    assert body["stopped"]["arm-2"]["state"] == "not_stopped"
    assert body["all_stopped"] is False


def test_empty_fleet_never_claims_all_stopped():
    client, _ = _client(live=(), all_peers=())
    body = client.post("/api/safety/estop").json()
    assert body["all_stopped"] is False
    assert body["targeted"] == []


def test_signed_rail_failure_does_not_degrade_broadcast():
    client, _ = _client(signed={"signed": False, "error": "safety mesh unavailable"})
    body = client.post("/api/safety/estop").json()
    # broadcast half still fully reported
    assert body["all_stopped"] is True
    assert body["counts"]["stopped"] == 2
    # signed half honestly absent
    assert body["lockout_engaged"] is False
    assert body["signed_rail"]["signed"] is False


def test_estop_lands_in_the_audit_trail():
    client, bridge = _client()
    client.post("/api/safety/estop")
    calls = [c for c in bridge.record_activity.call_args_list if c.args and c.args[0] == "estop"]
    assert calls, "an e-stop that leaves no audit trace is unauditable"
    assert calls[0].kwargs["ok"] is True


def test_resume_requires_override_code():
    client, bridge = _client()
    assert client.post("/api/safety/resume", json={}).status_code == 422
    assert client.post("/api/safety/resume", json={"override_code": "  "}).status_code == 422
    bridge.signed_resume.assert_not_called()
    body = client.post("/api/safety/resume", json={"override_code": "s3cret"}).json()
    bridge.signed_resume.assert_called_once_with("s3cret")
    assert body["status"] == "ok"
