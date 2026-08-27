"""Lifecycle lands in the audit trail.

Task/stop/e-stop were recorded, but the user's actual day - spawning
peers, killing them, recording sessions, submitting training - left no
trace, so 'who started this peer' was as unanswerable as 'who moved
that arm' used to be. Run with --no-cov.
"""

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


def _client():
    app = create_app()
    app.state.bridge.record_activity = mock.Mock()
    return TestClient(app), app


def test_spawn_records_activity():
    client, app = _client()
    with mock.patch.object(app.state.devices, "spawn", return_value={"peer_id": "so101-x", "pid": 1}):
        r = client.post("/api/devices/spawn", json={"robot_name": "so101", "mode": "sim"})
    assert r.status_code == 200
    call = app.state.bridge.record_activity.call_args
    assert call.args == ("api", "spawn")
    assert call.kwargs["target"] == "so101-x"
    assert call.kwargs["ok"] is True


def test_failed_spawn_records_failure():
    client, app = _client()
    with mock.patch.object(app.state.devices, "spawn", return_value={"error": "port taken"}):
        client.post("/api/devices/spawn", json={"robot_name": "so101"})
    assert app.state.bridge.record_activity.call_args.kwargs["ok"] is False


def test_despawn_records_activity():
    client, app = _client()
    with mock.patch.object(app.state.devices, "despawn", return_value={"stopped": True}):
        client.post("/api/devices/despawn", json={"peer_id": "so101-x"})
    call = app.state.bridge.record_activity.call_args
    assert call.args == ("api", "despawn")
    assert call.kwargs["target"] == "so101-x"


def test_training_submit_records_job_id():
    client, app = _client()
    fake = {"status": "success", "text": "ok", "data": {"job_id": "mock-42"}}
    with mock.patch("strands_robots.dashboard.training.submit", return_value=fake):
        client.post("/api/training/submit", json={"provider": "mock", "dataset_root": "/d"})
    call = app.state.bridge.record_activity.call_args
    assert call.args == ("training", "submit")
    assert call.kwargs["target"] == "mock-42"
    assert call.kwargs["ok"] is True


def test_record_session_open_records_activity():
    client, app = _client()
    with mock.patch.object(app.state.record, "open", return_value={"session": "s1"}):
        client.post("/api/record/open", json={"dataset": "cagatay/cubes", "task": "pick"})
    call = app.state.bridge.record_activity.call_args
    assert call.args == ("record", "session_open")
    assert call.kwargs["target"] == "cagatay/cubes"


def test_autospawn_poll_results_are_audited():
    """The watcher spawns robots with nobody at the keyboard - its actions
    need the audit trail most, and poll() already reports what it did."""
    from strands_robots.dashboard.server import _audit_autospawn

    bridge = mock.Mock()
    _audit_autospawn(bridge, {"spawned": ["so101-auto"], "despawned": ["so101-old"]})
    calls = bridge.record_activity.call_args_list
    assert calls[0].args == ("api", "spawn")
    assert calls[0].kwargs["target"] == "so101-auto"
    assert "auto-spawn" in calls[0].kwargs["detail"]
    assert calls[1].args == ("api", "despawn")
    assert calls[1].kwargs["target"] == "so101-old"

    bridge2 = mock.Mock()
    _audit_autospawn(bridge2, None)
    _audit_autospawn(bridge2, {"skipped": "autospawn disabled"})
    bridge2.record_activity.assert_not_called()
