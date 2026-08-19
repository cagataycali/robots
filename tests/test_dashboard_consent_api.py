"""U18 through the API: a refusal arrives answerable, and an answer is minimal.

Run with --no-cov.
"""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.server import create_app

HF_REFUSAL = (
    "pretrained_name_or_path='HashtagRobotics/smolvla-tic-tac-toe' not in allowlist. "
    "Set STRANDS_MESH_HF_REPO_ALLOW to add an org/repo prefix."
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Temp auth store (an enrolled passkey on this machine would 401 us), temp
    settings, temp .env — a test must never write the operator's real grants."""
    from strands_robots.dashboard import auth, config_api
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config_api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _client():
    app = create_app()
    app.state.bridge.record_activity = mock.Mock()
    return TestClient(app), app


def test_consent_state_starts_empty():
    client, _ = _client()
    body = client.get("/api/consent").json()
    assert body["trust_remote_code"] is False
    assert body["hf_repo_allow"] == []
    assert "trust_remote_code" in body["kinds"] and "hf_repo_allow" in body["kinds"]


def test_approving_trust_writes_env_and_process():
    import os

    client, app = _client()
    r = client.post("/api/consent", json={"kind": "trust_remote_code", "subject": "lerobot_local"})
    assert r.status_code == 200
    body = r.json()
    assert body["granted"] is True
    assert body["respawn_required"] is True
    assert "STRANDS_TRUST_REMOTE_CODE" in body["env_written"]
    assert os.environ["STRANDS_TRUST_REMOTE_CODE"] == "1"
    assert client.get("/api/consent").json()["trust_remote_code"] is True
    assert app.state.bridge.record_activity.call_args.args == ("api", "consent")


def test_approving_a_repo_grants_that_repo_only():
    client, _ = _client()
    r = client.post("/api/consent", json={"kind": "hf_repo_allow", "subject": "Org/model-v1"})
    assert r.json()["granted"] is True
    assert client.get("/api/consent").json()["hf_repo_allow"] == ["Org/model-v1"]


def test_second_approval_of_the_same_thing_changes_nothing():
    client, _ = _client()
    client.post("/api/consent", json={"kind": "hf_repo_allow", "subject": "Org/model-v1"})
    again = client.post("/api/consent", json={"kind": "hf_repo_allow", "subject": "Org/model-v1"}).json()
    assert again["granted"] is False
    assert again["already_granted"] is True
    assert again["respawn_required"] is True


def test_a_client_cannot_name_the_variable_or_the_value():
    client, _ = _client()
    r = client.post(
        "/api/consent",
        json={"kind": "hf_repo_allow", "subject": "Org/ok", "env_var": "PATH", "value": "/evil"},
    )
    assert r.status_code == 200
    state = client.get("/api/consent").json()
    assert state["hf_repo_allow"] == ["Org/ok"]
    import os

    assert "/evil" not in os.environ.get("PATH", "")


def test_unknown_kind_is_refused():
    client, _ = _client()
    assert client.post("/api/consent", json={"kind": "sudo"}).status_code == 422


def test_hostile_subject_is_not_granted():
    client, _ = _client()
    r = client.post("/api/consent", json={"kind": "hf_repo_allow", "subject": "Org/x;rm -rf /"})
    assert r.json()["granted"] is False
    assert client.get("/api/consent").json()["hf_repo_allow"] == []


def test_failed_spawn_carries_needs_consent():
    client, app = _client()
    with mock.patch.object(
        app.state.devices, "spawn", return_value={"peer_id": "so101-x", "pid": 4242}
    ), mock.patch.object(
        app.state.devices,
        "settle",
        return_value={"status": "failed", "exit_code": 1, "reason": HF_REFUSAL, "log_tail": []},
    ):
        body = client.post("/api/devices/spawn", json={"robot_name": "so101", "mode": "sim"}).json()
    assert body["needs_consent"]["kind"] == "hf_repo_allow"
    assert body["needs_consent"]["subject"] == "HashtagRobotics/smolvla-tic-tac-toe"
    # the old shape survives for clients that never heard of consent
    assert body["error"] == HF_REFUSAL


def test_spawn_failure_without_a_refusal_has_no_consent_key():
    client, app = _client()
    with mock.patch.object(
        app.state.devices, "spawn", return_value={"peer_id": "so101-x", "pid": 1}
    ), mock.patch.object(
        app.state.devices,
        "settle",
        return_value={"status": "failed", "exit_code": 1, "reason": "port 8091 in use", "log_tail": []},
    ):
        body = client.post("/api/devices/spawn", json={"robot_name": "so101", "mode": "sim"}).json()
    assert "needs_consent" not in body


def test_task_refusal_carries_needs_consent():
    client, app = _client()
    app.state.bridge.peers = {"so101-arm-1": {"stale": False}}
    app.state.bridge.send_cmd_async = mock.AsyncMock(return_value={"ok": False, "error": HF_REFUSAL})
    body = client.post("/api/robots/so101-arm-1/task", json={"instruction": "pick the cube"}).json()
    assert body["ok"] is False
    assert body["needs_consent"]["scope"] == "hf_repo_allow:HashtagRobotics/smolvla-tic-tac-toe"
    assert "trust" not in body["needs_consent"]["kind"]
