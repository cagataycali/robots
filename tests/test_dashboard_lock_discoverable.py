"""Q81 follow-through: a guard nobody can find is a guard nobody uses.

The lock shipped as an env var only, which means it existed for whoever read the commit. These tests pin
that it is discoverable in the two places an operator actually looks -- the permissions screen and the env
list -- and that flipping it from the UI really lands.
"""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.agent_motion import TASK_CONFIRM_ENV, task_confirm_required
from strands_robots.dashboard.consent import granted_state
from strands_robots.dashboard.server import create_app


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth, config_api
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config_api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv(TASK_CONFIRM_ENV, raising=False)
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _client():
    app = create_app()
    app.state.bridge.record_activity = mock.Mock()
    return TestClient(app), app


def test_the_lock_is_reported_in_both_states_not_only_when_on():
    """A grant can be listed only when granted -- there is nothing to say otherwise. A LOCK is the
    opposite: the operator has to be told it exists before they can choose it."""
    assert granted_state({})["locks"]["task_requires_confirm"] is False
    assert granted_state({TASK_CONFIRM_ENV: "on"})["locks"]["task_requires_confirm"] is True
    assert granted_state({})["locks"]["task_requires_confirm_env"] == TASK_CONFIRM_ENV


def test_it_is_not_filed_as_a_permission():
    """It tightens; every other key in that payload loosens. Filing it under the grants would make
    'revoke' mean two opposite things on one screen."""
    state = granted_state({TASK_CONFIRM_ENV: "1"})
    assert "task_requires_confirm" not in state["kinds"]
    assert state["trust_remote_code"] is False and state["agent_physical_motion"] is False


def test_the_permissions_endpoint_carries_it():
    client, _ = _client()
    body = client.get("/api/consent").json()
    assert body["locks"]["task_requires_confirm"] is False


def test_the_env_screen_lists_it_even_when_unset():
    """INTERESTING_ENV exists so an operator can discover what the dashboard reads; the lock was
    readable nowhere before this."""
    from strands_robots.dashboard.config_api import INTERESTING_ENV, env_key_allowed, env_view

    assert TASK_CONFIRM_ENV in INTERESTING_ENV
    assert env_key_allowed(TASK_CONFIRM_ENV), "listed but not writable would be a dead row"
    assert any(row.get("key") == TASK_CONFIRM_ENV for row in env_view())


def test_turning_it_on_from_the_ui_actually_locks_the_route():
    """The whole point: the button must change what the SERVER does, not just what the screen says."""
    client, app = _client()
    app.state.bridge.peers = {"so101-arm-1": {"stale": False, "presence": {"hw": "so_follower"}}}
    app.state.bridge.send_cmd_async = mock.AsyncMock(return_value={"ok": True})

    assert client.post("/api/robots/so101-arm-1/task", json={"instruction": "go"}).status_code == 200
    r = client.post("/api/config", json={"env": {TASK_CONFIRM_ENV: "1"}})
    assert r.status_code == 200
    assert task_confirm_required() is True
    assert client.post("/api/robots/so101-arm-1/task", json={"instruction": "go"}).status_code == 403
    # and the play path still works, which is what makes turning it on cheap
    assert client.post("/api/robots/so101-arm-1/task", json={"instruction": "go", "confirmed": True}).status_code == 200


def test_turning_it_off_clears_rather_than_deletes():
    """An absent line lets a stale value from a shell profile or a launchd plist win the next restart --
    a change that silently does not hold."""
    from strands_robots.dashboard.config_api import ENV_FILE

    client, _ = _client()
    client.post("/api/config", json={"env": {TASK_CONFIRM_ENV: "1"}})
    client.post("/api/config", json={"env": {TASK_CONFIRM_ENV: ""}})
    assert task_confirm_required() is False
    assert TASK_CONFIRM_ENV in ENV_FILE.read_text()


def test_the_screen_shows_the_row_in_both_states():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "strands_robots/dashboard/frontend/src/components/ConsentSettings.tsx"
    ).read_text()
    # rendered on the presence of `locks`, never on the flag itself
    assert "state?.locks ?" in src
    assert "task_requires_confirm\n" not in src.split("state?.locks ?")[0][-200:]
    assert "turn on" in src and "turn off" in src
    # and it does not borrow the consent endpoints to write a restriction
    assert "/api/config" in src.split("const setLock")[1][:400]
