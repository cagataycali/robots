"""The other half of the motion asymmetry: POST /api/robots/{peer}/task itself.

agent_motion_allowed guards the in-process fleet tool. The play button's confirmation lives in the BROWSER,
so the route was guarded by nothing -- anything holding the API token (a script, a shell, an LLM handed the
token, whoever finds it after the public tunnel leaks it) could start real motion with one curl.

That stays the default on purpose: the token is the operator, and enforcing a claim a client can simply
assert would be theatre with an outage attached. So this is an opt-in ANTI-ACCIDENT lock, and these tests
pin both halves of the deal: OFF changes nothing, ON refuses an unconfirmed real-motion POST.
"""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.agent_motion import (
    TASK_CONFIRM_ENV,
    task_confirm_required,
    task_post_allowed,
)
from strands_robots.dashboard.server import create_app

ARM = {"stale": False, "presence": {"hw": "so_follower"}}
SIM = {"stale": False, "presence": {"robot_type": "sim"}}
ON = {TASK_CONFIRM_ENV: "1"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.delenv(TASK_CONFIRM_ENV, raising=False)
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _client(peers):
    app = create_app()
    app.state.bridge.record_activity = mock.Mock()
    app.state.bridge.peers = peers
    app.state.bridge.send_cmd_async = mock.AsyncMock(return_value={"ok": True})
    return TestClient(app), app


# --- the pure rule ---------------------------------------------------------------------------------


def test_off_by_default_is_a_promise_not_an_accident():
    assert task_confirm_required({}) is False
    v = task_post_allowed(peer=ARM, confirmed=False, target="so101-arm-1", env={})
    assert v["allowed"] is True and v["gated"] is False


def test_on_refuses_an_unconfirmed_real_motion_post():
    v = task_post_allowed(peer=ARM, confirmed=False, target="so101-arm-1", env=ON)
    assert v["allowed"] is False
    # The refusal has to say all three exits, or it is a wall with no door.
    assert "Press play" in v["reason"]
    assert '"confirmed": true' in v["reason"]
    assert TASK_CONFIRM_ENV in v["reason"]
    assert "Nothing was sent" in v["reason"]


def test_on_never_blocks_a_confirmed_click_or_a_simulation():
    assert task_post_allowed(peer=ARM, confirmed=True, target="a", env=ON)["allowed"] is True
    assert task_post_allowed(peer=SIM, confirmed=False, target="sim-a", env=ON)["allowed"] is True


def test_a_peer_with_no_presence_yet_is_treated_as_real():
    """peer_is_physical's silence rule, and this lock must not invent a softer one: a peer that cannot
    be SHOWN to be a sim counts as real, so the unconfirmed POST is refused. That is the fail-safe
    direction for a motion guard, and the route has already 404'd a genuinely unknown id via
    require_peer -- peer=None here means a known peer whose presence has not arrived yet.

    (I wrote this test the other way round first and the code was right: worth keeping as the pin.)"""
    v = task_post_allowed(peer=None, confirmed=False, target="?", env=ON)
    assert v["allowed"] is False
    assert "cannot be shown to be a sim" in v["reason"]


# --- through the route ----------------------------------------------------------------------------


def test_the_route_still_runs_an_unconfirmed_task_by_default():
    client, _ = _client({"so101-arm-1": ARM})
    r = client.post("/api/robots/so101-arm-1/task", json={"instruction": "pick the cube"})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_the_route_refuses_before_sending_when_locked(monkeypatch):
    monkeypatch.setenv(TASK_CONFIRM_ENV, "1")
    client, app = _client({"so101-arm-1": ARM})
    r = client.post("/api/robots/so101-arm-1/task", json={"instruction": "pick the cube"})
    assert r.status_code == 403
    # THE POINT: not one byte reached the robot. A gate that refuses after sending is decoration.
    assert app.state.bridge.send_cmd_async.await_count == 0


def test_the_confirmed_marker_gets_through_but_never_onto_the_wire(monkeypatch):
    monkeypatch.setenv(TASK_CONFIRM_ENV, "1")
    client, app = _client({"so101-arm-1": ARM})
    r = client.post("/api/robots/so101-arm-1/task", json={"instruction": "pick the cube", "confirmed": True})
    assert r.status_code == 200
    (_target, cmd), _kw = app.state.bridge.send_cmd_async.await_args
    assert "confirmed" not in cmd, "the marker is a dashboard concern; the robot must never see it"
    assert cmd["instruction"] == "pick the cube"


def test_the_lock_does_not_touch_stopping(monkeypatch):
    """Stopping is never gated anywhere in this dashboard -- the one invariant that must survive every
    new guard, because a lock that can trap a moving arm is worse than no lock."""
    monkeypatch.setenv(TASK_CONFIRM_ENV, "1")
    client, app = _client({"so101-arm-1": ARM})
    assert client.post("/api/robots/so101-arm-1/stop").status_code == 200
    assert app.state.bridge.send_cmd_async.await_count == 1


def test_the_browser_sends_the_marker():
    """If play ever stops sending it, the lock turns into 'the dashboard cannot run anything' -- the
    failure an operator would report as the feature being broken."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "strands_robots/dashboard/frontend/src/lib/useTask.ts"
    text = src.read_text()
    assert "confirmed: true" in text
    assert "/task" in text
