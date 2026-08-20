"""U16: the deploy snippet must recreate the EXACT rig, or refuse.

The generated file is the contract that the dashboard's spawn form and a
plain `python robot.py` on an edge device describe the same object. The
strongest test therefore EXECUTES the generated source (with strands_robots
stubbed) and asserts the factory receives exactly the profile's arguments -
string comparison on generated code rots; a call-capture does not.

Run with --no-cov.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.deploy import render_snippet, snippet_filename

ARM_1 = {
    "cameras": {
        "top": {"fps": 30, "height": 1080, "index_or_path": 2, "type": "opencv", "width": 1920},
        "wrist": {"fps": 30, "height": 1080, "index_or_path": 1, "type": "opencv", "width": 1920},
    },
    "mode": "real",
    "name": "so101-arm-1",
    "peer_id": "so101-arm-1",
    "port": "/dev/cu.usbmodem5AB01818061",
    "robot_id": "follower_arm",
    "robot_name": "so101",
    "serial_number": "5AB0181806",
}


def _run_generated(source: str) -> tuple[tuple, dict, list[str]]:
    """Execute a generated snippet with Robot + time.sleep stubbed.

    Returns (args, kwargs, printed). The snippet's `while True: sleep(1)`
    keep-alive is broken by making sleep raise.
    """
    calls: list[tuple[tuple, dict]] = []
    printed: list[str] = []

    class _Robot:
        def __init__(self, *a, **kw):
            calls.append((a, kw))

        def connect_eagerly(self):
            return True, {}, None

    fake = types.ModuleType("strands_robots")
    fake.Robot = _Robot

    class _Stop(Exception):
        pass

    def _sleep(_s):
        raise _Stop

    import time as _time

    with mock.patch.dict(sys.modules, {"strands_robots": fake}), \
         mock.patch.object(_time, "sleep", _sleep), \
         mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
        try:
            exec(compile(source, "generated.py", "exec"), {"__name__": "__main__"})
        except _Stop:
            pass
    assert calls, "generated snippet never constructed a Robot"
    return calls[0][0], calls[0][1], printed


def test_real_arm_snippet_reconstructs_the_exact_profile():
    result = render_snippet(ARM_1, hub_host="192.168.1.42", now=0)
    args, kwargs, printed = _run_generated(result["snippet"])
    assert args == ("so101",)
    assert kwargs == {
        "mode": "real",
        "port": "/dev/cu.usbmodem5AB01818061",
        "id": "follower_arm",
        "cameras": ARM_1["cameras"],
        "mesh": True,
        "peer_id": "so101-arm-1",
    }
    assert any("so101-arm-1 online" in line for line in printed)


def test_hub_host_lands_in_zenoh_connect_and_env_is_setdefault():
    import os

    src = render_snippet(ARM_1, hub_host="192.168.1.42")["snippet"]
    env_before = dict(os.environ)
    try:
        os.environ.pop("ZENOH_CONNECT", None)
        os.environ["STRANDS_MESH_CAMERA_HZ"] = "2"  # edge box's own choice
        _run_generated(src)
        assert os.environ["ZENOH_CONNECT"] == "tcp/192.168.1.42:7447"
        assert os.environ["STRANDS_MESH_CAMERA_HZ"] == "2", "setdefault must not clobber"
    finally:
        os.environ.clear()
        os.environ.update(env_before)


def test_no_hub_host_leaves_zenoh_connect_commented():
    src = render_snippet(ARM_1)["snippet"]
    assert '# os.environ.setdefault("ZENOH_CONNECT"' in src
    live = [l for l in src.splitlines() if l.startswith('os.environ.setdefault("ZENOH_CONNECT"')]
    assert not live, "same-machine deploy must not point the peer at itself"


def test_measured_role_and_camera_warning_reach_the_header():
    profile = {**ARM_1, "role": "follower", "role_source": "measured", "role_volts": 12.6}
    src = render_snippet(profile)["snippet"]
    assert "FOLLOWER at 12.6V" in src
    assert "Camera indices are PER-MACHINE" in src
    # a NAMED role (no measurement) must not be presented as measured
    named = {**ARM_1, "role": "leader", "role_source": "name"}
    assert "LEADER" not in render_snippet(named)["snippet"]


def test_sim_payload_renders_without_port_or_connect_block():
    result = render_snippet({"robot_name": "so101", "mode": "sim", "peer_id": "sim-1"})
    args, kwargs, _ = _run_generated(result["snippet"])
    assert kwargs == {"mode": "sim", "mesh": True, "peer_id": "sim-1"}
    assert "connect_eagerly" not in result["snippet"]


@pytest.mark.parametrize(
    "payload, missing",
    [
        ({"mode": "real", "port": "/dev/x", "peer_id": "p"}, "robot_name"),
        ({"robot_name": "so101", "mode": "real", "peer_id": "p"}, "port"),
        ({"robot_name": "so101", "mode": "hover", "peer_id": "p"}, "mode"),
        ({"robot_name": "so101", "mode": "sim"}, "peer_id"),
    ],
)
def test_unrunnable_payload_is_refused_not_guessed(payload, missing):
    result = render_snippet(payload)
    assert missing in result["error"]


def test_filename_is_shell_safe():
    assert snippet_filename("so101 arm/1!") == "so101-arm-1-.py"
    assert snippet_filename("") == "robot.py"


# ---------------------------------------------------------------- route ----


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _client():
    from strands_robots.dashboard.server import create_app

    app = create_app()
    app.state.devices.profiles.get = mock.Mock(
        side_effect=lambda k: dict(ARM_1) if k == "5AB0181806" else None
    )
    return TestClient(app)


def test_route_renders_a_remembered_profile():
    body = _client().post("/api/deploy/snippet", json={"serial": "5AB0181806"}).json()
    assert body["filename"] == "so101-arm-1.py"
    assert "so101-arm-1" in body["snippet"]


def test_route_404s_unknown_serial_and_422s_no_payload():
    client = _client()
    assert client.post("/api/deploy/snippet", json={"serial": "nope"}).status_code == 404
    assert client.post("/api/deploy/snippet", json={}).status_code == 422


def test_route_withholds_loopback_hub_host():
    # TestClient reaches the app on http://testserver - a real hostname, so it
    # IS offered; loopback must not be.
    body = _client().post("/api/deploy/snippet", json={"serial": "5AB0181806"}).json()
    assert "tcp/testserver:7447" in body["snippet"]
    body2 = _client().post(
        "/api/deploy/snippet",
        json={"serial": "5AB0181806", "hub_host": "robots.example.com"},
    ).json()
    assert "tcp/robots.example.com:7447" in body2["snippet"]
