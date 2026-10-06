"""A deploy snippet for a real arm carries the spawn route's mesh-auth decision, not a looser one.

The spawn route refuses a ``mode=real`` child when the dashboard runs with wire auth off unless
the operator set ``STRANDS_DASH_REAL_SPAWN_WITHOUT_MESH_AUTH``. The snippet starts the same arm
on another box, so it refuses in the same place, renders no baked-in multicast or local-dev line,
and the mesh no longer lets the local-dev flag acknowledge an auth-off link that leaves loopback.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard import deploy, device_manager
from strands_robots.mesh import _zenoh_config

_REAL = {"robot_name": "so101", "mode": "real", "port": "/dev/ttyACM0", "peer_id": "arm-1"}
_SIM = {"robot_name": "so101", "mode": "sim", "peer_id": "sim-1"}
_ACK = {device_manager.REAL_SPAWN_ACK_ENV: "1"}
_WARNING = "# WARNING: mesh wire auth is off for a real arm, acknowledged by STRANDS_DASH_REAL_SPAWN_WITHOUT_MESH_AUTH"


@pytest.mark.parametrize(
    "mesh_env",
    [
        {"STRANDS_MESH_LOCAL_DEV": "1"},
        {"STRANDS_MESH_AUTH_MODE": "none", device_manager.INSECURE_ACK_ENV: "1"},
        {"STRANDS_MESH_AUTH_MODE": "mtsl"},  # a posture the resolver cannot read: refused, not guessed
    ],
)
def test_a_real_snippet_refuses_where_the_spawn_route_refuses(mesh_env):
    out = deploy.render_snippet(_REAL, hub_host="127.0.0.1", mesh_env=mesh_env)
    assert out == {"error": device_manager.real_spawn_posture_refusal(mesh_env)}
    assert "snippet" not in out


@pytest.mark.parametrize(
    ("payload", "mesh_env", "present", "absent"),
    [
        # mTLS dashboard: no auth-off line and no multicast default, for either mode.
        (_REAL, {}, [], ["STRANDS_MESH_LOCAL_DEV", "STRANDS_MESH_MULTICAST", "WARNING"]),
        (_SIM, {}, [], ["STRANDS_MESH_LOCAL_DEV", "STRANDS_MESH_MULTICAST"]),
        # A live multicast value is the dashboard's own posture and is carried over.
        (_REAL, {"STRANDS_MESH_MULTICAST": "true"}, ["'STRANDS_MESH_MULTICAST', 'true'"], ["LOCAL_DEV"]),
        # The operator's explicit yes: the auth-off posture is rendered, under a warning.
        (
            _REAL,
            {"STRANDS_MESH_LOCAL_DEV": "1", **_ACK},
            [_WARNING + "\nos.environ.setdefault('STRANDS_MESH_LOCAL_DEV', '1')"],
            [device_manager.REAL_SPAWN_ACK_ENV + "'"],
        ),
        # A sim snippet still carries a live local-dev posture as it is.
        (_SIM, {"STRANDS_MESH_LOCAL_DEV": "1"}, ["'STRANDS_MESH_LOCAL_DEV', '1'"], ["WARNING"]),
    ],
)
def test_the_rendered_mesh_block_is_the_dashboards_live_posture(payload, mesh_env, present, absent):
    snippet = deploy.render_snippet(payload, hub_host="127.0.0.1", mesh_env=mesh_env)["snippet"]
    for text in present:
        assert text in snippet, text
    for text in absent:
        assert text not in snippet, text


def test_resolve_mesh_env_drops_auth_off_keys_for_an_unacknowledged_real_arm():
    live = {"STRANDS_MESH_LOCAL_DEV": "1", device_manager.INSECURE_ACK_ENV: "1"}
    keys = dict(deploy.resolve_mesh_env(live, "real"))
    assert "STRANDS_MESH_LOCAL_DEV" not in keys and device_manager.INSECURE_ACK_ENV not in keys
    assert dict(deploy.resolve_mesh_env(live, "sim"))["STRANDS_MESH_LOCAL_DEV"] == "1"
    assert "STRANDS_MESH_MULTICAST" not in dict(deploy.resolve_mesh_env({}, "sim"))


@pytest.mark.parametrize(
    ("env", "outcome"),
    [
        ({}, "none"),  # the session's own default endpoints are loopback
        ({"ZENOH_CONNECT": "tcp/127.0.0.1:7447", "ZENOH_LISTEN": "tcp/[::1]:7448"}, "none"),
        ({"ZENOH_CONNECT": "tcp/localhost:7447"}, "none"),
        ({"ZENOH_CONNECT": "tcp/192.168.1.20:7447"}, ValueError),
        ({"ZENOH_LISTEN": "tcp/0.0.0.0:7447"}, ValueError),
        ({"ZENOH_CONNECT": "tcp/hub.lab:7447"}, ValueError),
        ({"STRANDS_MESH_MULTICAST": "true"}, ValueError),
        ({"ZENOH_CONNECT": "tcp/192.168.1.20:7447", "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE": "1"}, "none"),
    ],
)
def test_local_dev_acknowledges_auth_off_only_on_loopback(monkeypatch, env, outcome):
    for name in (
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
        "STRANDS_MESH_MULTICAST",
        "ZENOH_CONNECT",
        "ZENOH_LISTEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if outcome is ValueError:
        with pytest.raises(ValueError, match="STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"):
            _zenoh_config.resolve_auth_mode()
    else:
        assert _zenoh_config.resolve_auth_mode() == outcome
