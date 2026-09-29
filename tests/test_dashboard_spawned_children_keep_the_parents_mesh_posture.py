"""A robot the dashboard spawns runs in the mesh posture the dashboard runs in; a real arm never loses auth.

The three child scripts (``_SPAWNER``, ``_COLLECT_SPAWNER``, ``_REPLAY_SPAWNER``) opened with
``os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")`` and ``STRANDS_MESH_MULTICAST=true``
before they looked at ``cfg["mode"]``. ``STRANDS_MESH_LOCAL_DEV`` flips the mesh auth default to
``none`` AND stands in for the ``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE`` acknowledgement, so a
physical arm started from a dashboard whose own environment never mentioned the variable joined
the mesh with no mTLS and no ACL, and multicast scouting re-opened the discovery surface
``scouting_block`` closes by default (f005, CWE-1188 / CWE-306).

Now the child environment is composed in the parent (``child_env``): nothing is added to the
mesh posture, so a child inherits exactly what the dashboard has. A ``mode=real`` spawn is
refused before any process exists when that posture is unauthenticated, unless the operator
set the acknowledgement variable themselves.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from strands_robots.dashboard import device_manager
from strands_robots.dashboard.device_manager import (
    INSECURE_ACK_ENV,
    DeviceManager,
    child_env,
    real_spawn_posture_refusal,
)

MESH_POSTURE_ENVS = ("STRANDS_MESH_LOCAL_DEV", "STRANDS_MESH_MULTICAST", "STRANDS_MESH_AUTH_MODE", INSECURE_ACK_ENV)


@pytest.mark.parametrize("script_name", ["_SPAWNER", "_COLLECT_SPAWNER", "_REPLAY_SPAWNER"])
def test_no_child_script_touches_the_mesh_posture(script_name: str) -> None:
    script = getattr(device_manager, script_name)
    for name in MESH_POSTURE_ENVS:
        assert name not in script, f"{script_name} sets {name} itself"


def test_a_child_inherits_the_parents_posture_and_nothing_more() -> None:
    parent = {"PATH": "/usr/bin", "STRANDS_MESH_LOCAL_DEV": "1", "STRANDS_MESH_MULTICAST": "true"}
    env = child_env("sim", parent)
    assert env["STRANDS_MESH_LOCAL_DEV"] == "1"
    assert env["STRANDS_MESH_MULTICAST"] == "true"
    assert env["STRANDS_MESH"] == "true"
    assert env["STRANDS_MESH_CAMERA_HZ"] == "5"
    assert env["STRANDS_ROBOTS_NO_DYLD_SHIM"] == "1"
    assert parent == {"PATH": "/usr/bin", "STRANDS_MESH_LOCAL_DEV": "1", "STRANDS_MESH_MULTICAST": "true"}


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_a_child_of_an_authenticated_parent_gets_no_dev_shortcut(mode: str) -> None:
    env = child_env(mode, {"PATH": "/usr/bin"})
    assert "STRANDS_MESH_LOCAL_DEV" not in env
    assert "STRANDS_MESH_MULTICAST" not in env
    assert "STRANDS_MESH_AUTH_MODE" not in env


def test_an_operator_value_wins_over_the_child_defaults() -> None:
    env = child_env("sim", {"STRANDS_MESH_CAMERA_HZ": "2", "STRANDS_MESH": "false"})
    assert env["STRANDS_MESH_CAMERA_HZ"] == "2"
    assert env["STRANDS_MESH"] == "false"


# --- the refusal ------------------------------------------------------------------------------


def test_mtls_is_the_default_and_needs_no_acknowledgement() -> None:
    assert real_spawn_posture_refusal({}) is None
    assert real_spawn_posture_refusal({"STRANDS_MESH_AUTH_MODE": "mtls", "STRANDS_MESH_LOCAL_DEV": "1"}) is None


@pytest.mark.parametrize(
    "env",
    [
        {"STRANDS_MESH_LOCAL_DEV": "1"},
        {"STRANDS_MESH_LOCAL_DEV": "true"},
        {"STRANDS_MESH_LOCAL_DEV": "yes", "STRANDS_MESH_AUTH_MODE": "none"},
        {"STRANDS_MESH_AUTH_MODE": "none", INSECURE_ACK_ENV: "1", "STRANDS_MESH_LOCAL_DEV": "0"},
    ],
    ids=["local-dev-1", "local-dev-true", "local-dev-and-none", "none-acked-only-through-the-generic-knob"],
)
def test_an_unauthenticated_parent_cannot_spawn_real_hardware_by_default(env: dict[str, str]) -> None:
    why = real_spawn_posture_refusal(env)
    assert why is not None
    assert why.startswith("refused:")
    assert "real" in why and "mTLS" in why
    assert INSECURE_ACK_ENV in why and "STRANDS_MESH_AUTH_MODE=mtls" in why
    assert "Nothing was started" in why


def test_the_dashboards_own_acknowledgement_lets_a_real_spawn_through() -> None:
    env = {"STRANDS_MESH_LOCAL_DEV": "1", device_manager.REAL_SPAWN_ACK_ENV: "1"}
    assert real_spawn_posture_refusal(env) is None


def test_a_misspelt_auth_mode_refuses_rather_than_guessing() -> None:
    why = real_spawn_posture_refusal({"STRANDS_MESH_AUTH_MODE": "mtsl"})
    assert why is not None and "mtsl" in why


# --- through DeviceManager.spawn --------------------------------------------------------------


class _Proc:
    pid = 4242
    stdout = None

    def poll(self) -> int | None:
        return None


@pytest.fixture
def popen_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _Proc:
        calls.append({"argv": argv, **kwargs})
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(device_manager, "_drain", lambda *a, **k: None)
    monkeypatch.setattr(device_manager.bus_claim, "bus_holders", lambda port, **k: [])
    monkeypatch.setattr(device_manager.camera_liveness, "stamp_device_names", lambda cams, roster: cams)
    monkeypatch.setattr(DeviceManager, "_roster_for_stamp", lambda self: [])
    return calls


def test_spawn_refuses_a_real_arm_when_the_dashboard_runs_without_auth(
    popen_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    monkeypatch.delenv("STRANDS_MESH_AUTH_MODE", raising=False)
    monkeypatch.delenv(device_manager.REAL_SPAWN_ACK_ENV, raising=False)
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    out = dm.spawn("so101", "real", port="/dev/ttyACM0", remember=False)
    assert "error" in out and "mTLS" in out["error"]
    assert popen_calls == [], "no process may exist before the posture is settled"
    assert dm.robots == {}


def test_spawn_under_local_dev_still_starts_sims(
    popen_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    out = dm.spawn("so101", "sim", peer_id="lab-sim", remember=False)
    assert out.get("peer_id") == "lab-sim"
    assert popen_calls[0]["env"]["STRANDS_MESH_LOCAL_DEV"] == "1"


def test_a_real_arm_from_an_mtls_dashboard_starts_in_mtls(
    popen_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("STRANDS_MESH_LOCAL_DEV", raising=False)
    monkeypatch.delenv("STRANDS_MESH_MULTICAST", raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    out = dm.spawn("so101", "real", port="/dev/ttyACM0", peer_id="arm-1", remember=False)
    assert out.get("peer_id") == "arm-1"
    assert out.get("mesh_auth") == "mtls"
    env = popen_calls[0]["env"]
    assert "STRANDS_MESH_LOCAL_DEV" not in env
    assert "STRANDS_MESH_MULTICAST" not in env
    assert env["STRANDS_MESH_AUTH_MODE"] == "mtls"


def test_an_acknowledged_insecure_real_spawn_is_labelled_as_such(
    popen_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    monkeypatch.setenv(device_manager.REAL_SPAWN_ACK_ENV, "1")
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    out = dm.spawn("so101", "real", port="/dev/ttyACM0", peer_id="arm-2", remember=False)
    assert out.get("peer_id") == "arm-2"
    assert out.get("mesh_auth") == "none (acknowledged)"


@pytest.mark.parametrize("spawner", ["collect", "replay"])
def test_one_shot_sim_children_get_the_same_env(
    spawner: str, popen_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    monkeypatch.delenv("STRANDS_MESH_MULTICAST", raising=False)
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    if spawner == "collect":
        out = dm.collect(dataset_root=str(tmp_path), n_episodes=1)
    else:
        monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
        (tmp_path / "local" / "x").mkdir(parents=True)
        out = dm.replay("local/x", episode=0, root=str(tmp_path / "local" / "x"))
    assert "error" not in out, out
    env = popen_calls[0]["env"]
    assert env["STRANDS_MESH_LOCAL_DEV"] == "1"
    assert "STRANDS_MESH_MULTICAST" not in env
