"""A human's yes does not outlive the moment it was given for, or the robot it was given about.

A grant deposited by the dashboard hook sat in a process-global set until a byte-identical call
spent it. Nothing removed it when the approved call failed above the gate (a bad calibration file,
a target outside the arm's travel), when the peer left the fleet, or when the operator walked away
for the night, so a yes given at 09:00 was spendable at 17:00 by whoever drove the agent then
(f026, CWE-613). Now every grant carries the time it was deposited and the target it names: it
expires after ``STRANDS_DASH_MOTION_GRANT_TTL_S`` (15 minutes by default), and the mesh bridge
forgets a peer's grants the moment the peer ages out of the fleet snapshot or the mesh is
re-pointed. A grant that was never spent is a decision that was never acted on.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from strands_robots import _motion_grants
from strands_robots._motion_grants import (
    GRANT_TTL_ENV,
    consume_grant,
    deposit_grant,
    forget_grants_for_peer,
    grant_ttl_s,
    pending_grants,
)

TASK = {"action": "task", "target": "arm-1", "instruction": "wave"}
POSE = {"action": "move_motor", "port": "/dev/ttyACM0", "motor_name": "shoulder_pan", "position": 12.0}


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(GRANT_TTL_ENV, raising=False)
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    yield
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()


# --- the knob -----------------------------------------------------------------------------------


def test_the_default_ttl_is_fifteen_minutes() -> None:
    assert grant_ttl_s() == 900.0


@pytest.mark.parametrize("raw", ["0", "-5", "nan", "inf", "1e999", "soon", ""])
def test_an_unusable_ttl_falls_back_rather_than_removing_the_bound(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRANT_TTL_ENV, raw)
    assert grant_ttl_s() == 900.0


def test_an_operator_ttl_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRANT_TTL_ENV, "30")
    assert grant_ttl_s() == 30.0


# --- expiry -------------------------------------------------------------------------------------


def test_a_fresh_grant_is_spent_once() -> None:
    deposit_grant("fleet", TASK)
    assert consume_grant("fleet", TASK) is True
    assert consume_grant("fleet", TASK) is False


def test_an_expired_grant_is_not_spendable(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(_motion_grants.time, "monotonic", lambda: clock["now"])
    monkeypatch.setenv(GRANT_TTL_ENV, "60")
    deposit_grant("pose_tool", POSE)
    clock["now"] += 61
    assert consume_grant("pose_tool", POSE) is False, "a yes from over a minute ago is not a yes now"
    assert pending_grants() == [], "the expired grant is gone, not merely refused"


def test_a_grant_inside_the_window_is_spendable(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(_motion_grants.time, "monotonic", lambda: clock["now"])
    monkeypatch.setenv(GRANT_TTL_ENV, "60")
    deposit_grant("pose_tool", POSE)
    clock["now"] += 59
    assert consume_grant("pose_tool", POSE) is True


def test_the_ttl_is_read_when_the_grant_is_spent_not_when_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(_motion_grants.time, "monotonic", lambda: clock["now"])
    deposit_grant("pose_tool", POSE)
    clock["now"] += 120
    monkeypatch.setenv(GRANT_TTL_ENV, "60")
    assert consume_grant("pose_tool", POSE) is False, (
        "an operator tightening the window tightens it for grants already given"
    )


def test_a_deposit_sweeps_the_expired_grants_of_other_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(_motion_grants.time, "monotonic", lambda: clock["now"])
    monkeypatch.setenv(GRANT_TTL_ENV, "60")
    deposit_grant("pose_tool", POSE)
    clock["now"] += 61
    deposit_grant("fleet", TASK)
    assert [g["target"] for g in pending_grants()] == ["arm-1"]


# --- teardown with the peer ---------------------------------------------------------------------------


def test_a_peers_grants_leave_with_it() -> None:
    deposit_grant("fleet", TASK)
    deposit_grant("fleet", {**TASK, "target": "arm-2"})
    deposit_grant("arm_1", {"action": "execute", "instruction": "wave"})  # a bound proxy tool IS its peer
    deposit_grant("pose_tool", POSE)
    assert forget_grants_for_peer("arm-1") == 2
    assert consume_grant("fleet", TASK) is False
    assert consume_grant("arm_1", {"action": "execute", "instruction": "wave"}) is False
    assert consume_grant("fleet", {**TASK, "target": "arm-2"}) is True
    assert consume_grant("pose_tool", POSE) is True, "a serial port is not a peer; its grant stays"


def test_forgetting_an_unknown_peer_forgets_nothing() -> None:
    deposit_grant("fleet", TASK)
    assert forget_grants_for_peer("nobody") == 0
    assert forget_grants_for_peer("") == 0
    assert consume_grant("fleet", TASK) is True


def test_pending_grants_say_what_is_outstanding_without_exposing_the_key() -> None:
    deposit_grant("fleet", TASK)
    (grant,) = pending_grants()
    assert grant["tool"] == "fleet" and grant["target"] == "arm-1" and grant["action"] == "task"
    assert grant["age_s"] >= 0 and grant["expires_in_s"] <= grant_ttl_s()
    assert "key" not in grant


def test_the_mesh_bridge_forgets_a_peers_grants_when_it_ages_out_of_the_fleet() -> None:
    from strands_robots.dashboard.mesh_bridge import PEER_TTL_S, MeshBridge

    bridge = MeshBridge(peer_id="dash")
    bridge._running = True
    gone = time.time() - PEER_TTL_S - 5
    bridge.peers = {
        "arm-1": {"peer_id": "arm-1", "last_seen": gone, "first_seen": gone - 60, "presence": {}},
        "arm-2": {"peer_id": "arm-2", "last_seen": time.time(), "first_seen": gone, "presence": {}},
    }
    deposit_grant("fleet", TASK)
    deposit_grant("fleet", {**TASK, "target": "arm-2"})
    snapshot = bridge.snapshot()
    assert "arm-1" not in snapshot["peers"] and "arm-2" in snapshot["peers"]
    assert consume_grant("fleet", TASK) is False, "the robot the yes was about is no longer on the fleet"
    assert consume_grant("fleet", {**TASK, "target": "arm-2"}) is True
    forgotten = [e for e in bridge.activity_log() if e["action"] == "grants_forgotten"]
    assert forgotten and forgotten[0]["target"] == "arm-1" and forgotten[0]["detail"]["count"] == 1


def test_re_pointing_the_mesh_forgets_every_peers_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest import mock

    from strands_robots.dashboard.mesh_bridge import MeshBridge

    bridge = MeshBridge(peer_id="dash")
    bridge._loop = mock.MagicMock(is_closed=lambda: True)
    monkeypatch.setattr(bridge, "stop", lambda: None)
    monkeypatch.setattr(bridge, "start", lambda loop: True)
    bridge.peers = {"arm-1": {"peer_id": "arm-1", "last_seen": time.time()}}
    deposit_grant("fleet", TASK)
    deposit_grant("pose_tool", POSE)
    assert bridge.restart() is True
    assert consume_grant("fleet", TASK) is False
    assert consume_grant("pose_tool", POSE) is True
