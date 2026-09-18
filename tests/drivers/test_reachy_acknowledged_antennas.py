"""Acknowledged antenna targets preserve gates; HTTP OK is not physical completion."""

import math
import time

import pytest

from strands_robots.drivers.reachy import ReachyDriver


@pytest.fixture
def driver(monkeypatch):
    d = ReachyDriver()
    d._connected = True
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    d._on_joints({"head_joint_positions": [0.0] * 7, "antennas_joint_positions": [0.0, 0.0]})
    monkeypatch.setattr(d, "_send_cmd", lambda *a: pytest.fail("ack path must not fall back to WS"))
    return d


def test_acknowledged_pair_is_one_rest_target_in_right_left_radians(driver, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "_daemon_post", lambda path, data: calls.append((path, data)) or {"status": "ok"})
    result = driver.send_action({"antenna_right": 10, "antenna_left": -20}, require_ack=True)
    assert calls == [("/api/move/set_target", {"target_antennas": [math.radians(10), math.radians(-20)]})]
    assert result["status"] == "success"
    assert result["content"][0]["json"]["acknowledgement"] == "daemon_target_handler_ok"
    assert result["content"][0]["json"]["motion_verified"] is False


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "ignored", "reason": "move_running"},
        {"error": "timeout"},
        {},
        {"status": "success"},
        {"status": "ok", "error": "failed"},
    ],
)
def test_only_exact_ok_is_acknowledged_and_never_retried(driver, monkeypatch, reply):
    calls = []
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: calls.append(a) or reply)
    result = driver.send_action({"antenna_right": 10, "antenna_left": 20}, require_ack=True)
    assert result["status"] == "error"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "action",
    [
        {},
        {"antenna_right": 10},
        {"antenna_right": 10, "head_yaw": 0},
        {"antenna_right": float("nan"), "antenna_left": 0},
        {"antenna_right": 0, "antenna_left": 0, "head_pitch": 999},
        {"antenna_right": 0, "antenna_left": 0, "bogus": 1},
    ],
)
def test_ack_gates_precede_post(driver, monkeypatch, action):
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: pytest.fail("invalid action posted"))
    assert driver.send_action(action, require_ack=True)["status"] == "error"


@pytest.mark.parametrize("stamp", [None, float("nan"), True, 0, 3700.0])
def test_ack_requires_recent_telemetry(driver, monkeypatch, stamp):
    driver._joints_received_at = stamp
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: pytest.fail("stale telemetry posted"))
    assert driver.send_action({"antenna_right": 10, "antenna_left": 20}, require_ack=True)["status"] == "error"


def test_ack_requires_connection(driver, monkeypatch):
    driver._connected = False
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: pytest.fail("disconnected driver posted"))
    assert driver.send_action({"antenna_right": 10, "antenna_left": 20}, require_ack=True)["status"] == "error"


def test_ack_flag_is_boolean(driver, monkeypatch):
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: pytest.fail("invalid flag posted"))
    assert driver.send_action({"antenna_right": 10, "antenna_left": 20}, require_ack="true")["status"] == "error"


@pytest.mark.parametrize("step", [-3600, 3600])
def test_fresh_ack_survives_wall_clock_steps(driver, monkeypatch, step):
    wall_stamp = driver._joints["t"]
    monkeypatch.setattr(time, "time", lambda: wall_stamp + step)
    monkeypatch.setattr(time, "monotonic", lambda: 100.1)
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: {"status": "ok"})
    assert driver.send_action({"antenna_right": 1, "antenna_left": 1}, require_ack=True)["status"] == "success"


def test_old_receipt_refuses_even_when_wall_clock_looks_fresh(driver, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: driver._joints["t"])
    monkeypatch.setattr(time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: pytest.fail("expired receipt posted"))
    assert driver.send_action({"antenna_right": 1, "antenna_left": 1}, require_ack=True)["status"] == "error"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"head_joint_positions": [0.0] * 7, "antennas_joint_positions": [0.0]},
        {"head_joint_positions": [0.0] * 7, "antennas_joint_positions": [float("nan"), 0.0]},
        {"head_joint_positions": ["bad"], "antennas_joint_positions": [0.0, 0.0]},
    ],
)
def test_bad_joint_message_cannot_refresh_ack_permission(driver, monkeypatch, payload):
    driver._on_joints(payload)
    monkeypatch.setattr(driver, "_daemon_post", lambda *a: pytest.fail("invalid telemetry posted"))
    assert driver.send_action({"antenna_right": 1, "antenna_left": 1}, require_ack=True)["status"] == "error"


def test_cleanup_clears_receipt(driver):
    driver.cleanup()
    assert driver._joints_received_at is None
