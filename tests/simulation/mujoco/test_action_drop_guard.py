"""Tests for the strict action-drop guard.

The guard turns the silent "policy emits unmappable action keys -> arm frozen"
failure (so101 1..6, cosmos3 Cartesian midtrain, wrong-frame poses) into a loud,
actionable warning/error. Controlled by STRANDS_SIM_ACTION_DROP_MODE
(warn|raise|off) + STRANDS_SIM_ACTION_DROP_THRESHOLD.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation

# Canonical SO101 (robotstudio_so101/so101_new_calib.xml) names its actuators
# AND joints "1".."6" (bare numerics). Verified on a clean upstream asset
# cache (Thor). A semantic-named so101.xml only appears in polluted local
# caches; do not assume it.
_SO_JOINTS = ["1", "2", "3", "4", "5", "6"]
_WRONG_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture
def sim():
    s = Simulation(tool_name="drop_test", mesh=False)
    s.create_world()
    s.add_robot(name="arm", data_config="so101")
    yield s
    s.cleanup()


class TestApplyActionReturnsStats:
    def test_all_valid_keys_zero_dropped(self, sim):
        mj = sim._mj
        m, d = sim._world._model, sim._world._data
        action = {j: 0.1 for j in _SO_JOINTS}
        applied, dropped = sim._apply_action_by_name(m, d, action, "arm/", mj)
        assert applied == len(_SO_JOINTS)
        assert dropped == []

    def test_unmappable_keys_reported(self, sim):
        mj = sim._mj
        m, d = sim._world._model, sim._world._data
        # Cartesian EE keys match no SO101 actuator/joint; only "1" resolves
        action = {"ee_x": 0.1, "ee_y": 0.0, "ee_z": 0.2, "1": 0.1}
        applied, dropped = sim._apply_action_by_name(m, d, action, "arm/", mj)
        assert applied == 1  # only joint "1" resolved
        assert set(dropped) == {"ee_x", "ee_y", "ee_z"}

    def test_wrong_joint_names_all_dropped(self, sim):
        mj = sim._mj
        m, d = sim._world._model, sim._world._data
        # The regression we caught on Thor: semantic keys against the
        # canonical numeric-named so101_new_calib.xml -> every key drops.
        action = {k: 0.1 for k in _WRONG_KEYS}
        applied, dropped = sim._apply_action_by_name(m, d, action, "arm/", mj)
        assert applied == 0
        assert len(dropped) == 6


class TestDropModeRaise:
    def test_raise_on_full_drop(self, sim, monkeypatch):
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_MODE", "raise")
        cartesian = {"ee_x": 0.1, "ee_y": 0.0, "ee_z": 0.2, "gripper": 0.0}
        with pytest.raises(RuntimeError, match="action-space mismatch"):
            sim.send_action(cartesian, robot_name="arm", n_substeps=1)

    def test_no_raise_on_valid_action(self, sim, monkeypatch):
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_MODE", "raise")
        good = {j: 0.1 for j in _SO_JOINTS}
        # must not raise
        sim.send_action(good, robot_name="arm", n_substeps=1)

    def test_no_raise_below_threshold(self, sim, monkeypatch):
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_MODE", "raise")
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_THRESHOLD", "0.5")
        # 1 dropped of 6 = 17% < 50% -> no raise
        action = {j: 0.1 for j in _SO_JOINTS}
        action["bogus_joint"] = 0.0  # 1/7 dropped
        sim.send_action(action, robot_name="arm", n_substeps=1)

    def test_custom_threshold_triggers(self, sim, monkeypatch):
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_MODE", "raise")
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_THRESHOLD", "0.1")
        action = {j: 0.1 for j in _SO_JOINTS}
        action["bogus_joint"] = 0.0  # 1/7 = 14% > 10%
        with pytest.raises(RuntimeError, match="action-space mismatch"):
            sim.send_action(action, robot_name="arm", n_substeps=1)


class TestDropModeOff:
    def test_off_never_raises(self, sim, monkeypatch):
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_MODE", "off")
        cartesian = {"ee_x": 0.1, "ee_y": 0.0, "ee_z": 0.2}  # all drop
        # must not raise even at 100% drop
        sim.send_action(cartesian, robot_name="arm", n_substeps=1)


class TestDropModeWarnDefault:
    def test_default_is_warn_not_raise(self, sim, monkeypatch):
        monkeypatch.delenv("STRANDS_SIM_ACTION_DROP_MODE", raising=False)
        cartesian = {"ee_x": 0.1, "ee_y": 0.0, "ee_z": 0.2}  # all drop
        # default warn mode: must NOT raise (backwards compatible)
        sim.send_action(cartesian, robot_name="arm", n_substeps=1)


class TestEmptyActionNoFalsePositive:
    def test_empty_action_no_drop(self, sim, monkeypatch):
        monkeypatch.setenv("STRANDS_SIM_ACTION_DROP_MODE", "raise")
        # empty dict -> applied=0, dropped=[] -> no-op, not a "drop"
        sim.send_action({}, robot_name="arm", n_substeps=1)
