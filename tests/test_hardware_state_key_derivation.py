# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Policy action columns must bind only to keys the driver can actually command.

``_initialize_policy`` derived the policy's ``robot_state_keys`` as "every
observation key that is not a declared camera name". Those keys are the index
basis for the policy's action output (``_tensor_to_action_dicts`` maps tensor
column i to key i), so anything that slips into the list becomes a joint the
policy believes it commands - and shifts every subsequent column onto the wrong
joint.

The filter over-collects, verified against the installed lerobot 0.6.0:

* ``<cam>_depth`` is keyed ``f"{cam_key}_depth"``, not ``cam_key``, so the camera
  filter misses it and an ndarray depth frame enters the state vector. Emitted by
  every depth-capable driver (``so_follower``, ``koch_follower``,
  ``omx_follower``, ``openarm_follower``, ``rebot_b601_follower``, ``hope_jr``).
* ``openarm_follower`` additionally emits ``<motor>.vel`` and ``<motor>.torque``,
  triple-counting every joint.

Meanwhile ``SOFollower.send_action`` keeps ONLY ``"<motor>.pos"`` entries, so
those extra columns are silently discarded by the driver after having displaced
the real ones.

The derivation now prefers the driver's own declared ``action_features``, which is
exactly the set ``send_action`` accepts, and the fallback keeps scalars only.
No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import pytest

from strands_robots.hardware_robot import Robot

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_POS_KEYS = [f"{m}.pos" for m in _MOTORS]


class _Config:
    def __init__(self, cameras: dict[str, object] | None = None) -> None:
        self.cameras = cameras or {}


class _Driver:
    """Stand-in for a lerobot Robot driver."""

    def __init__(self, action_features=None, cameras=None) -> None:
        self._action_features = action_features
        self.config = _Config(cameras)

    @property
    def action_features(self):
        if self._action_features is None:
            raise AttributeError("action_features")
        return self._action_features


def _robot(driver: _Driver) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "so101"
    hw.robot = driver
    return hw


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class TestDeclaredFeaturesWithMixedChannels:
    """A driver may DECLARE more channels than ``send_action`` consumes.

    lerobot's ``openarm_follower`` with ``use_velocity_and_torque=True`` sets
    ``action_features == _motors_ft``, emitting ``<motor>.pos`` AND ``.vel`` AND
    ``.torque`` - 21 entries for a 7-DOF arm - while its ``send_action`` does
    ``if key.endswith(".pos")`` and discards the rest (verified in the installed
    lerobot source). Binding is positional, so all 21 would make a 7-D
    checkpoint's columns land on pos/vel/torque triples: ``joint_1 <- model[0]``,
    ``joint_2 <- model[3]``, ``joint_3 <- model[6]``, joints 4..7 zero-filled.

    Preferring ``action_features`` was the right call, but it routed past the very
    channel filter the fallback branch applies - and whose docstring names this
    exact driver.
    """

    _JOINTS = [f"joint_{i}" for i in range(1, 8)]

    def _mixed_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for joint in self._JOINTS:
            features[f"{joint}.pos"] = float
            features[f"{joint}.vel"] = float
            features[f"{joint}.torque"] = float
        return features

    def test_only_the_commandable_pos_channel_is_bound(self):
        features = self._mixed_features()
        driver = _Driver(action_features=features, cameras={"front": {}})
        observation = {**dict.fromkeys(features, 0.0), "front": _frame()}

        keys = _robot(driver)._derive_robot_state_keys(observation)

        assert keys == [f"{j}.pos" for j in self._JOINTS]
        assert len(keys) == 7, f"a 7-DOF arm must bind 7 columns, got {len(keys)}"

    def test_the_dropped_channels_are_logged(self):
        features = self._mixed_features()
        driver = _Driver(action_features=features)
        observation = dict.fromkeys(features, 0.0)

        with caplog_at_info() as records:
            _robot(driver)._derive_robot_state_keys(observation)

        messages = " ".join(records)
        assert "send_action does not consume" in messages
        assert "joint_1.vel" in messages or "joint_1.torque" in messages

    def test_a_pos_only_driver_is_unaffected(self):
        """The SO-101 case must not change: no mixed family, nothing dropped."""
        driver = _Driver(action_features=dict.fromkeys(_POS_KEYS, float))
        observation = dict.fromkeys(_POS_KEYS, 0.0)

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS

    def test_a_driver_declaring_no_pos_channel_keeps_its_own_keys(self):
        """Do not force a '.pos' convention on a driver that has none."""
        features = {"a": float, "b": float}
        driver = _Driver(action_features=features)

        keys = _robot(driver)._derive_robot_state_keys(dict.fromkeys(features, 0.0))

        assert keys == ["a", "b"]

    def test_a_declared_key_whose_value_is_an_array_never_takes_a_column(self):
        """A declared key observed as an ndarray is not a joint command."""
        driver = _Driver(action_features={**dict.fromkeys(_POS_KEYS, float), "depth": tuple})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "depth": _frame()}

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS


@contextlib.contextmanager
def caplog_at_info():
    """Collect INFO records from the hardware_robot logger."""
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("strands_robots.hardware_robot")
    handler = _Handler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class TestDeclaredActionFeaturesWin:
    def test_binds_to_the_drivers_declared_action_keys(self):
        driver = _Driver(action_features=dict.fromkeys(_POS_KEYS, float), cameras={"front": {}})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "front": _frame()}

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS

    def test_declared_order_is_preserved_not_observation_order(self):
        """Action columns bind by index, so the ORDER is the contract."""
        reversed_keys = list(reversed(_POS_KEYS))
        driver = _Driver(action_features=dict.fromkeys(reversed_keys, float))
        # Observation arrives in a different order than the driver declares.
        observation = dict.fromkeys(_POS_KEYS, 0.0)

        assert _robot(driver)._derive_robot_state_keys(observation) == reversed_keys

    def test_depth_frame_is_excluded(self):
        """Regression: '<cam>_depth' is not a camera NAME, so the old filter kept it."""
        driver = _Driver(action_features=dict.fromkeys(_POS_KEYS, float), cameras={"front": {}})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "front": _frame(), "front_depth": _frame()}

        keys = _robot(driver)._derive_robot_state_keys(observation)

        assert "front_depth" not in keys
        assert keys == _POS_KEYS

    def test_extra_per_motor_channels_are_excluded(self):
        """Regression: openarm_follower emits .vel/.torque alongside .pos."""
        driver = _Driver(action_features=dict.fromkeys(_POS_KEYS, float))
        observation = dict.fromkeys(_POS_KEYS, 0.0)
        for motor in _MOTORS:
            observation[f"{motor}.vel"] = 0.0
            observation[f"{motor}.torque"] = 0.0

        keys = _robot(driver)._derive_robot_state_keys(observation)

        assert keys == _POS_KEYS
        assert not any(k.endswith((".vel", ".torque")) for k in keys)

    def test_declared_key_absent_from_the_observation_is_dropped(self):
        """Binding a key the observation never reports would zero-fill that column."""
        driver = _Driver(action_features=dict.fromkeys([*_POS_KEYS, "phantom.pos"], float))
        observation = dict.fromkeys(_POS_KEYS, 0.0)

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS

    def test_the_binding_is_logged(self, caplog):
        driver = _Driver(action_features=dict.fromkeys(_POS_KEYS, float), cameras={"front": {}})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "front": _frame(), "front_depth": _frame()}

        with caplog.at_level(logging.INFO):
            _robot(driver)._derive_robot_state_keys(observation)

        msgs = [r.getMessage() for r in caplog.records]
        assert any("declared action_features" in m for m in msgs), msgs
        # The dropped keys are named, so a mis-binding is diagnosable from the log.
        assert any("front_depth" in m for m in msgs), msgs


class TestFallbackWhenNoDeclaredFeatures:
    def test_falls_back_to_observation_scalars_minus_cameras(self):
        driver = _Driver(action_features=None, cameras={"front": {}})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "front": _frame()}

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS

    def test_fallback_still_excludes_non_scalar_values(self):
        """A depth frame must never occupy an action column, even in the fallback."""
        driver = _Driver(action_features=None, cameras={"front": {}})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "front": _frame(), "front_depth": _frame()}

        keys = _robot(driver)._derive_robot_state_keys(observation)

        assert keys == _POS_KEYS

    def test_fallback_excludes_bools(self):
        """A driver status flag (is_homed, ...) is not a joint command."""
        driver = _Driver(action_features={})
        observation = {**dict.fromkeys(_POS_KEYS, 0.0), "is_homed": True}

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS

    def test_warns_when_declared_features_match_nothing(self, caplog):
        """A total mismatch is a real misconfiguration; do not fall back silently."""
        driver = _Driver(action_features={"totally.different": float})
        observation = dict.fromkeys(_POS_KEYS, 0.0)

        with caplog.at_level(logging.WARNING):
            keys = _robot(driver)._derive_robot_state_keys(observation)

        assert keys == _POS_KEYS  # fallback still produces a usable binding
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("reports none of them" in m for m in msgs), msgs

    def test_non_dict_action_features_are_ignored(self):
        driver = _Driver(action_features=["not", "a", "dict"])
        observation = dict.fromkeys(_POS_KEYS, 0.0)

        assert _robot(driver)._derive_robot_state_keys(observation) == _POS_KEYS


class TestAgainstTheRealDriverContract:
    def test_real_so_follower_declares_exactly_the_pos_keys(self):
        """Pin the assumption this fix rests on, against the INSTALLED lerobot."""
        pytest.importorskip("lerobot")
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        driver = SOFollower(SO101FollowerConfig(port="/dev/null", id="test"))

        # action_features is the set send_action accepts; it must be the .pos keys.
        assert set(driver.action_features) == set(_POS_KEYS)
