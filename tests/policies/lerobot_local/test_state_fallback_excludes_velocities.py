# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The observation-derived state ordering must not interleave velocity siblings.

When the configured ``robot_state_keys`` match nothing in the observation (the
generic ``joint_0..joint_N`` vs named-joint mismatch), the ordering falls back
to the observation's own scalar keys. The MuJoCo backend emits a velocity
sibling beside every joint position -- ``simulation/mujoco/rendering.py``::

    obs[jnt_name] = float(data.qpos[model.jnt_qposadr[jnt_id]])
    obs[f"{jnt_name}.vel"] = float(data.qvel[model.jnt_dofadr[jnt_id]])

-- so the observation's insertion order is ``[pos0, vel0, pos1, vel1, ...]``.
Taking it unfiltered made ``observation.state`` twice the DOF count, then
truncated it to the model's declared state dim, so HALF the slots held
velocities and the trailing joints were dropped entirely.

Measured on a 6-DoF arm driven to distinct known values::

    resolved order        ['1','1.vel','2','2.vel','3','3.vel','4','4.vel','5','5.vel','6','6.vel']
    first 6 (6-dim model) [0.2981, -0.4700, -0.1973, 0.6561, 0.4978, -0.5439]
    CORRECT positions     [0.2981, -0.1973, 0.4978, 0.0987, -0.3974, 0.2474]
    dropped joints        ['wrist_flex', 'wrist_roll', 'gripper']

A wrong state vector, with nothing raised and nothing logged about it.

A ``.vel`` key with NO position companion is KEPT: ``embodiments.json`` declares
LeKiwi body-frame base velocities ``x.vel`` / ``y.vel`` / ``theta.vel`` with no
``x`` / ``y`` / ``theta`` position key, so a blanket suffix drop would corrupt
those instead. Pairing is decided per key.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# A 6-DoF SO-style arm, in MJCF declaration order.
_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_POSITIONS = [0.2981, -0.1973, 0.4978, 0.0987, -0.3974, 0.2474]
_VELOCITIES = [-0.4700, 0.6561, -0.5439, 0.3310, 0.1200, -0.2900]

# Keys that match nothing a named-joint sim reports -> forces the fallback.
_GENERIC_KEYS = [f"joint_{i}" for i in range(6)]


def _mujoco_observation() -> dict[str, float]:
    """An observation in exactly the key order mujoco/rendering.py emits."""
    obs: dict[str, float] = {}
    for name, pos, vel in zip(_JOINTS, _POSITIONS, _VELOCITIES, strict=True):
        obs[name] = pos
        obs[f"{name}.vel"] = vel
    return obs


def _make_policy(*, state_dim: int = 6, robot_state_keys: list[str] | None = _GENERIC_KEYS):
    """A policy that appears loaded, with a declared ``observation.state`` dim."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")
    policy._loaded = True
    policy._device = torch.device("cpu")
    state_feat = MagicMock()
    state_feat.shape = (state_dim,)
    policy._input_features = {"observation.state": state_feat}
    action_feat = MagicMock()
    action_feat.shape = (6,)
    policy._output_features = {"action": action_feat}
    if robot_state_keys is not None:
        policy.set_robot_state_keys(list(robot_state_keys))
    return policy


def _state_of(policy, observation: dict[str, float]) -> list[float]:
    """The ``observation.state`` vector the policy would feed the model."""
    batch = policy._build_observation_batch(observation, "pick up the cube")
    return [round(float(v), 4) for v in batch["observation.state"].flatten().tolist()]


class TestTheStateVectorHoldsPositionsOnly:
    def test_the_state_vector_is_the_joint_positions(self):
        """Regression: it was [pos0, vel0, pos1, vel1, pos2, vel2]."""
        assert _state_of(_make_policy(), _mujoco_observation()) == _POSITIONS

    def test_no_velocity_reading_reaches_the_state_vector(self):
        state = _state_of(_make_policy(), _mujoco_observation())

        assert not (set(state) & set(_VELOCITIES)), f"velocity value(s) in state: {state}"

    def test_no_joint_is_dropped_by_truncation(self):
        """Half the slots being velocities pushed the last 3 joints out."""
        state = _state_of(_make_policy(), _mujoco_observation())

        for name, pos in zip(_JOINTS, _POSITIONS, strict=True):
            assert pos in state, f"{name} is missing from observation.state"

    def test_declaration_order_is_preserved(self):
        obs = _mujoco_observation()

        assert _state_of(_make_policy(), obs) == [obs[j] for j in _JOINTS]

    def test_the_width_is_the_dof_count_not_twice_it(self):
        policy = _make_policy(state_dim=12)

        # 12 declared dims, 6 real joints: the 6 unreported dims are zero-filled
        # in place rather than back-filled with velocities.
        assert _state_of(policy, _mujoco_observation()) == _POSITIONS + [0.0] * 6

    def test_the_other_build_path_agrees(self):
        """``_to_lerobot_observation`` (processor path) resolves the same order."""
        policy = _make_policy()

        state = policy._to_lerobot_observation(_mujoco_observation())["observation.state"]

        assert [round(float(v), 4) for v in np.asarray(state).flatten().tolist()] == _POSITIONS


class TestUnpairedVelocityKeysSurvive:
    """LeKiwi declares x.vel/y.vel/theta.vel with no position companion."""

    def test_base_velocities_with_no_position_companion_are_kept(self):
        policy = _make_policy(state_dim=4)
        obs = {"shoulder_pan.pos": 0.1, "x.vel": 0.5, "y.vel": -0.2, "theta.vel": 0.05}

        assert _state_of(policy, obs) == [0.1, 0.5, -0.2, 0.05]

    def test_a_mobile_manipulator_keeps_base_and_drops_arm_siblings(self):
        """Both rules apply to one observation: pair -> drop, unpaired -> keep."""
        policy = _make_policy(state_dim=4)
        obs = {"elbow": 0.3, "elbow.vel": 9.9, "x.vel": 0.5, "theta.vel": 0.05}

        assert _state_of(policy, obs) == [0.3, 0.5, 0.05, 0.0]

    def test_a_position_key_named_with_the_pos_suffix_is_not_a_companion(self):
        """``elbow.pos`` does not pair with ``elbow.vel`` - only ``elbow`` does."""
        policy = _make_policy(state_dim=2)
        obs = {"elbow.pos": 0.3, "elbow.vel": 1.5}

        assert _state_of(policy, obs) == [0.3, 1.5]


class TestConfiguredKeysAreHonoredVerbatim:
    def test_an_explicitly_configured_vel_key_is_not_filtered(self):
        """An operator naming a velocity is stating the model's input."""
        policy = _make_policy(state_dim=2, robot_state_keys=["elbow", "elbow.vel"])
        obs = {"elbow": 0.3, "elbow.vel": 1.5}

        assert _state_of(policy, obs) == [0.3, 1.5]

    def test_a_partial_match_still_uses_the_configured_order(self):
        """One configured key present is enough - no fallback, no filtering."""
        policy = _make_policy(state_dim=3, robot_state_keys=["elbow", "elbow.vel", "absent"])
        obs = {"elbow": 0.3, "elbow.vel": 1.5}

        assert _state_of(policy, obs) == [0.3, 1.5, 0.0]


class TestTheNoConfiguredKeysPathIsFilteredToo:
    """With no configured keys the ordering is ENTIRELY observation-derived.

    Only the processor path reaches it: ``_build_batch_from_strands_format``
    refuses an empty ``robot_state_keys`` outright (pinned below), while
    ``_to_lerobot_observation`` resolves the order from the observation.
    """

    def test_a_policy_with_no_robot_state_keys_gets_positions_only(self):
        policy = _make_policy(robot_state_keys=None)
        assert policy.robot_state_keys == []

        state = policy._to_lerobot_observation(_mujoco_observation())["observation.state"]

        assert [round(float(v), 4) for v in np.asarray(state).flatten().tolist()] == _POSITIONS

    def test_the_other_path_still_refuses_empty_keys(self):
        """Unchanged: it cannot map an observation without them."""
        policy = _make_policy(robot_state_keys=None)

        with pytest.raises(ValueError, match="robot_state_keys is empty"):
            _state_of(policy, _mujoco_observation())


class TestUnchangedBehaviour:
    def test_the_mismatch_warning_still_fires(self, caplog):
        """Filtering must not quiet the misconfiguration that caused it."""
        policy = _make_policy()

        with caplog.at_level(logging.WARNING):
            _state_of(policy, _mujoco_observation())

        assert any("robot_state_keys" in r.getMessage() for r in caplog.records), caplog.records

    def test_the_degradation_telemetry_is_still_set(self):
        policy = _make_policy()
        _state_of(policy, _mujoco_observation())

        assert policy.generic_state_keys_used is True

    def test_strict_keys_still_raises_before_filtering(self):
        policy = _make_policy()
        policy.strict_keys = True

        with pytest.raises(ValueError, match="strict_keys=True"):
            _state_of(policy, _mujoco_observation())

    def test_a_velocity_only_observation_is_unchanged(self):
        """No position companion anywhere: nothing to drop, so nothing changes."""
        policy = _make_policy(state_dim=2)
        obs = {"a.vel": 1.0, "b.vel": 2.0}

        assert _state_of(policy, obs) == [1.0, 2.0]

    def test_an_observation_with_no_velocities_is_unchanged(self):
        policy = _make_policy()
        obs = dict(zip(_JOINTS, _POSITIONS, strict=True))

        assert _state_of(policy, obs) == _POSITIONS

    def test_the_task_key_is_still_excluded(self):
        policy = _make_policy(state_dim=6)
        obs = _mujoco_observation() | {"task": "pick up the cube"}

        assert _state_of(policy, obs) == _POSITIONS
