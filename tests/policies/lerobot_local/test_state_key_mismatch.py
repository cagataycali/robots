# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""State-key mismatch must be LOUD, never a silent zero/empty state vector.

When ``robot_state_keys`` is auto-filled with generic names (``joint_0..N``,
derived from the model action dim) but the sim/robot reports named joints
(``shoulder_pan`` ...), none of the configured keys match the observation.
Both observation-to-batch paths previously handled this silently:

  * the preprocessor/VLA path (:meth:`_to_lerobot_observation`) fell back to
    the observation's scalar keys with no warning, and
  * the strands-native path (:meth:`_build_batch_from_strands_format`) dropped
    ``observation.state`` entirely,

so the policy ran open-loop (state conditioned on zeros / missing) with
``action_errors=0`` and no error log. These tests pin that the mismatch is now
loud: ``strict_keys=True`` raises an actionable error naming the actual
observation keys, and ``strict_keys=False`` warns once and falls back so the
state is populated rather than silently zeroed or dropped.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

NAMED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(policy_type="molmoact2", *, strict_keys=False):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type=policy_type, strict_keys=strict_keys)
    policy._input_features = {"observation.images.base": _visual(), "observation.state": _state(6)}
    policy._device = torch.device("cpu")
    # Generic auto-generated keys that match NO key in the named-joint obs.
    policy.robot_state_keys = [f"joint_{i}" for i in range(6)]
    return policy


def _named_obs():
    obs = {"base": np.zeros((224, 224, 3), np.uint8)}
    obs.update({name: 0.5 for name in NAMED_JOINTS})
    return obs


class TestStrictKeysRaises:
    """strict_keys=True turns a state-key mismatch into a clear, actionable error."""

    def test_vla_path_raises_with_actionable_message(self):
        policy = _policy("molmoact2", strict_keys=True)
        with pytest.raises(ValueError) as exc:
            policy._to_lerobot_observation(_named_obs())
        msg = str(exc.value)
        # Names the actual observation keys ...
        assert "shoulder_pan" in msg
        # ... and points at the two documented remedies.
        assert "embodiment=" in msg
        assert "set_robot_state_keys" in msg

    def test_strands_native_path_raises_with_actionable_message(self):
        policy = _policy("act", strict_keys=True)
        with pytest.raises(ValueError) as exc:
            policy._build_batch_from_strands_format(_named_obs(), {})
        msg = str(exc.value)
        assert "shoulder_pan" in msg
        assert "embodiment=" in msg
        assert "set_robot_state_keys" in msg


class TestNonStrictWarnsAndFallsBack:
    """strict_keys=False warns once and falls back so state is never silently lost."""

    def test_vla_path_populates_state_from_named_joints(self, caplog):
        policy = _policy("molmoact2", strict_keys=False)
        with caplog.at_level("WARNING"):
            out = policy._to_lerobot_observation(_named_obs())
        assert "observation.state" in out
        np.testing.assert_array_equal(out["observation.state"], np.full(6, 0.5, dtype=np.float32))
        assert any("robot_state_keys" in r.message for r in caplog.records)
        # The degraded binding is also surfaced as machine-detectable telemetry
        # that run_policy / eval_policy report.
        assert policy.generic_state_keys_used is True

    def test_strands_native_path_no_longer_drops_state(self, caplog):
        policy = _policy("act", strict_keys=False)
        with caplog.at_level("WARNING"):
            batch = policy._build_batch_from_strands_format(_named_obs(), {})
        # Pre-fix this key was silently absent -> open-loop rollout.
        assert "observation.state" in batch
        assert batch["observation.state"].flatten().tolist() == [0.5] * 6
        assert any("robot_state_keys" in r.message for r in caplog.records)
        assert policy.generic_state_keys_used is True

    def test_warns_at_most_once(self, caplog):
        policy = _policy("act", strict_keys=False)
        with caplog.at_level("WARNING"):
            policy._build_batch_from_strands_format(_named_obs(), {})
            policy._build_batch_from_strands_format(_named_obs(), {})
        mismatch_warnings = [r for r in caplog.records if "robot_state_keys" in r.message]
        assert len(mismatch_warnings) == 1


class TestMatchingKeysUnaffected:
    """When the configured keys match the observation, behavior is unchanged and silent."""

    def test_matched_keys_populate_state_without_warning(self, caplog):
        policy = _policy("act", strict_keys=False)
        policy.robot_state_keys = NAMED_JOINTS
        with caplog.at_level("WARNING"):
            batch = policy._build_batch_from_strands_format(_named_obs(), {})
        assert batch["observation.state"].flatten().tolist() == [0.5] * 6
        assert not any("robot_state_keys" in r.message for r in caplog.records)
        assert policy.generic_state_keys_used is False
