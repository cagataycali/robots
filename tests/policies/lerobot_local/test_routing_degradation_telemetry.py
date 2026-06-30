# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Heuristic remap fallbacks must be observable, not silent.

The non-declarative ``_to_lerobot_observation`` / ``_resolve_camera_targets``
remap path keeps a run alive even when it cannot bind the observation to the
model's inputs by name: a camera whose name matches no declared image feature
is routed to a free slot positionally, and ``observation.state`` is composed
from the observation's own scalar keys when none of ``robot_state_keys`` match
(the generic ``joint_0..N`` fallback). Either makes the robot move on
meaningless inputs while the rollout still reports ``status="success"`` and
``success_rate ~ 0``.

These tests pin that each fallback (a) flips a machine-readable flag on the
policy (``positional_fallback_used`` / ``generic_state_keys_used``) that
``run_policy`` / ``eval_policy`` surface in their JSON, and (b) emits a WARNING,
while a correctly-named observation leaves both flags False and stays quiet.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(input_features, robot_state_keys):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="molmoact2")
    policy._input_features = input_features
    policy.robot_state_keys = list(robot_state_keys)
    return policy


def test_flags_default_false_on_fresh_policy():
    policy = _policy({"observation.state": _state()}, ["shoulder_pan"])
    assert policy.positional_fallback_used is False
    assert policy.generic_state_keys_used is False


class TestPositionalCameraFallback:
    FEATURES = {
        "observation.images.top": _visual(),
        "observation.images.wrist": _visual(),
        "observation.state": _state(),
    }
    STATE_KEYS = [f"j{i}" for i in range(6)]

    def _obs(self, cam_names):
        obs = {name: np.zeros((224, 224, 3), np.uint8) for name in cam_names}
        obs.update({f"j{i}": 0.0 for i in range(6)})
        return obs

    def test_unmatched_camera_names_flip_flag_and_warn(self, caplog):
        # Cameras "cam0"/"cam1" match no declared image feature by name -> the
        # frames are routed positionally into the two free slots.
        policy = _policy(self.FEATURES, self.STATE_KEYS)
        with caplog.at_level(logging.WARNING):
            out = policy._to_lerobot_observation(self._obs(["cam0", "cam1"]))
        assert policy.positional_fallback_used is True
        # Frames still land in the declared image slots (run is not aborted).
        routed = sorted(k for k, v in out.items() if isinstance(v, np.ndarray) and v.ndim >= 2)
        assert routed == ["observation.images.top", "observation.images.wrist"]
        assert any("routing positionally" in r.message for r in caplog.records)

    def test_matching_camera_names_leave_flag_false_and_quiet(self, caplog):
        policy = _policy(self.FEATURES, self.STATE_KEYS)
        with caplog.at_level(logging.WARNING):
            out = policy._to_lerobot_observation(self._obs(["top", "wrist"]))
        assert policy.positional_fallback_used is False
        routed = sorted(k for k, v in out.items() if isinstance(v, np.ndarray) and v.ndim >= 2)
        assert routed == ["observation.images.top", "observation.images.wrist"]
        assert not any("routing positionally" in r.message for r in caplog.records)


class TestGenericStateKeysFallback:
    FEATURES = {"observation.state": _state(dim=6)}

    def _obs(self, scalar_keys):
        return {k: float(i) for i, k in enumerate(scalar_keys)}

    def test_mismatched_state_keys_flip_flag_and_warn(self, caplog):
        # robot_state_keys are real joint names, but the observation is keyed
        # "1".."6" (the SO arm's bus ids) -> none match -> state composed from
        # the observation's own scalar keys, which the caller cannot see.
        policy = _policy(self.FEATURES, ["shoulder_pan", "shoulder_lift", "elbow", "wrist", "wrist_roll", "gripper"])
        with caplog.at_level(logging.WARNING):
            out = policy._to_lerobot_observation(self._obs([str(i) for i in range(1, 7)]))
        assert policy.generic_state_keys_used is True
        # State is still composed (not silently dropped) so the run continues.
        assert out["observation.state"].shape == (6,)
        assert any("robot_state_keys" in r.message for r in caplog.records)

    def test_matching_state_keys_leave_flag_false_and_quiet(self, caplog):
        keys = ["shoulder_pan", "shoulder_lift", "elbow", "wrist", "wrist_roll", "gripper"]
        policy = _policy(self.FEATURES, keys)
        with caplog.at_level(logging.WARNING):
            policy._to_lerobot_observation(self._obs(keys))
        assert policy.generic_state_keys_used is False
        assert not any("robot_state_keys" in r.message for r in caplog.records)
