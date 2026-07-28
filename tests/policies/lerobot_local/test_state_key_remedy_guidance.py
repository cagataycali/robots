# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The state-key mismatch message must recommend the fix that actually works.

The mismatch diagnostic used to end with one fixed sentence for every caller:
"Pass embodiment='<name>' (e.g. embodiment='so101') or call
set_robot_state_keys([...])". On a REAL arm that first suggestion is actively
harmful. lerobot's ``SOFollower`` keys joints as ``"<motor>.pos"``, and the
SO-arm embodiments hardcode ``obs_rename`` onto ``observation.images.image`` --
so against a checkpoint declaring any other image feature (an ACT SO-101
checkpoint trained on ``observation.images.laptop``, say) the embodiment cannot
be configured at all. Following the advice printed by the warning therefore led
straight into the pipeline-configuration failure.

The remedy is now chosen from the observation's own shape: ``.pos`` keys mean
hardware and get ``set_robot_state_keys``; bare joint names mean sim and get an
embodiment, which additionally carries the sim-radian to model-degree conversion.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

_HARDWARE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
_SIM_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(*, strict_keys=False):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="molmoact2", strict_keys=strict_keys)
    policy._input_features = {"observation.images.base": _visual(), "observation.state": _state(6)}
    policy._device = torch.device("cpu")
    # Generic auto-generated keys matching NOTHING in either observation shape.
    policy.robot_state_keys = [f"joint_{i}" for i in range(6)]
    return policy


def _obs(keys):
    obs = {"base": np.zeros((224, 224, 3), np.uint8)}
    obs.update(dict.fromkeys(keys, 0.5))
    return obs


class TestHardwareObservationGuidance:
    def test_recommends_set_robot_state_keys_with_the_actual_keys(self):
        remedy = LerobotLocalPolicy._state_key_remedy(_HARDWARE_KEYS)

        assert "set_robot_state_keys" in remedy
        # The actual key names must be in the message, so the fix is copy-pasteable.
        assert "shoulder_pan.pos" in remedy

    def test_does_not_recommend_an_embodiment_on_hardware(self):
        """Regression: the old text told hardware callers to pass embodiment='so101'."""
        remedy = LerobotLocalPolicy._state_key_remedy(_HARDWARE_KEYS)

        assert "embodiment='so101'" not in remedy
        # It must actively steer away from it, not merely omit it.
        assert "Do NOT reach for a sim embodiment" in remedy

    def test_warning_emitted_on_a_hardware_observation_carries_the_hardware_remedy(self, caplog):
        """End to end through the real resolution path, not just the helper."""
        policy = _policy()

        import logging

        with caplog.at_level(logging.WARNING):
            order = policy._resolve_state_order(_obs(_HARDWARE_KEYS), _HARDWARE_KEYS)

        assert order == _HARDWARE_KEYS  # fell back to the observation's own keys
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("set_robot_state_keys" in m for m in msgs), msgs
        assert not any("embodiment='so101'" in m for m in msgs), msgs


class TestSimObservationGuidance:
    def test_recommends_an_embodiment_for_bare_joint_names(self):
        remedy = LerobotLocalPolicy._state_key_remedy(_SIM_KEYS)

        assert "embodiment=" in remedy
        # The reason an embodiment (not just state keys) matters in sim.
        assert "radian" in remedy

    def test_still_offers_set_robot_state_keys_as_the_alternative(self):
        assert "set_robot_state_keys" in LerobotLocalPolicy._state_key_remedy(_SIM_KEYS)

    def test_numeric_sim_joint_names_are_treated_as_sim(self):
        remedy = LerobotLocalPolicy._state_key_remedy([str(i) for i in range(1, 7)])

        assert "embodiment=" in remedy
        assert "Do NOT reach for a sim embodiment" not in remedy


class TestGuidanceIsRobust:
    def test_empty_observation_keys_do_not_crash(self):
        assert LerobotLocalPolicy._state_key_remedy([])

    def test_non_string_keys_do_not_crash(self):
        """Observation keys are normally strings, but must not be assumed to be."""
        assert LerobotLocalPolicy._state_key_remedy([1, None, "shoulder_pan"])

    def test_remedy_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        for keys in (_HARDWARE_KEYS, _SIM_KEYS, []):
            remedy = LerobotLocalPolicy._state_key_remedy(keys)
            assert remedy.isascii(), remedy

    def test_strict_keys_error_also_carries_the_hardware_remedy(self):
        """strict_keys raises instead of warning; the guidance must travel with it."""
        policy = _policy(strict_keys=True)

        with pytest.raises(ValueError) as excinfo:
            policy._resolve_state_order(_obs(_HARDWARE_KEYS), _HARDWARE_KEYS)

        assert "set_robot_state_keys" in str(excinfo.value)
