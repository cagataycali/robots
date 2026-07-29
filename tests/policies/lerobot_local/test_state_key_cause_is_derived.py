# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""An all-missing state-key diagnostic must name a cause it actually checked.

When none of the configured ``robot_state_keys`` appear in the observation, the
message used to end in one fixed explanation for every caller: "This usually
means generic auto-generated keys (joint_0..joint_N) were paired with a
robot/sim that reports named joints." It was asserted without looking at the
configured keys, so it was true only for the one case it describes.

The callers it misled are the ones who most need it right. Configuring the sim
``so101`` embodiment against a real SO arm is the canonical mistake - the sim
configuration names the MuJoCo asset's numeric joints ``'1'..'6'`` while the arm
reports ``'<motor>.pos'`` - and those operators were told the keys they had just
chosen deliberately were auto-generated ``joint_N`` placeholders.

A bare ``joint_`` prefix is not a sound test for a placeholder either: the
registry ships ``kinova_gen3``, whose real joint names are ``joint_1..joint_7``.
The loader emits the consecutive zero-based run ``joint_0..joint_{n-1}``, so that
run - not the prefix - is what identifies a key the library invented.

These tests pin that the cause is read from the configured keys, that the
generic case keeps the explanation written for it, and that nothing else about
the guard (remedy, fallback ordering, telemetry, warn-once, strict raise)
changed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.embodiment import EMBODIMENT_MAP
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy, _state_key_cause

# lerobot SOFollower motor feature keys - what a real SO arm reports.
HARDWARE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# The shape the loader fills in when a checkpoint declares a dim and the caller
# named no joints: consecutive and zero-based.
GENERIC_SENTENCE = (
    "This usually means generic auto-generated keys (joint_0..joint_N) were "
    "paired with a robot/sim that reports named joints."
)
NAMED_SENTENCE = "The configured keys describe a different robot/sim than the one reporting this observation."


def _loader_placeholders(n: int) -> list[str]:
    """The keys the loader auto-generates for an ``n``-dim checkpoint."""
    return [f"joint_{i}" for i in range(n)]


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(configured, *, strict_keys=True):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="molmoact2", strict_keys=strict_keys)
    policy._input_features = {"observation.images.base": _visual(), "observation.state": _state(6)}
    policy._device = torch.device("cpu")
    policy.robot_state_keys = list(configured)
    return policy


def _obs(keys=tuple(HARDWARE_KEYS)):
    obs = {"base": np.zeros((224, 224, 3), np.uint8)}
    obs.update(dict.fromkeys(keys, 0.5))
    return obs


def _diagnostic(configured, obs_keys=tuple(HARDWARE_KEYS)) -> str:
    """The all-missing message raised for ``configured`` against ``obs_keys``."""
    policy = _policy(configured)
    with pytest.raises(ValueError) as excinfo:
        policy._resolve_state_order(_obs(obs_keys), list(obs_keys))
    return str(excinfo.value)


class TestCauseIsReadFromTheConfiguredKeys:
    """The sentence must match whether the keys really are placeholders."""

    @pytest.mark.parametrize(
        ("label", "configured", "expect_generic"),
        [
            ("loader placeholders", _loader_placeholders(6), True),
            ("sim so101 numeric joints", EMBODIMENT_MAP["so101"].state_keys, False),
            ("sim so100 named joints", EMBODIMENT_MAP["so100"].state_keys, False),
            ("kinova_gen3 one-based joint names", EMBODIMENT_MAP["kinova_gen3"].state_keys, False),
            ("aloha bimanual named joints", EMBODIMENT_MAP["aloha"].state_keys, False),
        ],
    )
    def test_only_the_placeholder_shape_is_called_auto_generated(self, label, configured, expect_generic):
        msg = _diagnostic(configured)
        assert (GENERIC_SENTENCE in msg) is expect_generic, f"{label}: wrong cause in {msg!r}"
        assert (NAMED_SENTENCE in msg) is not expect_generic, f"{label}: wrong cause in {msg!r}"

    def test_no_shipped_configuration_is_described_as_auto_generated(self):
        """Every registry entry names a real robot's joints, never a placeholder.

        ``kinova_gen3`` / ``openarm_real`` are the entries that make this worth
        asserting: their joint names carry the ``joint_`` prefix, so a prefix
        test would describe a shipped robot's own naming as invented.
        """
        for name, embodiment in EMBODIMENT_MAP.items():
            keys = embodiment.state_keys
            if not keys:
                continue
            assert keys != _loader_placeholders(len(keys)), f"{name} collides with the loader's placeholder run"
            assert _state_key_cause(list(keys)) == NAMED_SENTENCE, name

    def test_the_placeholder_case_keeps_the_explanation_written_for_it(self):
        """The generic wording is unchanged for the callers it was correct for."""
        assert GENERIC_SENTENCE in _diagnostic(_loader_placeholders(6))


class TestPlaceholderRecognition:
    """Only the loader's own output counts as a placeholder run."""

    @pytest.mark.parametrize("dim", [1, 2, 6, 7, 14, 29])
    def test_the_loader_run_is_recognised_at_every_width(self, dim):
        assert _state_key_cause(_loader_placeholders(dim)) == GENERIC_SENTENCE

    @pytest.mark.parametrize(
        "configured",
        [
            pytest.param(["joint_1", "joint_2", "joint_3"], id="one-based-run"),
            pytest.param(["joint_1", "joint_0"], id="reordered-run"),
            pytest.param(["joint_0", "joint_2"], id="non-consecutive-run"),
            pytest.param(["joint_0", "Jaw"], id="mixed-with-a-named-key"),
            pytest.param(["joint_0.pos", "joint_1.pos"], id="suffixed-run"),
            pytest.param([], id="empty"),
        ],
    )
    def test_anything_else_is_not_claimed_to_be_auto_generated(self, configured):
        assert _state_key_cause(configured) == NAMED_SENTENCE

    @pytest.mark.parametrize("configured", [_loader_placeholders(3), ["1", "2"], []])
    def test_every_cause_is_one_plain_ascii_sentence(self, configured):
        cause = _state_key_cause(configured)
        cause.encode("ascii")
        assert cause.endswith(".")
        assert "\n" not in cause


class TestTheRestOfTheDiagnosticIsUnchanged:
    """Only the cause sentence moved; the guard's contract did not."""

    @pytest.mark.parametrize("configured", [_loader_placeholders(6), EMBODIMENT_MAP["so101"].state_keys])
    def test_the_registry_checked_remedy_still_follows_the_cause(self, configured):
        msg = _diagnostic(configured)
        cause = _state_key_cause(list(configured))
        assert cause in msg
        # so_real declares exactly the hardware observation's keys, so the
        # remedy must still name it - and must come after the cause.
        assert msg.index(cause) < msg.index("so_real")
        assert "set_robot_state_keys" in msg

    def test_the_observed_keys_are_still_quoted_back(self):
        msg = _diagnostic(EMBODIMENT_MAP["so101"].state_keys)
        assert "gripper.pos" in msg
        assert "'1', '2'" in msg

    def test_the_lenient_path_still_warns_once_and_falls_back(self, caplog):
        policy = _policy(EMBODIMENT_MAP["so101"].state_keys, strict_keys=False)
        obs = _obs()
        with caplog.at_level("WARNING"):
            first = policy._resolve_state_order(obs, list(HARDWARE_KEYS))
            second = policy._resolve_state_order(obs, list(HARDWARE_KEYS))
        assert first == HARDWARE_KEYS
        assert second == HARDWARE_KEYS
        assert policy.generic_state_keys_used is True
        warnings = [r for r in caplog.records if NAMED_SENTENCE in r.getMessage()]
        assert len(warnings) == 1

    def test_a_configured_key_that_is_present_still_wins(self):
        """The guard only fires when NOTHING matches, placeholders included."""
        configured = [*_loader_placeholders(5), "gripper.pos"]
        policy = _policy(configured)
        assert policy._resolve_state_order(_obs(), list(HARDWARE_KEYS)) == configured
