# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A state-key mismatch must recommend a binding the observation can actually use.

The mismatch diagnostics ended in one fixed sentence for every caller: "Pass
embodiment='<name>' (e.g. embodiment='so101') or call set_robot_state_keys([...])
with the robot's actual joint names". Against a real SO arm that example is not
merely unhelpful - it is a loop. lerobot's ``SOFollower`` keys joints as
``'<motor>.pos'``, while the ``so101`` configuration declares the MuJoCo asset's
numeric joints ``'1'..'6'``; none of those are present, so following the printed
advice lands on the same all-missing guard that printed it. Its
``state_units='degrees'`` would also convert units the hardware reports natively.

The registry already ships the configuration that does bind that observation
(``so_real``), and it was never named.

So the remedy is now chosen from the observation: an ``embodiment=`` is offered
only when the registry confirms every one of its ``state_keys`` is present, all
qualifying configurations are listed when the observation cannot tell them apart
(the real SO, Koch and OMX arms report identical ``.pos`` keys), and when nothing
matches no embodiment is offered at all.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.embodiment import (
    EMBODIMENT_MAP,
    matching_embodiments,
    state_key_remedy,
)
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# lerobot SOFollower motor feature keys - what a real SO arm reports.
HARDWARE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
# robotstudio_so101 MuJoCo asset joints - what the so101 configuration declares.
SIM_SO101_KEYS = ["1", "2", "3", "4", "5", "6"]


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(*, strict_keys=False):
    """A policy whose robot_state_keys are the generic auto-filled names.

    ``joint_0..joint_5`` match neither observation shape, which is the mismatch
    both diagnostics exist to report.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="molmoact2", strict_keys=strict_keys)
    policy._input_features = {"observation.images.base": _visual(), "observation.state": _state(6)}
    policy._device = torch.device("cpu")
    policy.robot_state_keys = [f"joint_{i}" for i in range(6)]
    return policy


def _obs(keys):
    obs = {"base": np.zeros((224, 224, 3), np.uint8)}
    obs.update(dict.fromkeys(keys, 0.5))
    return obs


class TestRecommendedEmbodimentBindsTheObservation:
    """The whole point: following the advice must not land back on the guard."""

    @pytest.mark.parametrize("keys", [HARDWARE_KEYS, SIM_SO101_KEYS], ids=["hardware", "sim"])
    def test_every_named_embodiment_resolves_the_observation(self, keys):
        """Configure each recommended embodiment and re-run the resolution."""
        candidates = matching_embodiments(keys)
        assert candidates, f"no embodiment offered for {keys}"

        for name in candidates:
            policy = _policy(strict_keys=True)
            policy.robot_state_keys = list(EMBODIMENT_MAP[name].state_keys)
            # Would raise the very mismatch the remedy is escaping.
            assert policy._resolve_state_order(_obs(keys), keys) == policy.robot_state_keys

    def test_the_old_fixed_example_would_not_have(self):
        """Regression: 'e.g. embodiment=so101' was a loop on a real arm."""
        policy = _policy(strict_keys=True)
        policy.robot_state_keys = list(EMBODIMENT_MAP["so101"].state_keys)

        with pytest.raises(ValueError, match="None of the configured robot_state_keys"):
            policy._resolve_state_order(_obs(HARDWARE_KEYS), HARDWARE_KEYS)

        assert "so101" not in matching_embodiments(HARDWARE_KEYS)


class TestRemedyContent:
    def test_hardware_observation_is_pointed_at_the_hardware_configuration(self):
        remedy = state_key_remedy(HARDWARE_KEYS)

        assert "so_real" in remedy
        # The sim configuration must not be recommended for a .pos observation.
        assert "embodiment='so101'" not in remedy

    def test_indistinguishable_configurations_are_all_listed(self):
        """so_real / koch_real / omx_real declare identical .pos keys."""
        remedy = state_key_remedy(HARDWARE_KEYS)

        for name in ("so_real", "koch_real", "omx_real"):
            assert name in remedy, remedy

    def test_a_sim_observation_gets_its_single_configuration(self):
        remedy = state_key_remedy(SIM_SO101_KEYS)

        assert "embodiment='so101'" in remedy
        assert "so_real" not in remedy

    def test_no_embodiment_is_offered_when_none_matches(self):
        remedy = state_key_remedy(["alpha", "beta", "gamma"])

        assert "embodiment=" not in remedy
        assert "set_robot_state_keys(['alpha', 'beta', 'gamma'])" in remedy

    def test_short_key_lists_are_quoted_verbatim_so_the_fix_is_pasteable(self):
        assert f"set_robot_state_keys({HARDWARE_KEYS!r})" in state_key_remedy(HARDWARE_KEYS)

    def test_a_long_key_list_is_not_repeated_inline(self):
        """A 29-joint humanoid must still be named, without printing 29 keys twice."""
        keys = list(EMBODIMENT_MAP["unitree_g1"].state_keys)
        remedy = state_key_remedy(keys)

        assert "unitree_g1" in remedy
        assert "with the observed keys above" in remedy
        assert keys[0] not in remedy


class TestMatchingEmbodiments:
    def test_a_partial_match_does_not_qualify(self):
        """A configuration missing even one key would reproduce the mismatch."""
        assert "so_real" not in matching_embodiments(HARDWARE_KEYS[:-1])

    def test_extra_observation_keys_do_not_disqualify(self):
        """Observations carry more than the model's state (velocities, base pose)."""
        assert "so_real" in matching_embodiments([*HARDWARE_KEYS, "battery", "x.vel"])

    def test_aliases_are_not_offered_as_separate_candidates(self):
        """EMBODIMENT_MAP holds aliases too; one spelling per configuration."""
        candidates = matching_embodiments(SIM_SO101_KEYS)

        assert candidates == ["so101"]

    def test_result_is_sorted_so_the_message_is_deterministic(self):
        candidates = matching_embodiments(HARDWARE_KEYS)

        assert candidates == sorted(candidates)


class TestRobustness:
    def test_non_string_keys_are_ignored_rather_than_rejected(self):
        assert matching_embodiments([1, None, *SIM_SO101_KEYS]) == ["so101"]
        assert "embodiment='so101'" in state_key_remedy([1, None, *SIM_SO101_KEYS])

    def test_an_observation_with_no_state_keys_says_nothing_can_bind_it(self):
        remedy = state_key_remedy([])

        assert "no scalar state keys" in remedy
        # Offering a remedy here would be advice that cannot work.
        assert "embodiment=" not in remedy

    @pytest.mark.parametrize(
        "keys", [HARDWARE_KEYS, SIM_SO101_KEYS, ["alpha"], []], ids=["hardware", "sim", "unknown", "empty"]
    )
    def test_remedy_is_plain_ascii(self, keys):
        """AGENTS.md: user-facing strings are ASCII only."""
        remedy = state_key_remedy(keys)

        assert remedy.isascii(), remedy


class TestBothDiagnosticsCarryIt:
    """The all-missing and partial-missing guards must not drift apart."""

    def test_all_missing_warning_names_a_matching_embodiment(self, caplog):
        import logging

        policy = _policy()

        with caplog.at_level(logging.WARNING):
            order = policy._resolve_state_order(_obs(HARDWARE_KEYS), HARDWARE_KEYS)

        assert order == HARDWARE_KEYS  # fell back to the observation's own keys
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("so_real" in m for m in warnings), warnings
        assert not any("embodiment='so101'" in m for m in warnings), warnings

    def test_all_missing_error_names_a_matching_embodiment(self):
        policy = _policy(strict_keys=True)

        with pytest.raises(ValueError, match="so_real"):
            policy._resolve_state_order(_obs(HARDWARE_KEYS), HARDWARE_KEYS)

    def test_partial_missing_error_names_a_matching_embodiment(self):
        """One configured key present, the rest absent - the tendon-gripper case."""
        policy = _policy(strict_keys=True)
        policy.robot_state_keys = [*HARDWARE_KEYS[:5], "left/gripper"]
        observation = _obs(HARDWARE_KEYS)

        with pytest.raises(ValueError) as excinfo:
            policy._collect_state_values(observation, policy.robot_state_keys)

        assert "so_real" in str(excinfo.value)
        assert "embodiment='<name>'" not in str(excinfo.value)

    def test_partial_missing_remedy_ignores_camera_keys(self):
        """The camera in the observation must not be offered as a state key."""
        policy = _policy(strict_keys=True)
        policy.robot_state_keys = [*HARDWARE_KEYS[:5], "left/gripper"]

        with pytest.raises(ValueError) as excinfo:
            policy._collect_state_values(_obs(HARDWARE_KEYS), policy.robot_state_keys)

        assert "'base'" not in str(excinfo.value)
