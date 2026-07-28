# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The zero-fill for a short action dim is a COMMAND, and must say so.

When a checkpoint's action dim is shorter than the robot's action keys,
``_tensor_to_action_dicts`` fills the trailing keys with ``0.0`` and those keys go
straight into the action dict. The diagnostic said those actuators "are
zero-filled and will not move" - the opposite of what happens.

On a real follower ``<motor>.pos = 0.0`` is an ABSOLUTE target. Verified against
the installed lerobot with a 1000..3000 calibration::

    DEGREES     0.0 -> 2000 ticks  (mid-point of travel)
    RANGE_0_100 0.0 -> 1000 ticks  (range_min = one hard end -> closed gripper)

and ``SOFollower.send_action`` forwards every ``.pos`` entry to
``bus.sync_write("Goal_Position", goal_pos)`` with no "no command" sentinel - it
filters on the key SUFFIX only, never on the value. So the unmatched actuators are
driven, hard, every control tick.

The fill itself stays the default (an existing test pins it, and it keeps the action
dict shape stable for index-based consumers), but the diagnostic now states the
consequence, and ``pad_short_actions=False`` omits the keys instead - which IS a
genuine no-op, because lerobot's comprehension never sees an absent key.
"""

from __future__ import annotations

import logging

import pytest
import torch

from strands_robots.policies.lerobot_local.embodiment import diagnose_action_dim
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

_KEYS = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class TestTheDiagnosticTellsTheTruth:
    def test_it_says_the_actuators_are_commanded(self):
        """Regression: it claimed they "will not move"."""
        warning = diagnose_action_dim(5, 6, name="so_real")

        assert warning is not None
        assert "COMMANDED TO 0.0" in warning
        assert "will not move" not in warning

    def test_it_explains_what_zero_means_on_hardware(self):
        """The number alone is not actionable; the units are the hazard."""
        warning = diagnose_action_dim(5, 6, name="so_real")

        assert "mid-point" in warning
        assert "end stop" in warning or "end-stop" in warning

    def test_it_names_the_way_out(self):
        assert "pad_short_actions=False" in diagnose_action_dim(5, 6, name="so_real")

    def test_it_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        assert diagnose_action_dim(5, 6, name="so_real").isascii()

    def test_matching_dims_produce_no_warning(self):
        assert diagnose_action_dim(6, 6, name="so_real") is None

    def test_the_over_dim_branch_is_unchanged(self):
        warning = diagnose_action_dim(8, 6, name="so_real")

        assert warning is not None
        assert "action dim 8" in warning


class TestPadShortActionsDefaultIsUnchanged:
    def test_padding_is_the_default(self):
        """An existing test pins the fill; it must remain the default."""
        policy = LerobotLocalPolicy()
        policy.set_robot_state_keys(_KEYS)

        emitted = policy._tensor_to_action_dicts(torch.zeros(5))[0]

        assert set(emitted) == set(_KEYS)
        assert emitted["gripper.pos"] == 0.0

    def test_the_flag_defaults_true(self):
        assert LerobotLocalPolicy().pad_short_actions is True


class TestPadShortActionsFalseOmits:
    def test_unmatched_keys_are_omitted_entirely(self):
        """An absent key is a genuine no-op for lerobot's send_action."""
        policy = LerobotLocalPolicy(pad_short_actions=False)
        policy.set_robot_state_keys(_KEYS)

        emitted = policy._tensor_to_action_dicts(torch.zeros(5))[0]

        assert len(emitted) == 5
        assert "gripper.pos" not in emitted
        assert list(emitted) == _KEYS[:5], "the surviving keys must keep their declared order"

    def test_the_gripper_is_the_key_that_gets_dropped(self):
        """The SO gripper is last, and its 0.0 is a closed end-stop - the hazard."""
        policy = LerobotLocalPolicy(pad_short_actions=False)
        policy.set_robot_state_keys(_KEYS)

        assert "gripper.pos" not in policy._tensor_to_action_dicts(torch.zeros(5))[0]

    @pytest.mark.parametrize("dim", [1, 3, 5])
    def test_exactly_the_model_dim_is_emitted(self, dim):
        policy = LerobotLocalPolicy(pad_short_actions=False)
        policy.set_robot_state_keys(_KEYS)

        assert len(policy._tensor_to_action_dicts(torch.zeros(dim))[0]) == dim

    def test_a_matching_dim_is_unaffected_by_the_flag(self):
        policy = LerobotLocalPolicy(pad_short_actions=False)
        policy.set_robot_state_keys(_KEYS)

        emitted = policy._tensor_to_action_dicts(torch.zeros(6))[0]

        assert set(emitted) == set(_KEYS)

    def test_values_are_still_plain_floats(self):
        """The Policy contract: every value is a python float, never np/torch."""
        policy = LerobotLocalPolicy(pad_short_actions=False)
        policy.set_robot_state_keys(_KEYS)

        emitted = policy._tensor_to_action_dicts(torch.zeros(5))[0]

        assert all(type(v) is float for v in emitted.values())

    def test_the_warning_still_fires_when_omitting(self, caplog):
        """Omitting is safer, not silent: the dim mismatch is still a mistake."""
        policy = LerobotLocalPolicy(pad_short_actions=False)
        policy.set_robot_state_keys(_KEYS)

        with caplog.at_level(logging.WARNING):
            policy._tensor_to_action_dicts(torch.zeros(5))

        assert any("action dim 5" in r.getMessage() for r in caplog.records), caplog.records
