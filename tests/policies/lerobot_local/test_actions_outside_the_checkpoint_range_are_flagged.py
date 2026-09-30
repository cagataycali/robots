"""An action far outside what the checkpoint's own action stats record is flagged, or clipped.

π0-FAST LIBERO (whose action stats span [-0.94, 1.0]) emitted a pitch of 131
and a gripper of 2062 on an out-of-distribution scene, and strands forwarded
them unchecked (PI-017): a joint-space policy would have driven the robot there.
``out_of_range_actions`` = ``"warn"`` (default: log once per episode),
``"clip"`` (clip to the recorded range) or ``"off"``.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

KEYS = ["x", "pitch", "gripper"]


def _policy(mode: str = "warn") -> LerobotLocalPolicy:
    policy = LerobotLocalPolicy(out_of_range_actions=mode)
    policy.set_robot_state_keys(KEYS)
    # the checkpoint's action stats: every column recorded in [-1, 1]
    policy._processor_bridge = types.SimpleNamespace(  # type: ignore[assignment]
        recorded_value_ranges=lambda key: [(-1.0, 1.0)] * 3, reset=lambda: None
    )
    return policy


def test_an_action_far_outside_the_recorded_range_is_warned_once(caplog) -> None:
    policy = _policy()
    caplog.set_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.policy")

    first = policy._tensor_to_action_dicts(torch.tensor([0.5, 131.0, 2062.0]))[0]
    policy._tensor_to_action_dicts(torch.tensor([0.5, 131.0, 2062.0]))

    assert first == {"x": 0.5, "pitch": 131.0, "gripper": 2062.0}  # forwarded unchanged
    warned = [r.getMessage() for r in caplog.records if "far outside the range" in r.getMessage()]
    assert len(warned) == 1
    assert "column 1 = 131" in warned[0] and "column 2 = 2062" in warned[0]
    assert policy.out_of_range_action_count == 2


def test_clip_clips_to_the_recorded_range() -> None:
    action = _policy("clip")._tensor_to_action_dicts(torch.tensor([0.5, 131.0, -40.0]))[0]
    assert action == {"x": 0.5, "pitch": 1.0, "gripper": -1.0}


def test_an_action_near_the_range_is_not_flagged(caplog) -> None:
    policy = _policy()
    caplog.set_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.policy")
    policy._tensor_to_action_dicts(torch.tensor([1.5, -2.5, 0.0]))  # within one recorded range beyond
    assert policy.out_of_range_action_count == 0
    assert not [r for r in caplog.records if "far outside the range" in r.getMessage()]


def test_off_forwards_and_says_nothing() -> None:
    policy = _policy("off")
    assert policy._tensor_to_action_dicts(torch.tensor([0.0, 999.0, 0.0]))[0]["pitch"] == 999.0
    assert policy.out_of_range_action_count == 0


def test_a_padded_model_width_compares_only_the_recorded_columns() -> None:
    policy = _policy("clip")
    wide = torch.tensor([0.5, 0.2, 0.1] + [500.0] * 29)  # pi0 / pi05 pad to 32
    action = policy._tensor_to_action_dicts(wide)[0]
    assert action == {"x": 0.5, "pitch": pytest.approx(0.2), "gripper": pytest.approx(0.1)}
    assert policy.out_of_range_action_count == 0


def test_a_checkpoint_without_action_stats_is_left_alone() -> None:
    policy = LerobotLocalPolicy(out_of_range_actions="clip")
    policy.set_robot_state_keys(KEYS)
    assert policy._tensor_to_action_dicts(torch.tensor([0.0, 999.0, 0.0]))[0]["pitch"] == 999.0


def test_reset_starts_a_new_episode_of_warnings() -> None:
    policy = _policy()
    policy._tensor_to_action_dicts(torch.tensor([0.0, 999.0, 0.0]))
    policy.reset()
    assert policy.out_of_range_action_count == 0 and policy._out_of_range_warned is False


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="out_of_range_actions"):
        LerobotLocalPolicy(out_of_range_actions="ignore")
