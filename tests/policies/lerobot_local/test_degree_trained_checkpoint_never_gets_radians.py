"""A degree-trained SO-arm checkpoint never meets a radian state unconverted.

Most SO-100/SO-101 fine-tunes on the Hub were recorded through LeRobot's
driver, which writes joints in degrees and the gripper in 0..100; every
simulator reports the same joints in radians. ``run_policy`` given only
``pretrained_name_or_path`` used to pack the radian state natively and apply
the model's degree actions as radians: on Isaac, a pi0.5 SO-101 fine-tune
(``tsangb34/pi05-so101-stack-white_bowls-100episodes``) held joint 4 at its
1.658 rad limit and joint 5 at -2.793 rad for all 150 frames, and the rollout
reported success. These tests pin the guard in ``get_actions``: the registered
simulation embodiment is applied when the caller named none, and anything it
cannot resolve is refused before a single action is emitted.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from strands_robots.policies.lerobot_local.embodiment import (
    DEGREE_LIKE_SPAN,
    EmbodimentMap,
    degree_like_columns,
    registered_sim_embodiment,
)
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

# The observation.state stats tsangb34/pi05-so101-stack-white_bowls-100episodes ships.
_DEGREE_STATS = {
    "min": torch.tensor([-30.4, -102.5, -94.3, 15.5, -19.5, 0.7]),
    "max": torch.tensor([32.7, 76.7, 96.6, 100.5, 21.1, 50.0]),
    "mean": torch.tensor([-1.2, -10.5, 3.1, 77.0, -0.9, 5.7]),
    "std": torch.tensor([13.9, 64.3, 65.9, 17.1, 6.9, 8.4]),
}
# A radian-recorded SO-101 dataset (the MuJoCo joint ranges), gripper included.
_RADIAN_STATS = {
    "min": torch.tensor([-1.9, -1.7, -1.6, -1.6, -2.7, -0.17]),
    "max": torch.tensor([1.9, 1.7, 1.6, 1.6, 2.7, 1.74]),
}
_SO101_KEYS = ["1", "2", "3", "4", "5", "6"]
# The so101 MJCF joint limits in radians (joint 4 upper 1.6581 is where the arm pinned).
_SO101_LIMITS = [(-1.92, 1.92), (-1.75, 1.75), (-1.69, 1.69), (-1.66, 1.6581), (-2.79, 2.79), (-0.175, 1.745)]


def _bridge(state_stats: dict | None, *, with_preprocessor: bool = True) -> ProcessorBridge:
    step = SimpleNamespace(_tensor_stats={"observation.state": state_stats} if state_stats else {})
    # Without a preprocessor the stats still arrive, on the postprocessor's unnormalizer.
    pipeline = SimpleNamespace(steps=[step])
    bridge = ProcessorBridge(
        preprocessor=pipeline if with_preprocessor else None, postprocessor=None if with_preprocessor else pipeline
    )
    bridge.apply_embodiment = MagicMock(name="apply_embodiment")  # type: ignore[method-assign]
    return bridge


def _policy(bridge: ProcessorBridge | None, **kwargs) -> LerobotLocalPolicy:
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="org/so101-pi05-degrees", **kwargs)
    policy._loaded = True
    policy._processor_bridge = bridge
    feature = SimpleNamespace(shape=(6,))
    policy._input_features = {
        "observation.state": feature,
        "observation.images.front": SimpleNamespace(shape=(3, 480, 640)),
        "observation.images.wrist": SimpleNamespace(shape=(3, 480, 640)),
    }
    policy._output_features = {"action": feature}
    return policy


def _sim_observation(values: list[float] | None = None) -> dict:
    values = values or [0.0, -0.4, 0.6, 1.2, 0.0, 0.3]
    return dict(zip(_SO101_KEYS, values, strict=True))


def test_the_stats_of_a_degree_trained_so101_read_as_degrees() -> None:
    ranges = list(zip(_DEGREE_STATS["min"].tolist(), _DEGREE_STATS["max"].tolist(), strict=True))
    assert degree_like_columns(ranges, 6) == [0, 1, 2, 3, 4, 5]


def test_a_radian_dataset_with_a_0_to_100_gripper_is_not_degree_trained() -> None:
    ranges = [(-1.9, 1.9), (-1.7, 1.7), (-1.6, 1.6), (-1.6, 1.6), (-2.7, 2.7), (0.0, 100.0)]
    assert degree_like_columns(ranges, 6) == []


def test_padding_columns_past_the_robot_are_not_read() -> None:
    ranges = [(-1.0, 1.0)] * 6 + [(-500.0, 500.0)] * 26
    assert degree_like_columns(ranges, 6) == []
    assert DEGREE_LIKE_SPAN == pytest.approx(2 * math.pi)


def test_the_so101_sim_joint_names_resolve_to_the_so101_embodiment_only() -> None:
    registered = registered_sim_embodiment(["6", "5", "4", "3", "2", "1"])
    assert registered is not None and registered.name == "so101"
    assert registered_sim_embodiment(["shoulder_pan.pos"]) is None


def test_the_bridge_reports_min_max_then_quantiles_then_mean_std() -> None:
    ranges = _bridge(_DEGREE_STATS).recorded_value_ranges("observation.state")
    assert ranges is not None and ranges[3] == pytest.approx((15.5, 100.5))
    quantiles = {"q01": torch.tensor([-2.0]), "q99": torch.tensor([3.0]), "mean": torch.tensor([0.0])}
    assert _bridge(quantiles).recorded_value_ranges("observation.state") == [pytest.approx((-2.0, 3.0))]
    mean_std = {"mean": torch.tensor([1.0]), "std": torch.tensor([2.0])}
    assert _bridge(mean_std).recorded_value_ranges("observation.state") == [pytest.approx((-5.0, 7.0))]
    assert _bridge(None).recorded_value_ranges("observation.state") is None


def test_run_policys_binding_adopts_the_so101_embodiment_for_a_degree_checkpoint() -> None:
    policy = _policy(_bridge(_DEGREE_STATS))
    policy.set_robot_state_keys(list(_SO101_KEYS))  # what run_policy does on every backend

    policy._guard_joint_units(_sim_observation())

    assert policy.embodiment_adopted == "so101"
    assert policy._embodiment is not None
    assert (policy._embodiment.state_units, policy._embodiment.action_units) == ("degrees", "degrees")
    # The cameras keep the routing the model's declared features ask for, not
    # the registered map's LIBERO-style image / wrist_image.
    assert policy._embodiment.obs_rename == {"front": "observation.images.front", "wrist": "observation.images.wrist"}


def test_a_degree_action_is_commanded_inside_the_joint_limits() -> None:
    """The hardware-safety pin: what reaches send_action is radians inside every limit."""
    policy = _policy(_bridge(_DEGREE_STATS))
    policy.set_robot_state_keys(list(_SO101_KEYS))
    policy._guard_joint_units(_sim_observation())

    # The action magnitudes the agent run recorded (degrees, gripper 0..100).
    degrees = torch.tensor([[5.1, 28.7, 22.3, 80.2, 4.4, 2.4]])
    [action] = policy._tensor_to_action_dicts(degrees)

    for key, (low, high) in zip(_SO101_KEYS, _SO101_LIMITS, strict=True):
        assert low <= action[key] <= high, (key, action[key])
    assert action["4"] == pytest.approx(math.radians(80.2), abs=1e-4)


def test_get_actions_refuses_before_inference_when_nothing_can_convert() -> None:
    """A declared native map on the so101 sim joints is refused, and the model is never called."""
    native = EmbodimentMap(name="so101_native", state_keys=list(_SO101_KEYS), action_keys=list(_SO101_KEYS))
    policy = _policy(_bridge(_DEGREE_STATS), embodiment=native)
    policy._embodiment = native
    policy._policy = MagicMock(name="model")

    with pytest.raises(ValueError, match=r"trained on degrees.*Nothing was commanded.*embodiment='so101'"):
        asyncio.run(policy.get_actions(_sim_observation(), "pick up the red cube"))
    policy._policy.select_action.assert_not_called()
    policy._policy.predict_action_chunk.assert_not_called()


def test_unrecognised_keys_near_zero_are_refused_and_named() -> None:
    policy = _policy(_bridge(_DEGREE_STATS))
    policy.set_robot_state_keys(["a", "b", "c", "d", "e", "f"])
    observation = dict(zip("abcdef", [0.1, -0.4, 0.6, 1.2, 0.0, 0.3], strict=True))

    with pytest.raises(ValueError, match=r"state_units='degrees'"):
        policy._guard_joint_units(observation)


def test_state_that_is_already_in_degrees_passes() -> None:
    policy = _policy(_bridge(_DEGREE_STATS))
    policy.set_robot_state_keys(["a", "b", "c", "d", "e", "f"])
    policy._guard_joint_units(dict(zip("abcdef", [-1.0, -95.0, 88.0, 70.0, 0.0, 12.0], strict=True)))
    assert policy.embodiment_adopted is None


def test_a_real_arm_through_the_lerobot_driver_is_left_alone() -> None:
    policy = _policy(_bridge(_DEGREE_STATS))
    motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    observation = {f"{motor}.pos": 0.5 for motor in motors}
    policy._guard_joint_units(observation)
    assert policy.embodiment_adopted is None and policy._embodiment is None


@pytest.mark.parametrize("stats", [_RADIAN_STATS, None])
def test_radian_or_absent_stats_change_nothing(stats: dict | None) -> None:
    policy = _policy(_bridge(stats))
    policy.set_robot_state_keys(list(_SO101_KEYS))
    policy._guard_joint_units(_sim_observation())
    assert policy.embodiment_adopted is None and policy._embodiment is None


def test_a_converting_embodiment_the_caller_declared_is_trusted() -> None:
    policy = _policy(_bridge(_DEGREE_STATS), embodiment="so101")
    policy._embodiment = registered_sim_embodiment(_SO101_KEYS)
    policy._guard_joint_units(_sim_observation())
    assert policy.embodiment_adopted is None


def test_an_adoption_that_cannot_configure_is_refused_with_the_routing_remedy() -> None:
    policy = _policy(_bridge(_DEGREE_STATS, with_preprocessor=False))
    policy.set_robot_state_keys(list(_SO101_KEYS))
    with pytest.raises(ValueError, match=r"applying it failed.*camera_key_map="):
        policy._guard_joint_units(_sim_observation())
    assert policy._embodiment is None and policy.embodiment_adopted is None


def test_rebinding_the_state_keys_re_arms_the_guard() -> None:
    policy = _policy(_bridge(_DEGREE_STATS))
    policy.set_robot_state_keys(["a", "b", "c", "d", "e", "f"])
    policy._guard_joint_units(dict(zip("abcdef", [-1.0, -95.0, 88.0, 70.0, 0.0, 12.0], strict=True)))
    policy.set_robot_state_keys(["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="trained on degrees"):
        policy._guard_joint_units(dict(zip("abcdef", [0.0] * 6, strict=True)))
