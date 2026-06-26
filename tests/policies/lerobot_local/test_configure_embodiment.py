"""Behavior tests for ``LerobotLocalPolicy._configure_embodiment``.

``_configure_embodiment`` is the load-time hook that wires a declarative
:class:`EmbodimentMap` into LeRobot's processor pipeline (rename + pack-state
step) and latches the model's action-tensor index -> actuator-name mapping
onto ``robot_state_keys``. The full ``_load_model`` path mocks this method out,
so these tests drive it directly with a mock :class:`ProcessorBridge` and
synthetic model features to pin its observable contract across every branch:

* an explicit ``EmbodimentMap`` spec is validated, injected into the bridge,
  and its ``action_keys`` become ``robot_state_keys``;
* with no spec but non-joint ``robot_state_keys``, a trivial pad-policy map is
  synthesised and wired the same way;
* with neither a spec nor usable keys (or only ``joint_*`` keys), it is a
  no-op that leaves the policy on the legacy heuristic remap path;
* an unknown embodiment name surfaces as a ``RuntimeError`` rather than a bare
  ``ValueError`` escaping the load path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _feature(dim: int) -> MagicMock:
    """A stand-in for a LeRobot feature spec exposing a ``shape`` tuple."""
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def _make_policy(**kwargs) -> LerobotLocalPolicy:
    """Construct a policy without touching the heavy ``_load_model`` path."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(**kwargs)


def _wire_features(policy: LerobotLocalPolicy, *, state_dim: int, action_dim: int) -> MagicMock:
    """Attach synthetic model features + a mock processor bridge to *policy*."""
    policy._input_features = {
        "observation.images.top": _feature(3),
        "observation.state": _feature(state_dim),
    }
    policy._output_features = {"action": _feature(action_dim)}
    bridge = MagicMock(name="ProcessorBridge")
    policy._processor_bridge = bridge
    return bridge


def test_explicit_map_is_validated_injected_and_sets_action_keys():
    """An explicit EmbodimentMap is wired into the bridge and drives action naming."""
    arm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    embodiment = EmbodimentMap(
        name="so100_test",
        obs_rename={"image": "observation.images.top"},
        state_keys=list(arm),
        action_keys=list(arm),
        dim_policy="strict",
    )
    policy = _make_policy(embodiment=embodiment)
    bridge = _wire_features(policy, state_dim=6, action_dim=6)

    policy._configure_embodiment()

    # The exact map instance is injected into the pipeline once.
    bridge.apply_embodiment.assert_called_once()
    assert bridge.apply_embodiment.call_args.args[0] is embodiment
    assert bridge.apply_embodiment.call_args.kwargs["input_features"] is policy._input_features
    # The model's action index -> actuator-name mapping latches onto action_keys.
    assert policy._embodiment is embodiment
    assert policy.robot_state_keys == arm


def test_map_synthesised_from_non_joint_robot_state_keys():
    """No spec + non-joint robot_state_keys -> a trivial pad map is built and wired."""
    motors = ["motor_0", "motor_1", "motor_2", "motor_3", "motor_4", "motor_5"]
    policy = _make_policy(embodiment=None)
    policy.robot_state_keys = list(motors)
    bridge = _wire_features(policy, state_dim=6, action_dim=6)

    policy._configure_embodiment()

    bridge.apply_embodiment.assert_called_once()
    synthesised = policy._embodiment
    assert isinstance(synthesised, EmbodimentMap)
    assert synthesised.name == "<from robot_state_keys>"
    assert synthesised.dim_policy == "pad"
    assert synthesised.state_keys == motors
    # action_keys mirror the supplied keys, so robot_state_keys is preserved.
    assert policy.robot_state_keys == motors


def test_no_spec_and_no_keys_is_a_noop():
    """Neither a spec nor usable keys -> stay on the legacy heuristic path."""
    policy = _make_policy(embodiment=None)
    policy.robot_state_keys = []
    bridge = _wire_features(policy, state_dim=6, action_dim=6)

    policy._configure_embodiment()

    assert policy._embodiment is None
    bridge.apply_embodiment.assert_not_called()


def test_only_joint_prefixed_keys_is_a_noop():
    """Generic ``joint_*`` keys carry no actuator identity -> no synthesis."""
    policy = _make_policy(embodiment=None)
    policy.robot_state_keys = ["joint_0", "joint_1", "joint_2"]
    bridge = _wire_features(policy, state_dim=3, action_dim=3)

    policy._configure_embodiment()

    assert policy._embodiment is None
    bridge.apply_embodiment.assert_not_called()


def test_unknown_embodiment_name_raises_runtime_error():
    """A bad registry name fails the load loudly as RuntimeError, not bare ValueError."""
    policy = _make_policy(embodiment="definitely_not_a_real_embodiment_xyz")
    _wire_features(policy, state_dim=6, action_dim=6)

    with pytest.raises(RuntimeError, match="Failed to load embodiment"):
        policy._configure_embodiment()

    assert policy._embodiment is None
