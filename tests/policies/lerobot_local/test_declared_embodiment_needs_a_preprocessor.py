"""A declared embodiment is refused when there is no preprocessor to apply it to.

``_configure_embodiment`` wires an :class:`EmbodimentMap` into LeRobot's pipeline
by installing a rename step and a ``strands_pack_state`` step - both on the
PREprocessor. ``ProcessorBridge.from_pretrained`` skips a pipeline whose config
the checkpoint omits ("If either doesn't exist, that pipeline is skipped"), and
``is_active`` is ``has_preprocessor or has_postprocessor``, so a checkpoint
shipping only ``policy_postprocessor.json`` reaches the configure hook with
nothing to configure.

Applying nothing there is not a degraded version of the caller's request. The
policy would take the legacy path - ``observation.state`` composed in the sim's
own units, cameras bound by name/position rather than by the declared
``obs_rename`` - while ``_tensor_to_action_dicts`` still converts the returned
action with ``model_action_to_sim``, i.e. exactly half of a
``*_units="degrees"`` map (so100 / so101). These cells pin that a DECLARED map
is refused with the cause named, that the policy does not claim it anyway, and
that the two shapes which were always fine are untouched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import (
    POSTPROCESSOR_CONFIG,
    PREPROCESSOR_CONFIG,
    ProcessorBridge,
)

ARM = ["1", "2", "3", "4", "5", "6"]


def _feature(dim: int) -> MagicMock:
    """A stand-in for a LeRobot feature spec exposing a ``shape`` tuple."""
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def _degrees_map() -> EmbodimentMap:
    """A so101-shaped map: sim radians in, model degrees out, both sides declared."""
    return EmbodimentMap(
        name="so101_test",
        obs_rename={"front": "observation.images.top"},
        state_keys=list(ARM),
        action_keys=list(ARM),
        dim_policy="pad",
        state_units="degrees",
        action_units="degrees",
        gripper_index=5,
        gripper_joint_range=[-0.175, 1.745],
    )


def _policy(**kwargs) -> LerobotLocalPolicy:
    """Construct a policy and give it synthetic model features, no heavy load."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(**kwargs)
    policy._input_features = {
        "observation.images.top": _feature(3),
        "observation.state": _feature(6),
    }
    policy._output_features = {"action": _feature(6)}
    return policy


def _bridge(*, preprocessor: object | None) -> ProcessorBridge:
    """A REAL bridge, so ``has_preprocessor`` is decided by the shipped property.

    The postprocessor is a plain sentinel: ``has_postprocessor`` only asks whether
    one is present, and it is what makes such a bridge ``is_active`` - the state a
    postprocessor-only checkpoint arrives in.
    """
    return ProcessorBridge(preprocessor=preprocessor, postprocessor=object())


def test_a_declared_embodiment_is_refused_when_no_preprocessor_can_carry_it():
    """No preprocessor means none of the declared map would run: say so."""
    policy = _policy(embodiment=_degrees_map())
    policy._processor_bridge = _bridge(preprocessor=None)

    with pytest.raises(ValueError):
        policy._configure_embodiment()


def test_the_refused_map_is_not_claimed_as_configured():
    """A half-applied map must not be latched: the action side reads ``_embodiment``."""
    policy = _policy(embodiment=_degrees_map())
    policy._processor_bridge = _bridge(preprocessor=None)
    policy.robot_state_keys = ["joint_0"]

    with pytest.raises(ValueError):
        policy._configure_embodiment()

    # _tensor_to_action_dicts converts only when _embodiment is set, so leaving it
    # None is what keeps both halves of the conversion consistent on the fallback.
    assert policy._embodiment is None
    assert policy.robot_state_keys == ["joint_0"]


def test_the_refusal_names_the_missing_pipeline_and_a_way_out():
    """The caller cannot act on "could not be configured" alone."""
    policy = _policy(embodiment=_degrees_map())
    policy._processor_bridge = _bridge(preprocessor=None)

    with pytest.raises(ValueError) as excinfo:
        policy._configure_embodiment()

    message = str(excinfo.value)
    assert "so101_test" in message
    assert PREPROCESSOR_CONFIG in message
    assert POSTPROCESSOR_CONFIG in message
    # The half that WOULD still have run, and the two remedies.
    assert "action_units" in message
    assert "embodiment=" in message


def test_a_map_synthesised_from_state_keys_still_takes_the_legacy_path():
    """Over-reach control: that map is native-unit and binds the same keys.

    Leaving it unapplied drops nothing the legacy path does not already do, and
    refusing it would discard a postprocessor the checkpoint really shipped.
    """
    policy = _policy()  # no declared spec
    policy.robot_state_keys = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    policy._processor_bridge = _bridge(preprocessor=None)

    policy._configure_embodiment()  # must not raise

    assert policy._embodiment is not None
    assert policy._embodiment.state_units == "native"


def test_a_declared_embodiment_is_applied_when_a_preprocessor_is_present():
    """Over-reach control: the normal path is untouched."""
    policy = _policy(embodiment=_degrees_map())
    policy._processor_bridge = _bridge(preprocessor=MagicMock(name="preprocessor"))

    with patch.object(ProcessorBridge, "apply_embodiment") as applied:
        policy._configure_embodiment()

    applied.assert_called_once()
    assert policy._embodiment is not None
    assert policy.robot_state_keys == ARM
