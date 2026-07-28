# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The state and action sides must agree on what a hardware observation is.

Two separately-written predicates decided "does this observation carry
``'<motor>.pos'`` hardware joint readings?":

* the ACTION side (``get_actions``' hardware key override) ran a plain-``float``
  pass and only tried a numpy pass ``if not _pos:`` - so a PARTIALLY matching
  first pass poisoned the retry and left the override unset;
* the STATE side (``PackStateProcessorStep``) used ONE combined pass and
  succeeded on the same observation.

When they disagreed, the model was fed raw hardware degrees while its action was
emitted WITH the sim radian->degree conversion applied - roughly 57x
under-scaled - and nothing warned, because the state side had succeeded.

The trigger is a numpy dtype detail: ``isinstance(np.float64(1.0), float)`` is
True but ``isinstance(np.float32(1.0), float)`` is False. So only an observation
MIXING ``float`` with ``np.float32`` tripped it, which is why it survived
testing.

Both sides now bind to one shared :func:`hardware_pos_keys`.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.embodiment import (
    EmbodimentMap,
    _is_joint_scalar,
    hardware_pos_keys,
)
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_POS_KEYS = [f"{motor}.pos" for motor in _MOTORS]

#: The only observation shape that tripped the split predicate: the plain-float
#: pass matches 3 of 6 keys, so it is non-empty (no numpy retry) but too short to
#: reach the embodiment's 6 actuators.
_MIXED_DTYPES = [1.0, 2.0, 3.0, np.float32(4.0), np.float32(5.0), np.float32(6.0)]


def _obs(values: list) -> dict:
    return dict(zip(_POS_KEYS, values, strict=True))


def _torch_scalar(value: float):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.Tensor, "__torch_function__"):
        pytest.skip("torch mock does not replicate 0-d tensor semantics")
    return torch.tensor(value)


def _torch_bool(value: bool = True):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.Tensor, "__torch_function__"):
        pytest.skip("torch mock does not replicate 0-d tensor semantics")
    return torch.tensor(value, dtype=torch.bool)


def _sim_policy(action_keys: list[str] | None = None) -> LerobotLocalPolicy:
    """A policy carrying a SIM embodiment.

    Only the hardware-key detection is exercised, and the empty checkpoint path
    skips the model load, so construction stays offline.
    """
    policy = LerobotLocalPolicy(policy_type="act")
    # so101 in sim: positional joint names, NOT '.pos' driver names.
    keys = action_keys if action_keys is not None else [str(i) for i in range(1, 7)]
    policy._embodiment = EmbodimentMap(name="so101", state_keys=list(keys), action_keys=list(keys))
    return policy


class TestEveryDtypeIsDetected:
    def test_plain_floats(self):
        assert hardware_pos_keys(_obs([1.0] * 6)) == _POS_KEYS

    def test_numpy_float32(self):
        assert hardware_pos_keys(_obs([np.float32(1.0)] * 6)) == _POS_KEYS

    def test_numpy_float64(self):
        assert hardware_pos_keys(_obs([np.float64(1.0)] * 6)) == _POS_KEYS

    def test_python_ints(self):
        assert hardware_pos_keys(_obs([1] * 6)) == _POS_KEYS

    def test_numpy_integers(self):
        assert hardware_pos_keys(_obs([np.int32(1)] * 6)) == _POS_KEYS

    def test_zero_dim_ndarray(self):
        assert hardware_pos_keys(_obs([np.array(1.0)] * 6)) == _POS_KEYS

    def test_zero_dim_torch_tensor(self):
        assert hardware_pos_keys(_obs([_torch_scalar(1.0)] * 6)) == _POS_KEYS

    def test_mixed_float_and_float32(self):
        assert hardware_pos_keys(_obs(_MIXED_DTYPES)) == _POS_KEYS


class TestNonJointValuesAreExcluded:
    def test_a_bool_is_not_a_joint_reading(self):
        """``isinstance(True, int)`` is True, so the exclusion has to be explicit.

        A driver status flag (``is_homed``, ...) must never occupy an actuator
        column, where ``True`` would be commanded as an absolute ``1.0``.
        """
        keys = hardware_pos_keys({"is_homed.pos": True, "shoulder_pan.pos": 1.0})

        assert keys == ["shoulder_pan.pos"]

    @pytest.mark.parametrize("value", [True, False])
    def test_is_joint_scalar_rejects_both_bools(self, value):
        assert _is_joint_scalar(value) is False

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: np.bool_(True),
            lambda: np.bool_(False),
            lambda: np.array(True),
            lambda: np.array(False, dtype=bool),
            _torch_bool,
        ],
        ids=["np.bool_ True", "np.bool_ False", "ndarray0d bool", "ndarray0d dtype=bool", "torch bool"],
    )
    def test_is_joint_scalar_rejects_every_bool_flavour(self, factory):
        """A python ``bool`` is only one of the ways an observation says "boolean".

        None of these is caught by the check for another: ``np.bool_`` is not a
        ``bool`` subclass and not an ``np.integer``, so it reaches the duck-typed
        0-d branch (``ndim`` is 0, ``item`` exists); a 0-d bool ``ndarray`` and a
        0-d ``torch.bool`` tensor arrive the same way. Each one accepted is an
        extra key in front of the real joints.
        """
        assert _is_joint_scalar(factory()) is False

    def test_a_multi_element_array_is_not_a_scalar(self):
        keys = hardware_pos_keys({"cloud.pos": np.zeros(3), "shoulder_pan.pos": 1.0})

        assert keys == ["shoulder_pan.pos"]

    def test_a_string_value_is_not_a_joint_reading(self):
        assert hardware_pos_keys({"a.pos": "1.0"}) == []

    def test_none_is_not_a_joint_reading(self):
        assert hardware_pos_keys({"a.pos": None}) == []

    def test_keys_without_the_pos_suffix_are_ignored(self):
        keys = hardware_pos_keys({"shoulder_pan": 1.0, "shoulder_pan.pos": 1.0})

        assert keys == ["shoulder_pan.pos"]

    def test_a_camera_frame_is_ignored(self):
        obs = {**_obs([1.0] * 6), "front": np.zeros((4, 4, 3), dtype=np.uint8)}

        assert hardware_pos_keys(obs) == _POS_KEYS


class TestOrderIsPreserved:
    def test_observation_order_is_kept(self):
        """Both sides bind positionally, so the ORDER is the contract."""
        reversed_keys = list(reversed(_POS_KEYS))

        assert hardware_pos_keys(dict.fromkeys(reversed_keys, 1.0)) == reversed_keys


class TestTheActionSideAgreesWithTheStateSide:
    @pytest.mark.parametrize(
        "values",
        [
            [1.0] * 6,
            [np.float32(1.0)] * 6,
            [np.float64(1.0)] * 6,
            [np.array(1.0)] * 6,
            _MIXED_DTYPES,
        ],
        ids=["float", "float32", "float64", "ndarray0d", "mixed"],
    )
    def test_the_action_side_binds_the_keys_the_state_side_packs(self, values):
        """The invariant the split predicate violated: same obs, same answer.

        ``hardware_pos_keys`` is what ``PackStateProcessorStep`` packs
        ``observation.state`` from, so the action side returning exactly those
        keys is what keeps the two unit systems from diverging.
        """
        observation = _obs(values)

        assert _sim_policy()._hardware_action_keys(observation) == hardware_pos_keys(observation)

    def test_mixed_dtypes_no_longer_fall_back_to_sim_action_keys(self):
        """The regression: 3 plain floats + 3 float32 left the override unset.

        The state side packed the raw hardware degrees, then the action side -
        seeing no hardware keys - emitted the command through the sim
        radian->degree conversion, ~57x under-scaled, silently.
        """
        assert _sim_policy()._hardware_action_keys(_obs(_MIXED_DTYPES)) == _POS_KEYS


class TestABooleanFlagNeverTakesAJointColumn:
    """A flag accepted as a joint reading shifts every joint by one column.

    Both sides slice ``[:len(actuators)]`` off this list, so one extra key in
    front does not fail loudly - it renames each reading to its neighbour and
    drops the last one. On hardware these are ABSOLUTE targets, and the flag's
    ``1.0`` lands in joint slot 0.
    """

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: True,
            lambda: np.bool_(True),
            lambda: np.array(True),
            _torch_bool,
        ],
        ids=["python bool", "np.bool_", "ndarray0d bool", "torch bool"],
    )
    def test_a_leading_flag_does_not_shift_the_joints(self, factory):
        observation = {"is_homed.pos": factory()}
        observation.update(_obs([float(i) for i in range(1, 7)]))

        assert hardware_pos_keys(observation) == _POS_KEYS

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: True,
            lambda: np.bool_(True),
            lambda: np.array(True),
            _torch_bool,
        ],
        ids=["python bool", "np.bool_", "ndarray0d bool", "torch bool"],
    )
    def test_both_sides_still_agree_when_a_flag_is_present(self, factory):
        """The shared predicate has to hold the invariant under a flag too.

        If only one side dropped the flag, the state vector and the actuator
        names would disagree by one column - the divergence this predicate was
        made single-source-of-truth to prevent.
        """
        observation = {"is_homed.pos": factory()}
        observation.update(_obs([float(i) for i in range(1, 7)]))

        assert _sim_policy()._hardware_action_keys(observation) == hardware_pos_keys(observation)


class TestTheOverrideStaysOffWhenItShould:
    def test_no_embodiment_means_no_override(self):
        policy = _sim_policy()
        policy._embodiment = None

        assert policy._hardware_action_keys(_obs([1.0] * 6)) is None

    def test_an_embodiment_without_action_keys_means_no_override(self):
        assert _sim_policy(action_keys=[])._hardware_action_keys(_obs([1.0] * 6)) is None

    def test_a_real_hardware_embodiment_needs_no_override(self):
        """``so101_real`` already declares '.pos' actuators - nothing to rename."""
        assert _sim_policy(action_keys=_POS_KEYS)._hardware_action_keys(_obs([1.0] * 6)) is None

    def test_a_sim_observation_is_not_mistaken_for_hardware(self):
        sim_obs = {str(i): 0.1 for i in range(1, 7)}

        assert _sim_policy()._hardware_action_keys(sim_obs) is None

    def test_too_few_hardware_joints_for_the_embodiment(self):
        """A partial hardware read must not bind a short, index-shifted action."""
        partial = {key: 1.0 for key in _POS_KEYS[:4]}

        assert _sim_policy()._hardware_action_keys(partial) is None
