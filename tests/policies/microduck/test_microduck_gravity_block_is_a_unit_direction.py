"""Slot two of the Microduck observation is the unit direction the layout declares.

``build_observation`` documents slot two as "a UNIT gravity direction in the base
frame", and holds ``base_quat`` to two properties on the way there: a width
(``_require_base_block``) and finiteness (``_non_finite_observation_error``). It
did not hold it to the third - that the four components describe a rotation - and
:func:`~strands_robots.policies.microduck.observation.quat_rotate_inverse`
implements a formula that is a rotation only for a UNIT quaternion: it mixes a
term quadratic in the components with a linear one, so scaling does not cancel.

A quaternion scaled by any positive factor encodes the SAME rotation. That is why
the library's single orientation domain
(:func:`~strands_robots.utils.coerce_orientation_quaternion`) accepts any
magnitude, and it says why in as many words: "every consumer either normalizes or
is scale-invariant". Measured on ``3c30c4b4`` for a 20 deg pitch about world +Y,
reading the same rotation at several magnitudes:

===============  =======================  ==========  =====================
``base_quat``    slot two                 magnitude   direction vs the unit
===============  =======================  ==========  =====================
unit             ``[ 0.342, 0, -0.940]``  1.000       0.000 deg
x 1.01           ``[ 0.349, 0, -0.938]``  1.001       0.393 deg
x 0.5            ``[ 0.086, 0, -0.985]``  0.989       15.038 deg
x 2              ``[ 1.368, 0, -0.759]``  1.564       40.986 deg
x 10             ``[34.202, 0,  5.031]``  34.570      78.368 deg
all zero         ``[ 0.000, 0, -1.000]``  1.000       upright, 20 deg off
===============  =======================  ==========  =====================

The last row is the worst of them: an all-zero quaternion is what an orientation
that was never written or was dropped on the wire spells, the formula returns the
vector it was handed unchanged, and for world ``-Z`` that is exactly the gravity
block of a PERFECTLY UPRIGHT base - the one attitude a locomotion policy most
needs to tell apart from a fall. It passes the width guard (4 components) and the
finiteness pass (all finite), so nothing else in the module sees it.

The contract was already settled everywhere else. Four world-to-body helpers
normalise internally - ``policies/wbc/control.quat_rotate_inverse``,
``policies/protomotions/state_utils.quat_rotate_inverse``,
``simulation/predicates._quat_rotate_inverse_wxyz`` and the Newton backend's copy
- and the two same-layer policy siblings refuse the degenerate case rather than
answering with a made-up rotation. This module was the fifth helper and the one
that had neither. ``TestTheSiblingHelpersAgree`` grades that agreement so the
family cannot drift apart again.

Normalising does not move the happy path: for a unit quaternion the divisor is
1.0 and the pre-existing formula is reproduced exactly, which
``test_a_unit_quaternion_is_answered_bit_for_bit`` pins against a local copy of
the pre-fix expression.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.policies.microduck import (
    MICRODUCK_DEFAULT_POSE,
    MICRODUCK_JOINT_NAMES,
    build_observation,
    projected_gravity,
    quat_rotate_inverse,
)
from strands_robots.policies.microduck.observation import _BASE_QUAT_LEN
from strands_robots.utils import MIN_QUATERNION_NORM

#: World gravity, the vector slot two is world ``-Z`` rotated from.
WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)

#: A 20 deg pitch about world +Y, wxyz - a plausible Microduck trunk attitude.
PITCH_20_DEG = 20.0
Q_PITCH_20 = np.array(
    [math.cos(math.radians(PITCH_20_DEG) / 2.0), 0.0, math.sin(math.radians(PITCH_20_DEG) / 2.0), 0.0],
    dtype=np.float32,
)

#: The magnitudes the same rotation is offered at. ``1.01`` is IMU drift, ``0.924``
#: is the norm a linear interpolation of two 90-deg-apart samples leaves behind.
SCALES = (0.5, 0.923879, 1.01, 2.0, 10.0)

#: The spellings of an orientation that carries no direction.
NO_DIRECTION = (
    ("all zero", [0.0, 0.0, 0.0, 0.0]),
    ("below the shared floor", [MIN_QUATERNION_NORM / 10.0, 0.0, 0.0, 0.0]),
)


def _reference_world_to_body(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """``R(q)^T @ vec`` built from the rotation matrix, normalising explicitly.

    An independent oracle: it constructs the matrix rather than reusing the
    Rodrigues-form expansion the helper under test evaluates.
    """
    w, x, y, z = (np.asarray(quat_wxyz, dtype=np.float64) / np.linalg.norm(quat_wxyz)).tolist()
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot.T @ np.asarray(vec, dtype=np.float64)


def _tilt_degrees(gravity: np.ndarray) -> float:
    """Angle between a base-frame gravity block and straight down, in degrees."""
    g = np.asarray(gravity, dtype=np.float64)
    cos = float(-g[2] / np.linalg.norm(g))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _obs_dict(base_quat: object) -> dict[str, object]:
    """A complete projected-gravity observation dict carrying ``base_quat``."""
    obs: dict[str, object] = {"base_ang_vel": [0.0, 0.0, 0.0], "base_quat": base_quat}
    for name in MICRODUCK_JOINT_NAMES:
        obs[name] = 0.0
        obs[f"{name}.vel"] = 0.0
    return obs


def _slot_two(base_quat: object) -> np.ndarray:
    """The gravity block ``build_observation`` writes for ``base_quat``."""
    observation = build_observation(
        _obs_dict(base_quat),
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=MICRODUCK_DEFAULT_POSE,
        last_action=np.zeros(len(MICRODUCK_JOINT_NAMES), dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
    )
    return np.asarray(observation[3:6], dtype=np.float64)


class TestASlotTwoBlockIsAUnitDirection:
    """The declared property of slot two, at every magnitude of one rotation."""

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_block_is_unit_length(self, scale: float) -> None:
        block = _slot_two((Q_PITCH_20 * scale).tolist())
        assert float(np.linalg.norm(block)) == pytest.approx(1.0, abs=1e-6), (
            f"base_quat scaled by {scale} (|q| = {float(np.linalg.norm(Q_PITCH_20 * scale)):.4f}) "
            f"put a gravity block of magnitude {float(np.linalg.norm(block)):.4f} into slot two, "
            f"which build_observation documents as a unit direction"
        )

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_block_names_the_attitude_the_quaternion_encodes(self, scale: float) -> None:
        block = _slot_two((Q_PITCH_20 * scale).tolist())
        assert _tilt_degrees(block) == pytest.approx(PITCH_20_DEG, abs=1e-3), (
            f"a base pitched {PITCH_20_DEG} deg, with base_quat scaled by {scale}, "
            f"reported an attitude {_tilt_degrees(block):.3f} deg off vertical"
        )

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_helper_matches_an_independent_rotation_matrix(self, scale: float) -> None:
        expected = _reference_world_to_body(Q_PITCH_20, WORLD_GRAVITY)
        got = np.asarray(quat_rotate_inverse((Q_PITCH_20 * scale).tolist(), WORLD_GRAVITY), dtype=np.float64)
        assert got == pytest.approx(expected, abs=1e-6), (
            f"scaled by {scale}, world -Z body-framed to {got.tolist()} instead of {expected.tolist()}"
        )


class TestAnOrientationThatWasNeverWrittenIsRefused:
    """A quaternion with no direction is refused, not read as upright."""

    @pytest.mark.parametrize(("label", "quat"), NO_DIRECTION, ids=[label for label, _ in NO_DIRECTION])
    def test_the_helper_refuses_it(self, label: str, quat: list[float]) -> None:
        with pytest.raises(ValueError, match="describes no rotation"):
            quat_rotate_inverse(quat, WORLD_GRAVITY)

    @pytest.mark.parametrize(("label", "quat"), NO_DIRECTION, ids=[label for label, _ in NO_DIRECTION])
    def test_build_observation_refuses_it(self, label: str, quat: list[float]) -> None:
        with pytest.raises(ValueError, match="describes no rotation"):
            _slot_two(quat)

    def test_the_refusal_names_what_the_block_would_have_claimed(self) -> None:
        """The message has to say why a zero orientation is not a harmless one."""
        with pytest.raises(ValueError, match="perfectly upright"):
            projected_gravity(np.zeros(_BASE_QUAT_LEN, dtype=np.float32))

    def test_the_width_and_finiteness_passes_do_accept_it(self) -> None:
        """Premise: no other guard in the module sees a zero quaternion.

        Both existing passes hold - it is four components and every one is
        finite - so this rule cannot be delegated to either of them.
        """
        zero = np.zeros(_BASE_QUAT_LEN, dtype=np.float32)
        assert zero.shape[0] == _BASE_QUAT_LEN, "premise: a zero quaternion is the declared width"
        assert bool(np.isfinite(zero).all()), "premise: a zero quaternion is finite"


class TestTheUnchangedBehaviour:
    """Controls: everything that held before still holds."""

    def test_a_unit_quaternion_is_answered_bit_for_bit(self) -> None:
        """The formula on the domain Pollen operates in is untouched."""
        for quat in ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]):
            q = np.asarray(quat, dtype=np.float32)
            assert float(np.linalg.norm(np.asarray(q, dtype=np.float64))) == 1.0, (
                f"premise: {quat} is exactly unit, so normalising divides by 1.0"
            )
            v = WORLD_GRAVITY
            # The pre-fix expression, evaluated locally.
            t = np.cross(q[1:4], v) * 2.0
            pre_fix = (v - q[0] * t + np.cross(q[1:4], t)).astype(np.float32)
            got = quat_rotate_inverse(q, v)
            assert np.array_equal(got, pre_fix), f"{quat}: {got.tolist()} != pre-fix {pre_fix.tolist()}"

    def test_a_non_finite_component_still_reaches_the_assembled_vector_pass(self) -> None:
        """A ``nan`` is named through the block it becomes, not refused here.

        The norm of a ``nan`` quaternion is ``nan``, which is not below the
        floor, so the value flows on and the assembled-vector pass reports it
        against ``projected_gravity (from base_quat)`` as it always has.
        """
        with pytest.raises(ValueError, match=r"projected_gravity \(from base_quat\)"):
            _slot_two([1.0, float("nan"), 0.0, 0.0])

    def test_an_identity_orientation_still_reads_as_world_gravity(self) -> None:
        assert _slot_two([1.0, 0.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-7)


class TestTheSiblingHelpersAgree:
    """The five world-to-body helpers read one rotation the same way."""

    def test_every_helper_is_scale_invariant(self) -> None:
        from strands_robots.policies.protomotions.state_utils import (
            quat_rotate_inverse as protomotions_helper,
        )
        from strands_robots.policies.wbc.control import quat_rotate_inverse as wbc_helper
        from strands_robots.simulation.newton.simulation import (
            _quat_rotate_inverse_wxyz as newton_helper,
        )
        from strands_robots.simulation.predicates import _quat_rotate_inverse_wxyz as predicates_helper

        q_wxyz = Q_PITCH_20.astype(np.float64)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        vec = np.array([0.0, 0.0, -1.0])
        helpers = {
            "microduck": lambda s: np.asarray(quat_rotate_inverse((q_wxyz * s).astype(np.float32), WORLD_GRAVITY)),
            "wbc": lambda s: np.asarray(wbc_helper(q_wxyz * s, vec)),
            "protomotions": lambda s: np.asarray(protomotions_helper((q_xyzw * s).astype(np.float32), vec)),
            "predicates": lambda s: np.asarray(predicates_helper(list(q_wxyz * s), list(vec))),
            "newton": lambda s: np.asarray(newton_helper(list(q_wxyz * s), list(vec))),
        }
        expected = _reference_world_to_body(q_wxyz, vec)
        for name, helper in helpers.items():
            for scale in (1.0, 2.0):
                got = np.asarray(helper(scale), dtype=np.float64)
                assert got == pytest.approx(expected, abs=1e-6), (
                    f"{name} read a quaternion scaled by {scale} as a different rotation: "
                    f"{got.tolist()} instead of {expected.tolist()}"
                )

    def test_both_policy_layer_helpers_refuse_a_zero_orientation(self) -> None:
        """The two same-layer siblings agree that a zero quaternion is refused."""
        from strands_robots.policies.protomotions.state_utils import (
            quat_rotate_inverse as protomotions_helper,
        )
        from strands_robots.policies.wbc.control import quat_rotate_inverse as wbc_helper

        for name, call in (
            ("microduck", lambda: quat_rotate_inverse(np.zeros(4, dtype=np.float32), WORLD_GRAVITY)),
            ("wbc", lambda: wbc_helper(np.zeros(4), np.array([0.0, 0.0, -1.0]))),
            (
                "protomotions",
                lambda: protomotions_helper(
                    np.zeros(4, dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)
                ),
            ),
        ):
            with pytest.raises(ValueError):
                call()
            assert name  # the loop variable names the helper in a failure
