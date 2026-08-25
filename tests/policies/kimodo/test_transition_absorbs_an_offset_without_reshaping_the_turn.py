"""The segment-transition ease absorbs a pose offset without reshaping the motion.

``_ease_onto_previous_pose`` decays the offset between a freshly sampled
segment's start pose and the pose last commanded. Its documented promise is that
"the motion keeps its own shape and velocity and only its starting offset is
absorbed", and for the linear channels (root position, joint angles) that is
plainly what the arithmetic does: one offset, scaled by the frame's weight, is
added to every eased frame. The correction a frame receives is therefore decided
by the weight alone.

The root's orientation is the rotational analogue of that, and these cells pin
it: the offset is the rotation carrying the segment's own start orientation onto
the pose last commanded, and it decays by pre-multiplication. Two consequences
are measurable and neither holds if each frame is instead interpolated toward
the previous orientation:

* a seam with no orientation offset to absorb leaves the root untouched, and
* a motion that turns keeps a uniform turn rate through the transition.

Why the existing transition suite cannot see this: its stub agent writes an
identity quaternion into every frame of every motion, so the root never rotates
there and a decaying offset is indistinguishable from an absolute pull. The
distinguishing input is a motion whose root actually turns.

The root pose is internal to the policy today - ``get_actions`` emits joint
targets only - so ``TestTheEmittedActionIsUntouched`` pins that this stays true.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoConfig, KimodoPolicy
from strands_robots.policies.kimodo.policy import _ease_onto_previous_pose

_ROOT = 7
_NUM_JOINTS = len(KIMODO_G1_JOINTS)
#: Rate the fixture motions yaw at, in degrees per native frame. 6 deg at 30 Hz
#: is a 180 deg/s turn - brisk for a humanoid, and large enough that a reshaped
#: turn rate is unmistakable rather than a rounding artefact.
_TURN_DEG_PER_FRAME = 6.0
#: Rotation of a fixture motion's first frame. Non-zero so the start
#: orientation the offset is measured against is not the identity.
_START_DEG = 30.0


def _quat_about_axis(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """Unit wxyz quaternion for a rotation of ``degrees`` about ``axis``."""
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    half = np.deg2rad(degrees) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * unit)], dtype=np.float32)


def _quat_about_z(degrees: float) -> np.ndarray:
    """Unit wxyz quaternion for a yaw of ``degrees`` about +Z."""
    return _quat_about_axis((0.0, 0.0, 1.0), degrees)


def _matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit wxyz quaternion.

    Matrix arithmetic is the independent oracle these cells need: the
    orientation the ease should emit is stated as a composition of matrices, so
    a slip in the quaternion product it is compared against has nowhere to hide.
    """
    w, x, y, z = (float(component) for component in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation axis (unit) and angle (radians) of a rotation matrix."""
    angle = float(np.arccos(np.clip((float(np.trace(matrix)) - 1.0) / 2.0, -1.0, 1.0)))
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    return axis / np.linalg.norm(axis), angle


def _rotation_matrix(axis: np.ndarray, radians: float) -> np.ndarray:
    """Rotation matrix for ``radians`` about a unit ``axis`` (Rodrigues)."""
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(radians) * skew + (1.0 - np.cos(radians)) * (skew @ skew)


def _yaw_degrees(quat: np.ndarray) -> float:
    """Yaw of a wxyz quaternion known to be a pure rotation about +Z."""
    return float(np.rad2deg(2.0 * np.arctan2(float(quat[3]), float(quat[0]))))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Shorter-arc angle between two unit wxyz quaternions, in degrees.

    Measured as ``4 * arcsin(d / 2)`` on the closer of ``a - b`` and ``a + b``,
    the two representations of one rotation. The textbook
    ``2 * arccos(|dot|)`` agrees analytically but not numerically: ``arccos`` is
    ill-conditioned at 1, so a float32 quaternion compared with itself reports
    around 0.04 deg of rotation and a "nothing moved" assertion cannot be
    written at all. The form below reports 0 for equal inputs.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    distance = min(float(np.linalg.norm(first - second)), float(np.linalg.norm(first + second)))
    return float(np.rad2deg(4.0 * np.arcsin(min(1.0, distance / 2.0))))


def _turning_motion(
    frames: int,
    *,
    deg_per_frame: float = _TURN_DEG_PER_FRAME,
    njoints: int = 4,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    start_deg: float = _START_DEG,
) -> np.ndarray:
    """A motion whose root turns at a constant rate and whose channels all advance.

    Args:
        frames: Number of native frames.
        deg_per_frame: Constant rotation increment per frame.
        njoints: Number of joint channels.
        axis: Axis the root turns about.
        start_deg: Rotation of the first frame. Non-zero by default so the start
            orientation the offset is measured from is not the identity, where
            inverting it or not would look the same.

    Returns:
        A ``(frames, 7 + njoints)`` float32 qpos block.
    """
    qpos = np.zeros((frames, _ROOT + njoints), dtype=np.float32)
    for index in range(frames):
        qpos[index, 0:3] = (0.10 * index, 0.0, 0.90)
        qpos[index, 3:7] = _quat_about_axis(axis, start_deg + deg_per_frame * index)
        qpos[index, _ROOT:] = 0.02 * index
    return qpos


def _previous_pose(
    motion: np.ndarray,
    *,
    yaw_offset: float,
    joint_offset: float = 0.30,
    base_deg: float = _START_DEG,
) -> np.ndarray:
    """The pose a seam must ease onto: the motion's start pose, displaced.

    The orientation is built from ``base_deg + yaw_offset`` rather than by
    recovering the yaw out of ``motion`` and rebuilding it: a float32 round trip
    through the recovered angle lands an ULP away, which would leave a 0.04 deg
    offset in the case that is supposed to carry none at all.

    Args:
        motion: The motion being eased.
        yaw_offset: Degrees of yaw between the motion's start orientation and
            the returned pose. Zero means there is no rotation to absorb.
        joint_offset: Radians added to every joint channel, so the seam still
            has something to absorb when ``yaw_offset`` is zero.
        base_deg: Yaw of the motion's own first frame.

    Returns:
        A ``(7 + njoints,)`` float32 pose.
    """
    pose = motion[0].copy()
    pose[3:7] = _quat_about_z(base_deg + yaw_offset)
    pose[_ROOT:] += joint_offset
    return pose


def _weights(count: int) -> list[float]:
    """The decay weights the ease applies, frame by frame."""
    return [1.0 - (index + 1) / (count + 1) for index in range(count)]


# --------------------------------------------------------------------------
# Premises: the fixtures really do exercise what the cases below claim.
# --------------------------------------------------------------------------
class TestThePremisesOfTheseCases:
    """Without these the cases could pass on a motion that never turns."""

    def test_the_fixture_motion_turns_at_a_constant_rate(self) -> None:
        motion = _turning_motion(12)
        yaws = [_yaw_degrees(motion[i, 3:7]) for i in range(12)]
        increments = np.diff(yaws)
        assert increments.min() > 0.0, "the fixture root must actually turn"
        assert np.allclose(increments, _TURN_DEG_PER_FRAME, atol=1e-3), increments

    def test_a_zero_yaw_offset_really_leaves_nothing_rotational_to_absorb(self) -> None:
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=0.0)
        assert _angle_between(pose[3:7], motion[0, 3:7]) == pytest.approx(0.0, abs=1e-9)
        assert float(np.abs(pose[_ROOT:] - motion[0, _ROOT:]).max()) > 0.1, (
            "the seam must still have a joint offset to absorb, or the whole ease is a no-op"
        )

    def test_the_angle_helper_reports_the_shorter_arc(self) -> None:
        assert _angle_between(_quat_about_z(0.0), _quat_about_z(90.0)) == pytest.approx(90.0, abs=1e-3)
        assert _angle_between(_quat_about_z(0.0), _quat_about_z(350.0)) == pytest.approx(10.0, abs=1e-3)

    def test_the_angle_helper_resolves_a_rotation_far_smaller_than_the_defect(self) -> None:
        """Otherwise "the root did not move" could not be asserted at all."""
        assert _angle_between(_quat_about_z(30.0), _quat_about_z(30.0)) == pytest.approx(0.0, abs=1e-9)
        assert _angle_between(_quat_about_z(30.0), _quat_about_z(30.001)) == pytest.approx(0.001, abs=1e-4)


# --------------------------------------------------------------------------
# The regression: a seam with no orientation offset must not move the root.
# --------------------------------------------------------------------------
class TestASeamWithNoOrientationOffsetLeavesTheRootAlone:
    """The clearest statement of the guarantee: no offset, no correction."""

    def test_the_root_orientation_is_untouched_while_the_joints_are_eased(self) -> None:
        """A joint-only offset must be absorbed by the joints and nothing else."""
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=0.0)

        eased = _ease_onto_previous_pose(motion, pose, 5)

        worst = max(_angle_between(eased[i, 3:7], motion[i, 3:7]) for i in range(len(motion)))
        assert worst < 1e-3, (
            f"the ease rotated the root by up to {worst:.4f} deg although the seam carried no orientation offset at all"
        )
        assert not np.array_equal(eased[:, _ROOT:], motion[:, _ROOT:]), "the joint offset should have been eased"

    def test_a_seam_that_carries_no_offset_at_all_emits_the_motion_verbatim(self) -> None:
        """Continuing from exactly the start pose is the identity on every channel.

        Byte-exact rather than approximate, and by construction: a motion whose
        first frame is the identity rotation makes the offset exactly the
        identity, which the interpolation and the product both carry through
        unchanged. The rotated-start case above is the same statement up to
        float32 rounding.
        """
        motion = _turning_motion(12, start_deg=0.0)

        eased = _ease_onto_previous_pose(motion, motion[0].copy(), 5)

        assert np.array_equal(eased, motion)


class TestTheCorrectionIsDecidedByTheWeightAlone:
    """The property that makes the ease shape-preserving on every channel."""

    @pytest.mark.parametrize("yaw_offset", [-40.0, -5.0, 25.0, 120.0, 200.0])
    @pytest.mark.parametrize("count", [1, 5, 8])
    def test_each_frame_is_rotated_by_the_weighted_share_of_the_offset(self, yaw_offset: float, count: int) -> None:
        """The angle between an eased frame and its own frame is ``weight * offset``."""
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=yaw_offset)
        total = _angle_between(pose[3:7], motion[0, 3:7])

        eased = _ease_onto_previous_pose(motion, pose, count)

        for index, weight in enumerate(_weights(count)):
            applied = _angle_between(eased[index, 3:7], motion[index, 3:7])
            assert applied == pytest.approx(weight * total, abs=0.05), (
                f"frame {index} was rotated {applied:.3f} deg where the weight "
                f"{weight:.3f} of a {total:.3f} deg offset is {weight * total:.3f} deg: the "
                "correction is being decided by the frame's own orientation"
            )

    def test_two_motions_that_start_alike_receive_the_same_correction(self) -> None:
        """Only the start pose and the seam decide the correction, not the motion."""
        slow = _turning_motion(12, deg_per_frame=1.0)
        fast = _turning_motion(12, deg_per_frame=11.0)
        assert np.array_equal(slow[0, 3:7], fast[0, 3:7]), "premise: the two motions start alike"
        pose = _previous_pose(slow, yaw_offset=-40.0)

        eased_slow = _ease_onto_previous_pose(slow, pose, 5)
        eased_fast = _ease_onto_previous_pose(fast, pose, 5)

        for index in range(5):
            applied_slow = _angle_between(eased_slow[index, 3:7], slow[index, 3:7])
            applied_fast = _angle_between(eased_fast[index, 3:7], fast[index, 3:7])
            assert applied_slow == pytest.approx(applied_fast, abs=0.05), (
                f"frame {index}: the slower motion was corrected by {applied_slow:.3f} deg and the "
                f"faster one by {applied_fast:.3f} deg, so the correction tracks the motion"
            )


class TestTheCorrectionIsAWorldFrameRotationOfTheOffset:
    """Which rotation, stated in matrices so the quaternion side has an oracle.

    The cases above measure how far each frame is rotated. These measure where
    it is rotated to, on a motion whose turn axis is not the offset's: two
    rotations about one axis commute, so a yaw-only fixture cannot tell a
    world-frame correction from a body-frame one, nor a product from its
    reverse.
    """

    #: A tilted turn axis and an unrelated orientation for the pose last
    #: commanded, so no two rotations in the fixture share an axis.
    MOTION_AXIS = (0.2, 0.9, -0.3)
    SEAM_AXIS = (0.6, -0.2, 0.75)
    SEAM_DEGREES = 55.0

    def _fixture(self) -> tuple[np.ndarray, np.ndarray]:
        motion = _turning_motion(12, axis=self.MOTION_AXIS, start_deg=17.0)
        pose = motion[0].copy()
        pose[3:7] = _quat_about_axis(self.SEAM_AXIS, self.SEAM_DEGREES)
        pose[_ROOT:] += 0.30
        return motion, pose

    def test_the_fixture_axes_are_not_parallel(self) -> None:
        """Premise: without this the convention cases below are vacuous."""
        motion, pose = self._fixture()
        seam_axis, _ = _axis_angle_from_matrix(_matrix_from_quat(pose[3:7]) @ _matrix_from_quat(motion[0, 3:7]).T)
        motion_axis = np.asarray(self.MOTION_AXIS, dtype=np.float64)
        motion_axis = motion_axis / np.linalg.norm(motion_axis)
        assert abs(float(np.dot(seam_axis, motion_axis))) < 0.9, "the two axes must differ"
        assert not np.allclose(motion[0, 3:7], _quat_about_axis((0.0, 0.0, 1.0), 0.0), atol=1e-6), (
            "the start orientation must not be the identity"
        )

    @pytest.mark.parametrize("count", [1, 5, 8])
    def test_every_eased_frame_is_its_own_frame_turned_by_the_decayed_offset(self, count: int) -> None:
        """``eased = R(weight * offset) * frame``, composed in the world frame."""
        motion, pose = self._fixture()
        offset = _matrix_from_quat(pose[3:7]) @ _matrix_from_quat(motion[0, 3:7]).T
        axis, angle = _axis_angle_from_matrix(offset)

        eased = _ease_onto_previous_pose(motion, pose, count)

        for index, weight in enumerate(_weights(count)):
            expected = _rotation_matrix(axis, weight * angle) @ _matrix_from_quat(motion[index, 3:7])
            actual = _matrix_from_quat(eased[index, 3:7])
            assert np.allclose(actual, expected, atol=1e-4), (
                f"frame {index} was not its own orientation turned by {weight:.3f} of the "
                f"seam offset; worst element differs by {float(np.abs(actual - expected).max()):.2e}"
            )


class TestAUniformTurnStaysUniformThroughTheTransition:
    """The rotational reading of "the motion keeps its own velocity"."""

    def test_the_yaw_rate_is_constant_across_the_eased_frames(self) -> None:
        """A constant turn plus a constant offset decay is a constant turn."""
        count = 5
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=-40.0)

        eased = _ease_onto_previous_pose(motion, pose, count)
        yaws = [_yaw_degrees(eased[i, 3:7]) for i in range(count + 1)]
        increments = np.diff(yaws)[:count]

        assert np.allclose(increments, increments[0], atol=0.05), (
            f"the eased yaw rate ramps instead of holding: {np.round(increments, 3).tolist()} deg/frame"
        )

    def test_the_yaw_rate_matches_the_linear_channels_own_rate(self) -> None:
        """Both channels advance by their own step plus one share of the offset."""
        count = 5
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=-40.0)
        total = _angle_between(pose[3:7], motion[0, 3:7])
        share = total / (count + 1)

        eased = _ease_onto_previous_pose(motion, pose, count)
        yaws = [_yaw_degrees(eased[i, 3:7]) for i in range(count + 1)]
        rotational = float(np.diff(yaws)[0])
        joint_steps = np.diff(eased[: count + 1, _ROOT], axis=0)

        assert abs(rotational) == pytest.approx(_TURN_DEG_PER_FRAME + share, abs=0.05)
        assert np.allclose(joint_steps, joint_steps[0], atol=1e-6), (
            f"premise: the linear channels advance uniformly, got {joint_steps}"
        )

    def test_a_wide_offset_never_reverses_the_turn(self) -> None:
        """Pulling each frame toward a distant orientation can invert the turn."""
        count = 5
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=200.0)

        eased = _ease_onto_previous_pose(motion, pose, count)
        yaws = [_yaw_degrees(eased[i, 3:7]) for i in range(count + 1)]
        increments = np.diff(yaws)[:count]

        assert (increments > 0.0).all(), (
            f"the root reversed direction inside the transition: {np.round(increments, 3).tolist()} deg/frame"
        )


# --------------------------------------------------------------------------
# Controls: behaviour that must NOT change.
# --------------------------------------------------------------------------
class TestTheSeamItselfIsUnchanged:
    """The first eased frame still lands where the ease has always put it."""

    @pytest.mark.parametrize("count", [1, 5, 8])
    def test_the_first_frame_is_one_share_short_of_the_pose_last_commanded(self, count: int) -> None:
        """Deliberately not pinned onto it: that would stall velocity for a frame."""
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=-40.0)
        total = _angle_between(pose[3:7], motion[0, 3:7])

        eased = _ease_onto_previous_pose(motion, pose, count)

        remaining = _angle_between(eased[0, 3:7], pose[3:7])
        assert remaining == pytest.approx(total / (count + 1), abs=0.05)

    def test_frames_after_the_window_are_the_sampler_s_own(self) -> None:
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=-40.0)

        eased = _ease_onto_previous_pose(motion, pose, 5)

        assert np.array_equal(eased[5:], motion[5:])


class TestTheLinearChannelsAndTheGuardsAreUnchanged:
    """The rest of the ease keeps working exactly as before."""

    def test_the_linear_channels_still_add_the_weighted_offset(self) -> None:
        count = 5
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=-40.0)
        offset = pose[_ROOT:] - motion[0, _ROOT:]

        eased = _ease_onto_previous_pose(motion, pose, count)

        for index, weight in enumerate(_weights(count)):
            expected = motion[index, _ROOT:] + weight * offset
            assert np.allclose(eased[index, _ROOT:], expected, atol=1e-6)

    def test_a_motion_whose_root_never_rotates_is_untouched(self) -> None:
        """The shape the existing transition suite drives: identity every frame."""
        motion = _turning_motion(12, deg_per_frame=0.0)
        pose = _previous_pose(motion, yaw_offset=0.0)

        eased = _ease_onto_previous_pose(motion, pose, 5)

        assert np.allclose(eased[:, 3:7], motion[:, 3:7], atol=1e-7)

    @pytest.mark.parametrize("frames", [0, -3])
    def test_a_non_positive_window_returns_the_motion_itself(self, frames: int) -> None:
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=-40.0)

        assert _ease_onto_previous_pose(motion, pose, frames) is motion

    def test_the_sampled_motion_is_not_modified_in_place(self) -> None:
        motion = _turning_motion(12)
        before = motion.copy()
        pose = _previous_pose(motion, yaw_offset=-40.0)

        _ease_onto_previous_pose(motion, pose, 5)

        assert np.array_equal(motion, before)

    def test_a_start_orientation_that_is_not_unit_names_the_same_rotation(self) -> None:
        """The ease normalises what it is handed, so a scale must not change it."""
        motion = _turning_motion(12)
        scaled = motion.copy()
        scaled[0, 3:7] = motion[0, 3:7] * 1.7
        pose = _previous_pose(motion, yaw_offset=-40.0)

        reference = _ease_onto_previous_pose(motion, pose, 5)
        scaled_result = _ease_onto_previous_pose(scaled, pose, 5)

        for index in range(5):
            deviation = _angle_between(scaled_result[index, 3:7], reference[index, 3:7])
            assert deviation < 1e-3, f"frame {index} moved {deviation:.4f} deg when the start pose was rescaled"

    def test_every_eased_frame_is_still_a_unit_quaternion(self) -> None:
        motion = _turning_motion(12)
        pose = _previous_pose(motion, yaw_offset=200.0)

        eased = _ease_onto_previous_pose(motion, pose, 5)

        assert np.allclose(np.linalg.norm(eased[:, 3:7], axis=1), 1.0, atol=1e-6)


class TestTheEmittedActionIsUntouched:
    """The policy emits joint targets only, and this change does not alter them."""

    def test_a_prompt_change_commands_the_weighted_joint_ease(self) -> None:
        """Every commanded value is the closed form of the linear ease."""
        count = 5
        centres = {"turn left": 0.10, "turn right": -1.40}

        class _TurningAgent:
            """Two motions with turning roots, so the root channel is live."""

            def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
                out = np.zeros((num_frames, _ROOT + _NUM_JOINTS), dtype=np.float32)
                for index in range(num_frames):
                    out[index, 3:7] = _quat_about_z(_TURN_DEG_PER_FRAME * index)
                out[:, _ROOT:] = centres[prompt] + 0.02 * np.arange(num_frames, dtype=np.float32)[:, None]
                return out

        policy = KimodoPolicy(
            config=KimodoConfig(num_frames=40, native_fps=30, tracker_fps=30, transition_frames=count),
            motion_agent=_TurningAgent(),
        )
        policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))
        segment = 8
        prompts = ["turn left"] * segment + ["turn right"] * segment
        # One tick per prompt: the frame cursor advances once per call, so each
        # action dict has to be taken once and then read across the joints.
        ticks = [asyncio.run(policy.get_actions({}, prompt))[0] for prompt in prompts]
        commanded = np.asarray(
            [[action[name] for name in KIMODO_G1_JOINTS] for action in ticks],
            dtype=np.float64,
        )

        second = _TurningAgent().sample("turn right", 40, 100, 7.5, None)
        last_commanded = commanded[segment - 1][0]
        offset = last_commanded - float(second[0, _ROOT])
        for index, weight in enumerate(_weights(count)):
            expected = float(second[index, _ROOT]) + weight * offset
            assert commanded[segment + index][0] == pytest.approx(expected, abs=1e-5), (
                f"eased tick {index} commanded {commanded[segment + index][0]:.6f} rad where the "
                f"linear ease gives {expected:.6f} rad"
            )
