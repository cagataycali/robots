"""MotionClip: construction validation and phase-based interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers.motion import MotionClip


def _ramp_clip(num_frames: int = 4, num_joints: int = 2, fps: float = 10.0) -> MotionClip:
    pos = np.linspace(0.0, 1.0, num_frames)[:, None] * np.ones((1, num_joints))
    return MotionClip.from_arrays(pos, fps=fps)


def test_metadata():
    clip = _ramp_clip(num_frames=5, num_joints=3, fps=25.0)
    assert clip.num_frames == 5
    assert clip.num_joints == 3
    assert clip.duration == pytest.approx(5 / 25.0)


def test_defaults_velocity_to_zeros():
    clip = _ramp_clip()
    _, vel, _ = clip.sample(0.0)
    np.testing.assert_array_equal(vel, np.zeros(clip.num_joints))


def test_sample_first_frame_exact():
    clip = _ramp_clip(num_frames=4, num_joints=2, fps=10.0)
    pos, _, phase = clip.sample(0.0, loop=True)
    np.testing.assert_allclose(pos, clip.frames_pos[0])
    assert phase == pytest.approx(0.0)


def test_sample_interpolates_between_frames():
    # 2 frames [0,0] -> [1,1] over 0.2s (fps=10 -> 2 frames). Halfway phase=0.5.
    pos = np.array([[0.0, 0.0], [1.0, 1.0]])
    clip = MotionClip.from_arrays(pos, fps=10.0)
    # duration = 2/10 = 0.2; t=0.05 -> phase 0.25 -> frame 0.5 -> halfway between f0 and f1
    p, _, phase = clip.sample(0.05, loop=True)
    assert phase == pytest.approx(0.25)
    np.testing.assert_allclose(p, [0.5, 0.5])


def test_loop_wraps_around():
    pos = np.array([[0.0], [1.0]])
    clip = MotionClip.from_arrays(pos, fps=10.0)
    # duration 0.2; t equal to one full period returns to start
    p0, _, ph0 = clip.sample(0.0, loop=True)
    p1, _, ph1 = clip.sample(clip.duration, loop=True)
    np.testing.assert_allclose(p0, p1)
    assert ph1 == pytest.approx(0.0)


def test_no_loop_clamps_to_final_frame():
    pos = np.array([[0.0], [1.0]])
    clip = MotionClip.from_arrays(pos, fps=10.0)
    p, _, phase = clip.sample(t=10.0, loop=False)  # far past the end
    np.testing.assert_allclose(p, [1.0])
    assert phase == pytest.approx(1.0)


def test_single_frame_clip_is_constant():
    clip = MotionClip.from_arrays([[0.3, -0.2]], fps=30.0)
    p, v, phase = clip.sample(5.0, loop=True)
    np.testing.assert_allclose(p, [0.3, -0.2])
    np.testing.assert_allclose(v, [0.0, 0.0])
    assert phase == 0.0


def test_rejects_non_2d_frames():
    with pytest.raises(ValueError, match="non-empty 2-D"):
        MotionClip.from_arrays([0.0, 1.0, 2.0])


def test_rejects_empty_frames():
    with pytest.raises(ValueError, match="non-empty 2-D"):
        MotionClip.from_arrays(np.zeros((0, 3)))


def test_rejects_mismatched_velocity_shape():
    with pytest.raises(ValueError, match="frames_vel shape"):
        MotionClip.from_arrays(np.zeros((4, 2)), frames_vel=np.zeros((4, 3)))


def test_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        MotionClip.from_arrays(np.zeros((4, 2)), fps=0.0)
