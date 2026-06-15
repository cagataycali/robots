"""Unit tests for the Cosmos 3 action de-normalization + EE-pose decode.

Pure NumPy (no GPU, no mujoco, no model): these pin the two honest steps that
turn the diffusers backend's raw ``[-1, 1]`` action into a physical absolute
end-effector trajectory - the de-normalization quantiles and the relative ->
absolute pose integration. They are the regression guard for the earlier defect
where the normalized chunk was fed straight into MuJoCo joint actuators
(physically meaningless: values landed arbitrarily inside/outside joint limits).
"""

import numpy as np
import pytest

from strands_robots.policies.cosmos3.action_decode import (
    _rot6d_to_matrix,
    decode_pose_trajectory,
    denormalize_quantile,
    load_action_stats,
)


def test_bundled_droid_stats_load_with_expected_width():
    stats = load_action_stats("droid_lerobot")
    assert set(stats) == {"q01", "q99"}
    # DROID raw unified action = 10D (3 trans + 6 rot + 1 grasp).
    assert stats["q01"].shape == (10,)
    assert stats["q99"].shape == (10,)
    # Translation deltas are ~+/-1.5 cm/step (why raw [-1,1] into joints is wrong).
    assert np.all(np.abs(stats["q01"][:3]) < 0.02)
    assert np.all(np.abs(stats["q99"][:3]) < 0.02)


def test_bundled_bridge_stats_load():
    stats = load_action_stats("bridge_orig_lerobot")
    assert stats["q01"].shape == (10,)


def test_missing_stats_raises_actionable_error():
    with pytest.raises(FileNotFoundError, match="No bundled Cosmos 3 action stats"):
        load_action_stats("not_a_real_domain")


def test_denormalize_quantile_matches_closed_form():
    """De-normalization inverts quantile normalization exactly:
    0.5 * (a + 1) * (q99 - q01) + q01 (mirrors cosmos_framework)."""
    q01 = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    q99 = np.array([1.0, 4.0, 6.0], dtype=np.float32)
    # a = -1 -> q01; a = +1 -> q99; a = 0 -> midpoint.
    out = denormalize_quantile(np.array([[-1.0, 1.0, 0.0]], dtype=np.float32), q01, q99)
    np.testing.assert_allclose(out[0], [-1.0, 4.0, 4.0], atol=1e-6)


def test_denormalize_round_trips_normalization():
    """denorm(norm(x)) == x for the quantile transform."""
    rng = np.random.default_rng(0)
    q01 = np.array([-0.0142, -0.0134, -0.0152], dtype=np.float32)
    q99 = np.array([0.0145, 0.0115, 0.0145], dtype=np.float32)
    x = rng.uniform(q01, q99, (5, 3)).astype(np.float32)
    norm = 2.0 * (x - q01) / (q99 - q01) - 1.0
    back = denormalize_quantile(norm, q01, q99)
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_denormalize_width_mismatch_raises():
    with pytest.raises(ValueError, match="does not match stats width"):
        denormalize_quantile(np.zeros((2, 4), dtype=np.float32), np.zeros(3), np.zeros(3))


def test_rot6d_to_matrix_is_orthonormal_for_identity():
    # rot6d identity = first two columns of I.
    rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    mat = _rot6d_to_matrix(rot6d)
    np.testing.assert_allclose(mat, np.eye(3), atol=1e-6)


def test_decode_pose_trajectory_anchors_at_initial_pose():
    """Index 0 of the absolute trajectory is the initial pose; a zero-translation
    identity-rotation delta keeps the EE put."""
    init = np.eye(4, dtype=np.float32)
    init[:3, 3] = [0.5, 0.1, 0.6]
    # 4 steps of identity rotation (rot6d=[1,0,0,0,1,0]) + zero translation.
    chunk = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32), (4, 1))
    poses = decode_pose_trajectory(chunk, init, rotation_dim=6)
    assert poses.shape == (5, 4, 4)
    np.testing.assert_allclose(poses[0], init, atol=1e-6)
    # No motion -> every pose equals the initial pose.
    for p in poses:
        np.testing.assert_allclose(p[:3, 3], init[:3, 3], atol=1e-6)


def test_decode_pose_trajectory_integrates_translation():
    """Per-step framewise translation accumulates along the trajectory."""
    init = np.eye(4, dtype=np.float32)
    # delta = +1cm x each step, identity rotation.
    chunk = np.tile(np.array([0.01, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32), (3, 1))
    poses = decode_pose_trajectory(chunk, init, rotation_dim=6)
    # backward_framewise with identity rotation -> straight accumulation.
    np.testing.assert_allclose(poses[1][:3, 3], [0.01, 0, 0], atol=1e-6)
    np.testing.assert_allclose(poses[2][:3, 3], [0.02, 0, 0], atol=1e-6)
    np.testing.assert_allclose(poses[3][:3, 3], [0.03, 0, 0], atol=1e-6)


def test_decode_pose_trajectory_rejects_bad_shape():
    with pytest.raises(ValueError, match="pose_chunk must be"):
        decode_pose_trajectory(np.zeros((4, 8), dtype=np.float32), np.eye(4), rotation_dim=6)
    with pytest.raises(ValueError, match="initial_pose must be"):
        decode_pose_trajectory(np.zeros((4, 9), dtype=np.float32), np.eye(3), rotation_dim=6)
