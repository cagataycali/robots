"""Unit tests for ``LiberoOffScreenRenderEngine.get_observation`` image rotation.

PR #169 changed the engine's image rotation from a baked-in 180°
``[::-1, ::-1]`` to V-flip ``[::-1, :]``. The change moves the second
flip into the policy's shared ``_apply_image_rotation_180_inplace``
helper (which now runs in BOTH local and service modes), so the engine
just produces OpenGL framebuffer convention like every other observation
producer.

Why this matters: pre-#169 the engine's 180° rotation collided with the
policy's service-mode ``image_rotation_180=True`` flag for ``libero_panda``,
double-rotating images back to OpenGL convention → out-of-distribution →
``success_rate=0`` on the docker GR00T server. Round 43's working LOCAL
eval got training-convention images (engine 180°, no policy rotation);
fixed by #169 to deliver OpenGL on the wire and let the policy rotate
once consistently.

These tests don't require running the real LIBERO env — they construct
the engine, inject a synthetic ``_latest_obs`` mimicking what
``OffScreenRenderEnv.reset/step`` returns, and check the
``get_observation`` output. Heavy integration coverage lives in
``tests_integ/benchmarks/libero/`` (separate suite, gated on the
``libero`` package install).
"""

from __future__ import annotations

import numpy as np
import pytest

# Lightweight import-skip: the engine module imports ``mujoco`` at the
# top of its lifecycle, but ``get_observation`` itself doesn't touch
# mujoco. We construct the engine without ever calling ``setup_libero_task``,
# so mujoco/libero/robosuite are not needed.
mujoco = pytest.importorskip("mujoco", reason="LiberoOffScreenRenderEngine module needs mujoco at construction")

from strands_robots.simulation.libero_offscreen_render.engine import (  # noqa: E402
    LiberoOffScreenRenderEngine,
)


def _make_engine_with_synthetic_obs(raw_obs: dict) -> LiberoOffScreenRenderEngine:
    """Construct an engine and inject ``raw_obs`` into ``_latest_obs``.

    Bypasses ``setup_libero_task`` (which would build a real OffScreenRenderEnv
    and require the libero package). We only test ``get_observation``'s
    transformation logic, which reads from ``_latest_obs``.
    """
    engine = LiberoOffScreenRenderEngine()
    engine._latest_obs = raw_obs  # type: ignore[assignment]
    # Mark the engine as having an env so get_observation doesn't bail
    # out via the "env not initialized" check. We never actually call
    # any env method.
    engine._env = object()  # type: ignore[assignment]
    return engine


class TestImageRotation:
    """#169: engine produces OpenGL-convention (V-flipped) images, not 180°.

    The downstream policy's ``image_rotation_180`` flag (set on
    ``libero_panda``'s data_config) applies the second flip to convert
    OpenGL → training convention. Pre-#169 the engine baked the full 180°,
    causing service-mode double-rotation back to OpenGL.
    """

    def test_agentview_image_v_flipped_not_180(self):
        """Engine output's top-left == raw bottom-left (V-flip), NOT
        raw bottom-right (which would indicate a 180° rotation).

        Sentinel for the #169 contract: engine should produce OpenGL
        convention, leaving the second flip to the policy."""
        h, w = 16, 8
        raw_img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = raw_img[0, 0].copy()
        original_top_right = raw_img[0, w - 1].copy()
        original_bottom_left = raw_img[h - 1, 0].copy()
        original_bottom_right = raw_img[h - 1, w - 1].copy()

        engine = _make_engine_with_synthetic_obs({"agentview_image": raw_img.copy()})

        obs = engine.get_observation()
        out = obs["image"]
        assert out.shape == (h, w, 3)

        # V-flip: output top-left == raw BOTTOM-left (NOT raw bottom-right).
        np.testing.assert_array_equal(out[0, 0], original_bottom_left)
        np.testing.assert_array_equal(out[0, w - 1], original_bottom_right)
        np.testing.assert_array_equal(out[h - 1, 0], original_top_left)
        np.testing.assert_array_equal(out[h - 1, w - 1], original_top_right)

        # Sentinel: must NOT be 180° rotation. If this assertion fires,
        # the engine is back to baking the full rotation, which collides
        # with the policy's image_rotation_180 in service mode (#169 bug).
        assert not np.array_equal(out[0, 0], original_bottom_right), (
            "engine output looks like a 180° rotation, not V-flip — this re-introduces the #169 double-rotation bug"
        )

    def test_wrist_image_also_v_flipped(self):
        """Both ``image`` and ``wrist_image`` get the same V-flip
        transformation (consistent with adapter contract)."""
        h, w = 8, 16
        raw_img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = raw_img[0, 0].copy()

        engine = _make_engine_with_synthetic_obs(
            {
                "agentview_image": raw_img.copy(),
                "robot0_eye_in_hand_image": raw_img.copy(),
            }
        )

        obs = engine.get_observation()
        out_wrist = obs["wrist_image"]
        # V-flip: output bottom-left == raw top-left.
        np.testing.assert_array_equal(out_wrist[h - 1, 0], original_top_left)

    def test_output_is_contiguous(self):
        """``np.ascontiguousarray`` materialises the flipped view as a
        fresh contiguous buffer. Required for downstream msgpack
        serialisation (service-mode policy round-trip)."""
        raw_img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        engine = _make_engine_with_synthetic_obs({"agentview_image": raw_img})

        obs = engine.get_observation()
        out = obs["image"]
        assert out.flags["C_CONTIGUOUS"], "rotated image must be C-contiguous"

    def test_skip_images_omits_rotation_work(self):
        """``skip_images=True`` short-circuits — no rotation, no copy."""
        raw_img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        engine = _make_engine_with_synthetic_obs({"agentview_image": raw_img})

        obs = engine.get_observation(skip_images=True)
        # No image keys returned at all when skipped.
        assert "image" not in obs
        assert "wrist_image" not in obs

    def test_no_image_in_raw_obs_skipped_silently(self):
        """If upstream raw obs is missing the camera key (e.g. a frame
        where rendering failed), get_observation skips silently rather
        than emitting a None or empty key — preserves the precondition
        that downstream consumers can assume well-formed image arrays
        when present."""
        engine = _make_engine_with_synthetic_obs(
            {
                "robot0_eef_pos": np.array([0.5, 0.0, 0.3]),
                # No agentview_image / robot0_eye_in_hand_image
            }
        )

        obs = engine.get_observation()
        assert "image" not in obs
        assert "wrist_image" not in obs


class TestStateExtraction:
    """Sanity that the rest of get_observation still produces the
    expected state schema for libero_panda. Not strictly part of #169
    but pinned here so the V-flip change doesn't accidentally drop
    state keys."""

    def test_eef_pose_state_extracted_correctly(self):
        """``robot0_eef_pos`` → ``x`` / ``y`` / ``z``,
        ``robot0_eef_quat`` (xyzw) → ``roll`` / ``pitch`` / ``yaw``.
        """
        engine = _make_engine_with_synthetic_obs(
            {
                "robot0_eef_pos": np.array([0.5, -0.1, 0.3]),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),  # identity (xyzw)
                "robot0_gripper_qpos": np.array([0.02, -0.02]),
            }
        )

        obs = engine.get_observation()
        assert obs["x"] == pytest.approx(0.5)
        assert obs["y"] == pytest.approx(-0.1)
        assert obs["z"] == pytest.approx(0.3)
        # Identity quat → all Euler angles ≈ 0.
        assert abs(obs["roll"]) < 1e-6
        assert abs(obs["pitch"]) < 1e-6
        assert abs(obs["yaw"]) < 1e-6
        # Gripper as 2-element list (matches LIBERO's
        # robot0_gripper_qpos shape).
        assert obs["gripper"] == [0.02, -0.02]
