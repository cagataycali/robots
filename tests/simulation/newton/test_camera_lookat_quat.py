"""Camera look-at quaternion math for the Newton backend.

These exercise the pure-math camera-orientation helpers
(``_look_at_quat`` / ``_quat_from_matrix``) directly. They need neither the
optional ``newton``/``warp`` packages nor a GPU, since the helpers operate only
on NumPy arrays, so they run in every environment.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from strands_robots.simulation.newton.simulation import (
    NewtonSimEngine,
    _short_joint_name,
)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Call the instance method without constructing a full engine.

    ``_look_at_quat`` only reads ``self`` to reach the ``_quat_from_matrix``
    static helper, so a tiny stand-in carrying that attribute is enough and
    avoids importing newton/warp (or touching a GPU).
    """
    stub = types.SimpleNamespace(_quat_from_matrix=NewtonSimEngine._quat_from_matrix)
    return NewtonSimEngine._look_at_quat(stub, eye, target, up)


def _quat_to_matrix(q):
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestShortJointName:
    def test_strips_hierarchical_path(self):
        assert _short_joint_name("so_arm100/worldbody/Base/Rotation") == "Rotation"

    def test_plain_name_unchanged(self):
        assert _short_joint_name("Jaw") == "Jaw"


class TestQuatFromMatrix:
    def test_identity_is_unit_quaternion(self):
        q = NewtonSimEngine._quat_from_matrix(np.eye(3))
        assert q == pytest.approx((0.0, 0.0, 0.0, 1.0))

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_180_degree_rotations_round_trip(self, axis):
        # A 180-degree rotation drives the trace negative, exercising each of
        # the three "largest diagonal" branches of the conversion.
        r = -np.eye(3)
        r[axis, axis] = 1.0
        q = NewtonSimEngine._quat_from_matrix(r)
        assert np.all(np.isfinite(q))
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert np.allclose(_quat_to_matrix(q), r, atol=1e-6)


class TestLookAtQuat:
    def test_oblique_view_points_camera_at_target(self):
        eye, target = (0.6, 0.6, 0.5), (0.0, 0.0, 0.15)
        q = _look_at(eye, target)
        assert np.all(np.isfinite(q))
        assert np.linalg.norm(q) == pytest.approx(1.0)
        # OpenGL convention: the camera looks down its local -Z, so -Z in world
        # space must equal the normalised eye->target direction.
        view = np.asarray(target) - np.asarray(eye)
        view /= np.linalg.norm(view)
        cam_neg_z = -_quat_to_matrix(q)[:, 2]
        assert np.allclose(cam_neg_z, view, atol=1e-6)

    def test_top_down_view_with_parallel_up_is_finite(self):
        # Regression: a camera directly above its target with the default
        # world-up has a view axis parallel to ``up``; cross(up, z) collapses
        # to ~0 and the old code normalised it into an all-NaN quaternion.
        q = _look_at((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        assert np.all(np.isfinite(q)), "top-down look-at must not produce NaNs"
        assert np.linalg.norm(q) == pytest.approx(1.0)
        cam_neg_z = -_quat_to_matrix(q)[:, 2]
        assert np.allclose(cam_neg_z, (0.0, 0.0, -1.0), atol=1e-6)

    def test_bottom_up_view_with_parallel_up_is_finite(self):
        q = _look_at((0.0, 0.0, -1.0), (0.0, 0.0, 0.0))
        assert np.all(np.isfinite(q))
        cam_neg_z = -_quat_to_matrix(q)[:, 2]
        assert np.allclose(cam_neg_z, (0.0, 0.0, 1.0), atol=1e-6)

    def test_coincident_eye_and_target_raises(self):
        with pytest.raises(ValueError, match="coincide"):
            _look_at((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    def test_explicit_parallel_up_falls_back(self):
        # Even when the caller explicitly passes an up vector parallel to the
        # view axis, the basis stays well-defined.
        q = _look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
        assert np.all(np.isfinite(q))
        assert np.linalg.norm(q) == pytest.approx(1.0)
