"""One world->body rotation, ``R(q)^T @ vec`` for a ``(w, x, y, z)`` quaternion.

``base_ang_vel`` is reported in the BODY frame on every backend - the IMU-gyro
convention locomotion and WBC controllers are trained against. MuJoCo's free-joint
angular velocity is already local; Newton and Isaac read it in the world frame and
rotate it through :func:`strands_robots.simulation.predicates._quat_rotate_inverse_wxyz`,
the same helper the ``base_velocity`` reward term uses to body-frame the linear
velocity. One implementation means the backends cannot drift apart, so the numeric
contract is pinned once here, against a rotation matrix built from first principles.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.simulation import predicates
from strands_robots.simulation.predicates import _quat_rotate_inverse_wxyz


def _about(axis: tuple[float, float, float], angle: float) -> list[float]:
    n = math.sqrt(sum(a * a for a in axis))
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), *(a / n * s for a in axis)]


def _reference(quat: list[float], vec: list[float]) -> list[float]:
    """Transpose of the rotation matrix of the normalised quaternion, applied to vec."""
    n = math.sqrt(sum(c * c for c in quat))
    w, x, y, z = (c / n for c in quat)
    r = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return [sum(r[k][i] * vec[k] for k in range(3)) for i in range(3)]


@pytest.mark.parametrize(
    ("quat", "vec", "expected"),
    [
        pytest.param([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], id="identity"),
        pytest.param(_about((0, 0, 1), math.pi / 2), [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], id="yaw90"),
        pytest.param(_about((0, 0, 1), math.pi / 2), [1.1, 2.2, 3.3], [2.2, -1.1, 3.3], id="yaw90-general"),
        # R_y(+90) = [[0,0,1],[0,1,0],[-1,0,0]]; R^T @ [0,0,1] = [-1,0,0].
        pytest.param(_about((0, 1, 0), math.pi / 2), [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], id="pitch90"),
        pytest.param(_about((1, 0, 0), math.pi), [0.0, 1.0, 2.0], [0.0, -1.0, -2.0], id="upside-down"),
        pytest.param([0.5, 0.5, -0.5, 0.5], [0.3, -1.2, 4.5], None, id="composite"),
        pytest.param(_about((0.3, -0.5, 0.8), 1.1), [0.4, 1.2, -0.7], None, id="oblique"),
        # A scaled quaternion encodes the same rotation, so a raw free-joint quat works.
        pytest.param([2.0, 2.0, -2.0, 2.0], [1.0, -2.0, 0.5], None, id="unnormalised"),
        # No rotation can be read from a ~zero norm: the vector comes back unchanged.
        pytest.param([0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0], id="zero-norm"),
        pytest.param([1e-12, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0], id="sub-threshold-norm"),
    ],
)
def test_the_rotation_matches_first_principles(quat, vec, expected):
    got = _quat_rotate_inverse_wxyz(quat, vec)

    want = expected if expected is not None else _reference(quat, vec)
    assert got == pytest.approx(want, abs=1e-9)
    if expected is None:
        # A frame change cannot alter how fast the base is moving.
        assert math.dist(got, [0, 0, 0]) == pytest.approx(math.dist(vec, [0, 0, 0]), abs=1e-9)


@pytest.mark.parametrize(
    "module", ["strands_robots.simulation.newton.simulation", "strands_robots.simulation.isaac.simulation"]
)
def test_every_backend_rotates_through_the_one_helper(module):
    """A backend-local copy of the rotation is how two backends' base_ang_vel drift."""
    backend = pytest.importorskip(module)

    assert backend._quat_rotate_inverse_wxyz is predicates._quat_rotate_inverse_wxyz
