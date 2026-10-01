"""``base_ang_vel`` is reported in the BODY frame, as the schema and the siblings do.

The observation schema reports a floating base as ``base_pos`` / ``base_quat`` /
``base_lin_vel`` / ``base_ang_vel``. Linear velocity is world-frame on all three
backends; **angular velocity is body-frame** - the IMU-gyro convention a locomotion
policy is trained against. MuJoCo's free-joint angular velocity is already local;
Newton and Isaac rotate theirs through the one helper pinned in
``tests/simulation/test_world_to_body_rotation.py``.

Isaac's ``articulation.get_angular_velocity()`` returns the **world** frame, and it
was emitted unrotated. So of the four base channels, this was the one whose numbers
silently disagreed with the other two backends - which falsifies the parity claim
the whole floating-base feature rests on ("observation mappings transfer unchanged
between backends").

It disagrees in the way hardest to notice, and that is what these tests are built
around: **for an upright, un-yawed base the two frames coincide**. A robot standing
still reads correct, a robot spinning about world Z reads correct, and the error
appears only once the base tilts or yaws - which is exactly when a locomotion
policy is depending on the signal. A test that only stood a robot up would have
passed on the broken code.

Nothing here needs Isaac Sim: the articulation is a stand-in returning a chosen
pose and twist.
"""

from __future__ import annotations

import math
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from tests.simulation._isaac_engine import isaac_engine


def _quat_about_axis(axis: tuple[float, float, float], angle: float) -> list[float]:
    """A (w,x,y,z) quaternion for ``angle`` radians about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), *(float(v) for v in a * s)]


class _Articulation:
    dof_names = ["j0"]

    def __init__(self, quat: list[float], ang_world: list[float]) -> None:
        self._quat = quat
        self._ang = ang_world

    def initialize(self, *a: Any, **k: Any) -> None:
        return None

    def get_joint_positions(self) -> Any:
        return np.zeros(1, dtype=np.float64)

    def get_joint_velocities(self) -> Any:
        return np.zeros(1, dtype=np.float64)

    def get_world_pose(self) -> Any:
        return np.array([0.0, 0.0, 0.9]), np.asarray(self._quat, dtype=np.float64)

    def get_linear_velocity(self) -> Any:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def get_angular_velocity(self) -> Any:
        return np.asarray(self._ang, dtype=np.float64)


def _observe(quat: list[float], ang_world: list[float]) -> dict[str, Any]:
    engine = isaac_engine()
    engine._world_created = True
    engine._world = types.SimpleNamespace()
    engine._obs_noise = {}
    engine._robots = {
        "bot": types.SimpleNamespace(  # type: ignore[dict-item]
            name="bot",
            joint_names=["j0"],
            articulation=_Articulation(quat, ang_world),
            fixed_base=False,
            usd_to_urdf_joint_names=None,
        )
    }
    return engine.get_observation("bot", skip_images=True)


class TestTheAngularVelocityIsRotatedIntoTheBodyFrame:
    def test_a_yawed_base_reports_the_body_frame_value(self) -> None:
        """90 degrees about Z: a world-frame X spin is a body-frame -Y spin."""
        quat = _quat_about_axis((0.0, 0.0, 1.0), math.pi / 2)

        obs = _observe(quat, [1.0, 0.0, 0.0])

        assert obs["base_ang_vel"] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)

    def test_a_pitched_base_reports_the_body_frame_value(self) -> None:
        """+90 degrees about Y: a world-frame +Z spin is a body-frame -X spin.

        The sign is the part worth stating. For R_y(+90) = [[0,0,1],[0,1,0],[-1,0,0]],
        body-frame is R^T @ v, so [0,0,1] becomes [-1,0,0]. Cross-checked against
        scipy's ``Rotation.from_euler("y", 90)`` and a first-principles matrix,
        both of which give [-1,0,0] - this expectation was written +1 first and the
        implementation was right.
        """
        quat = _quat_about_axis((0.0, 1.0, 0.0), math.pi / 2)

        obs = _observe(quat, [0.0, 0.0, 1.0])

        assert obs["base_ang_vel"] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)

    def test_an_upside_down_base_flips_the_sign(self) -> None:
        """180 degrees about X: Y and Z invert. The largest divergence there is."""
        quat = _quat_about_axis((1.0, 0.0, 0.0), math.pi)

        obs = _observe(quat, [0.0, 1.0, 2.0])

        assert obs["base_ang_vel"] == pytest.approx([0.0, -1.0, -2.0], abs=1e-9)

    def test_it_differs_from_the_raw_world_value(self) -> None:
        """Stated as the regression directly: passing Isaac's value through
        unrotated is what this refuses to do."""
        quat = _quat_about_axis((0.0, 0.0, 1.0), math.pi / 2)
        world = [1.0, 0.0, 0.0]

        obs = _observe(quat, world)

        assert obs["base_ang_vel"] != pytest.approx(world, abs=1e-6)


class TestTheCoincidingCasesAreWhyThisWasInvisible:
    """An upright, un-yawed base reads identically either way.

    These pass on the broken code, and that is the point of naming them: a test
    that stood a robot up and checked the gyro would have found nothing. They are
    kept because they are also the cases that must not regress.
    """

    def test_an_upright_base_is_unchanged(self) -> None:
        obs = _observe([1.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3])

        assert obs["base_ang_vel"] == pytest.approx([0.1, -0.2, 0.3], abs=1e-9)

    def test_a_spin_about_the_yaw_axis_of_an_upright_base_is_unchanged(self) -> None:
        """A yawing-but-upright base still agrees on the Z component."""
        quat = _quat_about_axis((0.0, 0.0, 1.0), 0.7)

        obs = _observe(quat, [0.0, 0.0, 1.5])

        assert obs["base_ang_vel"] == pytest.approx([0.0, 0.0, 1.5], abs=1e-9)


class TestTheOtherThreeChannelsAreUntouched:
    """Only the angular channel rotates. Rotating position or linear velocity
    would be a new bug in the opposite direction."""

    def test_base_pos_stays_world_frame(self) -> None:
        obs = _observe(_quat_about_axis((0.0, 0.0, 1.0), math.pi / 2), [1.0, 0.0, 0.0])

        assert obs["base_pos"] == pytest.approx([0.0, 0.0, 0.9], abs=1e-9)

    def test_base_lin_vel_stays_world_frame(self) -> None:
        """World on all three backends - Newton's own comment says so."""
        obs = _observe(_quat_about_axis((0.0, 0.0, 1.0), math.pi / 2), [1.0, 0.0, 0.0])

        assert obs["base_lin_vel"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)

    def test_base_quat_is_reported_verbatim(self) -> None:
        quat = _quat_about_axis((0.0, 1.0, 0.0), 0.4)

        obs = _observe(quat, [0.0, 0.0, 0.0])

        assert obs["base_quat"] == pytest.approx(quat, abs=1e-9)


class TestAFixedBaseStillReportsNothing:
    """The control from the original feature: these keys are for a base that moves."""

    def test_no_base_keys_for_a_welded_root(self) -> None:
        engine = isaac_engine()
        engine._world_created = True
        engine._world = types.SimpleNamespace()
        engine._obs_noise = {}
        engine._robots = {
            "arm": types.SimpleNamespace(  # type: ignore[dict-item]
                name="arm",
                joint_names=["j0"],
                articulation=_Articulation([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                fixed_base=True,
                usd_to_urdf_joint_names=None,
            )
        }

        obs = engine.get_observation("arm", skip_images=True)

        assert [k for k in obs if k.startswith("base_")] == []
