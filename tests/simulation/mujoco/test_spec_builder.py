"""Unit + parity tests for ``SpecBuilder`` - the MjSpec-based MJCF builder.

These tests exercise the new code path in isolation (not through the
``STRANDS_SIM_USE_MJSPEC`` flag) so they document the SpecBuilder contract
regardless of what ``_compile_world`` is configured to use today.

See IDEA.md for the full refactor plan and GH #121 for the tracking issue.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from strands_robots.simulation.models import (  # noqa: E402
    SimCamera,
    SimObject,
    SimWorld,
)
from strands_robots.simulation.mujoco.mjcf_builder import MJCFBuilder  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import (  # noqa: E402
    SpecBuilder,
    _geom_type,
    _normalize_size,
    _target_quat,
)

# -----------------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------------


class TestGeomType:
    def test_known_shapes_map_to_enum(self):
        assert _geom_type("box") == mujoco.mjtGeom.mjGEOM_BOX
        assert _geom_type("sphere") == mujoco.mjtGeom.mjGEOM_SPHERE
        assert _geom_type("cylinder") == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert _geom_type("capsule") == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert _geom_type("mesh") == mujoco.mjtGeom.mjGEOM_MESH
        assert _geom_type("plane") == mujoco.mjtGeom.mjGEOM_PLANE

    def test_ellipsoid_is_now_supported(self):
        """A bonus enabled by the SpecBuilder refactor - not in legacy."""
        assert _geom_type("ellipsoid") == mujoco.mjtGeom.mjGEOM_ELLIPSOID

    def test_unknown_shape_raises_with_helpful_list(self):
        with pytest.raises(ValueError, match="Unsupported shape"):
            _geom_type("hyperboloid")


class TestNormalizeSize:
    def test_box_halves_full_extents(self):
        assert _normalize_size("box", [0.2, 0.4, 0.6]) == [0.1, 0.2, 0.3]

    def test_sphere_uses_half_of_first(self):
        """SimObject convention is that size[0] is the diameter; we halve for radius."""
        assert _normalize_size("sphere", [0.1])[0] == pytest.approx(0.05)

    def test_cylinder_radius_and_half_height(self):
        out = _normalize_size("cylinder", [0.1, 0, 0.4])
        assert out[0] == pytest.approx(0.05)  # radius
        assert out[1] == pytest.approx(0.2)  # half-height

    def test_capsule_same_as_cylinder(self):
        assert _normalize_size("capsule", [0.1, 0, 0.4]) == _normalize_size("cylinder", [0.1, 0, 0.4])

    def test_plane_size_defaults_to_equal_xy(self):
        out = _normalize_size("plane", [2.0])
        assert out[0] == 2.0
        assert out[1] == 2.0

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Cannot normalize size"):
            _normalize_size("hyperboloid", [1.0])


class TestTargetQuat:
    def test_returns_none_for_degenerate(self):
        # Position == target.
        assert _target_quat([1, 2, 3], [1, 2, 3]) is None

    def test_returns_quat_for_valid_look_at(self):
        quat = _target_quat([1, 0, 0.5], [0, 0, 0])
        assert quat is not None
        assert len(quat) == 4
        # Normalised quaternion.
        norm = (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_forward_parallel_to_up_returns_none(self):
        # Looking straight down - forward is (0,0,-1), up is (0,0,1), parallel.
        assert _target_quat([0, 0, 1], [0, 0, 0]) is None


# -----------------------------------------------------------------------------
# SpecBuilder parity tests - identical compile results vs legacy MJCFBuilder
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_world() -> SimWorld:
    w = SimWorld()
    w.objects["cube"] = SimObject(
        name="cube",
        shape="box",
        position=[0, 0, 0.1],
        size=[0.1, 0.1, 0.1],
        color=[0.5, 0.5, 0.5, 1],
        is_static=False,
        mass=0.2,
    )
    w.objects["ball"] = SimObject(
        name="ball",
        shape="sphere",
        position=[0.5, 0, 0.1],
        size=[0.05],
        color=[1, 0, 0, 1],
        is_static=False,
        mass=0.1,
    )
    w.cameras["front"] = SimCamera(
        name="front",
        position=[1, 0, 0.5],
        target=[0, 0, 0],
        fov=60,
        width=640,
        height=480,
    )
    return w


class TestSpecBuilderParity:
    """Both code paths should produce MuJoCo models with the same structure."""

    def test_empty_world_compiles(self):
        w = SimWorld()
        spec = SpecBuilder.build(w)
        m = spec.compile()
        # empty world still has the ground + 2 lights + world-body = 1 body (the world itself).
        assert m.nbody >= 1

    def test_structural_parity_with_legacy_builder(self, sample_world: SimWorld):
        """Build the same world via both paths; compiled MjModel dimensions match."""
        m_legacy = mujoco.MjModel.from_xml_string(MJCFBuilder.build_objects_only(sample_world))
        m_spec = SpecBuilder.build(sample_world).compile()

        for attr in ("nbody", "ngeom", "ncam", "nu", "njnt", "nq", "nv"):
            legacy = getattr(m_legacy, attr)
            spec = getattr(m_spec, attr)
            assert legacy == spec, f"{attr} mismatch: legacy={legacy}, spec={spec}"

    def test_gravity_and_timestep_match(self, sample_world: SimWorld):
        sample_world.gravity = [0.0, 0.0, -5.0]
        sample_world.timestep = 0.004
        m_legacy = mujoco.MjModel.from_xml_string(MJCFBuilder.build_objects_only(sample_world))
        m_spec = SpecBuilder.build(sample_world).compile()
        assert m_legacy.opt.timestep == pytest.approx(m_spec.opt.timestep)
        assert np.allclose(m_legacy.opt.gravity, m_spec.opt.gravity)

    def test_body_positions_match(self, sample_world: SimWorld):
        m_legacy = mujoco.MjModel.from_xml_string(MJCFBuilder.build_objects_only(sample_world))
        m_spec = SpecBuilder.build(sample_world).compile()
        for name in ("cube", "ball"):
            a_id = mujoco.mj_name2id(m_legacy, mujoco.mjtObj.mjOBJ_BODY, name)
            b_id = mujoco.mj_name2id(m_spec, mujoco.mjtObj.mjOBJ_BODY, name)
            assert a_id >= 0 and b_id >= 0
            assert np.allclose(m_legacy.body_pos[a_id], m_spec.body_pos[b_id])

    def test_body_masses_match(self, sample_world: SimWorld):
        m_legacy = mujoco.MjModel.from_xml_string(MJCFBuilder.build_objects_only(sample_world))
        m_spec = SpecBuilder.build(sample_world).compile()
        for name in ("cube", "ball"):
            a_id = mujoco.mj_name2id(m_legacy, mujoco.mjtObj.mjOBJ_BODY, name)
            b_id = mujoco.mj_name2id(m_spec, mujoco.mjtObj.mjOBJ_BODY, name)
            assert m_legacy.body_mass[a_id] == pytest.approx(m_spec.body_mass[b_id], abs=1e-6)

    def test_camera_orientation_matches_within_float_precision(self):
        """The camera rotation produced by the two orientation paths
        (xyaxes vs quat-from-mju_mat2Quat) must be numerically equivalent.
        """
        test_cases = [
            ("at_origin", [0.0, 0.0, 0.0]),
            ("off_z", [0.5, 0.2, 0.0]),
            ("up_high", [1.0, 0.0, 0.5]),
        ]
        for case_name, target in test_cases:
            w = SimWorld()
            w.cameras["cam"] = SimCamera(name="cam", position=[1, 0, 0.5], target=target, fov=60, width=640, height=480)
            m_legacy = mujoco.MjModel.from_xml_string(MJCFBuilder.build_objects_only(w))
            m_spec = SpecBuilder.build(w).compile()

            rot_legacy = m_legacy.cam_mat0[0].reshape(3, 3)
            rot_spec = m_spec.cam_mat0[0].reshape(3, 3)
            assert np.max(np.abs(rot_legacy - rot_spec)) < 1e-4, f"{case_name}: camera rotation drift"


class TestSpecBuilderEllipsoidBonus:
    """Ellipsoid is NEW - the legacy MJCFBuilder rejects it. This locks the
    extra-shape capability the refactor unlocks.
    """

    def test_ellipsoid_compiles_via_spec_builder(self):
        w = SimWorld()
        w.objects["egg"] = SimObject(
            name="egg",
            shape="ellipsoid",
            position=[0, 0, 0.1],
            size=[0.1, 0.06, 0.04],
            color=[1, 0.5, 0, 1],
            is_static=False,
            mass=0.05,
        )
        spec = SpecBuilder.build(w)
        m = spec.compile()
        egg_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "egg")
        assert egg_id >= 0
