"""Unit tests for MJCFBuilder helpers - pure functions, no MuJoCo round-trip.

Targets the previously uncovered branches in
strands_robots/simulation/mujoco/mjcf_builder.py.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.simulation.models import SimObject
from strands_robots.simulation.mujoco.mjcf_builder import (
    MJCFBuilder,
    _camera_xyaxes_from_target,
    _sanitize_name,
)

# -- _sanitize_name ----------------------------------------------------------


class TestSanitizeName:
    @pytest.mark.parametrize(
        "name",
        ["alice", "alice_1", "alice.bob", "arm-0", "_leading", "a", "a" * 128],
    )
    def test_valid_names_pass_through(self, name):
        assert _sanitize_name(name) == name

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            " ",
            "a b",
            "alice<script>",
            "a/b",
            "a'xss",
            'a"xss',
            "a" * 129,
        ],
    )
    def test_invalid_names_rejected(self, bad):
        with pytest.raises(ValueError, match="Invalid simulation name"):
            _sanitize_name(bad)


# -- _camera_xyaxes_from_target ----------------------------------------------


def _axes_from_str(s: str) -> tuple[list[float], list[float]]:
    vals = [float(x) for x in s.split()]
    assert len(vals) == 6
    return vals[:3], vals[3:]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestCameraXYAxes:
    def test_looks_along_negative_z_axis(self):
        # Camera at (0,0,1) looking at (0,0,0) - forward is -Z in world.
        s = _camera_xyaxes_from_target([0, 0, 1], [0, 0, 0])
        assert s is not None
        # Near-parallel up+forward case falls back to world-X as right.
        right, image_up = _axes_from_str(s)
        # Axes must be unit length.
        assert _norm(right) == pytest.approx(1.0, abs=1e-5)
        assert _norm(image_up) == pytest.approx(1.0, abs=1e-5)
        # Orthogonal (dot == 0).
        assert _dot(right, image_up) == pytest.approx(0.0, abs=1e-5)

    def test_standard_topdown(self):
        # Overhead camera at (0,0,2) looking down at origin.
        s = _camera_xyaxes_from_target([0, 0, 2], [0, 0, 0])
        assert s is not None
        right, image_up = _axes_from_str(s)
        assert _norm(right) == pytest.approx(1.0, abs=1e-5)
        assert _norm(image_up) == pytest.approx(1.0, abs=1e-5)

    def test_side_view(self):
        # Camera at (2, 0, 0) looking at origin - forward is -X.
        s = _camera_xyaxes_from_target([2, 0, 0], [0, 0, 0])
        assert s is not None
        right, image_up = _axes_from_str(s)
        # image_up should have a strong Z-component (pointing toward world +Z).
        assert image_up[2] > 0.5

    def test_degenerate_target_equals_position(self):
        """Zero-length forward vector must return None."""
        s = _camera_xyaxes_from_target([1, 1, 1], [1, 1, 1])
        assert s is None

    def test_degenerate_near_zero_distance(self):
        s = _camera_xyaxes_from_target([0, 0, 0], [1e-12, 0, 0])
        assert s is None

    def test_forward_parallel_to_up_uses_fallback(self):
        """When forward is parallel to the ``up`` axis (vertical camera), we
        fall back to world-X as the right axis. The returned string must still
        contain valid unit vectors."""
        s = _camera_xyaxes_from_target([0, 0, 1], [0, 0, 0], up=(0.0, 0.0, 1.0))
        assert s is not None
        right, image_up = _axes_from_str(s)
        assert right == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)


# -- MJCFBuilder._object_xml shape branches ----------------------------------


class TestObjectXMLShapes:
    """Exercise every shape branch in the _object_xml body."""

    def _make(self, **kw):
        defaults = dict(
            name="probe",
            shape="box",
            position=[0.0, 0.0, 0.1],
            orientation=[1.0, 0.0, 0.0, 0.0],
            size=[0.05, 0.05, 0.05],
            color=[0.5, 0.5, 0.5, 1.0],
            mass=0.1,
            is_static=False,
            mesh_path=None,
        )
        defaults.update(kw)
        return SimObject(**defaults)

    def test_box_includes_geom(self):
        obj = self._make(shape="box")
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="box"' in xml
        assert 'name="probe_geom"' in xml
        # Mass→ half-size conversion: size[0]/2 = 0.025.
        assert "0.025" in xml

    def test_sphere_default_radius(self):
        obj = self._make(shape="sphere", size=[])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="sphere"' in xml

    def test_cylinder(self):
        obj = self._make(shape="cylinder", size=[0.04, 0.04, 0.12])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="cylinder"' in xml

    def test_capsule(self):
        obj = self._make(shape="capsule", size=[0.04, 0.04, 0.12])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="capsule"' in xml

    def test_mesh_requires_mesh_path(self):
        obj = self._make(shape="mesh", mesh_path="/tmp/does_not_matter.stl")
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="mesh"' in xml
        assert "mesh_probe" in xml

    def test_mesh_without_path_skips_geom(self):
        """Mesh shape with no mesh_path should not emit a <geom type='mesh'/>."""
        obj = self._make(shape="mesh", mesh_path=None)
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="mesh"' not in xml

    def test_plane_emits_plane_geom(self):
        obj = self._make(shape="plane", size=[1.0, 1.0], is_static=True)
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="plane"' in xml

    def test_static_object_has_no_freejoint(self):
        obj = self._make(is_static=True)
        xml = MJCFBuilder._object_xml(obj)
        assert "freejoint" not in xml

    def test_dynamic_object_has_freejoint(self):
        obj = self._make(is_static=False)
        xml = MJCFBuilder._object_xml(obj)
        assert 'name="probe_joint"' in xml
        assert "freejoint" in xml

    def test_name_is_sanitized(self):
        """Invalid names surface through _sanitize_name."""
        obj = self._make(name="bad name with spaces")
        with pytest.raises(ValueError, match="Invalid simulation name"):
            MJCFBuilder._object_xml(obj)
