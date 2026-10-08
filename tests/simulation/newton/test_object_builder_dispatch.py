"""Object-builder dispatch and quaternion-convention math for the Newton backend.

These exercise the pure, backend-agnostic logic that turns a
:class:`~strands_robots.simulation.models.SimObject` into Newton builder calls
(``NewtonSimEngine._add_object_to_builder``) and the wxyz->xyzw quaternion
convention conversion (``NewtonSimEngine._wxyz_to_wp_quat``).

Both run without the optional ``newton``/``warp`` packages or a GPU: the methods
only touch ``self._wp`` (the Warp module handle) and the builder, so a recording
stub for each is enough to assert the exact calls. This pins two contracts that
are easy to regress silently:

* the shape -> ``add_shape_*`` dispatch (including the static-vs-dynamic body
  branch), and
* the wxyz (SimRobot/MuJoCo) -> xyzw (Warp ``quatf``) component reorder, a
  classic footgun that produces a plausible-looking but wrong orientation.
"""

from __future__ import annotations

import types

import pytest

from strands_robots.simulation.models import SimObject, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine


class _RecordingWp:
    """Minimal stand-in for the Warp module, recording structural call args."""

    def transform(self, pos, quat):
        return ("xform", pos, quat)

    def vec3(self, *args):
        return ("vec3", tuple(args))

    def quat(self, *args):
        return ("quat", tuple(args))

    def quat_identity(self):
        return ("quat_identity",)


class _RecordingBuilder:
    """Newton ModelBuilder stand-in that records add_body / add_shape_* calls."""

    def __init__(self, body_id: int = 7) -> None:
        self.body_id = body_id
        self.calls: list[tuple] = []

    def add_body(self, **kwargs):
        self.calls.append(("add_body", kwargs))
        return self.body_id

    def add_shape_box(self, body, **kwargs):
        self.calls.append(("box", body, kwargs))

    def add_shape_sphere(self, body, **kwargs):
        self.calls.append(("sphere", body, kwargs))

    def add_shape_capsule(self, body, **kwargs):
        self.calls.append(("capsule", body, kwargs))

    def add_shape_cylinder(self, body, **kwargs):
        self.calls.append(("cylinder", body, kwargs))


def _engine_stub():
    """A tiny NewtonSimEngine stand-in carrying just ``_wp`` and the bound
    ``_wxyz_to_wp_quat`` helper, so ``_add_object_to_builder`` runs without
    importing newton/warp or touching a GPU.
    """
    wp = _RecordingWp()
    stub = types.SimpleNamespace(_wp=wp)
    stub._wxyz_to_wp_quat = lambda wxyz: NewtonSimEngine._wxyz_to_wp_quat(stub, wxyz)
    return stub


def _add(obj: SimObject, builder: _RecordingBuilder | None = None):
    stub = _engine_stub()
    builder = builder or _RecordingBuilder()
    NewtonSimEngine._add_object_to_builder(stub, builder, obj)
    return builder


class TestQuatConvention:
    """wxyz (SimRobot) -> xyzw (Warp quatf) reorder."""

    def test_identity_quaternion_reorders_to_xyzw(self):
        # wxyz identity (1,0,0,0) -> xyzw (0,0,0,1).
        stub = _engine_stub()
        assert NewtonSimEngine._wxyz_to_wp_quat(stub, [1.0, 0.0, 0.0, 0.0]) == ("quat", (0.0, 0.0, 0.0, 1.0))

    def test_general_quaternion_moves_w_to_last(self):
        # wxyz (w, x, y, z) = (0.1, 0.2, 0.3, 0.4) -> xyzw (0.2, 0.3, 0.4, 0.1).
        stub = _engine_stub()
        assert NewtonSimEngine._wxyz_to_wp_quat(stub, [0.1, 0.2, 0.3, 0.4]) == ("quat", (0.2, 0.3, 0.4, 0.1))


class TestStaticVsDynamicBody:
    """Static / massless objects attach to the world (body -1) with the object's
    own world transform; dynamic objects get a body and a zeroed shape transform.
    """

    def test_static_object_attaches_to_world_body(self):
        builder = _add(SimObject(name="ground", shape="box", size=[1.0, 1.0, 0.1], is_static=True))
        # No body created; the single shape call targets world body -1.
        assert all(c[0] != "add_body" for c in builder.calls)
        assert builder.calls[0][0] == "box"
        assert builder.calls[0][1] == -1

    def test_massless_object_treated_as_static(self):
        # mass <= 0 is the same "static" branch even without is_static set.
        builder = _add(SimObject(name="m0", shape="sphere", size=[0.05], mass=0.0))
        assert all(c[0] != "add_body" for c in builder.calls)
        assert builder.calls[0][1] == -1

    def test_static_shape_uses_object_world_transform(self):
        # Static shapes carry the full world xform (not quat_identity), since
        # there is no parent body frame to ride.
        builder = _add(SimObject(name="g", shape="box", size=[1, 1, 0.1], is_static=True))
        xform = builder.calls[0][2]["xform"]
        assert xform[2] != ("quat_identity",)

    def test_dynamic_object_creates_body_then_shape(self):
        builder = _add(SimObject(name="cube", shape="box", size=[0.2, 0.2, 0.2], mass=1.0))
        kinds = [c[0] for c in builder.calls]
        assert kinds == ["add_body", "box"]
        # Shape is attached to the created body id with a zeroed (identity) frame.
        assert builder.calls[1][1] == builder.body_id
        assert builder.calls[1][2]["xform"][2] == ("quat_identity",)

    def test_dynamic_body_forwards_mass(self):
        builder = _add(SimObject(name="cube", shape="box", size=[0.2, 0.2, 0.2], mass=2.5))
        assert builder.calls[0][1]["mass"] == 2.5


class TestShapeDispatch:
    """Each ``SimObject.shape`` maps to the matching ``add_shape_*`` call; the size
    arguments it receives are pinned across backends in
    ``tests/simulation/test_object_size_domain_across_backends.py``.
    """

    def test_color_truncated_to_rgb(self):
        # SimObject.color is RGBA by default; Newton shapes take an RGB tuple.
        builder = _add(SimObject(name="b", shape="box", size=[0.1, 0.1, 0.1], color=[0.1, 0.2, 0.3, 1.0], mass=1.0))
        box = next(c for c in builder.calls if c[0] == "box")
        assert box[2]["color"] == (0.1, 0.2, 0.3)

    def test_unsupported_shape_adds_no_geometry(self):
        # An unknown shape falls through the dispatch with no add_shape_* call;
        # for a dynamic object only the (empty) body is created.
        builder = _add(SimObject(name="blob", shape="ellipsoid", size=[0.1, 0.1, 0.1], mass=1.0))
        kinds = [c[0] for c in builder.calls]
        assert kinds == ["add_body"]


@pytest.mark.parametrize(
    ("shape", "expected"),
    [("box", "box"), ("sphere", "sphere"), ("capsule", "capsule"), ("cylinder", "cylinder")],
)
def test_each_primitive_routes_to_its_builder_method(shape, expected):
    builder = _add(SimObject(name="p", shape=shape, size=[0.1, 0.1, 0.1], mass=1.0))
    assert any(c[0] == expected for c in builder.calls)


@pytest.mark.parametrize(
    ("shape", "hint"),
    [("sfere", " Did you mean 'sphere'?"), ("cilinder", " Did you mean 'cylinder'?"), ("torus", "")],
)
def test_add_object_names_the_closest_newton_shape(shape, hint):
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = SimWorld()
    result = engine.add_object(name="obj", shape=shape)
    assert result["status"] == "error"
    assert result["content"][0]["text"] == (
        f"Unsupported shape {shape!r} for Newton backend.{hint} Supported: box, sphere, capsule, cylinder, mesh."
    )
