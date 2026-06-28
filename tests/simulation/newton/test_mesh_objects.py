"""Mesh-object support in the Newton backend's ``add_object`` (MuJoCo parity).

Three layers, each runnable in a different environment:

* ``add_object`` validation (no newton/warp/trimesh): the error paths return
  before any model rebuild, so they run against a light stand-in engine.
* mesh -> builder dispatch (no newton/warp/trimesh): ``_add_object_to_builder``
  only touches ``self._wp`` / ``self._nt`` / the builder / ``_load_mesh_geometry``,
  so stubs pin the exact ``add_shape_mesh`` call.
* ``_load_mesh_geometry`` parsing (needs ``trimesh``) and the full engine
  round-trip (needs newton + warp + GPU): load OBJ / STL, move, remove, and a
  60-step mock-policy rollout, all gated on availability.
"""

from __future__ import annotations

import importlib.util
import types

import numpy as np
import pytest

from strands_robots.simulation.models import SimObject
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_HAS_TRIMESH = importlib.util.find_spec("trimesh") is not None
_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


def _write_cube(path, exporter="obj"):
    """Write a unit cube to ``path`` via trimesh; return the path string."""
    import trimesh

    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(str(path))
    return str(path)


# --------------------------------------------------------------------------- #
# add_object validation (no optional deps)
# --------------------------------------------------------------------------- #
class TestAddObjectValidation:
    """Error paths return before the model rebuild, so a stub engine suffices."""

    def _stub(self):
        return types.SimpleNamespace(_world=types.SimpleNamespace(objects={}))

    def test_mesh_without_mesh_path_is_rejected(self):
        result = NewtonSimEngine.add_object(self._stub(), name="tool", shape="mesh")
        assert result["status"] == "error"
        assert "mesh_path" in result["content"][0]["text"]

    def test_mesh_with_missing_file_is_rejected(self):
        result = NewtonSimEngine.add_object(self._stub(), name="tool", shape="mesh", mesh_path="/no/such/file.obj")
        assert result["status"] == "error"
        assert "does not exist" in result["content"][0]["text"]

    def test_unsupported_shape_lists_supported_shapes(self):
        result = NewtonSimEngine.add_object(self._stub(), name="x", shape="ellipsoid")
        assert result["status"] == "error"
        assert "mesh" in result["content"][0]["text"]
        assert "Supported" in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# mesh -> builder dispatch (no optional deps)
# --------------------------------------------------------------------------- #
class _RecordingWp:
    def transform(self, pos, quat):
        return ("xform", pos, quat)

    def vec3(self, *args):
        return ("vec3", tuple(args))

    def quat(self, *args):
        return ("quat", tuple(args))

    def quat_identity(self):
        return ("quat_identity",)


class _RecordingBuilder:
    def __init__(self, body_id: int = 7) -> None:
        self.body_id = body_id
        self.calls: list[tuple] = []

    def add_body(self, **kwargs):
        self.calls.append(("add_body", kwargs))
        return self.body_id

    def add_shape_mesh(self, body, **kwargs):
        self.calls.append(("mesh", body, kwargs))


class _FakeMesh:
    """Stand-in for ``newton.Mesh`` recording its constructor args."""

    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices


def _mesh_engine_stub(vertices, indices):
    wp = _RecordingWp()
    stub = types.SimpleNamespace(_wp=wp, _nt=types.SimpleNamespace(Mesh=_FakeMesh))
    stub._wxyz_to_wp_quat = lambda wxyz: NewtonSimEngine._wxyz_to_wp_quat(stub, wxyz)
    stub._load_mesh_geometry = lambda mesh_path: (vertices, indices)
    return stub


def _dispatch(obj, vertices, indices):
    stub = _mesh_engine_stub(vertices, indices)
    builder = _RecordingBuilder()
    NewtonSimEngine._add_object_to_builder(stub, builder, obj)
    return builder


class TestMeshDispatch:
    def test_dynamic_mesh_creates_body_then_mesh_shape(self):
        verts = np.zeros((4, 3), dtype=np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        builder = _dispatch(SimObject(name="m", shape="mesh", mesh_path="/x.obj", mass=1.0), verts, idx)
        assert [c[0] for c in builder.calls] == ["add_body", "mesh"]
        mesh_call = builder.calls[1]
        assert mesh_call[1] == builder.body_id
        assert isinstance(mesh_call[2]["mesh"], _FakeMesh)
        assert mesh_call[2]["mesh"].indices is idx

    def test_static_mesh_attaches_to_world_body(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        builder = _dispatch(SimObject(name="m", shape="mesh", mesh_path="/x.obj", is_static=True), verts, idx)
        assert all(c[0] != "add_body" for c in builder.calls)
        assert builder.calls[0][0] == "mesh"
        assert builder.calls[0][1] == -1

    def test_size_becomes_per_axis_scale(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        builder = _dispatch(
            SimObject(name="m", shape="mesh", mesh_path="/x.obj", size=[2.0, 3.0, 4.0], mass=1.0),
            verts,
            idx,
        )
        scale = builder.calls[1][2]["scale"]
        assert scale == ("vec3", (2.0, 3.0, 4.0))

    def test_color_truncated_to_rgb(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        builder = _dispatch(
            SimObject(name="m", shape="mesh", mesh_path="/x.obj", color=[0.1, 0.2, 0.3, 1.0], mass=1.0),
            verts,
            idx,
        )
        assert builder.calls[1][2]["color"] == (0.1, 0.2, 0.3)


# --------------------------------------------------------------------------- #
# _load_mesh_geometry parsing (needs trimesh)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh (sim-newton extra) not installed")
class TestLoadMeshGeometry:
    def _stub(self):
        return types.SimpleNamespace(_mesh_cache={})

    def test_load_obj_returns_vertices_and_flat_indices(self, tmp_path):
        path = _write_cube(tmp_path / "cube.obj")
        verts, idx = NewtonSimEngine._load_mesh_geometry(self._stub(), path)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert idx.ndim == 1 and idx.size % 3 == 0
        assert verts.dtype == np.float32 and idx.dtype == np.int32

    def test_load_stl(self, tmp_path):
        path = _write_cube(tmp_path / "cube.stl")
        verts, idx = NewtonSimEngine._load_mesh_geometry(self._stub(), path)
        assert verts.size > 0 and idx.size % 3 == 0

    def test_result_is_cached_by_path(self, tmp_path):
        path = _write_cube(tmp_path / "cube.obj")
        stub = self._stub()
        first = NewtonSimEngine._load_mesh_geometry(stub, path)
        assert path in stub._mesh_cache
        # Second call returns the identical cached arrays (no re-parse).
        second = NewtonSimEngine._load_mesh_geometry(stub, path)
        assert second[0] is first[0] and second[1] is first[1]

    def test_missing_mesh_path_raises(self):
        with pytest.raises(ValueError, match="mesh_path"):
            NewtonSimEngine._load_mesh_geometry(self._stub(), None)


# --------------------------------------------------------------------------- #
# Full engine round-trip (needs newton + warp + GPU)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_HAS_NEWTON and _HAS_TRIMESH), reason="newton/warp/trimesh not installed")
class TestNewtonMeshRoundTrip:
    @pytest.fixture
    def engine(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        sim = NewtonSimEngine(solver="mujoco")
        sim.create_world()
        yield sim
        sim.destroy()

    def test_add_obj_mesh_object(self, engine, tmp_path):
        path = _write_cube(tmp_path / "cube.obj")
        result = engine.add_object(name="cube", shape="mesh", mesh_path=path, position=[0.2, 0.0, 0.1])
        assert result["status"] == "success"
        listing = engine.list_objects()["content"][0]["text"]
        assert "cube" in listing and "mesh" in listing and "cube.obj" in listing

    def test_add_stl_mesh_object(self, engine, tmp_path):
        path = _write_cube(tmp_path / "cube.stl")
        assert engine.add_object(name="c", shape="mesh", mesh_path=path)["status"] == "success"

    def test_move_mesh_object_preserves_pose(self, engine, tmp_path):
        path = _write_cube(tmp_path / "cube.obj")
        engine.add_object(name="cube", shape="mesh", mesh_path=path, position=[0.0, 0.0, 0.1])
        assert engine.move_object("cube", position=[0.3, 0.1, 0.2])["status"] == "success"
        assert engine._world.objects["cube"].position == [0.3, 0.1, 0.2]

    def test_remove_mesh_object(self, engine, tmp_path):
        path = _write_cube(tmp_path / "cube.obj")
        engine.add_object(name="cube", shape="mesh", mesh_path=path)
        assert engine.remove_object("cube")["status"] == "success"
        assert "cube" not in engine._world.objects

    def test_mesh_scene_steps_with_mock_policy(self, engine, tmp_path):
        path = _write_cube(tmp_path / "cube.obj")
        engine.add_robot("so100")
        engine.add_object(name="cube", shape="mesh", mesh_path=path, position=[0.25, 0.0, 0.05], mass=0.1)
        # 60-step rollout with the mock policy: no rebuild / step errors.
        engine.run_policy(robot_name="so100", policy_provider="mock", n_steps=60, control_frequency=50.0)
