"""``add_object`` reports the geom it compiled, not the request, for every shape.

A mesh consumes no ``size`` component -- ``_SIZE_LAYOUT["mesh"]`` is ``0``, and
``_normalize_size`` maps it to ``[]`` -- yet the success text echoed ``size=``
back. So the one shape whose extent the request never carries was the one shape
whose extent the result asserted: an omitted ``size`` reported ``[0.05, 0.05,
0.05]`` for an asset of any size (a 3.5 x 10.5 x 12.5 m generated room shell
reported as a 5 cm object), and an explicit vector was echoed as though it had
been honoured. ``_validate_size`` already records why that shape of report is a
defect -- a short vector "compiled a differently-sized object while reporting
success and echoed the requested ``[0.5]``".

The mesh row was fixed by reporting the extent read back off the compiled geom
(:func:`~strands_robots.simulation.mujoco.simulation._compiled_geom_extent`, over
MuJoCo's own ``geom_aabb``) plus the collision geometry, which for every mesh geom
is its convex hull rather than the triangles that render. Primitive shapes were
left echoing ``size`` because "there it *is* the extent, and the geom compiles to
it" -- which holds for ``box`` and ``ellipsoid`` and for no other shape:

* ``sphere`` consumes ``size[0]`` alone, so ``[0.05, 0.09, 0.2]`` reported y and
  z extents of 0.09 m and 0.2 m for a ball 0.05 m across in every axis.
* ``cylinder`` and ``capsule`` ignore ``size[1]``, so the report gave a circular
  cross-section a y extent its x extent contradicts.
* a ``capsule``'s hemispherical caps add its radius to each end, so a requested
  0.9 m height compiles to 0.95 m and the request under-reported it.
* ``plane`` keeps only its two visual half-widths -- ``size[1]`` mirrors
  ``size[0]`` when omitted and ``size[2]`` is replaced by the builder's own grid
  spacing -- so both a mirrored width and a discarded third component were
  reported as extents.

Every shape now reports through one owner
(:func:`~strands_robots.simulation.mujoco.simulation._compiled_geometry_detail`).
The compiled geometry is unchanged -- only what the result says about it is.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    Simulation,
    _compiled_geom_extent,
    _compiled_geometry_detail,
    _compiled_plane_half_widths,
)

# An open-topped channel: a thin floor plate spanned by two tall side walls.
# Concave by construction -- its convex hull is the full 0.3 x 0.4 x 0.4 m box,
# which fills the cavity between the walls. Written as an explicit triangle soup
# so the fixture needs no mesh library.
_CHANNEL_BOXES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((-0.20, -0.20, 0.00), (0.20, 0.20, 0.02)),  # floor plate, 2 cm thick
    ((-0.20, -0.20, 0.00), (-0.18, 0.20, 0.30)),  # left wall, 30 cm tall
    ((0.18, -0.20, 0.00), (0.20, 0.20, 0.30)),  # right wall
)
_CHANNEL_EXTENT = (0.3, 0.4, 0.4)

#: ``(shape, request, compiled extent)`` per primitive, with the compiled column
#: measured off ``geom_aabb`` rather than restated from the request -- the whole
#: point being that the two differ for four of the five rows. A row where they
#: differ is a row the pre-fix report got wrong.
_PRIMITIVES: tuple[tuple[str, list[float], tuple[float, float, float]], ...] = (
    ("box", [0.04, 0.08, 0.12], (0.04, 0.08, 0.12)),
    ("ellipsoid", [0.04, 0.08, 0.12], (0.04, 0.08, 0.12)),
    ("sphere", [0.05, 0.09, 0.2], (0.05, 0.05, 0.05)),
    ("sphere", [0.06], (0.06, 0.06, 0.06)),
    ("cylinder", [0.05, 0.1, 0.9], (0.05, 0.05, 0.9)),
    ("capsule", [0.05, 0.1, 0.9], (0.05, 0.05, 0.95)),
)

#: The rows above whose request holds a component the geom does not carry.
_MISREPORTED = tuple(
    (shape, request, compiled)
    for shape, request, compiled in _PRIMITIVES
    if [round(v, 6) for v in request] != [round(v, 6) for v in compiled]
)
_CHANNEL_CAVITY_FLOOR_Z = 0.02
_CHANNEL_WALL_TOP_Z = 0.30


def _write_box_soup(
    path: Path,
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> str:
    """Write ``boxes`` (``(lo, hi)`` corner pairs) as one OBJ triangle soup."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for lo, hi in boxes:
        base = len(verts) + 1  # OBJ indices are 1-based
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    verts.append(
                        (
                            hi[0] if sx else lo[0],
                            hi[1] if sy else lo[1],
                            hi[2] if sz else lo[2],
                        )
                    )
        corner = {(sx, sy, sz): base + sx * 4 + sy * 2 + sz for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)}
        for quad in (
            ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
            ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
            ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
            ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
        ):
            a, b, c, d = (corner[k] for k in quad)
            faces.append((a, b, c))
            faces.append((a, c, d))
    path.write_text(
        "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts) + "".join(f"f {a} {b} {c}\n" for a, b, c in faces),
        encoding="utf-8",
    )
    return str(path)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(block["text"] for block in result["content"] if "text" in block)


def _reported_extent(result: dict[str, Any]) -> list[float]:
    """Parse the ``extent=[...]`` vector out of an add_object success text."""
    match = re.search(r"extent=\[([^\]]+)\]", _text(result))
    assert match is not None, f"no extent reported: {_text(result)!r}"
    return [float(part) for part in match.group(1).split(",")]


@pytest.fixture
def channel_mesh(tmp_path: Path) -> str:
    return _write_box_soup(tmp_path / "channel.obj", _CHANNEL_BOXES)


@pytest.fixture
def sim():
    engine = Simulation(tool_name="test_mesh_add_reports_compiled_geometry", mesh=False)
    assert engine.create_world()["status"] == "success"
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


class TestAMeshReportsTheAssetsExtent:
    """The extent comes off the compiled geom, because the request has none."""

    def test_an_omitted_size_reports_the_assets_extent_not_the_5cm_default(self, sim, channel_mesh: str) -> None:
        """Pre-fix: ``size=[0.05, 0.05, 0.05]`` for a 0.3 x 0.4 x 0.4 m asset."""
        result = sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)
        assert result["status"] == "success", _text(result)
        assert _reported_extent(result) == pytest.approx(_CHANNEL_EXTENT, abs=1e-3)
        assert "size=" not in _text(result)

    def test_an_explicit_size_is_not_echoed_as_though_it_were_honoured(self, sim, channel_mesh: str) -> None:
        """A mesh consumes no component, so an echoed vector reports a fiction."""
        result = sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, size=[2.0, 2.0, 2.0])
        assert result["status"] == "success", _text(result)
        assert "2.0" not in _text(result)
        assert _reported_extent(result) == pytest.approx(_CHANNEL_EXTENT, abs=1e-3)

    def test_the_message_names_the_convex_hull_as_the_collision_geometry(self, sim, channel_mesh: str) -> None:
        """The property that makes a concave asset behave unlike it renders."""
        result = sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)
        assert "convex hull" in _text(result)


def _reported_size(result: dict[str, Any]) -> list[float]:
    """Parse the ``size=[...]`` vector out of an add_object success text."""
    match = re.search(r"size=\[([^\]]+)\]", _text(result))
    assert match is not None, f"no size reported: {_text(result)!r}"
    return [float(part) for part in match.group(1).split(",")]


class TestEveryShapeReportsTheExtentItCompiledTo:
    """The measurement, checked against the model for each shape in turn."""

    @pytest.mark.parametrize(("shape", "request_size", "compiled"), _PRIMITIVES)
    def test_the_reported_size_is_the_geoms_own_extent(
        self, sim, shape: str, request_size: list[float], compiled: tuple[float, float, float]
    ) -> None:
        result = sim.add_object("obj", shape=shape, size=request_size)
        assert result["status"] == "success", _text(result)
        assert _reported_size(result) == pytest.approx(compiled, abs=1e-6)
        assert _compiled_geom_extent(sim._mj, sim.mj_model, "obj_geom") == pytest.approx(compiled, abs=1e-6)
        assert "convex hull" not in _text(result)

    @pytest.mark.parametrize(("shape", "request_size", "compiled"), _MISREPORTED)
    def test_the_request_is_not_what_gets_reported(
        self, sim, shape: str, request_size: list[float], compiled: tuple[float, float, float]
    ) -> None:
        """The pre-fix failure: these four rows echoed the request verbatim.

        Read off the reported vector rather than searched for in the text, so a
        number that also appears in the mass or the position cannot satisfy it.
        """
        result = sim.add_object("obj", shape=shape, size=request_size)
        assert result["status"] == "success", _text(result)
        reported = _reported_size(result)
        assert reported != pytest.approx(request_size, abs=1e-9), _text(result)
        discarded = [v for v in request_size if round(v, 6) not in {round(c, 6) for c in compiled}]
        for value in discarded:
            assert value not in [pytest.approx(r, abs=1e-9) for r in reported], _text(result)

    def test_a_capsules_caps_are_counted_in_the_height_it_reports(self, sim) -> None:
        """``size[2]`` is the cylindrical section; the caps add ``size[0]``.

        The one row where the compiled extent exceeds every component of the
        request, so no echo of it could have been right.
        """
        diameter, height = 0.05, 0.9
        result = sim.add_object("rod", shape="capsule", size=[diameter, 0.0, height])
        assert result["status"] == "success", _text(result)
        assert _reported_size(result)[2] == pytest.approx(height + diameter, abs=1e-6)

    def test_an_extent_finer_than_a_tenth_of_a_millimetre_is_reported_exactly(self, sim) -> None:
        """The read-back is exact, so its resolution must not lose the request.

        Halving a float to a half-extent and doubling it back is exact, so a
        primitive reads back as precisely the number asked for. Quantising the
        report would state a value the geom does not have -- the class of report
        this read exists to remove.
        """
        result = sim.add_object("shim", shape="box", size=[0.12345, 0.2, 0.3])
        assert result["status"] == "success", _text(result)
        assert _reported_size(result) == pytest.approx([0.12345, 0.2, 0.3], abs=1e-9)


class TestAPlaneReportsTheOnlyGeometryItHas:
    """A plane is infinite for collision, so its bounding box describes nothing."""

    def test_the_report_names_the_visual_half_widths_not_the_infinite_bound(self, sim) -> None:
        result = sim.add_object("floor", shape="plane", size=[1.0, 2.0, 3.0], is_static=True)
        assert result["status"] == "success", _text(result)
        assert _reported_size(result) == pytest.approx([1.0, 2.0], abs=1e-6)
        assert "visual half-widths" in _text(result)
        assert "infinite for collision" in _text(result)
        # The discarded third component, and MuJoCo's own 2e10 m plane sentinel.
        assert "3.0" not in _text(result), _text(result)
        assert "e+" not in _text(result) and "20000000000" not in _text(result), _text(result)

    def test_an_omitted_width_reports_the_one_that_was_compiled_for_it(self, sim) -> None:
        """``size[1]`` mirrors ``size[0]``, which only the compiled geom knows."""
        result = sim.add_object("floor", shape="plane", size=[1.5], is_static=True)
        assert result["status"] == "success", _text(result)
        assert _reported_size(result) == pytest.approx([1.5, 1.5], abs=1e-6)
        assert _compiled_plane_half_widths(sim._mj, sim.mj_model, "floor_geom") == pytest.approx([1.5, 1.5], abs=1e-6)

    def test_the_infinite_bounding_box_is_what_the_extent_read_returns(self, sim) -> None:
        """Why the plane needs its own read rather than the shared one."""
        assert sim.add_object("floor", shape="plane", size=[1.0, 2.0], is_static=True)["status"] == "success"
        extent = _compiled_geom_extent(sim._mj, sim.mj_model, "floor_geom")
        assert extent is not None
        assert min(extent) > 1e9, extent


class TestAGeomThatCannotBeMeasuredSaysSo:
    """The fallback is "unavailable", never the request -- for all three forms."""

    @pytest.mark.parametrize("shape", ["box", "plane", "mesh"])
    def test_an_unresolvable_geom_reports_no_geometry_rather_than_the_request(self, sim, shape: str) -> None:
        detail = _compiled_geometry_detail(sim._mj, sim.mj_model, shape, "never_added_geom")
        assert "unavailable" in detail, detail
        assert not re.search(r"=\[", detail), detail


class TestTheReportedGeometryIsTheGeometryThatActsOnObjects:
    """Both halves of the report are checked against the compiled model."""

    def test_the_reported_extent_matches_the_geoms_own_bounding_box(self, sim, channel_mesh: str) -> None:
        assert sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)["status"] == "success"
        extent = _compiled_geom_extent(sim._mj, sim.mj_model, "channel_geom")
        assert extent == pytest.approx(_CHANNEL_EXTENT, abs=1e-3)

    def test_an_unresolvable_geom_reports_no_extent_rather_than_a_wrong_one(self, sim) -> None:
        assert _compiled_geom_extent(sim._mj, sim.mj_model, "never_added_geom") is None

    def test_a_non_string_geom_name_reports_no_extent_rather_than_crashing(self, sim) -> None:
        """The lookup routes through ``mj_name_to_id``, so it inherits its guard.

        Reaching ``mujoco.mj_name2id`` with a non-string name terminates the
        interpreter with SIGSEGV rather than raising, which no envelope can
        recover from.
        """
        not_a_name: Any = None
        assert _compiled_geom_extent(sim._mj, sim.mj_model, not_a_name) is None

    def test_a_concave_asset_collides_as_its_filled_hull(self, sim, channel_mesh: str) -> None:
        """The documented consequence, pinned so the warning cannot go stale.

        Dropped into the channel's open cavity, a ball comes to rest on the
        convex hull that spans the wall tops rather than on the interior floor
        2 cm up. This is MuJoCo's mesh-collision contract, not a defect -- the
        test exists so the sentence that documents it stays true.
        """
        import mujoco

        assert sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)["status"] == "success"
        assert (
            sim.add_object("ball", shape="sphere", size=[0.06], position=[0.0, 0.0, 0.45], mass=0.05)["status"]
            == "success"
        )
        model, data = sim.mj_model, sim.mj_data
        for _ in range(3000):
            mujoco.mj_step(model, data)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        rest_z = float(data.xpos[body_id][2])
        assert rest_z > _CHANNEL_WALL_TOP_Z - 0.05, f"expected a rest on the hull near the wall tops, got z={rest_z}"
        assert rest_z > _CHANNEL_CAVITY_FLOOR_Z + 0.1, f"the cavity is not load-bearing, yet z={rest_z}"
