"""A mesh asset whose declared ``scale`` is not finite is refused, not measured.

:func:`~strands_robots.simulation.isaac.mesh_assets.load_mesh_geometry`
already refuses a vertex coordinate that is not finite, and
:func:`~strands_robots.simulation.isaac.mesh_assets._non_finite_vertex_error`
states why: ``min``/``max`` order a NaN as neither smaller nor larger than
anything, so the running comparison
:func:`~strands_robots.simulation.isaac.mesh_assets.mesh_aabb` uses drops it
and reports the bounds of the coordinates that *are* finite - "the same
numbers a mesh declaring only those would produce, under no error at all".

``scale`` reaches that same comparison. ``mesh_aabb`` measures
``vertex * scale``, so a non-finite component poisons every vertex on that
axis and the comparison drops all of them, leaving the axis at its
``inf``/``-inf`` seed. The asset's own coordinates being finite is not
enough: the transform applied to them arrives from the caller, and it was
unchecked.

The reachable route is an MJCF ``<asset><mesh scale=...>``. MuJoCo compiles
such a model, so the reader cannot defer the question to the compiler, and
:func:`~strands_robots.simulation.isaac.loaders.load_mjcf_scene_objects`
measures the mesh's own bounds for a *mesh-only* body - one with no
collidable analytic geom - which is the branch the loader's own comment says
exists "so a mesh-only body can fall back to the mesh's own bounds".

Neither failure is screenable from the reported fields:

* A NaN axis gives a centre of NaN and, because the reported extent is
  floored at ``1e-4``, a **finite** size of 0.1 mm. A consumer screening the
  reported fields for non-finite values catches the centre and not the size,
  so a 0.2 m asset passes that screen as a plausible sub-millimetre one.
* An infinite axis gives an infinite centre and a size of NaN, because the
  extent is ``inf - inf``.

The classes below pin, in order: the refusal for both non-finite values on
every axis; that the MJCF loader refuses it end to end on the mesh-only body
that reaches the measurement; the premises that make the refusal necessary
(MuJoCo compiles it, the asset's own vertices are finite, and the extent
floor is what made the NaN case unscreenable); that a finite scale -
including the negative and zero ones - measures exactly what it did before;
that a body carrying collidable geometry takes its bound from that geometry
and never reaches the measurement - which is why the refusal for the value it
still carries lives at the read rather than here, pinned in
``test_mesh_asset_scale_is_refused_wherever_it_is_carried``; that the wording
has one owner per cause; and that the guard precedes the parse.
"""

from __future__ import annotations

import ast
import inspect
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.isaac import mesh_assets
from strands_robots.simulation.isaac.loaders import (
    _parse_axis,
    _parse_mjcf_mesh_assets,
    load_mjcf_scene_objects,
)
from strands_robots.simulation.isaac.mesh_assets import mesh_aabb

NAN = float("nan")
POS_INF = float("inf")
NEG_INF = float("-inf")

#: The refusal's stable phrase, shared by every axis and both values.
REFUSAL = "scale has a component that is not finite"

#: A closed tetrahedron with a 0.2 m extent on every axis. Four distinct
#: vertices is MuJoCo's own minimum for a mesh, so the same asset serves the
#: reader here and the compiler in the premise class.
TETRA: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.2, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, 0.0, 0.2),
)
FACES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

#: What ``mesh_aabb`` measures from :data:`TETRA` at unit scale. Float32 on
#: disk, so the values are the binary-STL round trip rather than exact.
UNIT_CENTER = (0.10000000149011612,) * 3
UNIT_SIZE = (0.20000000298023224,) * 3

#: Every axis, so a guard that checks only the first cannot pass.
AXES = (0, 1, 2)

#: Both non-finite values, which fail *differently*: NaN leaves the reported
#: size finite (floored), infinity does not.
NON_FINITE = (NAN, POS_INF, NEG_INF)


def _binary_stl(verts: tuple[tuple[float, float, float], ...]) -> bytes:
    out = [b"\0" * 80, struct.pack("<I", len(FACES))]
    for tri in FACES:
        out.append(struct.pack("<3f", 0.0, 0.0, 1.0))
        for idx in tri:
            out.append(struct.pack("<3f", *verts[idx]))
        out.append(struct.pack("<H", 0))
    return b"".join(out)


def _asset(tmp_path: Path, name: str = "asset") -> str:
    path = tmp_path / f"{name}.stl"
    path.write_bytes(_binary_stl(TETRA))
    return str(path)


def _scale(axis: int, value: float) -> tuple[float, float, float]:
    out = [1.0, 1.0, 1.0]
    out[axis] = value
    return (out[0], out[1], out[2])


#: A body with a mesh geom and *no* collidable analytic geom, so
#: ``load_mjcf_scene_objects`` falls through to ``mesh_aabb``.
_MESH_ONLY = """<mujoco model="probe">
  <asset><mesh name="m" file="asset.stl" scale="{scale}"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="widget" pos="0 0 0.5"><geom name="w_vis" type="mesh" mesh="m"/></body>
  </worldbody>
</mujoco>
"""

#: The same declaration on a body that *does* carry a collidable box. The
#: analytic bound wins, so the measurement is never reached.
_WITH_COLLISION = """<mujoco model="probe">
  <asset><mesh name="m" file="asset.stl" scale="{scale}"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="widget" pos="0 0 0.5">
      <geom name="w_col" type="box" size="0.1 0.1 0.05"/>
      <geom name="w_vis" type="mesh" mesh="m" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _scene(tmp_path: Path, template: str, scale: str) -> str:
    _asset(tmp_path)
    path = tmp_path / "scene.xml"
    path.write_text(template.format(scale=scale))
    return str(path)


def _widget(objects: list[Any]) -> Any:
    return next(obj for obj in objects if obj.name == "widget")


class TestANonFiniteScaleIsRefused:
    """Every axis, both values. A NaN or infinite scale is not a transform."""

    @pytest.mark.parametrize("axis", AXES)
    @pytest.mark.parametrize("value", NON_FINITE)
    def test_mesh_aabb_refuses_instead_of_reporting_bounds(self, tmp_path: Path, axis: int, value: float) -> None:
        with pytest.raises(ValueError, match=REFUSAL):
            mesh_aabb(_asset(tmp_path), _scale(axis, value))

    def test_a_scale_that_is_non_finite_on_every_axis_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=REFUSAL):
            mesh_aabb(_asset(tmp_path), (NAN, NAN, NAN))

    def test_the_refusal_names_the_asset_and_the_scale(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        with pytest.raises(ValueError) as excinfo:
            mesh_aabb(asset, (NAN, 1.0, 1.0))
        message = str(excinfo.value)
        assert asset in message
        assert "nan" in message.lower()

    def test_the_refusal_names_the_consequence_it_prevents(self, tmp_path: Path) -> None:
        # The reported extent's floor is why the NaN case was unscreenable, so
        # the message says so rather than only naming the offending value.
        with pytest.raises(ValueError, match=r"1e-4"):
            mesh_aabb(_asset(tmp_path), (NAN, 1.0, 1.0))


class TestTheMjcfSceneLoaderRefusesIt:
    """The reachable route: an ``<asset><mesh scale=...>`` on a mesh-only body."""

    @pytest.mark.parametrize("declared", ["nan 1 1", "1 nan 1", "1 1 nan", "nan nan nan", "inf 1 1"])
    def test_a_mesh_only_body_is_refused(self, tmp_path: Path, declared: str) -> None:
        with pytest.raises(ValueError, match=REFUSAL):
            load_mjcf_scene_objects(_scene(tmp_path, _MESH_ONLY, declared))

    def test_a_finite_declaration_is_measured_as_before(self, tmp_path: Path) -> None:
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, _MESH_ONLY, "1 1 1")))
        assert widget.size == UNIT_SIZE
        assert widget.position == pytest.approx((0.1, 0.1, 0.6))

    def test_the_declaration_reaches_the_measurement_through_the_registry(self, tmp_path: Path) -> None:
        # Premise for the route above: the scale the loader hands ``mesh_aabb``
        # is the one the ``<asset>`` element declared, parsed as three floats.
        path = _scene(tmp_path, _MESH_ONLY, "nan 1 1")
        registry = _parse_mjcf_mesh_assets(ET.parse(path).getroot(), str(tmp_path))
        _resolved, scale = registry["m"]
        assert math.isnan(scale[0])
        assert scale[1:] == (1.0, 1.0)


class TestThePremisesThatMakeTheRefusalNecessary:
    """Why the reader has to check: nobody upstream does, and nothing downstream can see it."""

    @pytest.mark.parametrize("declared", ["nan 1 1", "inf 1 1", "nan nan nan"])
    def test_mujoco_compiles_the_model_the_reader_must_refuse(self, tmp_path: Path, declared: str) -> None:
        # The reader cannot defer the question to the format's owner.
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path(_scene(tmp_path, _MESH_ONLY, declared))
        assert model.nmesh == 1

    def test_the_assets_own_coordinates_are_finite(self, tmp_path: Path) -> None:
        # So the vertex guard that already exists cannot catch this: the file
        # is clean and the transform is not.
        points, _counts, _indices = mesh_assets.load_mesh_geometry(_asset(tmp_path))
        assert all(math.isfinite(v) for point in points for v in point)

    def test_the_extent_floor_is_what_made_the_nan_case_unscreenable(self, tmp_path: Path) -> None:
        # ``zero`` is a finite, legitimate request to flatten an axis, and it
        # reports the same floored 1e-4 extent a NaN axis used to. That floor
        # is why screening the reported fields for non-finite values caught the
        # centre and not the size.
        center, size = mesh_aabb(_asset(tmp_path), (0.0, 1.0, 1.0))
        assert size[0] == 1e-4
        assert all(map(math.isfinite, center))
        assert all(map(math.isfinite, size))

    def test_the_mjcf_route_can_declare_exactly_these_two_values(self) -> None:
        # A malformed or wrong-arity ``scale`` falls back to unit scale, so
        # ``nan`` and ``inf`` are the only non-finite values the format's own
        # parse can hand the measurement.
        unit = (1.0, 1.0, 1.0)
        assert _parse_axis("garbage", default=unit) == unit
        assert _parse_axis("1 1", default=unit) == unit
        parsed = _parse_axis("nan inf -inf", default=unit)
        assert math.isnan(parsed[0])
        assert parsed[1] == POS_INF
        assert parsed[2] == NEG_INF


class TestAFiniteScaleIsUnchanged:
    """Every finite spelling measures exactly what it measured before."""

    @pytest.mark.parametrize(
        ("scale", "center", "size"),
        [
            ((1.0, 1.0, 1.0), UNIT_CENTER, UNIT_SIZE),
            (
                (2.0, 1.0, 0.5),
                (0.20000000298023224, 0.10000000149011612, 0.05000000074505806),
                (0.4000000059604645, 0.20000000298023224, 0.10000000149011612),
            ),
            ((-1.0, 1.0, 1.0), (-0.10000000149011612, 0.10000000149011612, 0.10000000149011612), UNIT_SIZE),
            (
                (0.0, 1.0, 1.0),
                (0.0, 0.10000000149011612, 0.10000000149011612),
                (1e-4, 0.20000000298023224, 0.20000000298023224),
            ),
        ],
        ids=["unit", "non-uniform", "mirrored", "flattened"],
    )
    def test_the_measured_bound_is_untouched(
        self,
        tmp_path: Path,
        scale: tuple[float, float, float],
        center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> None:
        assert mesh_aabb(_asset(tmp_path), scale) == (center, size)

    def test_a_millimetre_asset_still_measures_a_millimetre_bound(self, tmp_path: Path) -> None:
        # The idiom the registry's own docstring names, and the one a finiteness
        # guard must not mistake for a degenerate value.
        _center, size = mesh_aabb(_asset(tmp_path), (0.001, 0.001, 0.001))
        assert size == pytest.approx((0.0002, 0.0002, 0.0002))

    def test_the_default_scale_is_still_unit(self, tmp_path: Path) -> None:
        assert mesh_aabb(_asset(tmp_path)) == (UNIT_CENTER, UNIT_SIZE)


class TestABodyWithCollidableGeometryTakesItsBoundFromItsGeom:
    """The analytic bound wins, so such a body never reaches the measurement.

    That is why the measurement is the wrong place to hold the scale, not a
    reason to leave it unheld: the scale rides onto the scene object from this
    branch too, and the realization applies it to the visual prim's xform
    beside the mesh geom's position and orientation - two values the reader
    already refuses when they are not finite. The refusal for the third now
    lives where the scale is read rather than where it is measured, pinned in
    ``test_mesh_asset_scale_is_refused_wherever_it_is_carried``.

    A finite non-unit scale is used here so the claim is about which branch
    supplies the bound: at ``3 3 3`` the mesh would measure 0.6 m and the box
    is what is reported.
    """

    def test_the_reported_bound_comes_from_the_collidable_geom(self, tmp_path: Path) -> None:
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, _WITH_COLLISION, "3 3 3")))
        assert widget.position == pytest.approx((0.0, 0.0, 0.5))
        assert widget.size == pytest.approx((0.2, 0.2, 0.1))

    def test_the_declared_scale_still_rides_along_when_it_is_usable(self, tmp_path: Path) -> None:
        # The carry is what made the unchecked value reachable; it must survive
        # for a scale the guard accepts.
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, _WITH_COLLISION, "3 3 3")))
        assert widget.mesh_scale == pytest.approx((3.0, 3.0, 3.0))


class TestOneOwnerForEachWording:
    """Every finiteness refusal in the module is raised from a named factory.

    Derived rather than listed, so a third non-finite cause is held to the same
    rule the hour it lands instead of inheriting an exemption by being absent
    from a tuple.
    """

    def test_every_finiteness_refusal_is_raised_from_a_factory(self) -> None:
        tree = ast.parse(inspect.getsource(mesh_assets))
        raised: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            call = node.exc
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if "_non_finite" in call.func.id:
                raised.add(call.func.id)
        assert raised == {"_non_finite_vertex_error", "_non_finite_scale_error"}, raised

    def test_each_factory_carries_its_own_cause(self) -> None:
        # The two causes fail differently - a bad coordinate in the file versus
        # a bad transform from the caller - so they are two wordings, not one.
        vertex = mesh_assets._non_finite_vertex_error("m.stl", 0, (NAN, 0.0, 0.0))
        scale = mesh_assets._non_finite_scale_error("m.stl", (NAN, 1.0, 1.0))
        assert "vertex" in str(vertex)
        assert "scale" in str(scale)
        assert str(vertex) != str(scale)

    def test_the_scale_factory_is_raised_only_by_the_shared_owner(self) -> None:
        # Two consumers reach it now - the measurement and the loader's
        # passthrough - so the factory is raised from one owner they both call
        # rather than from each of them.
        raisers: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(mesh_assets))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Raise)
                    and isinstance(inner.exc, ast.Call)
                    and isinstance(inner.exc.func, ast.Name)
                    and inner.exc.func.id == "_non_finite_scale_error"
                ):
                    raisers.add(node.name)
        assert raisers == {"refuse_non_finite_scale"}, raisers

    def test_the_measurement_reaches_that_owner(self) -> None:
        called = {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(mesh_aabb).strip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_finite_scale" in called, called


class TestTheGuardPrecedesTheParse:
    """A transform that can never be applied is not worth reading a mesh for."""

    def test_a_missing_asset_is_still_named_by_its_own_message(self, tmp_path: Path) -> None:
        # The scale guard must not shadow the file contract for a caller whose
        # path is wrong and whose scale is fine.
        with pytest.raises(FileNotFoundError):
            mesh_aabb(str(tmp_path / "absent.stl"), (1.0, 1.0, 1.0))

    def test_the_scale_is_refused_without_reading_the_file(self, tmp_path: Path) -> None:
        # No asset on disk at all, so only a check that precedes the parse can
        # produce the scale refusal.
        with pytest.raises(ValueError, match=REFUSAL):
            mesh_aabb(str(tmp_path / "absent.stl"), (NAN, 1.0, 1.0))

    def test_the_guard_precedes_the_parse_call(self) -> None:
        # Graded on the calls rather than on the text: the docstring names
        # ``load_mesh_geometry`` before either statement runs, so a string
        # search finds the reference and not the ordering.
        tree = ast.parse(inspect.getsource(mesh_aabb).strip())
        lines = {
            node.func.id: node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_finite_scale" in lines, lines
        assert "load_mesh_geometry" in lines, lines
        assert lines["refuse_non_finite_scale"] < lines["load_mesh_geometry"]
