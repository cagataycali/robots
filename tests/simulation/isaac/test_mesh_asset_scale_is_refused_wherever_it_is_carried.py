"""A non-finite mesh scale is refused on every body that carries it, not only where it is measured.

:func:`~strands_robots.simulation.isaac.mesh_assets.mesh_aabb` refuses a scale
whose components are not all finite, because it measures ``vertex * scale`` and
``min``/``max`` order a NaN as neither smaller nor larger than anything, so the
running comparison drops every vertex on that axis.

The scale has a *second* consumer.
:func:`~strands_robots.simulation.isaac.loaders.load_mjcf_scene_objects` carries
it onto the :class:`~strands_robots.simulation.isaac.loaders.SceneObject` it
builds, and the Isaac realization applies it to the visual prim's xform. That
happens on every body that declares a mesh, while the *measurement* happens only
on the branch where no collidable analytic geom supplied the bound - so a check
that lived only at the measurement covered the minority of the bodies that carry
a scale.

The reason to hold the scale to the same test is the call it lands in. The
realization authors the visual prim with one
``_author_local_xform(translate=..., orient_wxyz=..., scale=...)``, whose three
arguments come from ``mesh_pos``, ``mesh_quat`` and ``mesh_scale``. The first two
are already refused when they are not finite. On a body with a collidable box,
``pos="nan 0 0"`` on the mesh geom and ``quat="nan 1 0 0"`` on the mesh geom were
both refused while ``scale="nan 1 1"`` on the asset it references was reported as
``mesh_scale=(nan, 1.0, 1.0)`` under ``success`` - and ``position``, ``size`` and
``offset`` all stayed finite and correct, because the collidable geom supplied
them. Every field a consumer would screen for a non-finite value was healthy.

The classes below pin, in order: the refusal on both templates for both values on
every axis; that the two siblings of the same xform call were already refused,
which is the reason this one is too; that a finite scale - negative, zero and
non-unit included - is unchanged on both templates; the premises that make the
refusal necessary (MuJoCo compiles the model, and the collidable body really does
take its bound from its geom rather than from the measurement); and that the
finiteness test has a single owner both consumers reach.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

from strands_robots.simulation.isaac import loaders, mesh_assets
from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects
from strands_robots.simulation.isaac.mesh_assets import mesh_aabb, refuse_non_finite_scale
from tests.simulation.isaac.test_mesh_asset_scale_must_be_finite import (
    _MESH_ONLY,
    _WITH_COLLISION,
    AXES,
    NAN,
    NON_FINITE,
    REFUSAL,
    _asset,
    _scale,
    _scene,
    _widget,
)

#: Both body shapes, named by which branch supplies the reported bound. The
#: measurement is reached only by the first; the carry happens on both.
TEMPLATES = (("mesh-only", _MESH_ONLY), ("with-collision", _WITH_COLLISION))

#: A finite scale that is not unit, so a control cannot pass by the value
#: happening to be the default.
FINITE_NON_UNIT = "3 3 3"


def _spell(scale: tuple[float, float, float]) -> str:
    return " ".join(repr(component) for component in scale)


class TestTheScaleIsRefusedOnEveryBodyThatCarriesIt:
    """Both templates, both non-finite values, every axis.

    The mesh-only rows were already refused at the measurement; the
    with-collision rows are the ones that reported success while carrying the
    value onto the scene object.
    """

    @pytest.mark.parametrize("shape,template", TEMPLATES, ids=[name for name, _ in TEMPLATES])
    @pytest.mark.parametrize("axis", AXES)
    @pytest.mark.parametrize("value", NON_FINITE, ids=["nan", "inf", "-inf"])
    def test_the_scene_is_refused(self, tmp_path: Path, shape: str, template: str, axis: int, value: float) -> None:
        scene = _scene(tmp_path, template, _spell(_scale(axis, value)))
        with pytest.raises(ValueError, match=REFUSAL):
            load_mjcf_scene_objects(scene)

    def test_the_refusal_names_the_asset_and_the_value(self, tmp_path: Path) -> None:
        scene = _scene(tmp_path, _WITH_COLLISION, "nan 1 1")
        with pytest.raises(ValueError) as caught:
            load_mjcf_scene_objects(scene)
        text = str(caught.value)
        assert "asset.stl" in text, text
        assert "nan" in text.lower(), text

    def test_the_refusal_names_the_visual_xform_as_well_as_the_measurement(self, tmp_path: Path) -> None:
        # The value reaches the xform on a body that never measures, so a
        # message about sizing alone would describe the wrong consumer.
        scene = _scene(tmp_path, _WITH_COLLISION, "nan 1 1")
        with pytest.raises(ValueError) as caught:
            load_mjcf_scene_objects(scene)
        text = str(caught.value)
        assert "visual prim" in text, text
        assert "extent floored at 1e-4" in text, text


class TestTheSiblingsOfTheSameXformCallWereAlreadyRefused:
    """Why this value is held to the same test: it travels with two that are.

    ``mesh_pos``, ``mesh_quat`` and ``mesh_scale`` are the three arguments of one
    ``_author_local_xform`` call in the Isaac realization. These pin that the
    first two are refused on a body whose bound comes from its collidable geom -
    the shape where the scale used to be carried unchecked.
    """

    def test_a_non_finite_mesh_geom_position_is_refused(self, tmp_path: Path) -> None:
        _asset(tmp_path)
        scene = tmp_path / "scene.xml"
        scene.write_text(_WITH_COLLISION.format(scale="1 1 1").replace('mesh="m"', 'mesh="m" pos="nan 0 0"'))
        with pytest.raises(ValueError, match="pos has a component that is not finite"):
            load_mjcf_scene_objects(str(scene))

    def test_a_non_finite_mesh_geom_orientation_is_refused(self, tmp_path: Path) -> None:
        _asset(tmp_path)
        scene = tmp_path / "scene.xml"
        scene.write_text(_WITH_COLLISION.format(scale="1 1 1").replace('mesh="m"', 'mesh="m" quat="nan 1 0 0"'))
        with pytest.raises(ValueError, match="quat has a component that is not finite"):
            load_mjcf_scene_objects(str(scene))

    def test_the_realization_applies_all_three_in_one_xform_call(self) -> None:
        # Derived, so the premise cannot rot into naming a call the realization
        # no longer makes. The three fields must be arguments of one call.
        source = inspect.getsource(loaders)
        assert "mesh_scale" in source, "the loader no longer carries a mesh scale"
        simulation = inspect.getsource(__import__("strands_robots.simulation.isaac.simulation", fromlist=["x"]))
        tree = ast.parse(simulation)
        together = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and {kw.arg for kw in node.keywords} >= {"translate", "orient_wxyz", "scale"}
        ]
        assert together, "no xform call takes the three fields together any more"
        rendered = {ast.unparse(node) for node in together}
        assert any("mesh_scale" in text and "mesh_quat" in text for text in rendered), rendered


class TestAFiniteScaleIsUnchanged:
    """Negative, zero and non-unit scales still load, on both templates.

    250 of the 1746 mesh assets the shipped registry declares carry a finite
    non-unit scale, so this is the population the guard must not touch.
    """

    @pytest.mark.parametrize("shape,template", TEMPLATES, ids=[name for name, _ in TEMPLATES])
    @pytest.mark.parametrize("scale", ("1 1 1", "3 3 3", "-1 2 1", "0 1 1"))
    def test_the_scene_loads(self, tmp_path: Path, shape: str, template: str, scale: str) -> None:
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, template, scale)))
        assert all(math.isfinite(component) for component in widget.mesh_scale)
        assert all(math.isfinite(component) for component in widget.size)

    def test_the_declared_scale_still_reaches_the_scene_object(self, tmp_path: Path) -> None:
        # The guard refuses a value; it must not stop a usable one being carried.
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, _WITH_COLLISION, FINITE_NON_UNIT)))
        assert widget.mesh_scale == pytest.approx((3.0, 3.0, 3.0))


class TestThePremisesThatMakeTheRefusalNecessary:
    """MuJoCo compiles it, and the collidable body really never measures."""

    @pytest.mark.parametrize("shape,template", TEMPLATES, ids=[name for name, _ in TEMPLATES])
    def test_mujoco_compiles_the_model(self, tmp_path: Path, shape: str, template: str) -> None:
        # The compiler only warns, so the reader cannot defer the question to it.
        mujoco = pytest.importorskip("mujoco")
        scene = _scene(tmp_path, template, "nan 1 1")
        assert mujoco.MjModel.from_xml_path(scene) is not None

    def test_the_collidable_body_takes_its_bound_from_its_geom(self, tmp_path: Path) -> None:
        # A 3x scale would give the mesh a 0.6 m bound; the box is what is
        # reported, which is what makes the measurement unreachable here.
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, _WITH_COLLISION, FINITE_NON_UNIT)))
        assert widget.size == pytest.approx((0.2, 0.2, 0.1))

    def test_the_mesh_only_body_does_measure_through_the_scale(self, tmp_path: Path) -> None:
        widget = _widget(load_mjcf_scene_objects(_scene(tmp_path, _MESH_ONLY, FINITE_NON_UNIT)))
        assert widget.size[0] == pytest.approx(0.6, abs=1e-3)


class TestTheFinitenessTestHasOneOwnerBothConsumersReach:
    """One test for the scale, so the two consumers cannot drift apart.

    Derived over the module rather than listed, so a third consumer is held to
    the same rule the hour it lands.
    """

    def test_only_the_owner_tests_the_scale_for_finiteness(self) -> None:
        tree = ast.parse(inspect.getsource(mesh_assets))
        owners: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr == "isfinite":
                    owners.add(node.name)
        assert owners, "the scan found no finiteness test at all - it is looking in the wrong place"
        assert owners == {"load_mesh_geometry", "refuse_non_finite_scale"}, owners

    def test_the_measurement_reaches_the_owner(self) -> None:
        called = {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(mesh_aabb).strip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_finite_scale" in called, called

    def test_the_loader_reaches_the_owner(self) -> None:
        called = {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(loaders.load_mjcf_scene_objects).strip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_finite_scale" in called, called

    def test_the_owner_is_reached_before_the_bound_is_built(self) -> None:
        # Refusing after the object is appended would report a scene the caller
        # cannot use; refusing after the measurement would keep the old hole.
        tree = ast.parse(inspect.getsource(loaders.load_mjcf_scene_objects).strip())
        guard = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "refuse_non_finite_scale"
        ]
        built = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SceneObject"
        ]
        measured = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "mesh_aabb"
        ]
        assert len(guard) == 1, guard
        assert built and measured, (built, measured)
        assert guard[0] < min(measured), (guard, measured)
        assert guard[0] < min(built), (guard, built)

    def test_the_owner_accepts_a_finite_scale_and_refuses_a_non_finite_one(self) -> None:
        # Only the raising half is asserted: the owner is annotated ``-> None``,
        # so "it returns nothing" is a static fact rather than a runtime one.
        refuse_non_finite_scale("m.stl", (-1.0, 0.0, 3.0))
        with pytest.raises(ValueError, match=REFUSAL):
            refuse_non_finite_scale("m.stl", (NAN, 1.0, 1.0))
