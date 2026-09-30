"""``randomize(randomize_colors=True)`` changes the colour the renderer uses, not only the one it reports.

``add_object(..., color=)`` binds a visual material (``/World/Looks/visual_material``),
and a bound material wins over the ``displayColor`` primvar the randomizer wrote:
on Isaac Sim 6.1 a red cube was reported recoloured to (0.18, 0.31, 0.82) and still
rendered (119, 37, 37). Each object now gets its own preview-surface material,
bound stronger than its descendants.
"""

from __future__ import annotations

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402


def _stage_with_material_bound_cubes() -> tuple[Usd.Stage, list[Usd.Prim]]:
    stage = Usd.Stage.CreateInMemory()
    shared = UsdShade.Material.Define(stage, "/World/Looks/visual_material")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/visual_material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.12, 0.1))
    shared.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    prims = []
    for name in ("red_cube", "blue_cube"):
        xf = UsdGeom.Xform.Define(stage, f"/World/Objects/{name}")
        UsdGeom.Cube.Define(stage, f"/World/Objects/{name}/geom")
        UsdShade.MaterialBindingAPI.Apply(xf.GetPrim()).Bind(shared)
        prims.append(xf.GetPrim())
    return stage, prims


def _diffuse(prim: Usd.Prim) -> tuple[float, ...]:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader = UsdShade.Shader(material.GetSurfaceOutput().GetConnectedSource()[0].GetPrim())
    return tuple(round(float(v), 3) for v in shader.GetInput("diffuseColor").Get())


def test_the_bound_material_carries_the_randomized_colour() -> None:
    _, (red, blue) = _stage_with_material_bound_cubes()
    IsaacSimulation._bind_randomized_color(red, "red_cube", [0.18, 0.31, 0.82], Gf, Sdf, UsdShade)
    assert _diffuse(red) == (0.18, 0.31, 0.82)
    # The geom child the renderer draws resolves to the same material.
    assert _diffuse(red.GetChild("geom")) == (0.18, 0.31, 0.82)


def test_recolouring_one_object_leaves_another_that_shared_its_material() -> None:
    _, (red, blue) = _stage_with_material_bound_cubes()
    IsaacSimulation._bind_randomized_color(red, "red_cube", [0.1, 0.9, 0.1], Gf, Sdf, UsdShade)
    assert _diffuse(blue) == (0.85, 0.12, 0.1)


def test_a_second_randomization_updates_the_same_material() -> None:
    stage, (red, _) = _stage_with_material_bound_cubes()
    IsaacSimulation._bind_randomized_color(red, "red_cube", [0.1, 0.2, 0.3], Gf, Sdf, UsdShade)
    IsaacSimulation._bind_randomized_color(red, "red_cube", [0.4, 0.5, 0.6], Gf, Sdf, UsdShade)
    assert _diffuse(red) == (0.4, 0.5, 0.6)
    looks = [p.GetName() for p in stage.GetPrimAtPath("/World/Looks").GetChildren()]
    assert looks.count("strands_randomized_red_cube") == 1
