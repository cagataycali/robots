"""A converted robot does not bring its MJCF's floor into the scene.

Menagerie's ``scene.xml`` puts a ``floor`` plane on ``<worldbody>``, and the Isaac
MJCF importer writes it under the robot's ``Geometry`` although ``add_robot``
converts with ``import_scene=False``. It then lies exactly on
``create_world()``'s ground plane, and the two coplanar surfaces z-fight: every
Isaac camera frame of the floor was streak noise. The post-import fix-up
deactivates every geom the MJCF attaches to the world, and nothing else.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
Usd = pytest.importorskip("pxr.Usd")
from pxr import UsdGeom, UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402

_MJCF = """<mujoco model="probe">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <geom name="table top" type="box" size=".3 .3 .02" pos="0 0 .4"/>
    <body name="base">
      <geom name="base_geom" type="box" size=".05 .05 .05"/>
      <body name="link"><joint name="j"/><geom type="capsule" size=".02" fromto="0 0 0 0 0 .2"/></body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def converted(tmp_path) -> tuple[str, str]:
    """An MJCF and a USD laid out the way the importer lays it out."""
    mjcf = tmp_path / "scene.xml"
    mjcf.write_text(_MJCF)
    usd = str(tmp_path / "scene.usda")
    stage = Usd.Stage.CreateNew(usd)
    root = UsdGeom.Xform.Define(stage, "/probe").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/probe/Geometry")
    UsdGeom.Plane.Define(stage, "/probe/Geometry/floor")
    UsdGeom.Cube.Define(stage, "/probe/Geometry/tn__tabletop_rA")  # "table top", name-mangled
    base = UsdGeom.Xform.Define(stage, "/probe/Geometry/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    stage.GetRootLayer().Save()
    return str(mjcf), usd


def test_the_worldbody_geoms_are_named_and_the_robots_are_not(converted) -> None:
    mjcf, _ = converted
    assert mjcf_assets._worldbody_geom_names(mjcf) == ["floor", "table top"]


def test_the_floor_is_deactivated_and_the_robot_is_left_alone(converted) -> None:
    mjcf, usd = converted

    dropped = mjcf_assets._deactivate_worldbody_geoms(usd, mjcf)

    assert sorted(dropped) == ["floor", "table top"]
    stage = Usd.Stage.Open(usd)
    assert not stage.GetPrimAtPath("/probe/Geometry/floor").IsActive()
    assert not stage.GetPrimAtPath("/probe/Geometry/tn__tabletop_rA").IsActive()
    assert stage.GetPrimAtPath("/probe/Geometry/base").IsActive()


def test_the_cache_key_moves_so_old_entries_are_rebuilt() -> None:
    # Entries converted before this fix carry the floor; the key has to change.
    assert "worldgeoms" in mjcf_assets._POSTPROCESS_VERSION


def _position_drives_are_not_the_point(monkeypatch) -> None:
    monkeypatch.setattr(mjcf_assets, "_author_position_drives", lambda usd, mjcf: None)


def test_a_conversion_that_asked_for_the_scene_keeps_its_floor(converted, monkeypatch) -> None:
    """``import_scene=True`` is a request for the floor; the fix-up must not undo it."""
    mjcf, usd = converted
    _position_drives_are_not_the_point(monkeypatch)

    mjcf_assets._post_import_fixups(usd, mjcf, import_scene=True)

    stage = Usd.Stage.Open(usd)
    assert stage.GetPrimAtPath("/probe/Geometry/floor").IsActive()
    assert stage.GetPrimAtPath("/probe/Geometry/tn__tabletop_rA").IsActive()


def test_a_robot_only_conversion_still_drops_its_floor(converted, monkeypatch) -> None:
    mjcf, usd = converted
    _position_drives_are_not_the_point(monkeypatch)

    mjcf_assets._post_import_fixups(usd, mjcf, import_scene=False)

    stage = Usd.Stage.Open(usd)
    assert not stage.GetPrimAtPath("/probe/Geometry/floor").IsActive()
    assert stage.GetPrimAtPath("/probe/Geometry/base").IsActive()

