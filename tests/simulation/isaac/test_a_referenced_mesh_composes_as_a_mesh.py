"""A referenced mesh asset composes as a ``Mesh``, not an empty ``Xform``.

``add_reference_to_stage`` defines a missing target prim as ``Xform`` and then
adds the reference. A local ``typeName`` beats the one a reference brings, and
every mesh USD ``convert_mesh_to_usd`` writes has a ``Mesh`` default prim, so
``add_object(shape="mesh")`` and ``load_scene``'s mesh visuals put an ``Xform``
carrying mesh attributes on the stage: the RTX renderer draws nothing, and the
collision APIs ``SingleGeometryPrim`` applies have no geometry under them.
Measured on Isaac Sim 6.0.1 and 6.1.0 (a live GPU probe run): the whole
subtree of ``/World/Objects/widget`` was one ``Xform`` prim.

The stage here is a real in-memory ``pxr`` stage and ``add_reference_to_stage``
is a faithful copy of the vendor function's define-then-reference semantics.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
Usd = pytest.importorskip("pxr.Usd")
from pxr import UsdGeom  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac.mesh_assets import convert_mesh_to_usd  # noqa: E402
from strands_robots.simulation.isaac.simulation import _adopt_referenced_type  # noqa: E402

_TETRA_OBJ = "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def _vendor_add_reference_to_stage(stage: Any, usd_path: str, prim_path: str, prim_type: str = "Xform") -> Any:
    """isaacsim.core.utils.stage.add_reference_to_stage, 6.0.1 / 6.1.0."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, prim_type)
    prim.GetReferences().AddReference(usd_path)
    return prim


@pytest.fixture
def mesh_usd(tmp_path: pathlib.Path) -> str:
    obj = tmp_path / "widget.obj"
    obj.write_text(_TETRA_OBJ, encoding="utf-8")
    return convert_mesh_to_usd(str(obj), cache_dir=str(tmp_path / "cache"))


class TestAReferencedMesh:
    def test_the_vendor_placeholder_type_hides_the_mesh(self, mesh_usd: str) -> None:
        """The defect, pinned so the premise is re-checked on a pxr upgrade."""
        stage = Usd.Stage.CreateInMemory()
        prim = _vendor_add_reference_to_stage(stage, mesh_usd, "/World/Objects/widget")
        assert prim.GetTypeName() == "Xform"
        assert not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim))

    def test_adopting_the_referenced_type_composes_a_mesh(self, mesh_usd: str) -> None:
        stage = Usd.Stage.CreateInMemory()
        prim = _vendor_add_reference_to_stage(stage, mesh_usd, "/World/Objects/widget")
        _adopt_referenced_type(prim)

        assert prim.IsA(UsdGeom.Mesh)
        assert len(UsdGeom.Mesh(prim).GetPointsAttr().Get()) == 4
        # Still transformable where add_object / load_scene pose it.
        UsdGeom.Xformable(prim).AddTranslateOp().Set((0.4, 0.0, 0.2))
        assert tuple(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()) == pytest.approx(
            (0.4, 0.0, 0.2)
        )

    def test_an_xform_asset_stays_an_xform(self, tmp_path: pathlib.Path) -> None:
        src = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
        root = UsdGeom.Xform.Define(src, "/Robot")
        UsdGeom.Mesh.Define(src, "/Robot/body")
        src.SetDefaultPrim(root.GetPrim())
        src.Save()
        stage = Usd.Stage.CreateInMemory()
        prim = _vendor_add_reference_to_stage(stage, str(tmp_path / "robot.usda"), "/World/Objects/robot")
        _adopt_referenced_type(prim)

        assert prim.GetTypeName() == "Xform"
        assert stage.GetPrimAtPath("/World/Objects/robot/body").IsA(UsdGeom.Mesh)

    def test_a_stand_in_loader_returning_none_is_a_no_op(self) -> None:
        _adopt_referenced_type(None)
