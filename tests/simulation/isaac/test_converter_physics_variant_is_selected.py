"""A converter-authored ``Physics`` variantSet composes with PhysX selected.

Isaac Sim 6.1's MJCF/URDF importers are built on mujoco-usd-converter 0.5, which
authors every physics schema a robot has (``ArticulationRootAPI``, joints,
drives, collision) behind a ``Physics`` variantSet ``{mujoco, none, physics,
physx}`` - with no default selection. Referenced as-is the robot composes with
no articulation root at all, and ``SingleArticulation(...)`` dies inside the
tensor API with ``'NoneType' object has no attribute 'is_homogeneous'``: every
``add_robot`` failed on 6.1. 6.0's converter (0.2) flattened physics inline, so
the selection has to be a no-op when the variantSet is absent.

The conversion cache half: the key covered the MJCF bytes but not the toolchain
that wrote the USD, so a 6.0 process sharing ``~/.strands_robots`` with a 6.1
one was handed the 6.1 (variant) layout and vice versa.

Needs ``pxr`` for a real in-memory stage; ``omni.usd`` is faked.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
Usd = pytest.importorskip("pxr.Usd")
from pxr import UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402
from strands_robots.simulation.isaac.simulation import _select_physics_variant  # noqa: E402


def _fake_omni_usd(monkeypatch: pytest.MonkeyPatch, stage) -> None:
    ctx = types.SimpleNamespace(get_stage=lambda: stage)
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: ctx  # type: ignore[attr-defined]
    omni = sys.modules.get("omni") or types.ModuleType("omni")
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)
    monkeypatch.setattr(omni, "usd", omni_usd, raising=False)


def _variant_robot(stage, path: str = "/World/Robots/so100", selection: str = ""):
    prim = stage.DefinePrim(path, "Xform")
    vset = prim.GetVariantSets().AddVariantSet("Physics")
    for name in ("mujoco", "none", "physics", "physx"):
        vset.AddVariant(name)
    vset.SetVariantSelection("physx")
    with vset.GetVariantEditContext():
        UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim(f"{path}/Base", "Xform"))
    vset.ClearVariantSelection()
    if selection:
        vset.SetVariantSelection(selection)
    return prim


def _has_articulation_root(stage) -> bool:
    return any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())


class TestTheUnselectedVariantGetsPhysx:
    def test_an_unselected_variant_composes_without_an_articulation_root(self):
        stage = Usd.Stage.CreateInMemory()
        _variant_robot(stage)
        assert not _has_articulation_root(stage)  # the failure mode itself

    def test_physx_is_selected_and_the_root_composes(self, monkeypatch):
        stage = Usd.Stage.CreateInMemory()
        prim = _variant_robot(stage)
        _fake_omni_usd(monkeypatch, stage)
        assert _select_physics_variant("/World/Robots/so100") == "physx"
        assert prim.GetVariantSets().GetVariantSet("Physics").GetVariantSelection() == "physx"
        assert _has_articulation_root(stage)


class TestNothingElseIsTouched:
    def test_an_explicit_selection_is_kept(self, monkeypatch):
        stage = Usd.Stage.CreateInMemory()
        prim = _variant_robot(stage, selection="mujoco")
        _fake_omni_usd(monkeypatch, stage)
        assert _select_physics_variant("/World/Robots/so100") is None
        assert prim.GetVariantSets().GetVariantSet("Physics").GetVariantSelection() == "mujoco"

    def test_a_flattened_60_layout_is_a_no_op(self, monkeypatch):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World/Robots/so100", "Xform")
        _fake_omni_usd(monkeypatch, stage)
        assert _select_physics_variant("/World/Robots/so100") is None

    def test_a_missing_prim_is_a_no_op(self, monkeypatch):
        _fake_omni_usd(monkeypatch, Usd.Stage.CreateInMemory())
        assert _select_physics_variant("/World/Robots/nope") is None


class TestTheImporterVersionString:
    def test_it_names_both_toolchain_packages(self):
        v = mjcf_assets._importer_version()
        assert v.startswith("isaacsim=") and ",mujoco-usd-converter=" in v
