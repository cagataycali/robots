"""A model that ``<attach>``es another MJCF converts to USD with the other model's meshes.

MuJoCo composes a ``<model>`` asset as its own spec, so the attached model keeps
its own ``<compiler meshdir>``. The Isaac MJCF importer resolves every mesh
against the ENTRY model's directory instead, so ``lekiwi`` (a base that attaches
``../so_arm100/so_arm100.xml``) failed with "Mesh Base file .../lekiwi/assets/
Base.stl" while MuJoCo loaded it. Such a model now reaches the importer as one
flattened file with absolute asset paths; any other model reaches it untouched.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("strands_robots.simulation.isaac")

import numpy as np  # noqa: E402

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402

_OBJ = "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"

_ARM = """<mujoco model="arm">
  <compiler angle="radian" meshdir="assets/"/>
  <default><default class="arm"><joint armature="0.1" range="-1.5 1.5"/></default></default>
  <asset><mesh name="link" file="link.obj"/></asset>
  <worldbody>
    <body name="Base">
      <body name="upper" childclass="arm">
        <joint name="shoulder"/>
        <geom type="mesh" mesh="link"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_BASE = """<mujoco model="base">
  <compiler angle="radian"/>
  <asset>
    <mesh name="plate" file="meshes/plate.obj"/>
    <model name="arm" file="../arm/arm.xml"/>
  </asset>
  <worldbody>
    <body name="chassis">
      <freejoint/>
      <geom type="mesh" mesh="plate"/>
      <body name="mount" pos="0 0 0.05"><attach model="arm" body="Base" prefix=""/></body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def attaching(tmp_path) -> str:
    """base/base/base.xml attaches base/arm/arm.xml, whose meshes sit in arm/assets/."""
    root = tmp_path / "base"
    (root / "base" / "meshes").mkdir(parents=True)
    (root / "arm" / "assets").mkdir(parents=True)
    (root / "base" / "meshes" / "plate.obj").write_text(_OBJ)
    (root / "arm" / "assets" / "link.obj").write_text(_OBJ)
    (root / "arm" / "arm.xml").write_text(_ARM)
    (root / "base" / "base.xml").write_text(_BASE)
    scene = root / "scene.xml"
    scene.write_text('<mujoco model="scene"><include file="base/base.xml"/></mujoco>')
    return str(scene)


def test_the_attached_model_is_found_through_an_include(attaching) -> None:
    arm = os.path.join(os.path.dirname(attaching), "arm", "arm.xml")
    assert mjcf_assets._attached_model_files(attaching) == [os.path.normpath(arm)]


def test_the_flattened_copy_names_each_mesh_where_its_own_model_keeps_it(attaching, tmp_path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    flat = mjcf_assets._flatten_attached_models(attaching, str(work))

    assert os.path.dirname(flat) == str(work)
    spec = mujoco.MjSpec.from_file(flat)  # held: a mesh handle outlives no spec
    files = {m.name: m.file for m in spec.meshes}
    root = os.path.dirname(attaching)
    assert files["link"] == os.path.join(root, "arm", "assets", "link.obj")
    assert files["plate"] == os.path.join(root, "base", "meshes", "plate.obj")


def test_the_flattened_copy_compiles_to_the_same_model(attaching, tmp_path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    flat = mjcf_assets._flatten_attached_models(attaching, str(work))
    a, b = mujoco.MjModel.from_xml_path(attaching), mujoco.MjModel.from_xml_path(flat)
    for field in ("nbody", "njnt", "ngeom", "nmesh"):
        assert getattr(a, field) == getattr(b, field), field
    # The attached model's default class still applies to its joint.
    np.testing.assert_allclose(a.jnt_range, b.jnt_range)
    np.testing.assert_allclose(a.dof_armature, b.dof_armature)
    assert b.dof_armature.max() == pytest.approx(0.1)


def test_a_model_that_attaches_nothing_is_handed_over_untouched(tmp_path) -> None:
    path = tmp_path / "plain.xml"
    path.write_text('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    assert mjcf_assets._flatten_attached_models(str(path), str(tmp_path)) == str(path)


class _Importer:
    seen: list[str] = []

    def __init__(self, config: Any) -> None:
        self._config = config

    def import_mjcf(self) -> str:
        # The file the importer is given must exist while it runs.
        assert os.path.isfile(self._config.mjcf_path)
        type(self).seen.append(open(self._config.mjcf_path, encoding="utf-8").read())
        stem = os.path.splitext(os.path.basename(self._config.mjcf_path))[0]
        out = os.path.join(self._config.usd_path, stem, f"{stem}.usda")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("#usda 1.0\n")
        return out


class _Config:
    mjcf_path = usd_path = fix_base = None
    import_scene = True


def test_the_importer_gets_the_flattened_copy_and_it_is_cleaned_up(attaching, tmp_path, monkeypatch) -> None:
    for name in ("isaacsim", "isaacsim.asset", "isaacsim.asset.importer", "isaacsim.asset.importer.mjcf"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = sys.modules["isaacsim.asset.importer.mjcf"]
    module.MJCFImporter = _Importer  # type: ignore[attr-defined]
    module.MJCFImporterConfig = _Config  # type: ignore[attr-defined]
    monkeypatch.setattr(mjcf_assets, "_author_position_drives", lambda *a, **k: None)
    _Importer.seen = []
    cache = tmp_path / "cache"

    result = mjcf_assets.convert_mjcf_to_usd(attaching, str(cache))

    assert os.path.isfile(result)
    assert os.path.join(os.path.dirname(attaching), "arm", "assets", "link.obj") in _Importer.seen[0]
    assert [n for n in os.listdir(cache) if n.startswith(".")] == []
