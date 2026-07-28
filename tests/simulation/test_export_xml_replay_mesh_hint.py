"""Regression tests: an unopenable mesh in replace_scene_mjcf names the cause.

``export_xml`` documents that ``spec.to_xml()`` writes mesh assets as BARE filenames
(``file="link3.stl"``) and emits no ``meshdir``, so the XML compiles only from the
directory holding those assets.

That caveat quietly breaks the natural recovery from a discarded scene - export
before a rebuild, re-apply afterwards - as soon as a ROBOT is in the scene, because
its URDF meshes are the ones that cannot be found:

    load_scene(...); add_robot("a")
    xml = export_xml()
    replace_scene_mjcf(xml=xml)
      -> "MJCF compile failed: Error: Error opening file 'link3.stl'"

Nothing tied that back to the documented export caveat, so the remedy was invisible.
The error now names it and lists the fixes, both of which are verified below:

* export BEFORE adding robots (an object-only scene round-trips cleanly), or
* inject a ``<compiler meshdir="...">`` pointing at the asset directory.
"""

from __future__ import annotations

import os

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_SCENE = """<mujoco model="s">
 <worldbody><light pos="0 0 3"/><geom name="f" type="plane" size="5 5 .1"/>
  <body name="cup" pos="0.5 0 0.4"><freejoint/>
   <geom name="cg" type="cylinder" size="0.04 0.05" mass="0.2"/></body>
 </worldbody>
 <sensor><framepos name="cp" objtype="body" objname="cup"/></sensor>
</mujoco>"""


def _export(sim) -> str:
    result = sim.export_xml()
    assert result["status"] == "success", result
    for block in result["content"]:
        if "json" in block and "xml" in block["json"]:
            return block["json"]["xml"]
    raise AssertionError("export_xml returned no xml payload")


@pytest.fixture
def scene_path(tmp_path):
    path = tmp_path / "s.xml"
    path.write_text(_SCENE)
    return str(path)


@pytest.fixture
def sim():
    s = Simulation(tool_name="export_replay_mesh_hint", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def test_replaying_a_robot_export_names_the_meshdir_cause(sim) -> None:
    """The core defect: a bare 'Error opening file' with no path to a fix."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    xml = _export(sim)

    result = sim.replace_scene_mjcf(xml=xml)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "Error opening file" in text, "MuJoCo's own reason must still be verbatim"
    assert "meshdir" in text
    assert "export_xml" in text


def test_an_unrelated_compile_error_gets_no_mesh_hint(sim) -> None:
    """Guard against pinning the hint to every failure."""
    result = sim.replace_scene_mjcf(xml="<mujoco><worldbody><geom type='nosuch' size='1'/></worldbody></mujoco>")
    assert result["status"] == "error"
    assert "meshdir" not in result["content"][0]["text"]


def test_the_world_survives_the_refused_replace(sim) -> None:
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    before = int(sim.mj_model.nbody)
    sim.replace_scene_mjcf(xml=_export(sim))
    assert int(sim.mj_model.nbody) == before
    assert sim.step(n_steps=20)["status"] == "success"


def test_remedy_export_before_adding_robots_round_trips(sim, scene_path) -> None:
    """First suggested fix: an object-only export replays cleanly.

    This is the working recovery from the discarded-scene warning: capture the
    loaded scene BEFORE any robot exists, then re-apply it after the rebuild.
    """
    assert sim.load_scene(scene_path=scene_path)["status"] == "success"
    xml = _export(sim)

    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cup") < 0, "expected the documented discard"

    assert sim.replace_scene_mjcf(xml=xml)["status"] == "success"
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cup") >= 0
    assert int(sim.mj_model.nsensor) == 1


def test_remedy_injecting_a_meshdir_makes_a_robot_export_replayable(sim) -> None:
    """Second suggested fix: point the compiler at the asset directory."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    xml = _export(sim)

    base = sim._world._backend_state.get("robot_base_xml") or ""
    mesh_dir = os.path.join(os.path.dirname(base), "assets")
    if not os.path.isdir(mesh_dir):
        pytest.skip("panda assets are not on this machine")

    patched = xml.replace('<compiler angle="radian"/>', f'<compiler angle="radian" meshdir="{mesh_dir}"/>', 1)
    assert patched != xml, "expected a <compiler> element to patch"
    assert sim.replace_scene_mjcf(xml=patched)["status"] == "success"


def test_an_object_only_export_needs_no_remedy(sim) -> None:
    """No meshes, no problem - the caveat is specific to mesh assets."""
    assert (
        sim.add_object(name="c", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.2)["status"] == "success"
    )
    assert sim.replace_scene_mjcf(xml=_export(sim))["status"] == "success"
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "c") >= 0
