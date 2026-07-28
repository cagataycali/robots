"""Regression tests: losing a raw-MJCF scene to a rebuild is not silent.

``eject_robot_from_scene`` (the FULL rebuild that removing ANY robot triggers)
reconstructs the scene declaratively from ``world.objects`` / ``world.cameras``.
Those registries cannot describe raw MJCF, so everything ``replace_scene_mjcf``
introduced is discarded. Measured on a scene with two hand-written bodies, a
sensor and a site:

    after replace_scene_mjcf   nbody=25 nsensor=1 nsite=1
    after remove_robot         nbody=12 nsensor=0 nsite=0

and ``remove_robot`` reported only ``"Robot 'b' removed."``. A sensor-conditioned
policy went blind mid-episode with nothing in the tool result to explain it.

Preserving arbitrary MJCF across the rebuild would mean tracking it in
``SimWorld``, which the registries deliberately do not model - so the fix is to
make the loss VISIBLE: the scene is flagged as agent-authored and ``remove_robot``
says what was dropped and how to avoid it. Scenes built through
``add_object`` / ``add_robot`` are unaffected and their message is byte-identical
to before.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_RAW_XML = """<mujoco>
 <worldbody>
  <light pos="0 0 3"/><geom name="ground" type="plane" size="5 5 .1"/>
  <body name="pole" pos="0 0 0.5"><joint name="h" type="hinge" axis="0 1 0"/>
   <geom name="pg" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" mass="1"/>
   <site name="tip" pos="0 0 -0.4"/></body>
 </worldbody>
 <sensor><jointpos name="sp" joint="h"/></sensor>
</mujoco>"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="raw_mjcf_scene_discard", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def _two_robots(sim) -> None:
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"


def test_raw_mjcf_scene_is_flagged(sim) -> None:
    assert not sim._world._backend_state.get("raw_mjcf_scene")
    assert sim.replace_scene_mjcf(xml=_RAW_XML)["status"] == "success"
    assert sim._world._backend_state.get("raw_mjcf_scene") is True


def test_remove_robot_warns_that_the_scene_was_discarded(sim) -> None:
    """The core defect: the loss used to be completely silent."""
    assert sim.replace_scene_mjcf(xml=_RAW_XML)["status"] == "success"
    _two_robots(sim)
    result = sim.remove_robot(name="b")

    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "replace_scene_mjcf" in text
    assert "DISCARDED" in text


def test_the_warning_describes_a_loss_that_really_happens(sim) -> None:
    """Guard against the warning outliving the behaviour it documents."""
    assert sim.replace_scene_mjcf(xml=_RAW_XML)["status"] == "success"
    _two_robots(sim)
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pole") >= 0
    assert int(sim.mj_model.nsensor) == 1

    assert sim.remove_robot(name="b")["status"] == "success"
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pole") < 0
    assert int(sim.mj_model.nsensor) == 0


def test_an_ordinary_scene_message_is_unchanged(sim) -> None:
    """No warning noise for scenes the registries CAN describe."""
    _two_robots(sim)
    result = sim.remove_robot(name="b")
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Robot 'b' removed."


def test_a_registry_built_scene_actually_survives(sim) -> None:
    """The counterpart the warning points users toward."""
    _two_robots(sim)
    assert (
        sim.add_object(name="cube", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.2)["status"]
        == "success"
    )
    assert sim.remove_robot(name="b")["status"] == "success"
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0


def test_robot_declared_sensors_still_survive(sim) -> None:
    """Sensors from a robot's own MJCF come back via spec.attach; only raw ones are lost."""
    assert sim.add_robot(name="g1", data_config="unitree_g1")["status"] == "success"
    assert sim.add_robot(name="h1", data_config="unitree_h1", position=[2, 0, 0])["status"] == "success"
    before = int(sim.mj_model.nsensor)
    assert before > 0, "expected the humanoid to declare IMU sensors"

    assert sim.remove_robot(name="h1")["status"] == "success"
    names = [mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(sim.mj_model.nsensor)]
    assert names, "the surviving robot's sensors were dropped"
    assert all((n or "").startswith("g1/") for n in names)
