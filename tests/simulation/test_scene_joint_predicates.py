"""Regression tests: joint predicates can score articulated SCENE joints.

``_joint_position`` sourced joint values from ``sim.get_observation()``, which
enumerates only a registered robot's ``joint_names``. An articulated scene joint
- a drawer slide, a door or cabinet hinge - never appears there, so the lookup
returned ``None`` and the referencing term degraded to ``False`` / ``0.0``.

That silently zeroed exactly the tasks these predicates exist for: a drawer
physically open at ``qpos=0.29`` failed its own ``> 0.15`` success check and
reported ``0.0`` progress, and every LIBERO ``(open X)`` / ``(closed X)`` goal
(which compiles to ``joint_above`` / ``joint_below``) scored a permanent 0% while
the task was solved.

The predicates now probe the backend's ``get_joint_state`` first - which reads
``mjData`` directly and so covers every joint in the scene - and fall back to the
observation dict when a backend has no such accessor.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation import predicates as P  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A cabinet whose drawer rides a named SLIDE joint. No robot is registered, so
# get_observation() yields nothing at all - the pre-fix blind spot.
_DRAWER_XML = """
<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="3 3 .1"/>
    <body name="cabinet" pos="0.5 0 0.2">
      <geom type="box" size=".15 .2 .2"/>
      <body name="drawer" pos="0 0 0">
        <joint name="drawer_slide" type="slide" axis="1 0 0" range="0 0.3"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
        <geom type="box" size=".14 .18 .08" rgba="0 1 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def drawer(tmp_path):
    path = tmp_path / "drawer.xml"
    path.write_text(_DRAWER_XML)
    sim = Simulation(tool_name="scene_joint_predicates", mesh=False)
    sim.load_scene(scene_path=str(path))
    yield sim
    sim.destroy()


def _set_drawer(sim, value: float) -> None:
    model, data = sim._world._model, sim._world._data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)


def test_observation_really_is_empty(drawer) -> None:
    """Pin the premise: the scene joint is invisible to get_observation."""
    assert drawer.get_observation("") == {}


def test_get_joint_state_reads_a_scene_joint(drawer) -> None:
    _set_drawer(drawer, 0.29)
    result = drawer.get_joint_state("drawer_slide")
    assert result["status"] == "success"
    payload = P._extract_json(result)
    assert payload["position"] == pytest.approx(0.29)
    assert payload["type"] == "slide"
    assert payload["dof_count"] == 1


def test_get_joint_state_rejects_unknown_joint(drawer) -> None:
    assert drawer.get_joint_state("no_such_joint")["status"] == "error"


def test_open_drawer_satisfies_joint_above(drawer) -> None:
    """The core defect: an open drawer must score its own success threshold."""
    _set_drawer(drawer, 0.29)
    check = P.PREDICATE_REGISTRY["joint_above"](joint="drawer_slide", value=0.15)
    assert check(drawer) is True


def test_closed_drawer_does_not_satisfy_joint_above(drawer) -> None:
    """No false positives: a closed drawer must still fail the threshold."""
    _set_drawer(drawer, 0.0)
    check = P.PREDICATE_REGISTRY["joint_above"](joint="drawer_slide", value=0.15)
    assert check(drawer) is False


def test_closed_drawer_satisfies_joint_below(drawer) -> None:
    _set_drawer(drawer, 0.0)
    check = P.PREDICATE_REGISTRY["joint_below"](joint="drawer_slide", value=0.05)
    assert check(drawer) is True


def test_joint_progress_reports_real_distance(drawer) -> None:
    """Was a dead 0.0 reward; must now track the actual joint value."""
    _set_drawer(drawer, 0.29)
    term = P.PREDICATE_REGISTRY["joint_progress"](joint="drawer_slide", target=0.0)
    assert term(drawer) == pytest.approx(-0.29)


def test_unknown_joint_still_degrades_to_false(drawer) -> None:
    """A typo'd spec must not start reporting success."""
    check = P.PREDICATE_REGISTRY["joint_above"](joint="drawer_slyde", value=0.15)
    assert check(drawer) is False


def test_robot_joints_still_resolve() -> None:
    """The robot path must be unaffected, by namespaced and bare name."""
    sim = Simulation(tool_name="scene_joint_predicates_robot", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="panda")
        model, data = sim._world._model, sim._world._data  # type: ignore[union-attr]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda/joint1")
        data.qpos[model.jnt_qposadr[jid]] = 0.8
        mujoco.mj_forward(model, data)

        for name in ("panda/joint1", "joint1"):
            assert P.PREDICATE_REGISTRY["joint_above"](joint=name, value=0.5)(sim) is True
            assert P.PREDICATE_REGISTRY["joint_above"](joint=name, value=1.5)(sim) is False
    finally:
        sim.destroy()
