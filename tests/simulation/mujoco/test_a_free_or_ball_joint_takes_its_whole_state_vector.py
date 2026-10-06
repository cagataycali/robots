"""A free or ball joint is written as its whole vector, and refuses a single number.

A free joint owns seven ``qpos`` slots ``[x, y, z, qw, qx, qy, qz]`` and six
``qvel`` slots; a ball joint four and three. ``set_joint_positions`` refused the
vector (``must be a number``) and accepted a single number, which it wrote into
``qpos[adr]`` alone and reported as success with ``y``, ``z`` and the quaternion
left where they were - so seating a free body, as the microduck kick guide
tells the reader to, had no supported call.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")

_SCENE = """
<mujoco model="free_and_ball">
  <worldbody>
    <body name="puck" pos="0 0 0.1">
      <freejoint name="puck_free"/>
      <geom type="sphere" size="0.03" mass="0.1"/>
    </body>
    <body name="arm" pos="0 0 0.5">
      <joint name="shoulder" type="ball"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.1"/>
      <body name="fore" pos="0 0 -0.2">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sim(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "scene.xml"
    path.write_text(_SCENE)
    s = Simulation()
    s.create_world()
    assert s.add_robot("rig", urdf_path=str(path))["status"] == "success"
    yield s
    s.destroy()


def _slots(sim, joint: str, vel: bool = False) -> list[float]:
    import mujoco

    m, d = sim._world._model, sim._world._data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"rig/{joint}")
    kind = int(m.jnt_type[jid])
    width = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}.get(kind, (1, 1))
    adr = int(m.jnt_dofadr[jid] if vel else m.jnt_qposadr[jid])
    return [float(v) for v in (d.qvel if vel else d.qpos)[adr : adr + width[vel]]]


@pytest.mark.parametrize(
    ("joint", "written", "expected"),
    [
        ("puck_free", [0.5, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0], [0.5, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
        # A non-unit quaternion is normalized on write, as MuJoCo itself reads it.
        ("puck_free", [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 2.0], [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        ("shoulder", [0.0, 3.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_the_whole_position_vector_is_written(sim, joint, written, expected):
    result = sim.set_joint_positions({joint: written}, robot_name="rig")
    assert result["status"] == "success", result
    assert _slots(sim, joint) == pytest.approx(expected)


def test_the_whole_velocity_vector_is_written(sim):
    twist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert sim.set_joint_velocities({"puck_free": twist, "shoulder": [1.0, 0.0, 0.0]}, robot_name="rig")["status"] == (
        "success"
    )
    assert _slots(sim, "puck_free", vel=True) == pytest.approx(twist)
    assert _slots(sim, "shoulder", vel=True) == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("method", "values", "says"),
    [
        ("set_joint_positions", {"puck_free": 0.5}, "free joint and takes 7 'positions' values [x, y, z, qw"),
        ("set_joint_positions", {"shoulder": 0.5}, "ball joint and takes 4 'positions' values [qw, qx, qy, qz]"),
        ("set_joint_velocities", {"puck_free": 0.5}, "free joint and takes 6 'velocities' values"),
        ("set_joint_positions", {"puck_free": [0.0, 0.0, 0.1]}, "must be a 7-element vector, got 3"),
        ("set_joint_positions", {"puck_free": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]}, "~zero norm"),
        ("set_joint_positions", {"puck_free": [0.0, 0.0, float("nan"), 1.0, 0.0, 0.0, 0.0]}, "positions['puck_free']"),
        ("set_joint_positions", {"elbow": [0.1, 0.2]}, "joint 'elbow' must be a number (it owns one slot)"),
        # The refusal is all-or-nothing: a valid sibling in the same call is not written.
        ("set_joint_positions", {"elbow": 0.4, "puck_free": 0.5}, "free joint"),
    ],
)
def test_a_value_of_the_wrong_shape_is_refused_and_nothing_moves(sim, method, values, says):
    before = {j: _slots(sim, j) for j in ("puck_free", "shoulder", "elbow")}
    result = getattr(sim, method)(values, robot_name="rig")
    assert result["status"] == "error"
    assert says in result["content"][0]["text"]
    assert {j: _slots(sim, j) for j in before} == before
