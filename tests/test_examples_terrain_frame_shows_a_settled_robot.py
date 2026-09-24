"""The terrain example captures its frame after the robot has stopped moving.

``examples/09_procedural_terrain.py`` renders one still per terrain kind and its
whole value is that the still shows a robot resting on the heightfield. A
floating base is spawned SEATED with its feet just clear of the surface (see
:meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine._seat_floating_bases_on_terrain`),
so it drops, and with nothing driving its joints its legs then fold under
gravity. That takes the better part of a second, not a handful of steps -
``tests/simulation/mujoco/test_get_robot_state_names_torque_only_actuation.py``
records the same thing for the shipped Go2, which "sinks from base z 0.445 to
0.20 within 2 s of the first ``step``".

Measured on ``5ebaa72``, when the example stepped a fixed 40 times before
capturing and called that settled ("Let the robot settle so it rests on the
heightfield (real contact)")::

    terrain   step  ncon  trunk_z  |base_lin_vel|
    rough       40     0   0.4581     0.802 m/s     <- nothing touching the ground
    stairs      40     2   0.4565     0.791 m/s
    pyramid     40     2   0.4965     0.791 m/s
    slope       40     2   0.4583     0.785 m/s
    rough      400     4   0.2332     0.007 m/s     <- at rest

Every frame was of a robot falling at ~0.8 m/s, 216-249 mm above the height it
settles to, and on ``rough`` with zero contacts at all.

How long the fall plus fold takes is a property of the model, not a constant -
the Go2 needs ~400 steps and the longer-legged base below needs ~2000 - so the
first test grades the settle budget the example ships against the physics rather
than against a number, and the second pins that the example measures the settle
instead of assuming it.

GL-free and asset-free: synthetic MJCF models, ``mesh=False``, nothing rendered.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import tempfile

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "examples" / "09_procedural_terrain.py"

# What this module means by "stopped moving", in m/s. Declared here rather than
# read from the example so the first test grades the example's settle against the
# physics even when the example publishes no threshold of its own. The two states
# are two orders of magnitude apart (0.8 m/s falling vs a few mm/s of contact
# jitter at rest), so the value needs no tuning per model.
_AT_REST_MPS = 0.01

# A base on a two-link leg with nothing driving its hinges: it drops onto the
# terrain and then buckles, which is the slow half of a quadruped's settle and
# takes ~2000 steps. Shaped after the drive-leg model in
# tests/simulation/mujoco/test_get_robot_state_names_torque_only_actuation.py.
_FOLDING_LEG = """
<mujoco model="folding_leg">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="trunk" pos="0 0 0.5">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.05" mass="2"/>
      <body name="thigh" pos="0 0 -0.05">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.2"/>
        <body name="shank" pos="0 0 -0.2">
          <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <geom name="foot" type="sphere" size="0.03" pos="0 0 -0.2" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hip" ctrlrange="-5 5"/>
    <motor joint="knee" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

# A rigid floating box: no joints to fold, so it settles as soon as it has
# dropped the few cm of seating clearance. The pair is what makes "the settle is
# measured" observable - one constant cannot be right for both.
_RIGID_BOX = """
<mujoco model="rigid_box">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="trunk" pos="0 0 0.2">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.05" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _load_example():
    # A leading digit makes the module unimportable by name; load it by path.
    spec = importlib.util.spec_from_file_location("procedural_terrain", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shipped_settle_budget() -> int:
    """The ``--steps`` default the example ships, read from its own source.

    The parser is built inside ``main()``, so the default is not reachable by
    calling anything; it is read out of the argparse call instead. The assertion
    that uses it is a physical one - this only locates the number.
    """
    tree = ast.parse(_EXAMPLE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--steps"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return int(ast.literal_eval(keyword.value))
    raise AssertionError(f"{_EXAMPLE.name} declares no --steps default")


def _world_with(xml: str, kind: str) -> tuple[Simulation, str]:
    directory = tempfile.mkdtemp()
    path = pathlib.Path(directory) / "robot.xml"
    path.write_text(xml, encoding="utf-8")
    sim = Simulation()
    assert sim.create_world(terrain=kind)["status"] != "error"
    assert sim.add_robot("floater", urdf_path=str(path))["status"] != "error"
    return sim, "floater"


def _speed(sim: Simulation, robot: str) -> float:
    return sum(float(v) ** 2 for v in sim.get_observation(robot)["base_lin_vel"]) ** 0.5


# One row per terrain kind, since the seating clearance and therefore the fall
# differ between a noise field and a stepped one.
@pytest.mark.parametrize("kind", ["rough", "stairs", "pyramid", "slope"])
def test_the_shipped_settle_budget_is_enough_for_the_robot_to_stop_moving(kind: str) -> None:
    budget = _shipped_settle_budget()
    sim, robot = _world_with(_FOLDING_LEG, kind)
    try:
        for _ in range(budget):
            sim.step()
        speed = _speed(sim, robot)
        assert speed <= _AT_REST_MPS, (
            f"after the example's {budget} settle steps on {kind} the base is still moving at "
            f"{speed:.3f} m/s, so the captured frame is of a falling robot"
        )
    finally:
        sim.destroy()


def test_the_settle_is_measured_so_it_adapts_to_the_model() -> None:
    module = _load_example()
    # The example's own threshold may be stricter than this module's but never
    # looser, or "settled" there would mean something this module calls moving.
    assert module.SETTLED_SPEED_MPS <= _AT_REST_MPS
    budget = _shipped_settle_budget()
    used = {}
    for name, xml in (("box", _RIGID_BOX), ("leg", _FOLDING_LEG)):
        sim, robot = _world_with(xml, "rough")
        try:
            used[name] = module.settle(sim, robot, budget)
            assert _speed(sim, robot) <= module.SETTLED_SPEED_MPS
        finally:
            sim.destroy()
    # A fixed step count cannot be right for both: the rigid box has no joints to
    # buckle and is done an order of magnitude sooner than the folding leg.
    assert used["box"] < used["leg"], used


def test_a_base_that_never_stops_moving_is_refused_rather_than_captured() -> None:
    module = _load_example()
    sim, robot = _world_with(_FOLDING_LEG, "rough")
    try:
        with pytest.raises(RuntimeError, match="still moving"):
            module.settle(sim, robot, module.SETTLE_SAMPLE_STEPS)
    finally:
        sim.destroy()
