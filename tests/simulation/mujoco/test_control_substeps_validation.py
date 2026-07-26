"""``control_substeps`` must be honored or rejected, never silently clamped.

``control_substeps`` is how many physics steps are integrated per applied
action. ``PolicyRunner._control_substeps`` exists precisely because integrating
a single ~2 ms ``mj_step`` per action leaves a position-servo arm ~10% of the
way to each target before the next action overwrites ``ctrl`` - the rollout
reports success while the policy looks like a no-op.

That helper resolved an explicit override with ``max(1, int(override))``, so the
public entry points accepted values they could not honor and reinstated the very
pathology the helper prevents:

* ``control_substeps=0`` / ``-5`` collapsed to 1 substep and returned
  ``status="success"`` (2 steps at 50 Hz advanced sim time 0.004 s instead of
  0.04 s - a 10x under-integration);
* ``control_substeps=2.7`` was truncated to 2 without a word;
* ``control_substeps=True`` acted as a silent 1 substep (``bool`` is an ``int``
  subclass);
* ``control_substeps=float("nan")`` reached ``int()`` inside the runner and
  surfaced as a bare ``ValueError`` wrapped in a dead-end "Policy failed"
  message that never names the parameter.

Every sibling knob on the same signatures was already guarded
(``action_horizon`` >= 1, ``n_episodes``/``max_steps`` positive ints,
``control_frequency`` > 0), so these assert the same contract for
``control_substeps`` on all three entry points that expose it, plus the loud
raise for callers driving ``PolicyRunner`` directly.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402

ROBOT_XML = """
<mujoco model="substep_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="10"/>
    <position name="elbow_act" joint="elbow" kp="10"/>
  </actuator>
</mujoco>
"""

# Values the rollout cannot honor: (value, why it used to slip through).
INVALID_SUBSTEPS = [
    pytest.param(0, id="zero"),
    pytest.param(-5, id="negative"),
    pytest.param(2.7, id="float-truncated"),
    pytest.param(True, id="bool-acts-as-one"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param("4", id="string"),
]


@pytest.fixture
def sim_with_robot():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    s = Simulation(tool_name="substeps_validation_sim", mesh=False)
    s.create_world()
    s.add_robot("arm1", urdf_path=path)
    yield s
    s.cleanup()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _spy_substeps(sim) -> list[int]:
    """Record the ``n_substeps`` every ``send_action`` is driven with."""
    captured: list[int] = []
    orig = sim.send_action

    def _spy(action, robot_name=None, n_substeps: int = 1):
        captured.append(int(n_substeps))
        return orig(action, robot_name=robot_name, n_substeps=n_substeps)

    sim.send_action = _spy  # type: ignore[method-assign]
    return captured


def _error_text(result: dict) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []) if "text" in block)


@pytest.mark.parametrize("value", INVALID_SUBSTEPS)
def test_run_policy_rejects_control_substeps_it_cannot_honor(sim_with_robot, value):
    """run_policy names the bad parameter and never steps physics."""
    captured = _spy_substeps(sim_with_robot)
    sim_time_before = sim_with_robot.get_state()["content"][0]

    result = sim_with_robot.run_policy(
        robot_name="arm1",
        policy_object=MockPolicy(),
        n_steps=2,
        control_frequency=50.0,
        control_substeps=value,
    )

    assert result["status"] == "error", result
    text = _error_text(result)
    assert "control_substeps" in text, text
    assert "positive integer" in text, text
    assert "run_policy" in text, text
    # Rejected at the entry point: no action was ever applied.
    assert captured == [], captured
    assert sim_with_robot.get_state()["content"][0] == sim_time_before


@pytest.mark.parametrize("value", INVALID_SUBSTEPS)
def test_eval_policy_rejects_control_substeps_it_cannot_honor(sim_with_robot, value):
    """eval_policy applies the identical contract to its sibling run_policy."""
    captured = _spy_substeps(sim_with_robot)

    result = sim_with_robot.eval_policy(
        robot_name="arm1",
        policy_object=MockPolicy(),
        n_episodes=1,
        max_steps=2,
        control_substeps=value,
    )

    assert result["status"] == "error", result
    text = _error_text(result)
    assert "control_substeps" in text, text
    assert "eval_policy" in text, text
    assert captured == [], captured


def test_evaluate_benchmark_rejects_control_substeps_before_any_lookup(sim_with_robot):
    """evaluate_benchmark rejects the value before resolving the benchmark.

    The unknown benchmark name would also be an error; the assertion is that the
    caller hears about the parameter they got wrong, and that the rejection
    happens before any benchmark lookup or policy construction.
    """
    result = sim_with_robot.evaluate_benchmark(
        benchmark_name="no_such_benchmark",
        robot_name="arm1",
        n_episodes=1,
        policy_object=MockPolicy(),
        control_substeps=0,
    )

    assert result["status"] == "error", result
    text = _error_text(result)
    assert "control_substeps" in text, text
    assert "evaluate_benchmark" in text, text
    assert "no_such_benchmark" not in text, text


def test_valid_control_substeps_is_honored(sim_with_robot):
    """A positive integer override still drives exactly that many substeps."""
    captured = _spy_substeps(sim_with_robot)

    result = sim_with_robot.run_policy(
        robot_name="arm1",
        policy_object=MockPolicy(),
        n_steps=3,
        control_frequency=50.0,
        control_substeps=7,
    )

    assert result["status"] == "success", result
    assert set(captured) == {7}, sorted(set(captured))


def test_omitted_control_substeps_still_derives_the_control_period(sim_with_robot):
    """``None`` keeps deriving substeps from the physics timestep (0.002 s)."""
    captured = _spy_substeps(sim_with_robot)

    result = sim_with_robot.run_policy(
        robot_name="arm1",
        policy_object=MockPolicy(),
        n_steps=3,
        control_frequency=50.0,
    )

    assert result["status"] == "success", result
    # 50 Hz control period (0.02 s) over a 0.002 s physics dt -> 10 substeps.
    assert set(captured) == {10}, sorted(set(captured))


@pytest.mark.parametrize("value", [0, -5, 2.7, True, float("nan")])
def test_policy_runner_raises_on_an_override_it_cannot_honor(sim_with_robot, value):
    """Direct PolicyRunner callers get a loud ValueError, not a silent clamp."""
    runner = PolicyRunner(sim_with_robot)
    with pytest.raises(ValueError, match="control_substeps"):
        runner._control_substeps(50.0, override=value)


def test_policy_runner_run_and_evaluate_share_the_substep_contract(sim_with_robot):
    """``run`` no longer carries an inline copy of the derivation.

    Both rollout paths resolve substeps through ``_control_substeps``, so an
    explicit override and the derived value agree between them.
    """
    runner = PolicyRunner(sim_with_robot)
    assert runner._control_substeps(50.0) == 10
    assert runner._control_substeps(25.0) == 20
    assert runner._control_substeps(50.0, override=3) == 3

    captured = _spy_substeps(sim_with_robot)
    policy = MockPolicy()
    policy.set_robot_state_keys(sim_with_robot.robot_joint_names("arm1"))
    result = runner.run("arm1", policy, n_steps=2, control_frequency=25.0, fast_mode=True)
    assert result["status"] == "success", result
    assert set(captured) == {20}, sorted(set(captured))
