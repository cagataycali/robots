"""``stop_policy`` with no ``robot_name`` stops the only rollout in flight.

Measured on ``Robot("so101", mode="sim")``: ``start_policy`` succeeded, then
``set_joint_positions`` and ``reset`` were refused with "Stop it first:
action='stop_policy'", and that remedy - typed as written - was refused
``stop_policy requires 'robot_name'``, twice. The tool named a remedy it
would not accept, and the one it would accept (``robot_name='so101'``) was
spelled ``name=`` in the sibling refusal.

The empty name is still never matched to the sole *robot* (a stop aimed at
the wrong robot reads as a stop that worked). It is matched to the sole
*rollout*: with exactly one policy running there is no wrong robot to aim
at. With several running the refusal names them; with none it says so and
names the robots. A gate only offers the bare remedy when it would resolve:
with several in flight it spells ``robot_name`` for each, so the sentence is
a call the tool accepts at either cardinality.

The envelope is read field by field rather than compared whole: what these
cells are about is *which robot* the stop resolved to and whether a rollout
really was in flight. ``exited`` - the MuJoCo override's report on whether the
worker was joined and is gone - is pinned in
``tests/simulation/test_stop_policy_waits_for_the_worker_to_exit.py``, across
all three of its values, and is not this file's subject.
"""

from __future__ import annotations

import re
import threading
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


def _text(result) -> str:
    return " ".join(c["text"] for c in result["content"] if "text" in c)


def _json(result) -> dict:
    return next(c["json"] for c in result["content"] if "json" in c)


def _wait_idle(sim: Simulation, name: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    fut = sim._policy_threads.get(name)
    while fut is not None and not fut.done() and time.time() < deadline:
        time.sleep(0.02)


@pytest.fixture
def sim(tmp_path):
    arm_path = tmp_path / "arm.xml"
    arm_path.write_text(ARM_XML)
    s = Simulation(tool_name="stop_policy_only_rollout", mesh=False)
    s.create_world()
    assert s.add_robot(name="arm", urdf_path=str(arm_path))["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


def _start(sim: Simulation, name: str) -> None:
    started = sim.start_policy(name, policy_provider="mock", duration=5.0, fast_mode=False)
    assert started["status"] == "success", started
    time.sleep(0.05)
    fut = sim._policy_threads.get(name)
    assert fut is not None and not fut.done()


class TestTheOnlyRolloutIsTheTarget:
    def test_an_empty_name_stops_the_one_policy_running(self, sim):
        _start(sim, "arm")
        result = sim._dispatch_action("stop_policy", {})
        assert result["status"] == "success", result
        answer = _json(result)
        assert (answer["robot"], answer["was_running"]) == ("arm", True), answer
        assert "Stopped on 'arm'" in _text(result)
        _wait_idle(sim, "arm")
        assert "No policies running" in _text(sim.list_policies_running())

    def test_the_gates_remedy_works_as_written(self, sim):
        """The sentence a refused mutation ends with is a call the tool accepts."""
        _start(sim, "arm")
        refused = sim._dispatch_action("reset", {})
        assert refused["status"] == "error"
        assert _text(refused).endswith("Stop it first: action='stop_policy'.")
        assert sim._dispatch_action("stop_policy", {})["status"] == "success"
        _wait_idle(sim, "arm")
        assert sim._dispatch_action("reset", {})["status"] == "success"

    def test_the_per_robot_remedy_spells_the_parameter_it_means(self, sim):
        _start(sim, "arm")
        refused = sim._dispatch_action("start_policy", {"robot_name": "arm"})
        assert refused["status"] == "error"
        assert "action='stop_policy', robot_name='arm'" in _text(refused)
        assert " name='arm'" not in _text(refused)


class TestAnythingElseIsRefusedNamingWhatIsRunning:
    def test_nothing_running_names_the_robots(self, sim):
        result = sim._dispatch_action("stop_policy", {})
        assert result["status"] == "error"
        assert _text(result) == "stop_policy requires 'robot_name'. No policy is running now; robots: 'arm'."

    def test_two_rollouts_are_named_not_guessed_between(self, sim, tmp_path):
        assert sim.add_robot(name="arm2", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
        _start(sim, "arm")
        _start(sim, "arm2")
        result = sim._dispatch_action("stop_policy", {})
        assert result["status"] == "error"
        text = _text(result)
        assert text.startswith("stop_policy requires 'robot_name': policies are running on ")
        assert "'arm'" in text and "'arm2'" in text and text.endswith("Name the one to stop.")
        # Neither was stopped by the refusal.
        assert set(sim._rollouts_in_flight()) == {"arm", "arm2"}
        assert sim.stop_policy("arm")["status"] == "success"
        _wait_idle(sim, "arm")
        # Now exactly one is in flight, and the bare call finds it.
        result = sim._dispatch_action("stop_policy", {})
        assert result["status"] == "success"
        assert _json(result)["robot"] == "arm2"

    def test_the_gates_remedy_works_as_written_with_several_running(self, sim, tmp_path):
        """A gate offers the bare remedy only where it resolves; else it names them.

        The gate's population (rollouts this thread is not driving) is not the
        resolver's (every rollout in flight), so the remedy is read back out of
        the refusal and every call it names is executed, rather than matching a
        sentence this test wrote itself.
        """
        assert sim.add_robot(name="arm2", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
        _start(sim, "arm")
        _start(sim, "arm2")
        refused = _text(sim._dispatch_action("reset", {}))
        # The bare form would be refused in this state, so it is not offered.
        assert "action='stop_policy'." not in refused
        named = re.findall(r"robot_name='([^']+)'", refused)
        assert set(named) == set(sim._rollouts_in_flight()) == {"arm", "arm2"}
        for name in named:
            assert sim._dispatch_action("stop_policy", {"robot_name": name})["status"] == "success"
            _wait_idle(sim, name)
        assert sim._dispatch_action("reset", {})["status"] == "success"

    def test_the_remedy_asks_the_resolver_not_the_gates_own_population(self, sim, tmp_path):
        """Two in flight with this thread driving one: the gate sees one, the resolver two.

        The gate exempts the rollout the calling thread drives, so deciding
        "is there exactly one?" from the gate's own population would offer the
        bare remedy in a state where ``stop_policy`` still refuses it. Recording
        this thread as ``arm``'s driver is what ``_drive_rollout`` does for a
        blocking ``run_policy``.
        """
        assert sim.add_robot(name="arm2", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
        _start(sim, "arm")
        _start(sim, "arm2")
        sim._rollout_driver_threads["arm"] = threading.get_ident()
        assert sim._rollouts_driven_by_other_threads() == ["arm2"]
        assert set(sim._rollouts_in_flight()) == {"arm", "arm2"}
        refused = _text(sim._dispatch_action("reset", {}))
        assert "action='stop_policy'." not in refused
        assert sim._dispatch_action("stop_policy", {})["status"] == "error"

    def test_an_explicit_name_is_never_overridden(self, sim, tmp_path):
        assert sim.add_robot(name="arm2", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
        _start(sim, "arm")
        result = sim.stop_policy("arm2")
        assert result["status"] == "success"
        answer = _json(result)
        assert (answer["robot"], answer["was_running"]) == ("arm2", False), answer
        assert sim._rollouts_in_flight() == ("arm",) or list(sim._rollouts_in_flight()) == ["arm"]
