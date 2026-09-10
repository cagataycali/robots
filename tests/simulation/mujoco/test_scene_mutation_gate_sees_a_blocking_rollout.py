"""The scene-mutation gate refuses for every rollout shape, not just Future-backed ones.

``_require_no_running_policy`` is the safety gate every scene mutation runs
first, and its own docstring states the hazard: an XML round-trip swaps
``world._model`` / ``world._data`` while a live rollout holds pointers into the
old arrays, so the worker segfaults on its next ``mj_step``.

It derived its population from ``self._policy_threads``, and that table records
only the rollouts ``start_policy`` submits - a blocking ``run_policy`` drives on
its caller's thread and registers no Future. So every mutation the gate lists
was refused during a Future-backed rollout and *accepted* during a blocking one,
on a scene that rollout was stepping, while ``list_policies_running`` reported
the rollout in flight for both. That is the two-sources drift #2833 closed for
the reporting surfaces - ``_active_policy_robots`` owns the union of the Future
table and the per-robot ``policy_running`` claim, and ``_rollouts_in_flight``
delegates there rather than re-deriving it - reached through the gate instead of
the report.

The fix points the gate at that same population, and exempts the rollout's own
driving thread: a rollout mutates the scene itself between episodes
(``PolicyRunner`` calls ``sim.reset()`` at the top of each one), and the driver
racing itself is not the hazard the gate is written against. That is the
reasoning that already makes ``self._lock`` an ``RLock``.

``TestBothRolloutShapesAgree`` pins the parity end to end against a real
rollout of each shape; the class above it pins each mutation verb against a
blocking-shaped claim directly, which needs no rollout and so runs in
milliseconds.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ROBOT_XML = """
<mujoco model="gate_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""

#: The global-scope mutations ``_require_no_running_policy``'s docstring lists,
#: each paired with a call that is otherwise valid on the fixture's world - so a
#: refusal can only come from the gate and not from argument validation.
GLOBAL_SCOPE_MUTATIONS = {
    "add_object": lambda sim: sim.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.05]),
    "move_object": lambda sim: sim.move_object("cube0", position=[0.3, 0.0, 0.05]),
    "remove_object": lambda sim: sim.remove_object("cube0"),
    "add_camera": lambda sim: sim.add_camera(name="cam", position=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.15]),
    "remove_camera": lambda sim: sim.remove_camera("cam0"),
    "set_gravity": lambda sim: sim.set_gravity([0.0, 0.0, -3.7]),
    "set_timestep": lambda sim: sim.set_timestep(0.004),
    "reset": lambda sim: sim.reset(),
}


@pytest.fixture
def robot_path(tmp_path):
    path = tmp_path / "gate_arm.xml"
    path.write_text(ROBOT_XML)
    return str(path)


@pytest.fixture
def sim(robot_path):
    """A live world holding one robot, one object and one camera, nothing running."""
    engine = Simulation(tool_name="test_scene_mutation_gate", mesh=False)
    assert engine.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert engine.add_robot("arm1", urdf_path=robot_path)["status"] == "success"
    assert engine.add_object(name="cube0", shape="box", position=[0.4, 0.0, 0.05])["status"] == "success"
    assert engine.add_camera(name="cam0", position=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.15])["status"] == "success"
    yield engine
    engine.cleanup(policy_stop_timeout=2.0)


def claim_as_blocking_rollout(engine: Simulation, robot_name: str = "arm1") -> None:
    """Put the world in the state a blocking ``run_policy`` on ANOTHER thread leaves.

    That shape is exactly "``policy_running`` is raised and no Future is
    registered", which is what ``_announce_rollout`` does for ``run_policy``.
    No driving-thread claim is recorded for the calling thread, so the caller is
    not the driver - the position any other thread is in while the rollout runs.
    """
    world = engine._world
    assert world is not None
    world.robots[robot_name].policy_running = True


def is_gate_refusal(result: dict, robot_name: str = "arm1") -> bool:
    """Whether ``result`` is the gate's refusal, and not some other error."""
    if result["status"] != "error":
        return False
    text = result["content"][0]["text"]
    return "while a policy is running" in text and robot_name in text and "stop_policy" in text


class TestABlockingRolloutIsGatedLikeAFutureBackedOne:
    """Every global-scope mutation is refused while a blocking rollout drives.

    Pre-fix every one of these returned ``status="success"`` and mutated the
    scene, because the gate read the Future table and a blocking rollout
    registers nothing in it.
    """

    @pytest.mark.parametrize("verb", sorted(GLOBAL_SCOPE_MUTATIONS))
    def test_mutation_is_refused(self, sim, verb):
        claim_as_blocking_rollout(sim)
        result = GLOBAL_SCOPE_MUTATIONS[verb](sim)
        assert is_gate_refusal(result), f"{verb}: {result}"

    def test_the_gate_and_the_report_agree_about_the_same_instant(self, sim):
        """The premise: the reporting half already saw this rollout.

        ``_rollouts_in_flight`` (and so ``list_policies_running``, the mesh
        ``status`` command and its state topic) named the robot while the gate
        let the mutation through - one engine answering two ways about one
        instant. Both halves now read the population ``_active_policy_robots``
        owns.
        """
        claim_as_blocking_rollout(sim)
        assert sim._rollouts_in_flight() == ("arm1",)
        assert "arm1" in sim.list_policies_running()["content"][0]["text"]
        assert is_gate_refusal(sim.reset())

    def test_a_finished_rollout_gates_nothing(self, sim):
        """Non-vacuity: the refusals above come from the claim, not from the fixture."""
        assert sim._rollouts_in_flight() == ()
        for verb, call in sorted(GLOBAL_SCOPE_MUTATIONS.items()):
            assert call(sim)["status"] == "success", verb


class TestTheDrivingThreadIsExempt:
    """The driver may mutate the scene it is stepping; another thread may not.

    ``PolicyRunner`` calls ``sim.reset()`` at the top of every episode, so a
    gate that refused the driver would break the episode boundary of every
    multi-episode rollout while doing nothing about the cross-thread race it
    exists to stop.
    """

    def test_the_driver_may_reset_between_episodes(self, sim):
        claim_as_blocking_rollout(sim)
        sim._rollout_driver_threads["arm1"] = threading.get_ident()
        assert sim.reset()["status"] == "success"

    def test_another_thread_may_not(self, sim):
        claim_as_blocking_rollout(sim)
        sim._rollout_driver_threads["arm1"] = threading.get_ident()
        verdict: list[dict] = []
        other = threading.Thread(target=lambda: verdict.append(sim.reset()))
        other.start()
        other.join(timeout=30.0)
        assert verdict and is_gate_refusal(verdict[0]), verdict


class TestThePerRobotScopeReadsTheSamePopulation:
    """The per-robot scope was on the Future table too.

    ``start_policy`` and the motion primitives gate per-robot rather than
    globally, because rollouts on *different* robots are safe to run
    concurrently - each writes a disjoint slice of ``data.ctrl[]``. Reading the
    Future table there let a second rollout be launched onto a robot a blocking
    ``run_policy`` was already driving, so two rollouts wrote the same slice and
    the engine reported ``Policy started`` for the second one.
    """

    def test_a_second_rollout_on_a_robot_already_driven_is_refused(self, sim):
        claim_as_blocking_rollout(sim)
        result = sim.start_policy(robot_name="arm1", policy_provider="mock", duration=1.0)
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "arm1" in text and "policy is running" in text and "stop_policy" in text, text

    def test_a_rollout_on_another_robot_does_not_block_one_here(self, sim, robot_path):
        """Non-vacuity: the per-robot scope is still per-robot after the change."""
        assert sim.add_robot("arm2", urdf_path=robot_path)["status"] == "success"
        claim_as_blocking_rollout(sim, "arm2")
        result = sim.start_policy(robot_name="arm1", policy_provider="mock", duration=1.0)
        assert result["status"] == "success", result
        sim.stop_policy("arm1")


class TestBothRolloutShapesAgree:
    """End to end against a real rollout of each shape, on real physics.

    The class above drives the gate from a synthesised claim; this one pins that
    the claim really is the shape a live ``run_policy`` leaves, and that the
    driving-thread bookkeeping survives a rollout the executor owns.
    """

    @pytest.mark.parametrize("shape", ["run_policy", "start_policy"])
    def test_a_concurrent_mutation_is_refused(self, sim, shape):
        if shape == "run_policy":
            driver = threading.Thread(
                target=lambda: sim.run_policy(
                    robot_name="arm1", policy_provider="mock", duration=6.0, control_frequency=20.0
                ),
                daemon=True,
            )
            driver.start()
        else:
            assert (
                sim.start_policy(robot_name="arm1", policy_provider="mock", duration=6.0, control_frequency=20.0)[
                    "status"
                ]
                == "success"
            )
        deadline = time.monotonic() + 20.0
        while sim._rollouts_in_flight() != ("arm1",) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sim._rollouts_in_flight() == ("arm1",), "rollout never reported in flight"
        try:
            assert is_gate_refusal(sim.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.05]))
        finally:
            sim.stop_policy("arm1")


class TestTheDrivingThreadBookkeepingReportsABadName:
    """A name the claim cannot key is still reported, not raised, on release.

    ``_drive_rollout`` records the driving thread before ``run_policy`` has
    judged the name, so the write is guarded against an unhashable one. The
    release in its ``finally`` has to be guarded the same way: ``dict.pop``
    hashes its key whenever the dict is non-empty, so with any other rollout in
    flight an unguarded pop raised ``TypeError`` out of the ``finally`` - and a
    raise there discards the unknown-robot error dict the rollout was returning.
    """

    def test_an_unhashable_name_is_reported_while_another_rollout_drives(self, sim):
        sim._rollout_driver_threads["arm1"] = threading.get_ident()
        try:
            result = sim.run_policy(robot_name=["arm1"], policy_provider="mock", duration=1.0)
        finally:
            sim._rollout_driver_threads.clear()
        assert result["status"] == "error", result
        assert "not found" in result["content"][0]["text"], result
