"""A fleet stop must not hold one robot's stop request behind another robot's join.

:meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.stop_policy`
waits (bounded by ``_POLICY_STOP_JOIN_TIMEOUT``) for the worker it just flagged
to exit, so its answer is true when the caller reads it. The mesh
``{"action": "stop"}`` fanout calls that verb once per robot, SEQUENTIALLY, so
the wait lands between one robot's stop request and the next robot's: a robot
whose policy server is wedged inside inference - where the cooperative flag is
not read - pays the full bound, and every robot behind it in the target list
keeps executing for that long.

Measured on this three-robot world with the wedged robot asked first, without
the pre-pass: the two healthy workers exited at t+1.007s and t+1.011s instead of
t+0.004s and t+0.009s. Lowering every target's flag up front (what
:meth:`cleanup` already does for its own multi-robot teardown) makes the workers
wind down concurrently while each ``stop_policy`` still reports its own verdict.

The pre-pass must not cost that verdict its evidence. A healthy worker exits
within a control tick of its flag going down; the fanout then spends up to the
join bound on the wedged robot before it asks the healthy one, by which time the
worker is done, pruned from the Future table, and its flag is already low - so
``stop_policy`` answered ``Was not running`` for a rollout this very stop had
halted, and the fleet stop's ``stopped`` list named only the robot that was still
executing. The pre-pass keeps the Futures it flagged until that robot's
``stop_policy`` reads them.
"""

import threading
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402

# The wedged robot burns the whole join bound, so a healthy worker that is only
# freed by that bound expiring lands near it. The margin between the two
# outcomes is two orders of magnitude, so the threshold is not tuned.
_HEALTHY_EXIT_BUDGET = 0.5


class _WedgedPolicy(MockPolicy):
    """A policy parked inside inference, where the cooperative stop flag is not read."""

    def __init__(self, gate: threading.Event, **kwargs):
        super().__init__(**kwargs)
        self._gate = gate
        self.entered = threading.Event()

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.entered.set()
        self._gate.wait()
        return await super().get_actions(observation_dict, instruction, **kwargs)


@pytest.fixture
def fleet():
    """Three robots, the first of them driven by a wedged policy server."""
    sim = Simulation()
    sim.create_world()
    for robot_type in ("so101", "panda", "koch"):
        sim.add_robot(robot_type)
    gate = threading.Event()
    wedged = _WedgedPolicy(gate)
    names = list(sim._world.robots)
    sim.start_policy(robot_name=names[0], policy_object=wedged, duration=30.0)
    for name in names[1:]:
        sim.start_policy(robot_name=name, policy_provider="mock", duration=30.0)
    assert wedged.entered.wait(10.0), "the wedged policy never reached inference"
    try:
        yield sim, names, gate
    finally:
        gate.set()
        sim.cleanup()


def test_the_healthy_robots_stop_without_waiting_for_the_wedged_one(fleet):
    sim, names, _ = fleet
    wedged_name, healthy = names[0], names[1:]
    # The fanout asks the wedged robot first: it is the first rollout registered,
    # and that is the order the population is reported in.
    assert list(sim._active_policy_robots()) == names

    exited: dict[str, float] = {}
    done = threading.Event()
    start = time.monotonic()

    def sampler():
        while not done.is_set():
            for name in names:
                future = sim._policy_threads.get(name)
                if name not in exited and (future is None or future.done()):
                    exited[name] = time.monotonic() - start
            time.sleep(0.002)

    watcher = threading.Thread(target=sampler, daemon=True)
    watcher.start()
    try:
        # Exactly what the mesh fanout does: lower every flag, then ask each robot.
        sim._request_policy_stop_all(names)
        for name in names:
            assert sim.stop_policy(name)["status"] == "success"
        deadline = time.time() + 5.0
        while time.time() < deadline and not all(n in exited for n in healthy):
            time.sleep(0.005)
    finally:
        done.set()
        watcher.join(timeout=2.0)

    for name in healthy:
        assert name in exited, f"{name} never exited"
        assert exited[name] < _HEALTHY_EXIT_BUDGET, (
            f"{name} kept executing for {exited[name]:.3f}s - it waited on "
            f"{wedged_name}'s join instead of its own stop request"
        )
    # The wedged robot is still held by its own server, and the answer says so.
    assert wedged_name not in exited


def test_a_robot_the_pre_pass_halted_is_reported_as_stopped_not_as_never_running(fleet):
    """The reviewer's sequence: wedged robot asked first, the healthy ones after its full join bound."""
    sim, names, _ = fleet
    wedged_name, healthy = names[0], names[1:]
    sim._request_policy_stop_all(names)
    # The healthy workers exit well inside the wedged robot's join bound...
    deadline = time.time() + _HEALTHY_EXIT_BUDGET
    while time.time() < deadline and not all(sim._policy_threads[n].done() for n in healthy):
        time.sleep(0.005)
    assert all(sim._policy_threads[n].done() for n in healthy)
    # ...and the wedged robot's stop_policy burns that bound and prunes them from the table.
    wedged = sim.stop_policy(wedged_name)
    assert next(c["json"] for c in wedged["content"] if "json" in c) == {
        "robot": wedged_name,
        "was_running": True,
        "exited": False,
    }
    assert all(n not in sim._policy_threads for n in healthy)
    # Each healthy robot is still answered for as halted BY THIS STOP, and joined.
    for name in healthy:
        answer = sim.stop_policy(name)
        verdict = next(c["json"] for c in answer["content"] if "json" in c)
        assert verdict == {"robot": name, "was_running": True, "exited": True}, verdict
        assert f"Stopped on '{name}'" in answer["content"][0]["text"]
    # The record is consumed: a SECOND stop on the same robot is a real "Was not running".
    second = sim.stop_policy(healthy[0])
    assert next(c["json"] for c in second["content"] if "json" in c)["was_running"] is False
    assert sim._pre_stopped == {}


def test_the_pre_pass_lowers_every_named_flag_and_joins_nothing(fleet):
    sim, names, _ = fleet
    assert all(sim._world.robots[n].policy_running for n in names)
    start = time.monotonic()
    sim._request_policy_stop_all([*names, "no_such_robot"])
    elapsed = time.monotonic() - start
    assert all(not sim._world.robots[n].policy_running for n in names)
    # Joining nothing: the wedged worker is still live, and this returned anyway.
    assert elapsed < _HEALTHY_EXIT_BUDGET
    assert not sim._policy_threads[names[0]].done()


def test_the_pre_pass_leaves_a_blocking_rollout_for_stop_policy_to_answer_for():
    # A blocking ``run_policy`` registers no Future, so its flag is the only
    # record that it was in flight. Pre-lowering it would make ``stop_policy``
    # answer "Was not running" for a rollout it really did halt, and the fleet
    # stop would name nothing under ``stopped``. It also has nothing to join.
    sim = Simulation()
    sim.create_world()
    sim.add_robot("so101")
    name = next(iter(sim._world.robots))
    driver = threading.Thread(
        target=lambda: sim.run_policy(robot_name=name, policy_provider="mock", duration=5.0),
    )
    driver.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline and name not in sim._active_policy_robots():
            time.sleep(0.01)
        assert name in sim._active_policy_robots()
        sim._request_policy_stop_all([name])
        assert sim._world.robots[name].policy_running, "the pre-pass must not flag a blocking rollout"
        answer = sim.stop_policy(name)
        verdict = next(c["json"] for c in answer["content"] if "json" in c)
        assert verdict == {"robot": name, "was_running": True, "exited": None}
    finally:
        driver.join(timeout=10)
        sim.cleanup()
