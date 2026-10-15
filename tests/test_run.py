import time

import pytest

from strands_robots import JointState, Robot, Simulation
from strands_robots.policy import SequencePolicy
from strands_robots.robot import RunResult


def test_robot_is_a_class_and_a_body_in_the_default_world():
    arm = Robot("so101")
    assert type(arm) is Robot and arm.sim is Simulation.default() and not arm.is_real
    d = arm.describe()
    assert d["id"] == "so101" and d["world"] == "mujoco" and d["real"] is False and d["task"] is None
    assert repr(arm) == "Robot('so101', id='so101', mujoco, dof=6)"
    arm.close()
    assert arm.sim is None and "so101" not in Simulation.default().robots


def test_run_for_steps_reaches_target_and_measures():
    arm = Robot("so101")
    res = arm.run(lambda o: {"shoulder_pan": 0.4, "gripper": 0.8}, hz=50, steps=60)
    assert isinstance(res, RunResult) and res.steps == 60 and res.stopped_by == "steps"
    assert res.seconds == pytest.approx(60 / 50)  # sim time, not wall time
    assert res.hz_achieved > 50  # faster than real time
    assert res.final is not None and res.final["shoulder_pan"] == pytest.approx(0.4, abs=0.02)


def test_run_for_seconds_sets_the_sim_step_count():
    arm = Robot("so101")
    t0 = arm.sim.time
    arm.run(lambda o: o.joints.positions, hz=25, seconds=1.0)
    assert arm.sim.time == pytest.approx(t0 + 1.0, abs=1e-9)


def test_run_stop_when_and_instruction():
    arm = Robot("so101")
    seen = []

    def pol(obs, instruction=None):
        seen.append(instruction)
        return {"elbow_flex": 1.0}

    res = arm.run(pol, instruction="bend", hz=50, stop_when=lambda o: o["elbow_flex"] > 0.5, steps=500)
    assert res.stopped_by == "stop_when" and 0 < res.steps < 500 and seen[0] == "bend"


def test_run_needs_a_bound_and_valid_hz():
    arm = Robot("so101")
    with pytest.raises(ValueError, match="seconds=, steps= or stop_when="):
        arm.run(lambda o: o.joints.positions)
    with pytest.raises(ValueError, match="hz"):
        arm.run(lambda o: o.joints.positions, hz=0, steps=1)
    with pytest.raises(ValueError, match="hz"):  # refused before the spec is resolved, so nothing is loaded
        arm.run("zenoh://127.0.0.1:1/nobody", hz=-1, steps=1)


def test_start_stop_background_task():
    arm = Robot("so101")
    task = arm.start(lambda o: {"wrist_roll": 0.5}, hz=200, seconds=30)
    assert task.status == "running"
    time.sleep(0.2)
    res = arm.stop()
    assert res is not None and res.stopped_by == "stopped" and res.steps > 5
    assert task.status == "done" and not task.running
    arm.start(lambda o: o.joints.positions, seconds=5)
    with pytest.raises(RuntimeError, match="already has a running task"):
        arm.start(lambda o: o.joints.positions, seconds=5)
    arm.stop()


def test_background_task_error_surfaces_in_result():
    arm = Robot("so101")

    def bad(obs):
        raise RuntimeError("policy exploded")

    task = arm.start(bad, steps=3)
    task._thread.join(2)
    assert task.status == "error"
    with pytest.raises(RuntimeError, match="policy exploded"):
        _ = task.result
    arm.close()  # teardown does not re-raise the task's error, and the body leaves the world
    assert arm.sim is None and "so101" not in Simulation.default().robots


def test_set_gains_takes_one_kp_and_kd_per_actuator():
    arm = Robot("so101")
    arm.set_gains([50.0] * 6, [1.0] * 6)
    with pytest.raises(ValueError, match="for 6 actuators"):
        arm.set_gains([50.0] * 5, [1.0] * 6)


def test_the_operator_gate_passes_a_simulated_body():
    arm = Robot("so101")
    assert arm.gate({"shoulder_pan": 0.2})[0] == pytest.approx(0.2)
    assert arm.confirm_act({"shoulder_pan": 0.2}).t > 0


def test_realtime_paces_a_sim_run_to_the_wall_clock():
    arm = Robot("so101")
    t0 = time.perf_counter()
    res = arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=6, realtime=True)
    wall = time.perf_counter() - t0
    assert res.steps == 6 and 0.2 <= wall <= 1.0  # 6 steps at 20 Hz ≈ 0.3 s, not the ~5 ms a free-running sim takes
    t0 = time.perf_counter()
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=6)
    assert time.perf_counter() - t0 < 0.2


def test_act_takes_the_joints_an_observation_measured():
    arm = Robot("so101")
    arm.act({"shoulder_pan": 0.4}, steps=200)
    measured = arm.observe().joints
    assert arm.coerce(measured) == pytest.approx(measured.positions)  # a hold at what was read
    with pytest.raises(KeyError, match="unknown joint"):
        arm.coerce(JointState.zeros(["shoulder_pan", "not_a_joint"]))


class _At30(SequencePolicy):
    hz = 30.0


def test_both_loops_replay_at_the_rate_the_policy_declares_and_refuse_another():
    arm = Robot("so101")
    assert arm.run(_At30([{"shoulder_pan": 0.1}]), seconds=1).hz_target == 30.0
    assert arm.sim.run({arm: _At30([{"shoulder_pan": 0.1}])}, seconds=1)[arm.id].hz_target == 30.0
    with pytest.raises(ValueError, match="runs at 30 Hz; hz=50"):
        arm.run(_At30([{"shoulder_pan": 0.1}]), hz=50, steps=1)
    with pytest.raises(ValueError, match="runs at 30 Hz; hz=50"):
        arm.sim.run({arm: _At30([{"shoulder_pan": 0.1}])}, hz=50, steps=1)
    other = Robot("so101", id="so101_2")

    class At20(SequencePolicy):
        hz = 20.0

    with pytest.raises(ValueError, match="different rates"):
        arm.sim.run({arm: _At30([{"shoulder_pan": 0.1}]), other: At20([{"shoulder_pan": 0.1}])}, steps=1)
