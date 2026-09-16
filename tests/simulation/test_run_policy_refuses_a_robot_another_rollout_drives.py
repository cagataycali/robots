"""``run_policy`` / ``eval_policy`` are refused while another thread's rollout holds the robot."""

import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import Simulation  # noqa: E402


def _text(result):
    return " ".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))


@pytest.fixture
def busy_arm():
    sim = Simulation()
    sim.create_world()
    sim.add_robot("so101")
    started = sim.start_policy(robot_name="so101", policy_provider="mock", duration=5.0)
    assert started["status"] == "success", _text(started)
    # Let the worker take its first frame so the claim is fully recorded.
    deadline = time.time() + 2.0
    while time.time() < deadline and "so101" not in sim._active_policy_robots():
        time.sleep(0.01)
    try:
        yield sim
    finally:
        sim.stop_policy("so101")
        sim.cleanup()


def test_run_policy_on_a_busy_robot_is_refused_with_the_stop_remedy(busy_arm):
    result = busy_arm.run_policy(robot_name="so101", policy_provider="mock", duration=0.2)
    assert result["status"] == "error"
    text = _text(result)
    assert "Cannot 'run_policy' on 'so101' while its policy is running" in text
    assert "action='stop_policy'" in text
    # The refused call did not touch the claim the live rollout holds.
    assert "so101" in busy_arm._active_policy_robots()


def test_eval_policy_on_a_busy_robot_is_refused(busy_arm):
    result = busy_arm.eval_policy(robot_name="so101", policy_provider="mock", n_episodes=1, max_steps=5)
    assert result["status"] == "error"
    assert "Cannot 'eval_policy' on 'so101' while its policy is running" in _text(result)


def test_a_policy_on_another_robot_is_not_refused(busy_arm):
    # stop_policy joins the worker before it answers, so so101 is free the
    # moment the call returns and this cell has no rollout of its own to wait
    # on. It used to do that wait itself, through the robot's entry in
    # _policy_threads, which the join now drops once the worker is gone. The
    # guarantee is still under test rather than assumed: that wait is what made
    # the restart below reliable, so a stop that answered while the worker was
    # still winding down would have start_policy refuse the robot.
    stopped = busy_arm.stop_policy("so101")
    assert stopped["status"] == "success", _text(stopped)
    busy_arm.add_robot("go2", position=[1.0, 0.0, 0.5])
    started = busy_arm.start_policy(robot_name="so101", policy_provider="mock", duration=5.0)
    assert started["status"] == "success", _text(started)
    result = busy_arm.run_policy(robot_name="go2", policy_provider="mock", duration=0.1)
    assert result["status"] == "success", _text(result)


def test_start_policy_worker_still_reaches_the_rollout_body():
    # The worker start_policy submits calls run_policy's body through
    # _drive_rollout; the driver-thread exemption must let it through.
    sim = Simulation()
    sim.create_world()
    sim.add_robot("so101")
    try:
        started = sim.start_policy(robot_name="so101", policy_provider="mock", duration=0.2)
        assert started["status"] == "success"
        fut = sim._policy_threads["so101"]
        result = fut.result(timeout=10)
        assert result["status"] == "success", _text(result)
        assert "Policy complete" in _text(result)
    finally:
        sim.cleanup()


def test_after_the_rollout_ends_run_policy_is_admitted_again():
    sim = Simulation()
    sim.create_world()
    sim.add_robot("so101")
    try:
        sim.start_policy(robot_name="so101", policy_provider="mock", duration=0.2)
        sim._policy_threads["so101"].result(timeout=10)
        result = sim.run_policy(robot_name="so101", policy_provider="mock", duration=0.1)
        assert result["status"] == "success", _text(result)
    finally:
        sim.cleanup()


def test_base_seam_default_admits_everything():
    from strands_robots.simulation.base import SimEngine

    assert SimEngine._require_no_running_policy(object(), "eval_policy", robot_name="x") is None


def test_the_running_rollout_is_not_truncated_by_a_second_one():
    # The consequence of admitting the second rollout: its ``finally`` lowered
    # the ``policy_running`` claim the first rollout reads as its stop signal,
    # so the first ended early reporting "Policy stopped" - a stop nobody asked
    # for - with a fraction of the steps its duration budgeted.
    sim = Simulation()
    sim.create_world()
    sim.add_robot("so101")
    try:
        sim.start_policy(robot_name="so101", policy_provider="mock", duration=1.0)
        fut = sim._policy_threads["so101"]
        deadline = time.time() + 2.0
        while time.time() < deadline and "so101" not in sim._active_policy_robots():
            time.sleep(0.01)
        sim.run_policy(robot_name="so101", policy_provider="mock", duration=0.2)
        text = _text(fut.result(timeout=20))
        assert "Policy complete" in text, text
        assert "Policy stopped" not in text, text
    finally:
        sim.cleanup()
