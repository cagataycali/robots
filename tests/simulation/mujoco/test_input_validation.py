"""Input validation regression tests for PR #85 fixes (T7, T9, T10).

These guard against silent data-integrity bugs and process-killing MuJoCo
aborts that were caught by autonomous local testing on PR #85.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty world for validation tests."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


@pytest.fixture
def sim_with_robot():
    """A simulation with a single robot for physics-validation tests."""
    sim = Simulation()
    sim.create_world()
    # Use a built-in registry robot — no network I/O
    res = sim.add_robot(name="panda", data_config="panda")
    if res["status"] != "success":
        pytest.skip(f"panda not available: {res['content'][0]['text']}")
    sim.reset()
    yield sim
    sim.destroy()


# --- T9: step validation --------------------------------------------------


class TestStepValidation:
    def test_step_negative_errors(self, sim_with_world):
        """step(n_steps=-5) must error and NOT decrement step_count."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps=-5)
        assert res["status"] == "error"
        assert "n_steps must be >= 0" in res["content"][0]["text"]
        assert sim_with_world._world.step_count == initial, "step_count must not change on rejected call"

    def test_step_zero_is_noop(self, sim_with_world):
        """step(n_steps=0) is a successful no-op."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps=0)
        assert res["status"] == "success"
        assert "no-op" in res["content"][0]["text"].lower()
        assert sim_with_world._world.step_count == initial

    def test_step_positive_still_works(self, sim_with_world):
        """Baseline: non-negative n_steps continues to work."""
        res = sim_with_world.step(n_steps=3)
        assert res["status"] == "success"
        assert sim_with_world._world.step_count == 3


# --- T7: raycast zero-direction guard -------------------------------------


class TestRaycastValidation:
    def test_zero_direction_errors_not_crash(self, sim_with_robot):
        """raycast with zero direction used to abort the interpreter. Now errors cleanly."""
        res = sim_with_robot.raycast(origin=[0, 0, 1], direction=[0, 0, 0])
        assert res["status"] == "error"
        assert "zero-length" in res["content"][0]["text"].lower()

    def test_wrong_length_direction_errors(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0, 1], direction=[0, 0])
        assert res["status"] == "error"
        assert "3 elements" in res["content"][0]["text"]

    def test_wrong_length_origin_errors(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0], direction=[0, 0, 1])
        assert res["status"] == "error"
        assert "3 elements" in res["content"][0]["text"]

    def test_valid_raycast_still_works(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0, 5], direction=[0, 0, -1])
        assert res["status"] == "success"

    def test_multi_raycast_zero_direction_isolates_error(self, sim_with_robot):
        """A zero-length direction in one ray must not abort the whole batch."""
        res = sim_with_robot.multi_raycast(
            origin=[0, 0, 5],
            directions=[[0, 0, -1], [0, 0, 0], [1, 0, -1]],
        )
        assert res["status"] == "success"
        # The JSON payload should show error on ray[1] only
        rays = res["content"][1]["json"]["rays"]
        assert len(rays) == 3
        assert rays[1].get("error") is not None
        assert "zero-length" in rays[1]["error"]


# --- T10: apply_force must reject missing-both --------------------------


class TestApplyForceValidation:
    def test_missing_both_force_and_torque_errors(self, sim_with_robot):
        """apply_force(body='link1') with no force/torque must error, not silent success."""
        res = sim_with_robot.apply_force(body_name="link1")
        assert res["status"] == "error"
        assert "at least one" in res["content"][0]["text"].lower()

    def test_explicit_zero_force_still_clears_latched(self, sim_with_robot):
        """Regression: apply_force(body, force=[0,0,0]) is the documented way to clear."""
        # First latch a force
        r1 = sim_with_robot.apply_force(body_name="link1", force=[10, 0, 0])
        assert r1["status"] == "success"
        # Then clear with explicit zero — this MUST remain valid
        r2 = sim_with_robot.apply_force(body_name="link1", force=[0, 0, 0])
        assert r2["status"] == "success"

    def test_wrong_length_force_errors(self, sim_with_robot):
        res = sim_with_robot.apply_force(body_name="link1", force=[1, 2])
        assert res["status"] == "error"
        assert "3-element" in res["content"][0]["text"]
