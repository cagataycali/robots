"""Regression test: run_policy accepts a pre-built Policy object.

Without this, every notebook or script that records multiple rollouts with
the same policy pays the ~10s create_policy cost on every call, and worse,
the first ~13s of the recording shows a frozen arm because the model is
still loading inside run_policy.
"""

from __future__ import annotations

import os
import time

import pytest


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)
def test_run_policy_reuses_policy_object() -> None:
    """Two rollouts with a single pre-built MockPolicy should both succeed."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.policies import MockPolicy
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])

    policy = MockPolicy()

    t0 = time.time()
    r1 = sim.run_policy(
        robot_name="arm",
        policy_object=policy,
        duration=0.3,
        control_frequency=20.0,
    )
    d1 = time.time() - t0
    assert r1["status"] == "success", r1

    t0 = time.time()
    r2 = sim.run_policy(
        robot_name="arm",
        policy_object=policy,
        duration=0.3,
        control_frequency=20.0,
    )
    d2 = time.time() - t0
    assert r2["status"] == "success", r2

    # Second call reuses policy; neither should be dramatically slower than the other.
    # (Both should be <2s for mock; if policy_object wasn't honoured, we'd rebuild.)
    assert d1 < 3.0 and d2 < 3.0, f"rollouts took {d1:.1f}s + {d2:.1f}s"

    sim.destroy()


def test_run_policy_object_param_exposed() -> None:
    """Signature check — policy_object must be in both base and MuJoCo variants."""
    import inspect

    from strands_robots.simulation import Simulation

    sig = inspect.signature(Simulation.run_policy)
    assert "policy_object" in sig.parameters
    # Default must be None so existing callers are unaffected
    assert sig.parameters["policy_object"].default is None

    # start_policy too
    sig2 = inspect.signature(Simulation.start_policy)
    assert "policy_object" in sig2.parameters
