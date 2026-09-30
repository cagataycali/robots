# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A non-string ``instruction`` is refused by every rollout facade, before anything moves.

``run_multi_policy`` already refused ``instructions=42``; the single-robot
facades ran the rollout and wrote ``42`` / ``None`` / a list into the summary,
the result metadata and the recorded ``task`` column. The facade set is read
from the class signatures, so one added later joins this table or fails.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import pytest

from strands_robots.policies.mock import MockPolicy

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ARM_XML = """
<mujoco model="arm">
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="30"/></actuator>
</mujoco>
"""

NOT_TEXT: list[Any] = [42, None, ["pick", "up"], {"task": "pick"}]


def _text(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


@pytest.fixture
def sim(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    engine = Simulation(tool_name="instruction_domain", mesh=False)
    try:
        engine.create_world()
        assert engine.add_robot(name="arm", urdf_path=str(xml_path))["status"] == "success"
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _facades(engine: Simulation) -> dict[str, Any]:
    common = {"robot_name": "arm", "control_frequency": 30.0}
    return {
        "run_policy": lambda v: engine.run_policy(
            policy_object=MockPolicy(), instruction=v, n_steps=3, fast_mode=True, **common
        ),
        "start_policy": lambda v: engine.start_policy(
            policy_object=MockPolicy(), instruction=v, n_steps=3, fast_mode=True, **common
        ),
        "eval_policy": lambda v: engine.eval_policy(
            policy_object=MockPolicy(), instruction=v, n_episodes=1, max_steps=3, **common
        ),
        "evaluate_benchmark": lambda v: engine.evaluate_benchmark(
            "__no_such_benchmark__", policy_object=MockPolicy(), instruction=v, n_episodes=1, **common
        ),
    }


def _wait_until_idle(engine: Simulation, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while "No policies running" not in _text(engine.list_policies_running()):
        assert time.monotonic() < deadline, "a rollout is still in flight"
        time.sleep(0.02)


def test_the_table_covers_every_facade_that_takes_an_instruction(sim):
    declared = {
        name
        for name, member in inspect.getmembers(Simulation, inspect.isfunction)
        if not name.startswith("_") and "instruction" in inspect.signature(member).parameters
    }
    assert declared == set(_facades(sim))


@pytest.mark.parametrize("facade", ["eval_policy", "evaluate_benchmark", "run_policy", "start_policy"])
@pytest.mark.parametrize("value", NOT_TEXT, ids=lambda v: type(v).__name__)
def test_a_non_string_is_refused_and_drives_nothing(sim, facade, value):
    applied: list[Any] = []
    real_send = sim.send_action
    sim.send_action = lambda *a, **k: (applied.append(a), real_send(*a, **k))[1]
    try:
        result = _facades(sim)[facade](value)
        _wait_until_idle(sim)
    finally:
        sim.send_action = real_send

    assert result["status"] == "error", result
    assert _text(result) == f"{facade}: 'instruction' must be a string (use \"\" for none), got {type(value).__name__}."
    assert applied == []


@pytest.mark.parametrize("value", ["", "pick up the cube"], ids=["default", "text"])
def test_a_string_still_runs(sim, value):
    result = _facades(sim)["run_policy"](value)
    assert result["status"] == "success", result
