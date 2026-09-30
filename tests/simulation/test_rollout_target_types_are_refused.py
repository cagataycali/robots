# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout refuses a ``robot_name`` or ``instruction`` that is not a string.

Before, both were used as given. ``run_policy(instruction=None)`` (a ``task``
lookup that missed) ran to ``status="success"`` and carried the ``None`` into
the result metadata and the recorded ``task`` column, while
``run_multi_policy`` refused the same value. ``run_policy(policy, ...)`` - the
hardware call shape - bound the policy to ``robot_name`` and was reported as an
unknown robot named by its ``repr``. Every single-robot rollout facade now
refuses both, before anything is driven, and says the same sentence.
"""

from __future__ import annotations

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
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

# (label, kwargs, the words the refusal must contain)
CASES: list[tuple[str, dict[str, Any], str]] = [
    ("instruction-none", {"robot_name": "arm", "instruction": None}, "'instruction' must be a string, got NoneType"),
    ("instruction-int", {"robot_name": "arm", "instruction": 42}, "'instruction' must be a string, got int"),
    ("instruction-list", {"robot_name": "arm", "instruction": ["pick"]}, "'instruction' must be a string, got list"),
    ("instruction-dict", {"robot_name": "arm", "instruction": {"task": "pick"}}, "got dict"),
    ("robot-name-list", {"robot_name": ["arm"], "instruction": "pick"}, "got list"),
    ("robot-name-int", {"robot_name": 7, "instruction": "pick"}, "'robot_name' must be a robot name string"),
    (
        "robot-name-policy",
        {"robot_name": MockPolicy(), "instruction": "pick"},
        "pass a pre-built policy as policy_object=",
    ),
]


def _text(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


@pytest.fixture
def sim(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    engine = Simulation(tool_name="rollout_target_types", mesh=False)
    try:
        engine.create_world()
        assert engine.add_robot(name="arm", urdf_path=str(xml_path))["status"] == "success"
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _call(engine: Simulation, facade: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    common: dict[str, Any] = {"control_frequency": 30.0}
    if facade in ("run_policy", "start_policy"):
        return getattr(engine, facade)(**kwargs, **common, n_steps=3, fast_mode=True)
    if facade == "eval_policy":
        return engine.eval_policy(**kwargs, **common, n_episodes=1, max_steps=3)
    return engine.evaluate_benchmark("__no_such_benchmark__", **kwargs, n_episodes=1)


def _wait_until_idle(engine: Simulation, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while "No policies running" not in _text(engine.list_policies_running()):
        assert time.monotonic() < deadline, "a rollout is still in flight"
        time.sleep(0.02)


@pytest.mark.parametrize("facade", ["run_policy", "start_policy", "eval_policy", "evaluate_benchmark"])
@pytest.mark.parametrize(("label", "kwargs", "words"), CASES, ids=[c[0] for c in CASES])
def test_a_non_string_target_is_refused_before_anything_is_driven(sim, facade, label, kwargs, words):
    applied: list[Any] = []
    real_send = sim.send_action
    sim.send_action = lambda *a, **k: (applied.append(a), real_send(*a, **k))[1]
    try:
        result = _call(sim, facade, kwargs)
        _wait_until_idle(sim)
    finally:
        sim.send_action = real_send

    assert result["status"] == "error", result
    assert _text(result).startswith(f"{facade}: "), result
    assert words in _text(result), result
    assert applied == []


def test_a_string_instruction_still_drives_the_arm(sim):
    result = sim.run_policy(robot_name="arm", instruction="pick", n_steps=3, control_frequency=30.0, fast_mode=True)
    assert result["status"] == "success", result
