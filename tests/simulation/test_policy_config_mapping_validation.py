"""Provider keyword bags must be rejected when they cannot be splatted.

``policy_config`` (provider kwargs for ``create_policy``) and ``policy_kwargs``
(per-call kwargs for ``Policy.get_actions``) are free-form dicts that reach
their consumer through ``**``, so a value of the wrong *shape* has no signature
to bounce off at the call the caller actually made:

* ``run_policy(policy_config="host=127.0.0.1")`` raised a bare ``TypeError``
  out of the library ("... argument after ** must be a mapping, not str"),
  naming an internal helper instead of the parameter to fix.
* ``start_policy(policy_config=[...])`` returned ``status="success"`` -- the
  splat failed on the background thread, inside the future, so the caller was
  told a policy had started that never produced a single action.
* ``run_policy(policy_kwargs="pick up the cube")`` failed only once the
  rollout was already under way, after a recorder could have been started.

Every one of those is now a structured error naming the offending parameter.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from strands_robots.policies import policy_mapping_error
from strands_robots.tools.run_policy import run_policy as run_policy_tool

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
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


@pytest.fixture
def sim_with_arm(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="policy_config_schema", mesh=False)
    try:
        sim.create_world()
        result = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert result["status"] == "success", result
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


class TestPolicyMappingError:
    """The shared shape check both the sim facade and the tool wrapper use."""

    @pytest.mark.parametrize(
        "value",
        [None, {}, {"host": "127.0.0.1", "port": 5555}, MappingProxyType({"port": 5555})],
        ids=["none", "empty-dict", "dict", "mapping"],
    )
    def test_splattable_values_are_accepted(self, value):
        assert policy_mapping_error(value) is None

    @pytest.mark.parametrize(
        ("value", "type_name"),
        [("host=1", "str"), ([("host", 1)], "list"), (5555, "int"), (object(), "object")],
    )
    def test_unsplattable_value_names_its_type_and_a_correct_example(self, value, type_name):
        message = policy_mapping_error(value, "policy_config")
        assert message is not None
        assert message.startswith("policy_config must be a dict of")
        assert f"got {type_name}" in message
        assert "create_policy" in message

    def test_message_describes_the_parameter_it_was_asked_about(self):
        message = policy_mapping_error("goal", "policy_kwargs")
        assert message is not None
        assert message.startswith("policy_kwargs must be a dict of")
        assert "get_actions" in message


class TestRolloutRejectsUnsplattablePolicyMapping:
    """The public rollout entry points surface the error themselves."""

    def test_run_policy_rejects_policy_config_instead_of_raising(self, sim_with_arm):
        result = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            policy_config="host=127.0.0.1",
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        assert "run_policy: policy_config must be a dict of" in result["content"][0]["text"]

    def test_run_policy_rejects_policy_kwargs_before_the_rollout_starts(self, sim_with_arm):
        result = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            policy_kwargs="pick up the cube",
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "run_policy: policy_kwargs must be a dict of" in text
        # Pre-fix this surfaced as "Policy failed: ... must be a mapping" from
        # inside the control loop, i.e. after the rollout had begun stepping.
        assert "Policy failed" not in text

    def test_start_policy_reports_the_error_instead_of_a_false_started(self, sim_with_arm):
        result = sim_with_arm.start_policy(
            robot_name="arm1",
            policy_provider="mock",
            policy_config=["host=127.0.0.1"],
            n_steps=4,
            control_frequency=30.0,
        )
        assert result["status"] == "error", result
        assert "start_policy: policy_config must be a dict of" in result["content"][0]["text"]
        # The rejected call must not have marked the robot as running: a
        # subsequent well-formed start is accepted.
        started = sim_with_arm.start_policy(
            robot_name="arm1", policy_provider="mock", n_steps=2, control_frequency=30.0
        )
        assert started["status"] == "success", started
        sim_with_arm.stop_policy(robot_name="arm1")

    def test_eval_policy_rejects_policy_config_before_running_episodes(self, sim_with_arm):
        result = sim_with_arm.eval_policy(
            robot_name="arm1",
            policy_provider="mock",
            policy_config="host=127.0.0.1",
            n_episodes=2,
            max_steps=4,
            control_frequency=30.0,
        )
        assert result["status"] == "error", result
        assert "eval_policy: policy_config must be a dict of" in result["content"][0]["text"]

    def test_agent_dispatch_surfaces_the_same_error(self, sim_with_arm):
        result = sim_with_arm._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm1",
                "policy_provider": "mock",
                "policy_config": "host=127.0.0.1",
                "n_steps": 4,
                "control_frequency": 30.0,
            },
        )
        assert result["status"] == "error", result
        assert "policy_config must be a dict of" in result["content"][0]["text"]

    def test_well_formed_policy_config_still_runs(self, sim_with_arm):
        result = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            policy_config={},
            policy_kwargs={},
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
        )
        assert result["status"] == "success", result


class TestRunPolicyToolRejectsUnsplattablePolicyMapping:
    """The multi-episode tool checks before it creates a dataset."""

    def test_tool_rejects_policy_config_without_starting_a_recording(self, tmp_path):
        for attr in ("_tool_func", "original_function", "__wrapped__", "func"):
            target = getattr(run_policy_tool, attr, None)
            if callable(target):
                run_policy = target
                break
        else:  # pragma: no cover - defensive against SDK churn
            run_policy = run_policy_tool

        calls: list[str] = []

        class _RecordingSim:
            def start_recording(self, **kwargs):
                calls.append("start_recording")
                return {"status": "success", "content": [{"text": "started"}]}

            def run_policy(self, **kwargs):
                calls.append("run_policy")
                return {"status": "success", "content": [{"text": "ok"}]}

            def stop_recording(self, **kwargs):
                calls.append("stop_recording")
                return {"status": "success", "content": [{"text": "stopped"}]}

        result = run_policy(
            simulation=_RecordingSim(),
            robot_name="arm1",
            n_episodes=2,
            n_steps=4,
            policy_config="host=127.0.0.1",
            dataset_root=str(tmp_path / "ds"),
            dataset_repo_id="local/ds",
        )
        assert result["status"] == "error", result
        assert "run_policy: policy_config must be a dict of" in result["content"][0]["text"]
        assert calls == []
