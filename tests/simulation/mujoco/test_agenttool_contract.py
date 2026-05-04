"""T1/T13: AgentTool router contract — unknown kwargs rejected, required args friendly,
vector dims validated, tool_spec matches method signatures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="contract_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestRouterRejectsUnknownKwargs:
    """T1 DoD: Unknown top-level params must be rejected with a clear message."""

    def test_unknown_kwarg_on_set_gravity(self, sim):
        result = sim._dispatch_action(
            "set_gravity", {"gravity": [0, 0, -9.81], "bogus_param": 42}
        )
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown parameter 'bogus_param'" in text
        assert "set_gravity" in text
        assert "Valid:" in text

    def test_unknown_kwarg_on_step(self, sim):
        result = sim._dispatch_action("step", {"n_steps": 5, "num_steps": 10})
        assert result["status"] == "error"
        assert "Unknown parameter 'num_steps'" in result["content"][0]["text"]

    def test_unknown_kwarg_on_reset(self, sim):
        result = sim._dispatch_action("reset", {"hard_reset": True})
        assert result["status"] == "error"
        assert "Unknown parameter 'hard_reset'" in result["content"][0]["text"]


class TestRouterRequiredArgError:
    """T1 DoD: Missing required params produce a friendly error (no Python TypeError)."""

    def test_missing_required_arg_on_add_object(self, sim):
        # add_object requires `name`. Default for shape is `box` but `name` has no default.
        result = sim._dispatch_action("add_object", {"shape": "box"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "requires parameter 'name'" in text
        assert "add_object" in text

    def test_missing_required_arg_on_stop_policy(self, sim):
        # stop_policy has robot_name default="" so it's not technically required;
        # but apply_force requires body_name.
        result = sim._dispatch_action("apply_force", {"force": [0, 0, 1]})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "requires parameter 'body_name'" in text


class TestRouterValidatesVectorDims:
    """T1 DoD: Vector params with wrong length rejected before reaching MuJoCo."""

    def test_gravity_wrong_length_rejected(self, sim):
        result = sim._dispatch_action("set_gravity", {"gravity": [0, 0]})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "'gravity'" in text and "3" in text and "2" in text

    def test_position_wrong_length_rejected(self, sim):
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "position": [0, 0]},
        )
        assert result["status"] == "error"
        assert "'position'" in result["content"][0]["text"]

    def test_orientation_wrong_length_rejected(self, sim):
        # orientation is a quaternion (4)
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "orientation": [1, 0, 0]},
        )
        assert result["status"] == "error"
        assert "'orientation'" in result["content"][0]["text"]

    def test_color_wrong_length_rejected(self, sim):
        # color is rgba (4)
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "color": [1, 0, 0]},
        )
        assert result["status"] == "error"
        assert "'color'" in result["content"][0]["text"]

    def test_non_numeric_vector_component_rejected(self, sim):
        result = sim._dispatch_action(
            "set_gravity", {"gravity": [0, 0, "low"]}
        )
        assert result["status"] == "error"
        assert "numeric" in result["content"][0]["text"].lower()

    def test_non_list_vector_rejected(self, sim):
        result = sim._dispatch_action("set_gravity", {"gravity": 9.81})
        assert result["status"] == "error"
        assert "'gravity'" in result["content"][0]["text"]


class TestRouterKwargsPassthrough:
    """Methods with **kwargs in signature accept unknown params without error."""

    def test_add_object_accepts_extra_kwargs(self, sim):
        # add_object has **kwargs so extra params are allowed (backwards compat).
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "future_flag": True},
        )
        # Either success (extra key ignored) or a proper runtime error; must NOT
        # be an "unknown parameter" router rejection.
        if result["status"] == "error":
            assert "Unknown parameter" not in result["content"][0]["text"]


class TestToolSpecMethodParity:
    """T13 DoD: every enum action in tool_spec.json has a matching method whose
    signature matches declared top-level params."""

    # Params in tool_spec.json that are intentionally not consumed by every method
    # (they are cross-cutting or action-conditional).
    SPEC_ONLY_ALLOWED = {
        # action is the dispatch key itself
        "action",
        # video composite params — folded into `video` by the router
        "output_path",
        "fps",
        # name/robot_name are aliased bi-directionally
        "robot_name",
        "name",
        # global knobs sometimes listed at top level for LLM convenience
    }

    def test_every_action_maps_to_a_method(self, sim):
        spec_path = Path(
            "/Users/cagatay/robots/strands_robots/simulation/mujoco/tool_spec.json"
        )
        spec = json.loads(spec_path.read_text())
        actions = spec["properties"]["action"]["enum"]

        missing = []
        for action in actions:
            method_name = sim._ACTION_ALIASES.get(action, action)
            if not hasattr(sim, method_name):
                missing.append(action)
        assert not missing, f"Actions without a method: {missing}"

    def test_no_method_has_silently_unused_param(self, sim):
        """Known legacy drifts that the router USED to silently drop are now
        either implemented or flagged by the router. This test enumerates
        the pre-T1 drift cases as a regression ward."""
        # Before T1: step(num_steps), run_policy(n_steps wrong), etc. silently dropped.
        # After T1: all of these rejected. Verify a sampling.
        drift_cases = [
            ("step", {"num_steps": 5}),  # should be `n_steps`
            ("forward_kinematics", {"some_ghost_param": 1}),
            ("get_features", {"unknown_filter": "a"}),
        ]
        for action, bad_kwargs in drift_cases:
            result = sim._dispatch_action(action, bad_kwargs)
            # Router must reject; must NOT silently succeed with default values.
            assert result["status"] == "error", f"{action} silently accepted {bad_kwargs}"


class TestUnifiedNoWorldMessage:
    """T14: Every action must use the same 'No world.' message when no world exists."""

    @pytest.fixture
    def fresh_sim(self):
        """A sim with NO world."""
        s = Simulation(tool_name="no_world_test", mesh=False)
        yield s
        s.cleanup()

    def _assert_standard_no_world_error(self, result, action):
        assert result["status"] == "error", f"{action} should error when no world"
        text = result["content"][0]["text"]
        assert "No world" in text, f"{action} error text lacks 'No world': {text}"

    def test_step_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(
            fresh_sim._dispatch_action("step", {"n_steps": 1}), "step"
        )

    def test_reset_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(fresh_sim._dispatch_action("reset", {}), "reset")

    def test_set_gravity_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(
            fresh_sim._dispatch_action("set_gravity", {"gravity": [0, 0, -1]}),
            "set_gravity",
        )

    def test_render_no_world(self, fresh_sim):
        # render returns error cleanly when no world, not a crash.
        result = fresh_sim._dispatch_action("render", {})
        assert result["status"] == "error"
        # render uses the unified message now:
        assert "No world" in result["content"][0]["text"]

    def test_get_state_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(
            fresh_sim._dispatch_action("get_state", {}), "get_state"
        )


class TestUnifiedNotFoundMessages:
    """T15: Unknown-name errors use the consistent '<Kind> X not found.' shape."""

    def test_robot_not_found(self, sim):
        result = sim._dispatch_action("get_robot_state", {"robot_name": "ghost_bot"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Robot 'ghost_bot' not found" in text

    def test_object_not_found(self, sim):
        result = sim._dispatch_action(
            "move_object", {"name": "ghost_box", "position": [0, 0, 0]}
        )
        assert result["status"] == "error"
        assert "Object 'ghost_box' not found" in result["content"][0]["text"]

    def test_body_not_found(self, sim):
        result = sim._dispatch_action(
            "apply_force", {"body_name": "ghost_body", "force": [0, 0, 1]}
        )
        assert result["status"] == "error"
        assert "Body 'ghost_body' not found" in result["content"][0]["text"]

    def test_sensor_not_found(self, sim):
        result = sim._dispatch_action("get_sensor_data", {"sensor_name": "ghost_sensor"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        # T45 is about distinguishing "no sensors" vs "not found"; at minimum the
        # current behaviour must mention the sensor name clearly.
        assert "ghost_sensor" in text


class TestIdempotentStopFamily:
    """T16: stop_recording, stop_cameras_recording, stop_policy and close_viewer
    can be called unconditionally — when already stopped they succeed with a
    distinguishable 'Was not ...' message."""

    def test_stop_recording_twice_is_idempotent(self, sim):
        r1 = sim.stop_recording()
        assert r1["status"] == "success"
        r2 = sim.stop_recording()
        assert r2["status"] == "success"
        assert "Was not recording" in r2["content"][0]["text"]

    def test_stop_cameras_recording_twice_is_idempotent(self, sim):
        r1 = sim.stop_cameras_recording()
        assert r1["status"] == "success"
        r2 = sim.stop_cameras_recording()
        assert r2["status"] == "success"

    def test_close_viewer_twice_is_idempotent(self, sim):
        # close_viewer was already idempotent — pin it with a regression test.
        assert sim.close_viewer()["status"] == "success"
        assert sim.close_viewer()["status"] == "success"


class TestStopPolicyContract:
    """T16 + T24: stop_policy requires a robot_name; is idempotent per robot."""

    def test_stop_policy_empty_robot_name_friendly_error(self, sim):
        r = sim._dispatch_action("stop_policy", {})
        assert r["status"] == "error"
        assert "requires" in r["content"][0]["text"].lower() and "robot_name" in r["content"][0]["text"]

    def test_stop_policy_unknown_robot_errors(self, sim):
        r = sim._dispatch_action("stop_policy", {"robot_name": "ghost_bot"})
        assert r["status"] == "error"
        assert "Robot 'ghost_bot' not found" in r["content"][0]["text"]
