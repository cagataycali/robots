"""The policy surface can be found from the schema and from a wrong first guess.

Measured on ``Robot("so101", mode="sim")`` with the guesses an agent makes
first: ``run_policy(policy="lerobot/act", task="wave")``. The refusal was
``Unknown parameter 'policy' ... Valid: [22 names]`` with ``observer``,
``stop_when`` and ``policy_object`` listed as valid - keys no JSON tool call
can fill - and no nearest-name hint, so the key it needed
(``policy_provider`` / ``policy_config``) was one of many to scan. The
``policy_provider`` schema entry said "See list_providers()", which is not an
action (the unknown-action suggester offers ``list_bodies``), so the provider
names were only revealed after a wrong guess; ``instruction`` carried no
description at all, so ``task=`` had nothing to steer it.

Now the refusal names the nearest valid keys and lists only the ones a tool
call can carry; the schema names every registered provider and says where the
model itself goes; ``instruction`` says what it is.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import list_providers
from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA, Simulation, _tool_call_can_carry


@pytest.fixture
def sim():
    s = Simulation(tool_name="policy_surface_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _text(result) -> str:
    return result["content"][0]["text"]


class TestTheRefusalPointsAtTheKeyThatWasMeant:
    def test_policy_is_answered_with_policy_provider_and_policy_config(self, sim):
        text = _text(sim._dispatch_action("run_policy", {"policy": "lerobot/act", "instruction": "wave"}))
        assert text.startswith("Unknown parameter 'policy' for action 'run_policy'. Did you mean: ")
        hint = text.split("Did you mean: ")[1].split("?")[0]
        assert "policy_provider" in hint and "policy_config" in hint

    def test_callables_and_live_objects_are_not_offered(self, sim):
        text = _text(sim._dispatch_action("run_policy", {"policy": "x", "instruction": "wave"}))
        valid = text.split("Valid: ")[1]
        for python_only in ("observer", "stop_when", "policy_object"):
            assert f"'{python_only}'" not in valid
        for reachable in ("instruction", "policy_provider", "policy_config", "max_steps", "video", "reset_between"):
            assert f"'{reachable}'" in valid

    def test_eval_policy_drops_on_frame_but_keeps_success_fn(self, sim):
        """``success_fn`` is a *name* (str) on eval_policy, so it stays."""
        valid = _text(sim._dispatch_action("eval_policy", {"task": "wave"})).split("Valid: ")[1]
        assert "'on_frame'" not in valid and "'policy_object'" not in valid
        assert "'success_fn'" in valid and "'instruction'" in valid

    def test_no_hint_when_nothing_is_close(self, sim):
        text = _text(sim._dispatch_action("set_gravity", {"gravity": [0, 0, -9.81], "bogus_param": 42}))
        assert text == "Unknown parameter 'bogus_param' for action 'set_gravity'. Valid: ['gravity']"


class TestToolCallCanCarry:
    @staticmethod
    def _param(annotation) -> inspect.Parameter:
        return inspect.Parameter("p", inspect.Parameter.KEYWORD_ONLY, annotation=annotation)

    @pytest.mark.parametrize(
        "annotation",
        [
            "Callable[[int], None] | None",
            "collections.abc.Callable[[Any], bool]",
            Callable[[int], None],
            "Policy | None",
        ],
    )
    def test_callables_and_policy_objects_cannot(self, annotation):
        assert _tool_call_can_carry(self._param(annotation)) is False

    @pytest.mark.parametrize(
        "annotation",
        [
            "str | None",
            "dict[str, Any] | None",
            "list[str]",
            int,
            bool,
            Any,
            "policy_name: str",
            inspect.Parameter.empty,
        ],
    )
    def test_everything_else_can(self, annotation):
        assert _tool_call_can_carry(self._param(annotation)) is True

    def test_a_name_that_merely_contains_policy_is_kept(self):
        assert _tool_call_can_carry(self._param("PolicyConfig | None")) is True


class TestTheSchemaNamesTheProviders:
    def test_mujoco_schema_lists_every_registered_provider(self):
        desc = _TOOL_SPEC_SCHEMA["properties"]["policy_provider"]["description"]
        missing = [name for name in list_providers() if name not in desc]
        assert missing == [], f"providers registered but not in the schema description: {missing}"
        assert "list_providers()" not in desc.split("(")[0]  # not presented as an action to call
        assert "policy_config" in desc  # says where the model itself goes

    def test_instruction_says_what_it_is(self):
        desc = _TOOL_SPEC_SCHEMA["properties"]["instruction"]["description"]
        assert "task" in desc and "run_policy" in desc

    def test_hardware_schema_lists_every_registered_provider(self):
        pytest.importorskip("lerobot")
        from strands_robots import Robot

        arm = Robot("so101", mode="real", port="/dev/cu.usbmodem-policy-surface-test")
        try:
            desc = arm.tool_spec["inputSchema"]["json"]["properties"]["policy_provider"]["description"]
        finally:
            arm.cleanup()
        missing = [name for name in list_providers() if name not in desc]
        assert missing == []
