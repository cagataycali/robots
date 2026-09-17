"""The policy surface can be found from the schema and from a wrong first guess.

Measured on ``Robot("so101", mode="sim")`` with the guesses an agent makes
first: ``run_policy(policy="lerobot/act", task="wave")``. The refusal was
``Unknown parameter 'policy' ... Valid: [22 names]`` with ``observer``,
``on_frame`` and ``policy_object`` listed as valid - keys no JSON tool call
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

from strands_robots.registry.policies import list_policy_providers
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
        for python_only in ("observer", "policy_object"):
            assert f"'{python_only}'" not in valid
        for reachable in ("instruction", "policy_provider", "policy_config", "max_steps", "video", "reset_between"):
            assert f"'{reachable}'" in valid

    def test_stop_when_stays_offered_because_a_tool_call_carries_its_dict(self, sim):
        """``stop_when`` is ``dict | Callable | None`` - the dict half is JSON.

        The schema publishes ``stop_when`` and documents the predicate DSL, so
        dropping it for the callable alternative would hide a knob the agent was
        told to use, and leave a typo of it pointing at unrelated names.
        """
        valid = _text(sim._dispatch_action("run_policy", {"policy": "x"})).split("Valid: ")[1]
        assert "'stop_when'" in valid
        text = _text(sim._dispatch_action("run_policy", {"stop_whn": {"predicate": "grasped"}}))
        assert text.split("Did you mean: ")[1].startswith("stop_when")

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

    @pytest.mark.parametrize(
        "annotation",
        [
            "dict[str, Any] | Callable[[SimEngine], bool] | None",
            "str | Callable[[int], None]",
            "Policy | str | None",
        ],
    )
    def test_a_union_with_one_json_alternative_is_kept(self, annotation):
        """One unreachable alternative does not make the parameter unreachable."""
        assert _tool_call_can_carry(self._param(annotation)) is True

    def test_an_annotation_of_only_none_is_kept(self):
        """Nothing is left to judge, so the "never hide a key" default applies."""
        assert _tool_call_can_carry(self._param(None)) is True

    def test_a_union_inside_a_subscript_does_not_keep_a_callable(self):
        """The ``|`` between a callback's own argument types is not an alternative."""
        annotation = "Callable[[Started | Step | Ended], None] | None"
        assert _tool_call_can_carry(self._param(annotation)) is False


class TestTheSchemaNamesTheProviders:
    """The exhaustive ``one of ...`` clause is graded against the shipped registry.

    The description is a literal written when the package was built, so its
    population is ``policies.json`` and not the process's live registry:
    :func:`strands_robots.policies.list_providers` also reports every
    ``register_policy`` call made at runtime, and in a whole-suite run those
    are the throwaway providers sibling tests register (``custom_test``,
    ``kwarg_test``, ``preflight_*``) - names no shipped schema can list and no
    caller can pass on a clean install. Measured: the cells below pass alone
    and failed under ``hatch run test`` for exactly those names.
    """

    def test_mujoco_schema_lists_every_registered_provider(self):
        desc = _TOOL_SPEC_SCHEMA["properties"]["policy_provider"]["description"]
        missing = [name for name in list_policy_providers() if name not in desc]
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
        missing = [name for name in list_policy_providers() if name not in desc]
        assert missing == [], f"providers registered but not in the schema description: {missing}"


class TestNoPublishedKnobIsHiddenByARefusal:
    """The narrowed "Valid:" list may not hide a knob the schema publishes.

    ``tests/simulation/mujoco/test_tool_spec_names_every_agent_knob.py`` grades
    the other direction - no refusal advertises a name the schema hides. Both
    are needed: dropping Python-only keys from the list is only safe while every
    key the agent was told to send survives it.
    """

    def test_every_published_parameter_of_every_action_is_still_offered(self, sim):
        published = set(_TOOL_SPEC_SCHEMA["properties"]) - {"action"}
        hidden: dict[str, list[str]] = {}
        for action in _TOOL_SPEC_SCHEMA["properties"]["action"]["enum"]:
            method = getattr(sim, action, None)
            if method is None:
                continue
            try:
                params = inspect.signature(method).parameters
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            expected = {
                name
                for name, param in params.items()
                if name in published and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
            }
            if not expected:
                continue
            text = _text(sim._dispatch_action(action, {"definitely_not_a_field": 1}))
            assert "Valid: " in text, (action, text)
            offered = text.split("Valid: ")[1]
            missing = sorted(name for name in expected if f"'{name}'" not in offered)
            if missing:
                hidden[action] = missing
        assert hidden == {}, f"published by the schema, hidden by the refusal: {hidden}"
