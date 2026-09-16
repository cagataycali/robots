"""Every plain parameter a sim action accepts is a property of the tool schema.

The unknown-parameter refusal lists the action's accepted names from its Python
signature, while the agent plans from the JSON schema. When the two disagreed
the agent was told "Valid: [... 'overwrite' ...]" for a knob the schema never
showed it - and could not find friction_range/mass_range at all, so
randomize_physics ran only at its defaults. Measured on
Robot("so101", mode="sim"): 7 agent-usable knobs accepted at runtime but absent
from the schema (overwrite, color_range, friction_range, mass_range,
max_frames_per_camera, action_key_map), 4 randomize booleans with no
description.

Parameters that only Python can pass (callables, policy objects, kwargs dicts,
observers) and the two internal S3 handles of stop_recording are exempt by
name; a new exemption is a decision, so it lands here.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots import Robot

# Python-only parameters: no agent can construct these from JSON.
_PYTHON_ONLY = {
    "policy_object",
    "policy_kwargs",
    "on_frame",
    "observer",
    "video",  # run_policy: the residual video kwargs are remapped (see _RUN_POLICY_RESIDUAL_VIDEO_KEYS)
    "control_substeps",
    "async_rtc",
    "rtc_inference_timeout_s",
    "max_onframe_failures",
    "reset_between",
    "wbc_install_torque_control",
}
# Remapped on the way in: the schema spells them differently on purpose.
_REMAPPED = {"torque": "torque_vec"}
# stop_recording's S3 handles are an operator pipeline concern, not an agent knob.
_OPERATOR_ONLY = {"bucket", "run_id"}


@pytest.fixture(scope="module")
def sim():
    robot = Robot("so101", mode="sim")
    yield robot
    robot.cleanup()


def test_every_agent_knob_is_in_the_schema(sim) -> None:
    schema = sim.tool_spec["inputSchema"]["json"]
    props = schema["properties"]
    missing: dict[str, list[str]] = {}
    for action in props["action"]["enum"]:
        fn = getattr(sim, action, None)
        if fn is None:
            continue
        for name, param in inspect.signature(fn).parameters.items():
            if name in ("self", "kwargs") or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                continue
            if name in _PYTHON_ONLY or name in _OPERATOR_ONLY:
                continue
            if _REMAPPED.get(name, name) not in props:
                missing.setdefault(action, []).append(name)
    assert missing == {}, f"accepted at runtime but invisible to the agent: {missing}"


def test_randomize_knobs_are_described(sim) -> None:
    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    for name in (
        "randomize_colors",
        "randomize_lighting",
        "randomize_physics",
        "randomize_positions",
        "position_noise",
        "color_range",
        "friction_range",
        "mass_range",
    ):
        assert props[name].get("description", "").startswith("randomize:"), name
    for name in ("color_range", "friction_range", "mass_range"):
        assert props[name]["type"] == "array" and props[name]["minItems"] == props[name]["maxItems"] == 2


def test_the_refusal_and_the_schema_agree_on_start_recording(sim) -> None:
    """The names the unknown-parameter refusal lists are all names the schema shows."""
    import asyncio

    async def call():
        last = None
        tool_use = {"toolUseId": "t", "name": "so101", "input": {"action": "start_recording", "output_dir": "/nowhere"}}
        async for ev in sim.stream(tool_use, {}):
            last = ev
        return last.tool_result

    result = asyncio.run(call())
    text = " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))
    assert result["status"] == "error" and "Valid:" in text
    listed = eval(text.split("Valid:", 1)[1].strip().rstrip("."))  # noqa: S307 - our own repr of a list of names
    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    assert [n for n in listed if n not in props] == []
