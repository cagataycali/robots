"""Every parameter the sim tool advertises to a model carries a real description.

Measured (deepdive D-048): 30 of the 98 parameters in ``tool_spec.json`` had no
description or a sub-25-character one (``n_steps``, ``robot_name``, ``width``,
``timestep``, ``gravity`` ...). The model sees only this schema, so an empty
description is a guess it has to make - and one it makes on every turn, since
the whole spec is re-sent each call. This pins the bar: each parameter names
its unit or vocabulary, and where it is not obvious, the action(s) it belongs to.
"""

from __future__ import annotations

from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA


def test_every_parameter_has_a_description_of_at_least_25_chars() -> None:
    short = {
        name: spec.get("description", "")
        for name, spec in _TOOL_SPEC_SCHEMA["properties"].items()
        if len(spec.get("description", "")) < 25
    }
    assert short == {}, f"undescribed tool parameters: {short}"


def test_step_count_spellings_point_at_each_other() -> None:
    """``step`` takes ``n_steps`` and ``set_gripper`` takes ``steps``; until one spelling wins, each says which action owns it."""
    props = _TOOL_SPEC_SCHEMA["properties"]
    assert "set_gripper" in props["steps"]["description"]
    assert "set_gripper" in props["n_steps"]["description"] and "step" in props["n_steps"]["description"]
