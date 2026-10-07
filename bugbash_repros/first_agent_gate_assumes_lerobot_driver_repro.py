"""
docs/start/first-agent.md "The gate in front of real motion" repro
==================================================================

The code block at docs/start/first-agent.md:76-88 reads literally:

    arm = Robot("so101", mode="real", port="/dev/null")
    agent = Agent(tools=[arm], callback_handler=None)
    result = agent("Run the mock policy on so101 for 2 seconds ...")

The paragraph above it (line 72) describes the behaviour of
*"A mode='real' robot built through the lerobot driver"* with
*"eight actions"* including the gated **execute** and **start**.

But `Robot("so101", mode="real", port="/dev/null")` with the kwargs the
code block gives returns a `FeetechDriver` (so101 ships a native driver;
first-real-arm.md:81 states that `driver='lerobot'` is "the default when
the robot has no native driver" — so101 is NOT such a robot, so the
default is `driver='strands'`).  `FeetechDriver`:

  * is NOT a subclass of `strands.tools.decorator.AgentTool`, so
    `Agent(tools=[arm])` logs ``unrecognized tool specification`` and
    registers no tool.
  * exposes 5 actions (``status, sensors, move_to, set_torque, stop``),
    none of which are the gated ``execute`` or ``start`` the paragraph
    names.

So the documented block cannot reach the operator gate at all: there is
no registered tool for the agent to invoke, and even if there were, the
native driver has no ``execute``/``start`` verbs to gate.  A reader who
copy-pastes the block watches `agent.tool_names == []`.

Running this file (``python first_agent_gate_assumes_lerobot_driver_repro.py``)
asserts the mismatch between the docs claim and what the default-driver
code actually produces.
"""

from __future__ import annotations

import io
import sys

from strands import Agent
from strands.tools.decorator import AgentTool as StrandsAgentTool
from strands_robots import Robot


def _build_agent_capturing_stderr(arm):
    cap = io.StringIO()
    old = sys.stderr
    sys.stderr = cap
    try:
        agent = Agent(tools=[arm], callback_handler=None)
    finally:
        sys.stderr = old
    return agent, cap.getvalue()


def main() -> None:
    # --- Branch A: the code block from docs/start/first-agent.md:76-78, verbatim
    arm_default = Robot("so101", mode="real", port="/dev/null")
    default_driver_cls = type(arm_default).__name__
    default_actions = tuple(
        arm_default.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    )
    default_is_agent_tool = isinstance(arm_default, StrandsAgentTool)

    agent_default, stderr_default = _build_agent_capturing_stderr(arm_default)
    tool_names_default = tuple(agent_default.tool_names)

    # --- Branch B: adding driver="lerobot" (what the paragraph is actually about)
    arm_lerobot = Robot("so101", mode="real", port="/dev/null", driver="lerobot")
    lerobot_driver_cls = type(arm_lerobot).__name__
    lerobot_actions = tuple(
        arm_lerobot.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    )
    lerobot_is_agent_tool = isinstance(arm_lerobot, StrandsAgentTool)

    agent_lerobot, stderr_lerobot = _build_agent_capturing_stderr(arm_lerobot)
    tool_names_lerobot = tuple(agent_lerobot.tool_names)

    print("=== docs/start/first-agent.md:76 (verbatim code block) ===")
    print(f"  Robot(...) type         : {default_driver_cls}")
    print(f"  tool_spec actions ({len(default_actions)}): {default_actions}")
    print(f"  isinstance AgentTool    : {default_is_agent_tool}")
    print(f"  Agent build stderr      : {stderr_default.strip()!r}")
    print(f"  agent.tool_names        : {tool_names_default}")
    print()
    print("=== Same call + driver='lerobot' (what the paragraph describes) ===")
    print(f"  Robot(...) type         : {lerobot_driver_cls}")
    print(f"  tool_spec actions ({len(lerobot_actions)}): {lerobot_actions}")
    print(f"  isinstance AgentTool    : {lerobot_is_agent_tool}")
    print(f"  Agent build stderr      : {stderr_lerobot.strip()!r}")
    print(f"  agent.tool_names        : {tool_names_lerobot}")

    # Hard assertions pinning the defect: the documented block cannot reach
    # the operator gate because no tool is registered on the agent.
    assert default_driver_cls == "FeetechDriver"
    assert default_is_agent_tool is False
    assert "execute" not in default_actions and "start" not in default_actions
    assert tool_names_default == ()
    assert "unrecognized tool specification" in stderr_default

    # And the one-word fix: adding driver="lerobot" matches the docs claim.
    assert lerobot_driver_cls == "Robot"
    assert lerobot_is_agent_tool is True
    assert "execute" in lerobot_actions and "start" in lerobot_actions
    assert tool_names_lerobot == ("so101",)
    assert stderr_lerobot.strip() == ""


if __name__ == "__main__":
    main()
    print("\nrepro passed: default-driver block from docs/start/first-agent.md:76 registers no tool, while adding driver='lerobot' matches the docs' 'eight actions' claim.")
