"""Slice B: the dashboard agent carries the SDK's own robot_mesh tool.

Three promises, each cheap to break silently:
 - robot_mesh is on the agent's tool list (the agent-native surface, not
   just the bespoke fleet wrapper);
 - it is NOT in the dashboard hook's MOTION_ACTIONS - robot_mesh raises its
   own SDK interrupt, and a second gate would ask the operator twice for
   one command;
 - the UI's "y"/"n" answer satisfies BOTH approval checks, because
   robot_mesh refuses any non-string response.
"""

from __future__ import annotations

from strands_robots.dashboard import agent_bridge as ab
from strands_robots.dashboard.agent_hitl import MOTION_ACTIONS, response_approves
from strands_robots.tools.robot_mesh import _interrupt_approves


def test_robot_mesh_is_on_the_agents_tool_list():
    tool = ab._robot_mesh_tool()
    assert tool is not None, "robot_mesh failed to import - the agent lost its native surface"
    name = getattr(tool, "tool_name", None) or getattr(tool, "__name__", "")
    assert "robot_mesh" in str(name)


def test_robot_mesh_is_not_double_gated():
    assert "robot_mesh" not in MOTION_ACTIONS, (
        "robot_mesh carries its own SDK interrupt (tool_context.interrupt in "
        "strands_robots/tools/robot_mesh.py); listing it in MOTION_ACTIONS would "
        "ask the operator twice for one command"
    )


def test_the_ui_answer_satisfies_both_gates():
    """The frontend sends the literal 'y' / 'n' (interruptConfirm.ts)."""
    for yes in ("y", "yes"):
        assert response_approves(yes), yes
        assert _interrupt_approves(yes), yes
    for no in ("n", "no", "", None, {"approve": True}, True):
        assert not _interrupt_approves(no), no
    assert not response_approves("n")
    # The dict shape still approves OUR hook (backward compat) but robot_mesh
    # refuses it - which is exactly why the UI standardised on strings.
    assert response_approves({"approve": True})


def test_default_prompt_teaches_the_declined_confirm():
    """The model must accept a no rather than retry the same command."""
    assert "robot_mesh" in ab.DEFAULT_SYSTEM_PROMPT
    assert "never retry" in ab.DEFAULT_SYSTEM_PROMPT
