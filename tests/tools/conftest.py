"""The agent-turn tests need strands (the [tools] extra); without it they are not collected."""

import importlib.util

collect_ignore_glob = (
    [] if importlib.util.find_spec("strands") else ["test_agent_turn.py", "test_sim_tool.py", "test_tool_gate.py"]
)
