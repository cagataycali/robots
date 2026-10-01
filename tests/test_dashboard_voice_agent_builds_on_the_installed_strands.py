"""The dashboard voice agent builds against the installed strands, and its stop tool ends the session.

strands-agents 1.57.2 promoted the bidirectional agent to ``strands.bidi`` and dropped the
built-in ``stop_conversation`` tool, so a console importing the old surface failed the moment an
operator opened the microphone. This builds the real ``BidiAgent`` (no network: the OpenAI model
connects on ``start()``) and drives the stop tool the voice model would call.
"""

from __future__ import annotations

import pytest

pytest.importorskip("strands.bidi")


def test_the_voice_agent_builds_and_its_stop_tool_cancels_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands.bidi import BidiAgent
    from strands.types.tools import ToolContext

    from strands_robots.dashboard.voice import build_voice_agent, stop_conversation

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    agent = build_voice_agent("openai")

    assert isinstance(agent, BidiAgent)
    assert set(agent.tool_names) == {"fleet", "stop_conversation"}
    assert not agent.cancel_signal.is_set()

    context = ToolContext(
        tool_use={"toolUseId": "t1", "name": "stop_conversation", "input": {}},
        agent=agent,
        invocation_state={},
    )
    assert stop_conversation(tool_context=context) == "Ending conversation"
    assert agent.cancel_signal.is_set()
