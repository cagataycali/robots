"""The missing-``policy_port`` refusal names the provider that needs it and the way to run with no server.

Measured with an agent driving ``Robot("so101", mode="real")`` and the prompt
"Wave the arm for 3 seconds" (no word about servers): the tool defaulted to a
server-dialing provider, answered ``policy_port is required to build a policy
(pass the port of the policy server, or use run_policy with a pre-built
policy_object)`` and the agent asked the operator for a port. ``run_policy`` and
``policy_object`` are not things this tool's caller can reach for, the provider
that wanted the port was one the caller never chose, and the in-process
providers the tool schema itself names were absent - so the one refusal that
could have led to ``mock`` led to a question instead.

The default is ``lerobot_local`` now, which dials nothing, so the default path
refuses on the missing checkpoint instead; the port refusal is graded on the
server-dialing provider a caller has to choose by name.
"""

from __future__ import annotations

import os

import pytest

from strands_robots import Robot


@pytest.fixture
def arm():
    robot = Robot("so101", mode="real", port=os.devnull)
    yield robot
    robot.cleanup()


def _text(result: dict) -> str:
    return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))


class TestTheRefusal:
    def test_names_the_chosen_server_provider(self, arm) -> None:
        result = arm._policy_port_error(None, "execute_task", "moveit2")
        assert result is not None and result["status"] == "error"
        text = _text(result)
        assert text.startswith("execute_task: policy_port is required - policy_provider 'moveit2' dials a policy server")

    def test_no_provider_is_called_the_default_in_the_port_refusal(self, arm) -> None:
        """The default builds in process, so no port refusal can be about a default."""
        text = _text(arm._policy_port_error(None, "start_task", "moveit2"))
        assert "(the default)" not in text

    def test_names_the_in_process_way(self, arm) -> None:
        text = _text(arm._policy_port_error(None, "execute_task", "moveit2"))
        assert "policy_provider='mock'" in text
        assert "'lerobot_local'" in text
        assert "With no server running" in text

    def test_does_not_name_a_verb_this_tool_lacks(self, arm) -> None:
        text = _text(arm._policy_port_error(None, "execute_task", "moveit2"))
        assert "run_policy" not in text
        assert "policy_object" not in text

    def test_an_in_process_provider_needs_no_port(self, arm) -> None:
        assert arm._policy_port_error(None, "execute_task", "mock") is None
        assert arm._policy_port_error(None, "execute_task", "lerobot_local") is None

    def test_the_tool_surface_refuses_the_default_on_its_missing_checkpoint(self, arm) -> None:
        """Through ``stream``: the default provider's refusal reaches the agent before any gate.

        The default is ``lerobot_local``, so with nothing else given the refusal
        is about the checkpoint, never about a port for a server nobody chose.
        """
        import asyncio

        async def collect():
            events = []
            async for ev in arm.stream(
                {"toolUseId": "t1", "name": arm.tool_name, "input": {"action": "execute", "instruction": "wave"}},
                {},
            ):
                events.append(ev)
            return events

        events = asyncio.run(collect())
        result = events[-1].tool_result if hasattr(events[-1], "tool_result") else events[-1]
        text = _text(result)
        assert "policy_provider='lerobot_local' builds its policy from pretrained_name_or_path" in text
        assert "policy_port is required" not in text
