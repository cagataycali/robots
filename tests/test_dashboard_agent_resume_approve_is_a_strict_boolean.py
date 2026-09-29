# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent-console consent answer is the JSON boolean ``true`` or it is a refusal.

Finding f025 (LOW). ``/ws/agent`` answered an interrupt with
``bool(frame.get("approve"))``, so the string ``"false"``, the string ``"no"``,
``1`` or a non-empty list all read as consent, and only the empty values did not.
The frame comes from the page, whose operator authored the request being
approved, so no boundary was crossed; but the motion gate must not turn a
misspelt refusal into a yes. ``approve`` and ``always`` are now strict: ``True``
or the string ``"true"`` mean yes, anything else means no.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import agent_console, routes_agent, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests.test_dashboard_sim_routes import FakeEngine  # noqa: E402

#: A browser always sends Origin on a socket handshake; this is the dashboard's own page.
OWN_PAGE = {"origin": "http://testserver"}

INTERRUPT: list[dict[str, Any]] = [
    {"type": "text", "text": "moving"},
    {"type": "interrupt", "id": "i1", "name": "sim_motion", "reason": {"detail": "2 -> 1.000 rad"}},
]
DONE: list[dict[str, Any]] = [{"type": "done", "stop_reason": "end_turn"}]


class RecordingConsole:
    def __init__(self) -> None:
        self.script: list[list[dict[str, Any]]] = [list(INTERRUPT), list(DONE)]
        self.prompts: list[Any] = []

    async def run(self, prompt: Any) -> AsyncIterator[dict[str, Any]]:
        self.prompts.append(prompt)
        for event in self.script.pop(0):
            yield event
            await asyncio.sleep(0)

    resume = staticmethod(agent_console.Console.resume)


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[Any]:
    monkeypatch.setenv("DASHBOARD_SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(sim_session, "_default_factory", FakeEngine)
    built = create_app()
    yield built
    for session in built.state.safety.store.all():
        built.state.safety.store.remove(session.id)


def _answer(app: Any, approve: Any, always: Any = False) -> dict[str, Any]:
    console = RecordingConsole()
    app.state.console_factory = lambda: console
    with TestClient(app, headers=OWN_PAGE) as client, client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "say", "text": "raise joint 2"})
        assert ws.receive_json()["type"] == "text"
        assert ws.receive_json()["type"] == "interrupt"
        ws.send_json({"type": "resume", "id": "i1", "approve": approve, "always": always})
        assert ws.receive_json()["type"] == "done"
    return console.prompts[1][0]["interruptResponse"]["response"]


class TestOnlyTrueIsConsent:
    @pytest.mark.parametrize("approve", ["false", "no", "0", "yes", 1, [False], {"approve": True}])
    def test_anything_that_is_not_the_boolean_true_is_a_refusal(self, app: Any, approve: Any) -> None:
        assert _answer(app, approve) == {"approve": False, "always": False}

    @pytest.mark.parametrize("approve", [True, "true"])
    def test_true_is_consent(self, app: Any, approve: Any) -> None:
        assert _answer(app, approve) == {"approve": True, "always": False}

    def test_always_is_held_to_the_same_rule(self, app: Any) -> None:
        assert _answer(app, True, always="false") == {"approve": True, "always": False}
        assert _answer(app, True, always=True) == {"approve": True, "always": True}


class TestTheParserItself:
    @pytest.mark.parametrize("value", [True, "true"])
    def test_yes(self, value: Any) -> None:
        assert routes_agent.strict_flag(value) is True

    @pytest.mark.parametrize("value", [False, None, "", "false", "True", "TRUE", "yes", 1, 0, [True], {"a": 1}])
    def test_no(self, value: Any) -> None:
        assert routes_agent.strict_flag(value) is False
