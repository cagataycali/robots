"""The dashboard agent console: peers-only tools, the e-stop over Safety, and the /ws/agent protocol. No model is called."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import agent_console, agent_hitl, routes_sim, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402

#: The dashboard's own page at the TestClient host: a browser always sends Origin on a socket handshake (f022).
OWN_PAGE = {"origin": "http://testserver"}
from tests.test_dashboard_sim_routes import FakeEngine  # noqa: E402


@pytest.fixture
def safety(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("DASHBOARD_SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(sim_session, "_default_factory", FakeEngine)
    s = routes_sim.Safety(sim_session.SessionStore())
    yield s
    for sess in s.store.all():
        s.store.remove(sess.id)


# -- the tools: no robot of its own ---------------------------------------------------------------


def _tools(safety) -> dict[str, Any]:
    return {t.tool_name: t for t in agent_console.build_tools(safety)}


#: The in-process simulation tools the console carried before it drove mesh peers only.
RETIRED_SIM_TOOLS = ("robots", "sim_sessions", "sim_start", "sim_state", "sim_set_joints", "sim_reset", "sim_stop")


def test_the_console_holds_no_in_process_sim_tool(safety):
    """Every robot the agent drives is a mesh peer: the seven session tools are gone, the e-stop stays."""
    t = _tools(safety)
    assert set(t) == {"emergency_stop"}
    assert not (set(t) & set(RETIRED_SIM_TOOLS))
    assert agent_console.expected_tool_names(None) == ["emergency_stop"]
    for name in RETIRED_SIM_TOOLS:
        assert not hasattr(agent_console, name)
    for gone in ("MotionGate", "Grants", "MOTION_TOOLS", "INTERRUPT_NAME", "response_approves"):
        assert not hasattr(agent_console, gone), f"{gone} belonged to the sim gate and left with it"


def test_the_system_prompt_names_no_sim_tool():
    # "robots" is a word; the retired TOOL is the backticked or underscore spelling
    for name in RETIRED_SIM_TOOLS[1:]:
        assert name not in agent_console.SYSTEM_PROMPT
    assert "`robots`" not in agent_console.SYSTEM_PROMPT and "Sim tab" not in agent_console.SYSTEM_PROMPT
    assert "spawn_robot" in agent_console.SYSTEM_PROMPT and "fleet" in agent_console.SYSTEM_PROMPT


def test_emergency_stop_latches_the_lockout_and_is_never_refused(safety):
    t = _tools(safety)
    out = t["emergency_stop"]()
    assert out["lockout"] == safety.lockout.as_fields() and safety.lockout.state == "locked"
    # a second stop while latched is still accepted: stopping is never gated
    assert t["emergency_stop"]()["lockout"]["state"] == "locked"


class _Bridge:
    def __init__(self, peers: dict[str, Any]):
        self.peers = peers

    def snapshot(self) -> dict[str, Any]:
        return {"peers": self.peers}


def test_asks_first_is_the_real_arm_proxies_only():
    """The badge is derived from the mesh: real arms ask, sims and hosts do not, no bridge asks nothing."""
    assert agent_console.asks_first(None) == []
    bridge = _Bridge(
        {
            "arm-1": {"presence": {"robot_type": "robot"}, "state": {"joints": {"j1": 0.0}}},
            "lane-so101": {"presence": {"robot_type": "sim"}},
            "lane-so101__so101": {"presence": {"robot_type": "sim"}, "state": {"joints": {"j1": 0.0}}},
            "dashboard-x-safety": {"presence": {"robot_type": "dashboard"}},
        }
    )
    assert agent_console.asks_first(bridge) == ["arm_1"]


def test_asks_first_survives_an_unreadable_bridge():
    class Broken:
        def snapshot(self):
            raise RuntimeError("mesh down")

    assert agent_console.asks_first(Broken()) == []


def test_resume_prompt_is_the_sdk_interrupt_response():
    assert agent_console.Console.resume("i1", True, True) == [
        {"interruptResponse": {"interruptId": "i1", "response": {"approve": True, "always": True}}}
    ]


def test_translate_flattens_sdk_events():
    assert agent_console._translate({"data": "hi"}) == [{"type": "text", "text": "hi"}]
    msg = {
        "message": {
            "role": "assistant",
            "content": [
                {"toolUse": {"name": "arm_1", "input": {"action": "state"}}},
                {"toolResult": {"status": "success", "content": [{"text": "x"}, {"json": {}}]}},
            ],
        }
    }
    assert agent_console._translate(msg) == [
        {"type": "tool_use", "name": "arm_1", "input": {"action": "state"}},
        {"type": "tool_result", "status": "success", "text": "x"},
    ]
    assert agent_console._translate({"event": {"contentBlockDelta": {}}}) == []


# -- the socket ---------------------------------------------------------------


class ScriptedConsole:
    """Yields a fixed event script; records every prompt it was given."""

    def __init__(self, script):
        self.script = script
        self.prompts: list[Any] = []

    async def run(self, prompt):
        self.prompts.append(prompt)
        for ev in self.script.pop(0):
            yield ev
            await asyncio.sleep(0)

    resume = staticmethod(agent_console.Console.resume)


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("DASHBOARD_SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(sim_session, "_default_factory", FakeEngine)
    a = create_app()
    yield a
    for s in a.state.safety.store.all():
        a.state.safety.store.remove(s.id)


def test_agent_info(app):
    with TestClient(app, headers=OWN_PAGE) as c:
        info = c.get("/api/agent").json()
    # no bridge in this app: nothing asks first, and the interrupt is the fleet hook's
    assert info["asks_first"] == [] and info["interrupt"] == agent_hitl.INTERRUPT_NAME
    assert not (set(info["tools"]) & set(RETIRED_SIM_TOOLS))
    assert info["model"] == agent_console.model_id()


@pytest.mark.parametrize("configured", [None, "eu.anthropic.claude-haiku-4-5-20251001-v1:0"])
def test_the_console_names_a_model_the_installed_sdk_knows(app, monkeypatch, configured):
    """Unset, the model is the SDK's own default - the console keeps no second copy of it."""
    from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID

    monkeypatch.delenv(agent_console.MODEL_ENV, raising=False)
    if configured is not None:
        monkeypatch.setenv(agent_console.MODEL_ENV, configured)
    expected = configured or DEFAULT_BEDROCK_MODEL_ID
    with TestClient(app, headers=OWN_PAGE) as c:
        assert c.get("/api/agent").json()["model"] == expected
    assert agent_console.default_model().config["model_id"] == expected


def test_ws_turn_interrupt_resume(app):
    console = ScriptedConsole(
        [
            [
                {"type": "text", "text": "moving"},
                {
                    "type": "interrupt",
                    "id": "i1",
                    "name": "physical_motion",
                    "reason": {"target": "arm-1", "action": "execute"},
                },
            ],
            [{"type": "tool_result", "status": "success", "text": "ok"}, {"type": "done", "stop_reason": "end_turn"}],
        ]
    )
    app.state.console_factory = lambda: console
    with TestClient(app, headers=OWN_PAGE) as c, c.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "say", "text": "raise joint 2"})
        assert ws.receive_json()["type"] == "text"
        it = ws.receive_json()
        assert it["type"] == "interrupt" and it["id"] == "i1"
        ws.send_json({"type": "resume", "id": "i1", "approve": True, "always": False})
        assert ws.receive_json()["type"] == "tool_result"
        assert ws.receive_json()["type"] == "done"
    assert console.prompts[0] == "raise joint 2"
    assert console.prompts[1] == [
        {"interruptResponse": {"interruptId": "i1", "response": {"approve": True, "always": False}}}
    ]


def test_ws_rejects_bad_frames_and_long_prompts(app):
    app.state.console_factory = lambda: ScriptedConsole([])
    with TestClient(app, headers=OWN_PAGE) as c, c.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "dance"})
        assert "say|resume" in ws.receive_json()["message"]
        ws.send_json({"type": "say", "text": "   "})
        assert ws.receive_json()["message"] == "say what?"
        ws.send_json({"type": "say", "text": "x" * (agent_console.MAX_PROMPT_CHARS + 1)})
        assert "longer than" in ws.receive_json()["message"]


def test_ws_without_an_agent_closes_with_a_reason(app):
    def boom():
        raise RuntimeError("no credentials")

    app.state.console_factory = boom
    with TestClient(app, headers=OWN_PAGE) as c, c.websocket_connect("/ws/agent") as ws:
        m = ws.receive_json()
        assert m["type"] == "error" and "no credentials" in m["message"]


def test_ws_stranger_is_closed(app, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr("strands_robots.dashboard.access.peer_is_loopback", lambda request: False)
    app.state.console_factory = lambda: ScriptedConsole([])
    # Accepted, then closed with 4401: that order is what carries the code to
    # the page, which shows the login screen on it.
    with (
        TestClient(app, headers=OWN_PAGE) as c,
        c.websocket_connect("/ws/agent", headers={"x-forwarded-for": "10.0.0.9"}) as ws,
        pytest.raises(WebSocketDisconnect) as exc,
    ):
        ws.receive_json()
    assert exc.value.code == 4401
