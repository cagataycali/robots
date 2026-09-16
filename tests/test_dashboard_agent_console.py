"""The dashboard agent console: gate, tools over Safety, and the /ws/agent protocol. No model is called."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import agent_console, routes_sim, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests.test_dashboard_sim_routes import FakeEngine  # noqa: E402


@pytest.fixture
def safety(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("DASHBOARD_SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(sim_session, "_default_factory", lambda robot: FakeEngine(robot))
    s = routes_sim.Safety(sim_session.SessionStore())
    yield s
    for sess in s.store.all():
        s.store.remove(sess.id)


# -- approvals ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (True, (True, False)),
        (False, (False, False)),
        ({"approve": True}, (True, False)),
        ({"approve": True, "always": True}, (True, True)),
        ({"approve": False, "always": True}, (False, False)),  # 'always' never widens a no
        ({"approve": "yes"}, (False, False)),  # only a real bool is a yes
        ("yes", (True, False)),
        ("sure", (False, False)),
        (None, (False, False)),
    ],
)
def test_response_approves(response, expected):
    assert agent_console.response_approves(response) == expected


# -- the gate -----------------------------------------------------------------


class FakeEvent:
    """A BeforeToolCallEvent stand-in: interrupt() returns the scripted answer."""

    def __init__(self, name: str, tool_input: dict, answer: Any = None):
        self.tool_use = {"name": name, "input": tool_input}
        self.answer = answer
        self.cancel_tool: Any = None
        self.interrupted_with: Any = None

    def interrupt(self, name, reason=None):
        self.interrupted_with = (name, reason)
        return self.answer


def test_gate_asks_only_for_motion_tools(monkeypatch):
    monkeypatch.setattr("strands_robots.tools._hitl_audit.log_operator_response", lambda *a, **k: None)
    gate = agent_console.MotionGate(agent_console.Grants())
    ev = FakeEvent("sim_state", {"session_id": "abc"})
    gate._gate(ev)
    assert ev.interrupted_with is None and ev.cancel_tool is None
    ev = FakeEvent("sim_stop", {"session_id": "abc"})
    gate._gate(ev)
    assert ev.interrupted_with is None, "stopping is never gated"


def test_gate_interrupt_carries_what_a_yes_moves(monkeypatch):
    monkeypatch.setattr("strands_robots.tools._hitl_audit.log_operator_response", lambda *a, **k: None)
    gate = agent_console.MotionGate(agent_console.Grants())
    ev = FakeEvent(
        "sim_set_joints", {"session_id": "abc", "positions": {"2": 1.0, "3": -0.25}}, answer={"approve": True}
    )
    gate._gate(ev)
    name, reason = ev.interrupted_with
    assert name == agent_console.INTERRUPT_NAME
    assert reason["session_id"] == "abc" and reason["detail"] == "2 → 1.000 rad, 3 → -0.250 rad"
    assert ev.cancel_tool is None


def test_gate_no_cancels_and_is_audited(monkeypatch):
    rows = []
    monkeypatch.setattr("strands_robots.tools._hitl_audit.log_operator_response", lambda *a, **k: rows.append((a, k)))
    gate = agent_console.MotionGate(agent_console.Grants())
    ev = FakeEvent("sim_set_joints", {"session_id": "abc", "positions": {"2": 1.0}}, answer={"approve": False})
    gate._gate(ev)
    assert "declined" in str(ev.cancel_tool)
    assert (
        rows and rows[0][0] == ("dashboard_agent_console", "sim_set_joints", "abc") and rows[0][1]["approved"] is False
    )


def test_gate_always_grants_that_session_only(monkeypatch):
    monkeypatch.setattr("strands_robots.tools._hitl_audit.log_operator_response", lambda *a, **k: None)
    grants = agent_console.Grants()
    gate = agent_console.MotionGate(grants)
    ev = FakeEvent(
        "sim_set_joints", {"session_id": "abc", "positions": {"1": 0.5}}, answer={"approve": True, "always": True}
    )
    gate._gate(ev)
    assert grants.covers("abc")
    again = FakeEvent("sim_set_joints", {"session_id": "abc", "positions": {"1": 0.0}}, answer={"approve": False})
    gate._gate(again)
    assert again.interrupted_with is None and again.cancel_tool is None, "granted: not asked again"
    other = FakeEvent("sim_set_joints", {"session_id": "xyz", "positions": {"1": 0.0}}, answer={"approve": False})
    gate._gate(other)
    assert other.interrupted_with is not None, "a different session still asks"


# -- the tools ----------------------------------------------------------------


def _tools(safety) -> dict[str, Any]:
    return {t.tool_name: t for t in agent_console.build_tools(safety)}


def test_tools_drive_sessions_through_safety(safety):
    t = _tools(safety)
    assert set(t) == {
        "robots",
        "sim_sessions",
        "sim_start",
        "sim_state",
        "sim_set_joints",
        "sim_reset",
        "sim_stop",
        "emergency_stop",
    }
    assert t["sim_sessions"]() == []
    snap = t["sim_start"](robot="so101")
    sid = snap["id"]
    assert snap["state"] == "running" and "model_path" not in snap
    assert t["sim_sessions"]()[0]["id"] == sid
    assert t["sim_set_joints"](session_id=sid, positions={"j0": 0.1})["status"] == "success"
    assert (
        safety.store.get(sid).command_log[-1][0] == "set_joints"
        if hasattr(safety.store.get(sid), "command_log")
        else True
    )
    assert t["sim_reset"](session_id=sid)["status"] == "success"
    with pytest.raises(ValueError):
        t["sim_set_joints"](session_id=sid, positions={})
    with pytest.raises(ValueError):
        t["sim_state"](session_id="nope")
    assert t["sim_stop"](session_id=sid) == {"ok": True}
    assert t["sim_sessions"]() == []


def test_tools_refused_under_estop_except_stopping(safety):
    t = _tools(safety)
    sid = t["sim_start"](robot="so101")["id"]
    t["emergency_stop"]()
    assert safety.lockout.state == "locked"
    with pytest.raises(PermissionError, match="e-stop"):
        t["sim_set_joints"](session_id=sid, positions={"j0": 0.1})
    with pytest.raises(PermissionError):
        t["sim_start"](robot="so101")
    assert t["sim_stop"](session_id=sid) == {"ok": True}, "stopping is never refused"


def test_sim_start_refuses_a_robot_without_an_asset(safety):
    with pytest.raises(ValueError, match="simulation asset"):
        _tools(safety)["sim_start"](robot="not-a-robot")


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
                {"toolUse": {"name": "sim_state", "input": {"session_id": "a"}}},
                {"toolResult": {"status": "success", "content": [{"text": "x"}, {"json": {}}]}},
            ],
        }
    }
    assert agent_console._translate(msg) == [
        {"type": "tool_use", "name": "sim_state", "input": {"session_id": "a"}},
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
    monkeypatch.setattr(sim_session, "_default_factory", lambda robot: FakeEngine(robot))
    a = create_app()
    yield a
    for s in a.state.safety.store.all():
        a.state.safety.store.remove(s.id)


def test_agent_info(app):
    with TestClient(app) as c:
        info = c.get("/api/agent").json()
    assert info["asks_first"] == ["sim_set_joints"] and info["interrupt"] == "sim_motion" and info["model"]


def test_ws_turn_interrupt_resume(app):
    console = ScriptedConsole(
        [
            [
                {"type": "text", "text": "moving"},
                {"type": "interrupt", "id": "i1", "name": "sim_motion", "reason": {"detail": "2 → 1.000 rad"}},
            ],
            [{"type": "tool_result", "status": "success", "text": "ok"}, {"type": "done", "stop_reason": "end_turn"}],
        ]
    )
    app.state.console_factory = lambda: console
    with TestClient(app) as c, c.websocket_connect("/ws/agent") as ws:
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
    with TestClient(app) as c, c.websocket_connect("/ws/agent") as ws:
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
    with TestClient(app) as c, c.websocket_connect("/ws/agent") as ws:
        m = ws.receive_json()
        assert m["type"] == "error" and "no credentials" in m["message"]


def test_ws_stranger_is_closed(app, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr("strands_robots.dashboard.access.peer_is_loopback", lambda request: False)
    app.state.console_factory = lambda: ScriptedConsole([])
    # Accepted, then closed with 4401: that order is what carries the code to
    # the page, which shows the login screen on it.
    with (
        TestClient(app) as c,
        c.websocket_connect("/ws/agent", headers={"x-forwarded-for": "10.0.0.9"}) as ws,
        pytest.raises(WebSocketDisconnect) as exc,
    ):
        ws.receive_json()
    assert exc.value.code == 4401
