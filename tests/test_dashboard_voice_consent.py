"""The voice surface: the gate covers it, and a spoken refusal can still become a decision.

Voice forwards audio and transcript ONLY, so a refusal raised inside the fleet tool is spoken once and
gone -- and the operator cannot grant a permission by talking. So the refusal is also pushed to the
browser as a needs_consent frame, and the grant stays a deliberate tap on the same ConsentSheet.
"""

from __future__ import annotations

import inspect

from strands_robots.dashboard import agent_bridge, voice
from strands_robots.dashboard.agent_motion import MOTION_ENV, agent_motion_allowed

ARM = {"presence": {"hw": "so_follower"}}


class _Bridge:
    def __init__(self):
        self.sent = []

    def snapshot(self):
        return {"peers": {"so101-arm-1": ARM, "sim-a": {"presence": {"robot_type": "sim"}}}}

    def send_cmd(self, target, cmd, timeout=0, source=""):
        self.sent.append((target, cmd))
        return {"ok": True}


def _call(fleet):
    return getattr(fleet, "original", None) or getattr(fleet, "__wrapped__", None) or fleet


def test_voice_uses_the_same_gated_fleet_tool():
    """A future refactor giving voice its own copy of the tool would silently unguard it -- the voice
    session is the surface where nobody would notice, because there is no error bubble to read."""
    src = inspect.getsource(voice.build_voice_agent)
    assert "_make_fleet_tool" in src
    assert "agent_bridge" in src or "from strands_robots.dashboard.agent_bridge" in src


def test_the_refusal_reaches_a_listener_with_a_classifiable_text(monkeypatch):
    from strands_robots.dashboard.consent import classify_refusal

    heard = []
    drop = agent_bridge.add_refusal_listener(heard.append)
    try:
        bridge = _Bridge()
        agent_bridge.set_bridge(bridge)
        monkeypatch.delenv(MOTION_ENV, raising=False)
        _call(agent_bridge._make_fleet_tool())(
            action="task", target="so101-arm-1", instruction="wave")
        assert len(heard) == 1, "voice has no other rail; the notification is the whole mechanism"
        need = classify_refusal(heard[0])
        assert need is not None and need.kind == "agent_physical_motion"
        assert bridge.sent == []
    finally:
        drop()


def test_nothing_is_announced_when_nothing_was_refused(monkeypatch):
    heard = []
    drop = agent_bridge.add_refusal_listener(heard.append)
    try:
        agent_bridge.set_bridge(_Bridge())
        monkeypatch.delenv(MOTION_ENV, raising=False)
        fleet = _call(agent_bridge._make_fleet_tool())
        fleet(action="peers")
        fleet(action="stop", target="so101-arm-1")
        fleet(action="task", target="sim-a", instruction="wave")
        assert heard == [], "a consent card for a turn that was allowed teaches clicking yes by reflex"
    finally:
        drop()


def test_a_broken_listener_cannot_break_a_turn(monkeypatch):
    def boom(_text):
        raise RuntimeError("browser gone")

    drop = agent_bridge.add_refusal_listener(boom)
    try:
        agent_bridge.set_bridge(_Bridge())
        monkeypatch.delenv(MOTION_ENV, raising=False)
        res = _call(agent_bridge._make_fleet_tool())(
            action="task", target="so101-arm-1", instruction="wave")
        assert res["status"] == "error"
        assert "MOVE REAL HARDWARE" in res["content"][0]["text"]
    finally:
        drop()


def test_the_remover_actually_removes():
    heard = []
    drop = agent_bridge.add_refusal_listener(heard.append)
    drop()
    drop()  # idempotent: a session may unregister on both the early-return and the finally path
    agent_bridge._notify_refusal("STRANDS_DASH_AGENT_PHYSICAL_MOTION")
    assert heard == [], "a leaked listener outlives its websocket and pins its closure forever"


def test_the_session_drops_its_listener_on_every_exit_path():
    src = inspect.getsource(voice.run_voice_session)
    # the early return when the model cannot be built, and the finally
    assert src.count("drop_listener()") >= 2
    assert src.index("drop_listener()") < src.index("runner.cancel()")


def test_the_voice_prompt_tells_the_model_the_refusal_is_final():
    """Otherwise the model reroutes: rewording, retrying, or picking a different arm -- which is the
    one failure mode of a gate that only says no to a tool call."""
    # Normalised: the prompt is a wrapped literal, so pinning line breaks would fail on a reflow
    # that changed nothing about what the model is told.
    p = " ".join(voice.VOICE_PROMPT.split())
    assert "do NOT retry" in p
    assert "spoken yes cannot grant it" in p
    assert "on screen" in p
    assert "Stopping is never refused" in p


def test_a_granted_machine_speaks_no_card(monkeypatch):
    heard = []
    drop = agent_bridge.add_refusal_listener(heard.append)
    try:
        bridge = _Bridge()
        agent_bridge.set_bridge(bridge)
        monkeypatch.setenv(MOTION_ENV, "1")
        res = _call(agent_bridge._make_fleet_tool())(
            action="task", target="so101-arm-1", instruction="wave")
        assert res["status"] == "success" and heard == []
        assert bridge.sent and bridge.sent[0][1]["action"] == "execute"
    finally:
        drop()
