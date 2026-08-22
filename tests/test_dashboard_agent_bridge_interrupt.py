"""The interrupt path through run_turn_blocking: park, resume, dismiss.

A gated tool call pauses the turn (interrupt event, history unsaved); the
human's answer resumes the SAME agent; a fresh prompt over an unanswered
confirm dismisses it and restores the pre-turn boundary. Fake agent only -
these tests must never run a model or reach the mesh.
"""

from __future__ import annotations

import queue

import pytest

from strands_robots.dashboard import agent_bridge as ab
from strands_robots.dashboard.agent_hitl import consume_grant

REASON = {"tool": "fleet", "action": "task", "target": "arm-1", "instruction": "wave"}


class _Interrupt:
    def __init__(self, id="int-1", name="physical_motion", reason=REASON):
        self.id, self.name, self.reason = id, name, reason


class _Result:
    def __init__(self, stop_reason="end_turn", interrupts=None, text="ok"):
        self.stop_reason = stop_reason
        self.interrupts = interrupts
        self.message = {"content": [{"text": text}]}


class _FakeAgent:
    """First call interrupts; a resume call completes."""

    def __init__(self):
        self.messages: list[dict] = []
        self.callback_handler = None
        self.calls: list = []

    def __call__(self, agent_input):
        self.calls.append(agent_input)
        if isinstance(agent_input, str):
            return _Result("interrupt", [_Interrupt()])
        return _Result("end_turn", text="the arm waved")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(ab, "get_agent", lambda: fake)
    saved: list = []
    monkeypatch.setattr(ab, "_save_history", lambda msgs: saved.append(msgs))
    ab._set_pending(None)
    assert not ab._turn_lock.locked(), "another test left the turn lock held"
    yield fake, saved
    ab._set_pending(None)
    if ab._turn_lock.locked():
        ab._turn_lock.release()


def _events(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_gated_turn_parks_with_an_interrupt_event(_clean):
    _, saved = _clean
    q: queue.Queue = queue.Queue()
    ab.run_turn_blocking("wave at me", q)

    events = _events(q)
    kinds = [e.get("type") for e in events]
    assert "interrupt" in kinds and "done" not in kinds
    ie = next(e for e in events if e["type"] == "interrupt")
    assert ie["id"] == "int-1"
    assert ie["reason"]["target"] == "arm-1"
    assert ab.pending_interrupt()["id"] == "int-1"
    assert not ab._turn_lock.locked(), "a parked turn must not hold the lock"
    assert saved == [], "an unanswered interrupt must NOT be saved to history"
    assert kinds[-1] == "__END__"


def test_resume_continues_the_same_agent_and_saves(_clean):
    fake, saved = _clean
    q: queue.Queue = queue.Queue()
    ab.run_turn_blocking("wave at me", q)
    _events(q)

    q2: queue.Queue = queue.Queue()
    ab.resume_interrupt_blocking("int-1", {"approve": True}, q2)
    events = _events(q2)
    assert [e["type"] for e in events][-2:] == ["done", "__END__"]
    assert next(e for e in events if e["type"] == "done")["text"] == "the arm waved"
    # The resume payload is the SDK's interruptResponse shape, on the SAME agent.
    assert fake.calls[-1] == [{"interruptResponse": {"interruptId": "int-1", "response": {"approve": True}}}]
    assert ab.pending_interrupt() is None
    assert saved, "a completed turn is saved"


def test_resume_with_wrong_id_is_refused(_clean):
    fake, _ = _clean
    q: queue.Queue = queue.Queue()
    ab.run_turn_blocking("wave at me", q)
    _events(q)

    q2: queue.Queue = queue.Queue()
    ab.resume_interrupt_blocking("other-id", True, q2)
    events = _events(q2)
    assert events[0]["type"] == "error"
    assert len(fake.calls) == 1, "a refused resume must not reach the agent"
    assert ab.pending_interrupt() is not None, "the real confirm is still answerable"


def test_new_prompt_over_pending_confirm_dismisses_it(_clean, monkeypatch):
    fake, _ = _clean
    q: queue.Queue = queue.Queue()
    ab.run_turn_blocking("wave at me", q)
    _events(q)
    assert ab.pending_interrupt() is not None

    rebuilt = []
    monkeypatch.setattr(ab, "_abandon_pending", lambda: (rebuilt.append(1), ab._set_pending(None))[1])
    q2: queue.Queue = queue.Queue()
    ab.run_turn_blocking("actually, list the peers", q2)
    events = _events(q2)
    assert any(e.get("type") == "notice" and "dismissed" in e.get("text", "") for e in events)
    assert rebuilt, "the parked turn is dropped so history stays at the pre-turn boundary"


def test_abandon_pending_resets_the_agent_not_the_history_file(monkeypatch):
    dropped = []
    monkeypatch.setattr(ab, "clear_history", lambda: dropped.append(1))
    with ab._agent_lock:
        ab._agent = object()
    ab._set_pending({"id": "int-1"})
    ab._abandon_pending()
    assert ab.pending_interrupt() is None
    with ab._agent_lock:
        assert ab._agent is None, "next turn rebuilds from disk = pre-turn boundary"
    assert dropped == [], "the history FILE survives - only the parked turn is dropped"


def test_fleet_tool_consumes_a_deposited_grant(monkeypatch):
    """The human's yes lets exactly one matching task through the backstop."""
    from strands_robots.dashboard.agent_hitl import deposit_grant
    from strands_robots.dashboard.agent_motion import MOTION_ENV

    sent: list = []

    class _Bridge:
        def snapshot(self):
            return {"peers": {"arm-1": {"presence": {"hw": "so_follower"}}}}

        def send_cmd(self, target, cmd, timeout=0, source=""):
            sent.append((target, cmd))
            return {"ok": True}

    ab.set_bridge(_Bridge())
    monkeypatch.delenv(MOTION_ENV, raising=False)
    fleet = ab._make_fleet_tool()
    call = getattr(fleet, "original", None) or getattr(fleet, "__wrapped__", None) or fleet

    deposit_grant("fleet", {"action": "task", "target": "arm-1", "instruction": "wave"})
    res = call(action="task", target="arm-1", instruction="wave")
    assert res["status"] == "success"
    assert sent and sent[0][1]["action"] == "execute"

    # One-shot: the identical call without a fresh yes is refused again.
    res2 = call(action="task", target="arm-1", instruction="wave")
    assert res2["status"] == "error"
    assert len(sent) == 1
    assert consume_grant("fleet", {"action": "task", "target": "arm-1", "instruction": "wave"}) is False
