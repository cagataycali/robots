"""Agent bridge/motion/robot_mesh of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_agent_bridge_interrupt.py, test_dashboard_agent_motion.py, test_dashboard_agent_robot_mesh.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import queue

import pytest

from strands_robots.dashboard import agent_bridge as ab
from strands_robots.dashboard.agent_hitl import MOTION_ACTIONS, consume_grant, response_approves
from strands_robots.dashboard.agent_motion import (
    MOTION_ENV,
    agent_motion_allowed,
    peer_is_physical,
)
from strands_robots.tools.robot_mesh import _interrupt_approves

# ============================================================================
# from tests/test_dashboard_agent_bridge_interrupt.py
# The interrupt path through run_turn_blocking: park, resume, dismiss.
# ============================================================================

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


RM_REASON = {
    "action": "broadcast",
    "target": "*ALL_PEERS*",
    "function": "",
    "command": {"action": "execute", "instruction": "wave"},
    "instruction": "",
    "warning": "Fleet-wide physical effect. Reply 'y' to approve, anything else to deny.",
}


def test_parking_is_name_agnostic_robot_mesh_interrupts_flow_verbatim(_clean):
    """robot_mesh raises its OWN interrupt names (robot_mesh-<action>-approval).

    The rail must forward name and reason untouched - a filter on
    'physical_motion' here would silently swallow every SDK-native confirm.
    """

    class _RmAgent(_FakeAgent):
        def __call__(self, agent_input):
            self.calls.append(agent_input)
            if isinstance(agent_input, str):
                return _Result(
                    "interrupt", [_Interrupt(id="rm-1", name="robot_mesh-broadcast-approval", reason=RM_REASON)]
                )
            return _Result("end_turn", text="broadcast sent")

    import queue as _q
    from unittest.mock import patch as _patch

    rm = _RmAgent()
    with _patch.object(ab, "get_agent", lambda: rm):
        q: _q.Queue = _q.Queue()
        ab.run_turn_blocking("wave everyone", q)
        events = _events(q)
        ie = next(e for e in events if e["type"] == "interrupt")
        assert ie["name"] == "robot_mesh-broadcast-approval"
        assert ie["reason"] == RM_REASON  # verbatim, including *ALL_PEERS* and the warning
        assert ab.pending_interrupt()["name"] == "robot_mesh-broadcast-approval"

        # And the resume delivers the literal string into the same turn.
        q2: _q.Queue = _q.Queue()
        ab.resume_interrupt_blocking("rm-1", "y", q2)
        assert rm.calls[-1] == [{"interruptResponse": {"interruptId": "rm-1", "response": "y"}}]
        assert any(e.get("type") == "done" for e in _events(q2))


def test_resume_payload_matches_the_sdk_contract():
    """The frame we build is the shape strands.types.interrupt documents.

    Pinned against the SDK's own TypedDict so an SDK rename fails HERE with
    a sentence, not in production with a silently ignored response.
    """
    from strands.types.interrupt import InterruptResponse, InterruptResponseContent

    assert set(InterruptResponseContent.__annotations__) == {"interruptResponse"}
    assert set(InterruptResponse.__annotations__) == {"interruptId", "response"}


# ============================================================================
# from tests/test_dashboard_agent_motion.py
# Q80: the chat box must not be a way around every motion gate on the play button.
# ============================================================================

ARM = {"presence": {"hw": "so_follower", "robot_type": "so101"}}
SIM = {"presence": {"robot_type": "sim"}}


def test_stopping_is_never_gated():
    """A safety brake that asks permission is not a safety brake."""
    for action in ("stop", "stop_all", "peers", "status"):
        v = agent_motion_allowed(action, peer=ARM, target="so101-arm-1", env={})
        assert v["allowed"] is True, action
        assert v["gated"] is False, action
        assert v["reason"] == ""


def test_a_task_on_a_real_arm_is_refused_and_nothing_is_sent():
    v = agent_motion_allowed("task", peer=ARM, target="so101-arm-1", env={})
    assert v["allowed"] is False
    assert v["physical"] is True
    r = v["reason"]
    # The model relays this text, so it has to read as an instruction to the HUMAN.
    assert "MOVE REAL HARDWARE" in r
    assert "Nothing was sent" in r
    assert "so101-arm-1" in r
    assert "so_follower" in r  # the evidence, not an assertion
    # Every refusal in this dashboard names the next step that needs no permission.
    assert "press play" in r and "simulated peer" in r
    assert MOTION_ENV in r
    # And it must promise the one thing an operator will need in a hurry.
    assert "everyone stop" in r


def test_a_sim_task_is_allowed_because_a_reflex_grant_is_a_dead_gate():
    v = agent_motion_allowed("task", peer=SIM, target="sim-a", env={})
    assert v["allowed"] is True
    assert v["physical"] is False
    assert v["gated"] is True  # it WAS considered, and passed
    for pres in ({"robot_type": "simulation"}, {"robot_type": "mujoco"}, {"sim": True}, {"mode": "sim"}):
        assert agent_motion_allowed("task", peer={"presence": pres}, env={})["allowed"] is True, pres


def test_the_grant_is_an_env_var_so_it_can_be_seen_and_taken_back():
    for value in ("1", "true", "TRUE", "yes", "on"):
        v = agent_motion_allowed("task", peer=ARM, target="a", env={MOTION_ENV: value})
        assert v["allowed"] is True, value
        assert v["granted"] is True
        assert v["physical"] is True, "granted does not make it pixels"
    for value in ("", "0", "no", "off", "maybe", " "):
        assert agent_motion_allowed("task", peer=ARM, env={MOTION_ENV: value})["allowed"] is False, value


def test_an_unclassifiable_peer_counts_as_metal():
    """runRisk's rule: a needless prompt costs a click, a missing one costs a collision."""
    for peer in (None, {}, {"presence": {}}, {"presence": {"robot_type": ""}}, {"presence": {"hw": ""}}, {"state": {}}):
        physical, why = peer_is_physical(peer)
        assert physical is True, peer
        assert why
        assert agent_motion_allowed("task", peer=peer, env={})["allowed"] is False, peer


def test_the_tool_actually_consults_it_before_building_a_command(monkeypatch):
    """The gate is worthless if the wiring skips it -- this pins the call site."""
    from strands_robots.dashboard import agent_bridge

    sent: list = []

    class _Bridge:
        def snapshot(self):
            return {"peers": {"so101-arm-1": ARM, "sim-a": SIM}}

        def send_cmd(self, target, cmd, timeout=0, source=""):
            sent.append((target, cmd))
            return {"ok": True}

    agent_bridge.set_bridge(_Bridge())
    monkeypatch.delenv(MOTION_ENV, raising=False)
    fleet = agent_bridge._make_fleet_tool()
    call = getattr(fleet, "original", None) or getattr(fleet, "__wrapped__", None) or fleet

    res = call(action="task", target="so101-arm-1", instruction="pick up the red cube")
    assert res["status"] == "error"
    assert "MOVE REAL HARDWARE" in res["content"][0]["text"]
    assert sent == [], "the refusal must happen BEFORE any command reaches the mesh"

    # A sim target goes through, so the gate did not just break the tool.
    res2 = call(action="task", target="sim-a", instruction="pick up the red cube")
    assert res2["status"] == "success"
    assert [t for t, _ in sent] == ["sim-a"]

    # And stopping the real arm still works with no grant at all.
    res3 = call(action="stop", target="so101-arm-1")
    assert res3["status"] == "success"
    assert sent[-1][1]["action"] == "stop"


def test_the_grant_lets_an_unattended_agent_through(monkeypatch):
    from strands_robots.dashboard import agent_bridge

    sent: list = []

    class _Bridge:
        def snapshot(self):
            return {"peers": {"so101-arm-1": ARM}}

        def send_cmd(self, target, cmd, timeout=0, source=""):
            sent.append((target, cmd))
            return {"ok": True}

    agent_bridge.set_bridge(_Bridge())
    monkeypatch.setenv(MOTION_ENV, "1")
    fleet = agent_bridge._make_fleet_tool()
    call = getattr(fleet, "original", None) or getattr(fleet, "__wrapped__", None) or fleet
    res = call(action="task", target="so101-arm-1", instruction="wave")
    assert res["status"] == "success"
    assert sent and sent[0][1]["action"] == "execute"


def test_an_unreadable_snapshot_does_not_become_permission(monkeypatch):
    from strands_robots.dashboard import agent_bridge

    class _Bridge:
        def snapshot(self):
            raise RuntimeError("mesh down")

        def send_cmd(self, target, cmd, timeout=0, source=""):
            raise AssertionError("must not be reached")

    agent_bridge.set_bridge(_Bridge())
    monkeypatch.delenv(MOTION_ENV, raising=False)
    fleet = agent_bridge._make_fleet_tool()
    call = getattr(fleet, "original", None) or getattr(fleet, "__wrapped__", None) or fleet
    res = call(action="task", target="so101-arm-1", instruction="wave")
    assert res["status"] == "error"
    assert "MOVE REAL HARDWARE" in res["content"][0]["text"]


# ============================================================================
# from tests/test_dashboard_agent_robot_mesh.py
# Slice B: the dashboard agent carries the SDK's own robot_mesh tool.
# ============================================================================


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
