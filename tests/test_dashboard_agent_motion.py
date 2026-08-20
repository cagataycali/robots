"""Q80: the chat box must not be a way around every motion gate on the ▶ button.

The agent's fleet tool makes the SAME send_cmd as ▶ ({"action": "execute"}) on the SAME peer. ▶ has a
confirm sheet naming "physical" (JOURNEYS #3) and, since Q79, refuses a policy that cannot drive that
robot. Typing a sentence into the dock had neither.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.agent_motion import (
    MOTION_ENV,
    agent_motion_allowed,
    peer_is_physical,
)

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
    assert "press ▶" in r and "simulated peer" in r
    assert MOTION_ENV in r
    # And it must promise the one thing an operator will need in a hurry.
    assert "everyone stop" in r


def test_a_sim_task_is_allowed_because_a_reflex_grant_is_a_dead_gate():
    v = agent_motion_allowed("task", peer=SIM, target="sim-a", env={})
    assert v["allowed"] is True
    assert v["physical"] is False
    assert v["gated"] is True  # it WAS considered, and passed
    for pres in ({"robot_type": "simulation"}, {"robot_type": "mujoco"},
                 {"sim": True}, {"mode": "sim"}):
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
    for peer in (None, {}, {"presence": {}}, {"presence": {"robot_type": ""}},
                 {"presence": {"hw": ""}}, {"state": {}}):
        physical, why = peer_is_physical(peer)
        assert physical is True, peer
        assert why
        assert agent_motion_allowed("task", peer=peer, env={})["allowed"] is False, peer


def test_the_tool_actually_consults_it_before_building_a_command(monkeypatch):
    """The gate is worthless if the wiring skips it — this pins the call site."""
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
