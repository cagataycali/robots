"""The tool factory and the motion gate read one precedence: metal wins (f030).

``peer_tools.classify_peer`` decided what tool a peer gets, and it read a
presence record ``robot_type`` first: a record saying ``sim`` got the sim
tool. ``agent_motion.peer_is_physical`` decided whether a motion needs a
human yes, and it read ``hw`` first: a record naming hardware is metal. The
two disagreed on one record, ``{"robot_type": "sim", "hw": "so101 @ ..."}``,
and the disagreement was not cosmetic. Only ``KIND_REAL`` proxies entered the
interrupt hook's table (``motion_actions_for``), so a peer the gate itself
called metal got a sim tool whose ``execute`` and ``start`` never reached the
gate at all: the rollout ran on the wire with no interrupt.

The same hole opened for a plain wire ``robot_type: "sim"`` claim this
dashboard did not launch: ``peer_is_physical`` answers metal for it (the claim
cannot be checked), but the sim tool it minted was outside the table.

No in-tree publisher emits the ambiguous record (``Mesh._build_presence``
writes ``robot_type`` from ``peer_type`` and ``hw`` from the hardware robot's
name; sims pass ``peer_type="sim"`` and have no ``hw``). Producing it takes a
hardware robot started with ``init_mesh(..., peer_type="sim")`` or a peer
writing its own presence body, both of which the gate was built to survive.

Now hardware evidence is read first by both, through one helper, and every
proxy whose peer the gate calls metal is in the interrupt table.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dashboard import agent_hitl, agent_motion, peer_tools

METAL_CLAIMING_SIM: dict[str, Any] = {
    "presence": {"robot_type": "sim", "hw": "so101 @ /dev/ttyACM0", "topics": ["health", "state"]},
    "presence_source": "wire",
    "sim_corroborated": False,
}
WIRE_SIM_CLAIM: dict[str, Any] = {
    "presence": {"robot_type": "sim", "hostname": "h", "topics": ["health"]},
    "state": {"joints": {"1": {"position": 0.0}}},
    "presence_source": "wire",
    "sim_corroborated": False,
}
LAUNCHED_SIM: dict[str, Any] = {**WIRE_SIM_CLAIM, "sim_corroborated": True}
REAL_ARM: dict[str, Any] = {
    "presence": {"robot_type": "robot", "hw": "so101 @ /dev/tty", "topics": ["health", "state"]},
    "state": {"joints": {"1": {}}},
    "presence_source": "wire",
}


def _send(peer_id: str, cmd: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "content": []}


def _intent(peers: dict[str, Any], peer_id: str, action: str = "execute") -> dict[str, Any] | None:
    proxies = peer_tools.build_peer_tools(peers, _send)
    by_peer = {t.peer_id: t for t in proxies}
    assert peer_id in by_peer, f"{peer_id} got no tool"
    tool = by_peer[peer_id]
    return agent_hitl.motion_intent(
        tool.tool_name,
        {"action": action, "instruction": "pick up the red cube"},
        peers,
        env={},
        extra_actions=peer_tools.motion_actions_for(proxies, peers),
        bound_targets={t.tool_name: t.peer_id for t in proxies},
    )


class TestOnePrecedence:
    def test_hardware_evidence_beats_a_sim_robot_type(self) -> None:
        # The disputed record: the gate says metal, so the factory must not mint a sim tool.
        physical, why = agent_motion.peer_is_physical(METAL_CLAIMING_SIM)
        assert physical and "so101" in why
        assert peer_tools.classify_peer("arm-x", METAL_CLAIMING_SIM) == peer_tools.KIND_REAL

    def test_hardware_evidence_beats_the_sim_child_rule_too(self) -> None:
        child = {**METAL_CLAIMING_SIM, "parent": "world-1"}
        assert peer_tools.classify_peer("world-1__so101", child) == peer_tools.KIND_REAL

    def test_both_read_the_same_helper(self) -> None:
        for presence in ({"hw": "so101"}, {"hw": "  "}, {"hw": 7}, {}, {"hw": ""}):
            evidence = agent_motion.hardware_evidence(presence)
            peer = {"presence": {"robot_type": "sim", **presence}}
            assert (evidence is not None) == agent_motion.peer_is_physical(peer)[0]
            assert (evidence is not None) == (peer_tools.classify_peer("p", peer) == peer_tools.KIND_REAL)


class TestTheGateTable:
    def test_a_metal_peer_with_a_sim_claim_is_interrupted_on_execute(self) -> None:
        peers = {"arm-x": METAL_CLAIMING_SIM}
        intent = _intent(peers, "arm-x")
        assert intent is not None and intent["target"] == "arm-x" and "so101" in intent["why_physical"]

    def test_an_unlaunched_wire_sim_claim_is_interrupted_on_execute(self) -> None:
        # The gate already answers metal for this peer; the factory kept it out of the table.
        peers = {"sim-9": WIRE_SIM_CLAIM}
        assert peer_tools.classify_peer("sim-9", WIRE_SIM_CLAIM) == peer_tools.KIND_SIM, "the sim surface stays"
        intent = _intent(peers, "sim-9")
        assert intent is not None and "did not launch it" in intent["why_physical"]

    def test_a_sim_this_dashboard_launched_is_not_interrupted(self) -> None:
        peers = {"sim-1": LAUNCHED_SIM}
        assert peer_tools.classify_peer("sim-1", LAUNCHED_SIM) == peer_tools.KIND_SIM
        assert _intent(peers, "sim-1") is None

    def test_a_real_arm_is_interrupted_as_before(self) -> None:
        assert _intent({"arm-1": REAL_ARM}, "arm-1") is not None

    def test_reads_are_never_gated(self) -> None:
        for action in ("status", "state", "stop"):
            assert _intent({"sim-9": WIRE_SIM_CLAIM}, "sim-9", action) is None

    @pytest.mark.parametrize(
        "peer",
        [METAL_CLAIMING_SIM, WIRE_SIM_CLAIM, LAUNCHED_SIM, REAL_ARM, {"presence": {"robot_type": "sim"}}],
        ids=["metal-claiming-sim", "wire-sim-claim", "launched-sim", "real-arm", "plain-sim"],
    )
    def test_every_proxy_on_a_peer_the_gate_calls_metal_is_in_the_table(self, peer: dict[str, Any]) -> None:
        peers = {"p": peer}
        proxies = peer_tools.build_peer_tools(peers, _send)
        table = peer_tools.motion_actions_for(proxies, peers)
        physical, _ = agent_motion.peer_is_physical(peer)
        for tool in proxies:
            assert (tool.tool_name in table) == physical, (tool.peer_kind, physical)
            if physical:
                assert table[tool.tool_name] >= {"execute", "start"}
