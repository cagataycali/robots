# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sim claim this dashboard did not launch is metal for every verb that moves, on both sides.

A wire ``robot_type: "sim"`` presence classifies the peer as a sim, so its
proxy tool offers the sim verbs (``reset``, ``step``, ...). The motion gate
already answers metal for that peer (the claim cannot be checked), but the
proxy's interrupt row was only ``execute`` / ``start``: the agent's ``reset``
never reached the gate and drove the peer's joints home with no one asked.
The browser trusted the claim too: ``runRisk`` read ``robot_type`` before
``hw`` and never looked at the bridge's ``sim_corroborated`` mark, so the card
skipped its confirm sheet.

Now one set, ``agent_motion.PHYSICAL_MOTION_ACTIONS``, is every proxy row and
the HTTP gate's core, the mesh's own wire gate gates nothing outside it, and
``runRisk`` reads hardware first and believes a wire sim claim only when the
server corroborated it.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import agent_hitl, agent_motion, peer_tools
from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.mesh.core import WIRE_MOTION_ACTIONS
from tests._dashboard_frontend import requires_node, run_frontend

UNLAUNCHED_SIM: dict[str, Any] = {
    "presence": {"robot_type": "sim", "hostname": "h", "topics": ["health"]},
    "state": {"joints": {"1": {"position": 0.0}}},
    "presence_source": "wire",
    "sim_corroborated": False,
}
LAUNCHED_SIM: dict[str, Any] = {**UNLAUNCHED_SIM, "sim_corroborated": True}


def _intent(peer: dict[str, Any], action: str) -> dict[str, Any] | None:
    peers = {"sim-9": peer}
    proxies = peer_tools.build_peer_tools(peers, lambda _p, _c: {"status": "success", "content": []})
    (tool,) = proxies
    assert tool.peer_kind == peer_tools.KIND_SIM, "the sim surface stays"
    return agent_hitl.motion_intent(
        tool.tool_name,
        {"action": action},
        peers,
        env={},
        extra_actions=peer_tools.motion_actions_for(proxies, peers),
        bound_targets={tool.tool_name: tool.peer_id},
    )


@pytest.mark.parametrize("action", sorted(agent_motion.PHYSICAL_MOTION_ACTIONS - {"teleop_receive"}))
def test_every_moving_sim_verb_on_an_unlaunched_sim_asks(action: str) -> None:
    intent = _intent(UNLAUNCHED_SIM, action)
    assert intent is not None and intent["action"] == action
    assert "did not launch it" in intent["why_physical"]


@pytest.mark.parametrize("action", ["reset", "step", "execute", "status", "stop"])
def test_a_sim_this_dashboard_launched_asks_nothing(action: str) -> None:
    assert _intent(LAUNCHED_SIM, action) is None


def test_the_gate_the_proxies_and_the_wire_read_one_set() -> None:
    peers = {"sim-9": UNLAUNCHED_SIM, "arm-1": {"presence": {"robot_type": "robot", "hw": "so101"}}}
    proxies = peer_tools.build_peer_tools(peers, lambda _p, _c: {})
    table = peer_tools.motion_actions_for(proxies, peers)
    assert set(table.values()) == {agent_motion.PHYSICAL_MOTION_ACTIONS}
    assert agent_motion.GATED_ACTIONS == agent_motion.PHYSICAL_MOTION_ACTIONS | {"task"}
    # The robot host's own gate may lag (it gates fewer verbs), never lead.
    assert WIRE_MOTION_ACTIONS <= agent_motion.PHYSICAL_MOTION_ACTIONS
    assert agent_motion.PHYSICAL_MOTION_ACTIONS - WIRE_MOTION_ACTIONS <= {"reset", "step"}


def _sample(peer_id: str, body: dict[str, Any]) -> Any:
    sample = mock.MagicMock(spec=["payload", "key_expr"])
    sample.payload.to_bytes.return_value = json.dumps(body).encode()
    sample.key_expr = f"strands/{peer_id}/presence"
    return sample


@pytest.mark.parametrize(("mode", "corroborated"), [("sim", True), ("real", False), (None, False)])
def test_the_presence_event_carries_its_provenance(mode: str | None, corroborated: bool) -> None:
    bridge = MeshBridge(peer_id="dash")
    bridge._running = True
    if mode is not None:
        bridge.managed_children = lambda: [{"peer_id": "sim-9", "mode": mode, "alive": True}]
    with mock.patch.object(bridge, "_emit") as emit:
        bridge._on_presence(_sample("sim-9", {"robot_id": "sim-9", "robot_type": "sim", "timestamp": time.time()}))
    (event,) = [c.args[0] for c in emit.call_args_list if c.args[0].get("type") == "presence"]
    assert event["presence_source"] == "wire"
    assert event["sim_corroborated"] is corroborated is bridge.peers["sim-9"]["sim_corroborated"]


@requires_node
def test_the_browser_believes_a_sim_claim_only_when_the_server_corroborated_it() -> None:
    got = run_frontend(
        """
const { runRisk } = await import('./runRisk.ts')
const { mergeMeshEvent } = await import('./meshPeers.ts')
const sim = { robot_id: 's', robot_type: 'sim' }
const peers = mergeMeshEvent({}, { type: 'presence', peer_id: 's', data: sim, presence_source: 'wire', sim_corroborated: false }, 1)
const launched = mergeMeshEvent({}, { type: 'presence', peer_id: 's', data: sim, presence_source: 'wire', sim_corroborated: true }, 1)
out({
  unlaunched: runRisk(sim, { presence_source: 'wire', sim_corroborated: false }),
  missingMark: runRisk(sim, { presence_source: 'wire' }),
  merged: runRisk(peers.s.presence, peers.s).physical,
  launched: runRisk(sim, { presence_source: 'wire', sim_corroborated: true }).physical,
  launchedMerged: runRisk(launched.s.presence, launched.s).physical,
  hwFirst: runRisk({ robot_id: 'a', robot_type: 'sim', hw: 'so101 @ /dev/ttyACM0' }, { presence_source: 'wire', sim_corroborated: true }),
  simHwUnlaunched: runRisk({ robot_id: 'm', robot_type: 'robot', hw: 'mujoco' }, { presence_source: 'wire' }).physical,
})
"""
    )
    assert got["unlaunched"]["physical"] is True and "cannot be checked" in got["unlaunched"]["reason"]
    assert got["missingMark"]["physical"] is True
    assert got["merged"] is True
    assert got["launched"] is False and got["launchedMerged"] is False
    assert got["hwFirst"]["physical"] is True and got["hwFirst"]["device"] == "so101 @ /dev/ttyACM0"
    assert got["simHwUnlaunched"] is True
