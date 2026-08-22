"""The proxy tools ride ONE motion gate and the agent follows the mesh.

Wiring rules pinned here: agent_hitl.motion_intent accepts derived per-proxy
gate rows and bound targets (a proxy IS its peer — the model cannot write or
omit its way around the binding); real-arm execute/start gate, sims and
stop/status never do; agent_bridge builds proxies from the live snapshot,
rebuilds when the fleet changes, and agent_status stops lying before the
first build.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard import agent_hitl as hitl
from strands_robots.dashboard import peer_tools as pt

REAL = {
    "presence": {"kind": "robot", "hw": "so101 on /dev/cu.usbmodemX"},
    "state": {"joints": {f"j{i}": 0.0 for i in range(6)}},
}
SIM = {"presence": {"kind": "robot", "robot_type": "mujoco"}, "state": {}}

PEERS = {"so101-real-689": REAL, "twin-1": SIM}
PROXY_MOTION = {"so101_real_689": frozenset({"execute", "start"})}
PROXY_TARGETS = {"so101_real_689": "so101-real-689", "twin_1": "twin-1"}
NO_GRANT = {"STRANDS_DASH_AGENT_PHYSICAL_MOTION": ""}


class TestMotionIntentWithProxies:
    def test_real_proxy_execute_gates_with_the_bound_target(self):
        reason = hitl.motion_intent(
            "so101_real_689",
            {"action": "execute", "instruction": "wave"},
            PEERS,
            NO_GRANT,
            extra_actions=PROXY_MOTION,
            bound_targets=PROXY_TARGETS,
        )
        assert reason is not None
        assert reason["target"] == "so101-real-689"
        assert reason["tool"] == "so101_real_689"

    def test_binding_beats_a_model_written_target(self):
        # the model cannot aim a bound proxy at a different peer to dodge the
        # gate... but a written target field is not part of the proxy's spec,
        # so if one appears the EXPLICIT field wins only when present — the
        # proxy never forwards it to the wire either way (map_invocation
        # ignores unknown fields for status/stop and refuses unknown actions).
        reason = hitl.motion_intent(
            "so101_real_689",
            {"action": "execute"},
            PEERS,
            NO_GRANT,
            extra_actions=PROXY_MOTION,
            bound_targets=PROXY_TARGETS,
        )
        assert reason is not None and reason["target"] == "so101-real-689"

    def test_sim_proxy_never_asks(self):
        # twin_1 is not in the derived motion table at all; even if it were,
        # peer_is_physical says sim.
        reason = hitl.motion_intent(
            "twin_1",
            {"action": "add_object", "name": "red_cube"},
            PEERS,
            NO_GRANT,
            extra_actions=PROXY_MOTION,
            bound_targets=PROXY_TARGETS,
        )
        assert reason is None

    def test_proxy_status_and_stop_never_ask(self):
        for verb in ("status", "stop"):
            reason = hitl.motion_intent(
                "so101_real_689",
                {"action": verb},
                PEERS,
                NO_GRANT,
                extra_actions=PROXY_MOTION,
                bound_targets=PROXY_TARGETS,
            )
            assert reason is None, verb

    def test_without_proxy_tables_behavior_is_unchanged(self):
        # the pre-proxy contract: unknown tool name = never gated.
        assert hitl.motion_intent("so101_real_689", {"action": "execute"}, PEERS, NO_GRANT) is None

    def test_fleet_task_still_gates_exactly_as_before(self):
        reason = hitl.motion_intent(
            "fleet",
            {"action": "task", "target": "so101-real-689", "instruction": "wave"},
            PEERS,
            NO_GRANT,
            extra_actions=PROXY_MOTION,
            bound_targets=PROXY_TARGETS,
        )
        assert reason is not None and reason["target"] == "so101-real-689"

    def test_robot_mesh_stays_absent_no_double_gate(self):
        # robot_mesh raises its OWN SDK interrupt; it must not gain a row here.
        assert "robot_mesh" not in hitl.MOTION_ACTIONS
        assert hitl.motion_intent(
            "robot_mesh", {"action": "tell", "target": "so101-real-689"}, PEERS, NO_GRANT,
            extra_actions=PROXY_MOTION, bound_targets=PROXY_TARGETS,
        ) is None


class TestHookCarriesTheTables:
    def test_hook_accepts_and_stores_proxy_tables(self):
        hook = hitl.MotionInterruptHook(lambda: PEERS, PROXY_MOTION, PROXY_TARGETS)
        assert hook._proxy_motion == dict(PROXY_MOTION)
        assert hook._proxy_targets == dict(PROXY_TARGETS)

    def test_hook_without_tables_still_constructs(self):
        hook = hitl.MotionInterruptHook(lambda: PEERS)
        assert hook._proxy_motion == {} and hook._proxy_targets == {}


class TestBridgeWiring:
    def test_peer_proxy_tools_uses_the_bridge_snapshot(self, monkeypatch):
        from strands_robots.dashboard import agent_bridge as ab

        class FakeBridge:
            def snapshot(self):
                return {"peers": PEERS}

            def send_cmd(self, *a, **k):
                return {"status": "success", "content": []}

        monkeypatch.setattr(ab, "_bridge", FakeBridge())
        tools = ab._peer_proxy_tools()
        assert sorted(t.tool_name for t in tools) == ["so101_real_689", "twin_1"]

    def test_no_bridge_means_no_proxies_not_an_error(self, monkeypatch):
        from strands_robots.dashboard import agent_bridge as ab

        monkeypatch.setattr(ab, "_bridge", None)
        assert ab._peer_proxy_tools() == []

    def test_unbuilt_agent_status_names_the_expected_robots(self, monkeypatch):
        from strands_robots.dashboard import agent_bridge as ab

        class FakeBridge:
            def snapshot(self):
                return {"peers": PEERS}

        monkeypatch.setattr(ab, "_bridge", FakeBridge())
        monkeypatch.setattr(ab, "_agent", None)
        status = ab.agent_status()
        assert status["built"] is False
        for name in ("fleet", "robot_mesh", "so101_real_689", "twin_1"):
            assert name in status["tools"], name

    def test_fleet_signature_changes_on_join_and_leave(self):
        sig_before = pt.fleet_signature(PEERS)
        grown = dict(PEERS)
        grown["new-arm"] = REAL
        assert pt.fleet_signature(grown) != sig_before
        shrunk = {"twin-1": SIM}
        assert pt.fleet_signature(shrunk) != sig_before
        assert pt.fleet_signature(dict(PEERS)) == sig_before

    def test_gateway_churn_does_not_change_the_signature(self):
        with_gw = dict(PEERS)
        with_gw["gateway-x"] = {"presence": {"kind": "gateway"}}
        assert pt.fleet_signature(with_gw) == pt.fleet_signature(PEERS)

    def test_prompt_teaches_the_native_tools(self):
        from strands_robots.dashboard import agent_bridge as ab

        assert "native tool" in ab.DEFAULT_SYSTEM_PROMPT
        assert "so101_real_689" in ab.DEFAULT_SYSTEM_PROMPT
