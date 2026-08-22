"""peer_tools: every fleet peer becomes a native AgentTool — the pure rules.

The dashboard cannot hold Robot('so101') in-process (the child process owns the
bus), so each peer gets a PROXY tool whose spec mirrors what the peer is and
whose invocation maps onto the validated mesh command family. These tests pin
the pure layer: classification, naming, specs, the invocation mapping, and the
derived motion-gate table.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from strands_robots.dashboard import peer_tools as pt

# ── fixture peers, shaped like _peers_snapshot()'s entries ───────────────────

REAL_ARM = {
    "role": "follower",
    "presence": {"kind": "robot", "hw": "so101 on /dev/cu.usbmodem5AB01584281"},
    "state": {"joints": {f"j{i}": 0.0 for i in range(6)}},
}
REAL_ARM_BY_JOINTS = {  # no hw string, no role — joints alone prove metal-or-unknown
    "presence": {"kind": "robot"},
    "state": {"joints": {f"j{i}": 0.0 for i in range(6)}},
}
SIM_PEER = {"presence": {"kind": "robot", "robot_type": "mujoco"}, "state": {"joints": {}}}
SIM_BY_MODE = {"presence": {"kind": "robot", "mode": "sim"}, "state": {}}
SIM_CHILD = {"presence": {"kind": "robot", "parent": "twin-1"}, "state": {"joints": {"j0": 0}}}
GATEWAY = {"presence": {"kind": "gateway"}, "state": {}}
HOST = {"presence": {"kind": "robot"}, "state": {"joints": {}}}

# ── the REAL wire payloads, transcribed from a live mesh (BUGS.md Q180) ──────
# Nothing on the mesh publishes a "kind" field: mesh/core.py:1008 builds
# presence as robot_id/robot_type/hostname/timestamp and appends topics. The
# fixtures above are synthetic, which is exactly how the gateway skip came to
# test a field that never arrives — these pin the payload as measured.
WIRE_GATEWAY = {
    "presence": {
        "robot_id": "gateway-cagatays-Mac-mini-1f3a",
        "robot_type": "gateway",
        "hostname": "cagatays-Mac-mini.local",
        "timestamp": 1787412345.6,
        "topics": ["health"],
    },
    "state": {},
    "cameras": {},
}
WIRE_REAL_ARM = {
    "presence": {
        "robot_id": "so101-real-689",
        "robot_type": "robot",
        "hostname": "cagatays-Mac-mini.local",
        "timestamp": 1787412345.6,
        "connected": True,
        "hw": "so101 on /dev/cu.usbmodem5AB01584281",
        "topics": ["health"],
    },
    "state": {"joints": {f"j{i}": 0.0 for i in range(6)}},
    "cameras": {"top": {}, "wrist": {}},
}
WIRE_SIM_CHILD = {
    "presence": {
        "robot_id": "so101-real-689-twin__so101",
        "robot_type": "sim",
        "hostname": "cagatays-Mac-mini.local",
        "timestamp": 1787412345.6,
        "topics": ["health"],
    },
    "state": {"joints": {f"j{i}": 0.0 for i in range(6)}},
    "cameras": {"front": {}, "wrist": {}},
}


class TestClassification:
    def test_real_arm_by_hw(self):
        assert pt.classify_peer("so101-real-689", REAL_ARM) == pt.KIND_REAL

    def test_real_arm_by_joints(self):
        assert pt.classify_peer("so101-leader", REAL_ARM_BY_JOINTS) == pt.KIND_REAL

    def test_sim_by_robot_type(self):
        assert pt.classify_peer("twin-1", SIM_PEER) == pt.KIND_SIM

    def test_sim_by_mode(self):
        assert pt.classify_peer("twin-2", SIM_BY_MODE) == pt.KIND_SIM

    def test_sim_child_peer_is_sim_even_with_joints(self):
        # a __<robot> child of a sim world must never classify as REAL from
        # its joints: its sim_call delegates to the parent Simulation.
        assert pt.classify_peer("twin-1__so101", SIM_CHILD) == pt.KIND_SIM

    def test_gateway_gets_no_tool(self):
        assert pt.classify_peer("gateway-mac-1", GATEWAY) == pt.KIND_SKIP

    # ── Q180: the wire's own payload, not a field we invented ────────────────
    def test_wire_gateway_gets_no_tool(self):
        # THE REGRESSION THIS PINS: the skip tested presence["kind"], which no
        # peer publishes, so a throwaway gateway session became an AgentTool
        # described to the model as a "Robot peer".
        assert pt.classify_peer("gateway-cagatays-Mac-mini-1f3a", WIRE_GATEWAY) == pt.KIND_SKIP

    def test_wire_gateway_carries_no_kind_field_at_all(self):
        # If a future presence payload starts carrying "kind", this test tells
        # the next reader that the ORIGINAL skip was reading a ghost.
        assert "kind" not in WIRE_GATEWAY["presence"]

    def test_wire_real_arm_is_still_real(self):
        assert pt.classify_peer("so101-real-689", WIRE_REAL_ARM) == pt.KIND_REAL

    def test_wire_sim_child_is_still_sim(self):
        assert pt.classify_peer("so101-real-689-twin__so101", WIRE_SIM_CHILD) == pt.KIND_SIM

    def test_gateway_is_absent_from_the_fleet_signature(self):
        # The churn half of Q180: a probe's birth/death rebuilt the agent
        # because its (peer_id, kind) pair entered the signature.
        peers = {"so101-real-689": WIRE_REAL_ARM, "gateway-cagatays-Mac-mini-1f3a": WIRE_GATEWAY}
        assert pt.fleet_signature(peers) == pt.fleet_signature({"so101-real-689": WIRE_REAL_ARM})

    def test_gateway_builds_no_tool(self):
        tools = pt.build_peer_tools(
            {"so101-real-689": WIRE_REAL_ARM, "gateway-cagatays-Mac-mini-1f3a": WIRE_GATEWAY},
            send_cmd=lambda *a, **k: {"status": "ok"},
        )
        assert [t.tool_name for t in tools] == ["so101_real_689"]

    def test_health_only_topics_without_a_type_is_a_coordinator(self):
        # Belt: a coordinator whose presence lost its robot_type still cannot
        # mint a motion tool — no joints, no cameras, health-only topics.
        peer = {"presence": {"topics": ["health"], "timestamp": 1.0}, "state": {}}
        assert pt.classify_peer("some-coordinator", peer) == pt.KIND_SKIP

    def test_health_only_topics_with_joints_is_not_skipped(self):
        # ...but a real robot advertising only health while publishing joints
        # is a robot: the belts must not swallow the fleet.
        peer = {"presence": {"topics": ["health"]}, "state": {"joints": {"j0": 0.0}}}
        assert pt.classify_peer("so101-quiet", peer) == pt.KIND_REAL

    def test_jointless_robot_is_host(self):
        assert pt.classify_peer("mystery-bot", HOST) == pt.KIND_HOST

    def test_unknown_peer_gets_no_tool(self):
        # the GATE fails closed (unknown = metal); the FACTORY fails quiet —
        # a tool for a peer we cannot describe would advertise an invented spec.
        assert pt.classify_peer("ghost", None) == pt.KIND_SKIP
        assert pt.classify_peer("", REAL_ARM) == pt.KIND_SKIP


class TestNaming:
    def test_dashes_become_underscores(self):
        assert pt.sanitize_tool_name("so101-real-689") == "so101_real_689"

    def test_double_underscore_child_survives(self):
        assert pt.sanitize_tool_name("so101-real-689-twin__so101") == "so101_real_689_twin__so101"

    def test_leading_digit_prefixed(self):
        assert pt.sanitize_tool_name("3dof-arm") == "p_3dof_arm"

    def test_python_keyword_escaped(self):
        assert pt.sanitize_tool_name("import") == "import_"

    def test_collision_gets_suffix(self):
        first = pt.sanitize_tool_name("a-b")
        second = pt.sanitize_tool_name("a.b", {first})
        assert first == "a_b" and second == "a_b_2" and first != second

    def test_empty_id_still_names(self):
        assert pt.sanitize_tool_name("") == "peer"


class TestSpecs:
    def test_real_spec_mirrors_hardware_robot_surface(self):
        spec = pt.peer_tool_spec("so101-real-689", pt.KIND_REAL, "so101_real_689")
        schema = spec["inputSchema"]["json"]
        assert schema["properties"]["action"]["enum"] == ["execute", "start", "status", "stop"]
        assert "instruction" in schema["properties"] and "policy_port" in schema["properties"]
        assert spec["name"] == "so101_real_689"
        assert "so101-real-689" in spec["description"]

    def test_real_spec_default_action_is_the_safe_one(self):
        spec = pt.peer_tool_spec("x", pt.KIND_REAL, "x")
        assert spec["inputSchema"]["json"]["properties"]["action"]["default"] == "status"

    def test_sim_spec_carries_the_published_actions(self):
        spec = pt.peer_tool_spec("twin-1", pt.KIND_SIM, "twin_1")
        enum = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        for verb in ("add_object", "add_camera", "list_objects", "register_urdf", "raycast"):
            assert verb in enum

    def test_sim_spec_never_advertises_what_the_wire_refuses(self):
        spec = pt.peer_tool_spec("twin-1", pt.KIND_SIM, "twin_1")
        enum = set(spec["inputSchema"]["json"]["properties"]["action"]["enum"])
        assert not (enum & pt.SIM_CALL_BLOCKED)

    def test_sim_blocklist_matches_mesh_security(self):
        from strands_robots.mesh import security as sec

        assert pt.SIM_CALL_BLOCKED == sec.SIM_CALL_BLOCKED_ACTIONS

    def test_sim_spec_source_is_the_shipped_tool_spec(self):
        raw = json.loads(
            (
                Path(pt.__file__).resolve().parents[1] / "simulation" / "mujoco" / "tool_spec.json"
            ).read_text()
        )
        spec = pt.peer_tool_spec("twin-1", pt.KIND_SIM, "twin_1")
        enum = set(spec["inputSchema"]["json"]["properties"]["action"]["enum"])
        assert enum == set(raw["properties"]["action"]["enum"]) - pt.SIM_CALL_BLOCKED

    def test_host_spec_offers_only_status_and_stop(self):
        spec = pt.peer_tool_spec("mystery-bot", pt.KIND_HOST, "mystery_bot")
        assert spec["inputSchema"]["json"]["properties"]["action"]["enum"] == ["status", "stop"]

    def test_skip_kind_has_no_spec(self):
        assert pt.peer_tool_spec("gw", pt.KIND_SKIP, "gw") is None


class TestInvocationMapping:
    def test_sim_action_maps_to_sim_call(self):
        cmd, err = pt.map_invocation(
            "twin-1",
            pt.KIND_SIM,
            {"action": "add_object", "name": "red_cube", "shape": "box", "color": [1, 0, 0, 1]},
        )
        assert err is None
        assert cmd["action"] == "sim_call" and cmd["sim_action"] == "add_object"
        assert cmd["sim_params"]["name"] == "red_cube"
        assert "action" not in cmd["sim_params"]

    def test_sim_robot_name_lifts_to_top_level(self):
        cmd, err = pt.map_invocation(
            "twin-1", pt.KIND_SIM, {"action": "get_robot_state", "robot_name": "so101"}
        )
        assert err is None
        assert cmd["robot_name"] == "so101" and "robot_name" not in cmd["sim_params"]

    def test_sim_rollout_refused_with_pointer(self):
        cmd, err = pt.map_invocation("twin-1", pt.KIND_SIM, {"action": "run_policy"})
        assert cmd is None and "execute" in err

    def test_sim_none_params_dropped(self):
        cmd, _ = pt.map_invocation("twin-1", pt.KIND_SIM, {"action": "list_objects", "seed": None})
        assert cmd["sim_params"] == {}

    def test_real_execute_maps_with_fields(self):
        cmd, err = pt.map_invocation(
            "so101-real-689",
            pt.KIND_REAL,
            {"action": "execute", "instruction": "wave", "policy_port": 5555, "duration": 10},
        )
        assert err is None
        assert cmd == {
            "action": "execute",
            "instruction": "wave",
            "policy_port": 5555,
            "duration": 10,
        }

    def test_real_status_and_stop_carry_nothing_extra(self):
        for verb in ("status", "stop"):
            cmd, err = pt.map_invocation("x", pt.KIND_REAL, {"action": verb, "instruction": "hm"})
            assert err is None and cmd == {"action": verb}

    def test_real_unknown_action_refused(self):
        cmd, err = pt.map_invocation("x", pt.KIND_REAL, {"action": "add_object"})
        assert cmd is None and "add_object" in err

    def test_host_cannot_execute(self):
        cmd, err = pt.map_invocation("x", pt.KIND_HOST, {"action": "execute"})
        assert cmd is None and err

    def test_missing_action_refused(self):
        cmd, err = pt.map_invocation("x", pt.KIND_REAL, {})
        assert cmd is None and "action" in err


FLEET = {
    "so101-real-689": REAL_ARM,
    "so101-leader": REAL_ARM_BY_JOINTS,
    "twin-1": SIM_PEER,
    "twin-1__so101": SIM_CHILD,
    "gateway-mac-1": GATEWAY,
    "dashboard-mac-2d0a": {"presence": {"kind": "dashboard"}},
}


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


async def _collect(agen):
    return [ev async for ev in agen]


class TestFactory:
    def test_builds_one_tool_per_tool_worthy_peer(self):
        tools = pt.build_peer_tools(FLEET, send_cmd=lambda *a, **k: {})
        names = sorted(t.tool_name for t in tools)
        assert names == ["so101_leader", "so101_real_689", "twin_1", "twin_1__so101"]

    def test_tool_specs_declare_their_names(self):
        tools = pt.build_peer_tools(FLEET, send_cmd=lambda *a, **k: {})
        for t in tools:
            assert t.tool_spec["name"] == t.tool_name
            assert t.tool_type == "robot"

    def test_invocation_reaches_send_cmd_with_the_mapped_command(self):
        sent: list = []

        def fake_send(peer, cmd, timeout=30.0, source=""):
            sent.append((peer, cmd, source))
            return {"status": "success", "content": [{"text": "ok"}]}

        tools = pt.build_peer_tools(FLEET, send_cmd=fake_send)
        twin = next(t for t in tools if t.tool_name == "twin_1")
        events = _run(
            _collect(
                twin.stream(
                    {"toolUseId": "t1", "input": {"action": "list_objects"}},
                    {},
                )
            )
        )
        assert sent == [("twin-1", {"action": "sim_call", "sim_action": "list_objects", "sim_params": {}}, "agent")]
        result = events[-1].tool_result if hasattr(events[-1], "tool_result") else events[-1]
        assert "ok" in json.dumps(result if isinstance(result, dict) else result.__dict__, default=str)

    def test_mapping_error_never_reaches_the_wire(self):
        sent: list = []
        tools = pt.build_peer_tools(FLEET, send_cmd=lambda *a, **k: sent.append(a) or {})
        twin = next(t for t in tools if t.tool_name == "twin_1")
        events = _run(
            _collect(twin.stream({"toolUseId": "t2", "input": {"action": "run_policy"}}, {}))
        )
        assert sent == []
        assert events  # the refusal came back as a result, not an exception

    def test_wire_exception_becomes_an_error_result(self):
        def boom(*a, **k):
            raise RuntimeError("zenoh down")

        tools = pt.build_peer_tools(FLEET, send_cmd=boom)
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        events = _run(_collect(arm.stream({"toolUseId": "t3", "input": {"action": "status"}}, {})))
        assert events  # refusal as result; no raise out of stream

    def test_proxies_expose_their_binding(self):
        tools = pt.build_peer_tools(FLEET, send_cmd=lambda *a, **k: {})
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        assert arm.peer_id == "so101-real-689" and arm.peer_kind == pt.KIND_REAL


class TestMotionGateTable:
    def test_only_real_arms_and_only_motion_verbs(self):
        tools = pt.build_peer_tools(FLEET, send_cmd=lambda *a, **k: {})
        table = pt.motion_actions_for(tools)
        assert table == {
            "so101_leader": frozenset({"execute", "start"}),
            "so101_real_689": frozenset({"execute", "start"}),
        }

    def test_stop_and_status_never_gated(self):
        tools = pt.build_peer_tools(FLEET, send_cmd=lambda *a, **k: {})
        for actions in pt.motion_actions_for(tools).values():
            assert "stop" not in actions and "status" not in actions
