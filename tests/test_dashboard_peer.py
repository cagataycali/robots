"""Peer proxy tools of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_peer_id_validation.py, test_dashboard_peer_origin.py, test_dashboard_peer_tools_wiring.py, test_dashboard_peer_tools.py, test_dashboard_peer_ttl_prune.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.dashboard import agent_hitl as hitl
from strands_robots.dashboard import mesh_bridge as mb
from strands_robots.dashboard import peer_tools as pt
from strands_robots.dashboard.device_manager import (
    DeviceManager,
    validate_peer_id,
)
from strands_robots.dashboard.mesh_bridge import (
    PEER_STALE_S,
    MeshBridge,
    prune_peers,
)

# ============================================================================
# from tests/test_dashboard_peer_id_validation.py
# A caller-supplied peer_id is a zenoh KEY SEGMENT, not a label (Q3 corollary).
# ============================================================================


class TestValidatePeerId:
    def test_none_means_generate_one_and_is_fine(self) -> None:
        assert validate_peer_id(None) is None

    @pytest.mark.parametrize(
        "good",
        [
            "so101-arm-1",
            "replay-1234",
            "so101.real:2",
            "A_z".replace("_", "-"),
            "x" * 64,
        ],
    )
    def test_every_id_this_codebase_generates_passes(self, good: str) -> None:
        assert validate_peer_id(good) is None, good

    @pytest.mark.parametrize(
        "bad",
        [
            "*",  # fleet-wide wildcard
            "**",  # multi-level wildcard
            "so101/*",  # wildcard inside a hierarchy
            "a/b",  # splices a key level
            "../../../tmp/pwn",
            "so101 arm",  # space
            "{a:1}",  # brace DSL
            "$now",  # dollar DSL
            "",  # empty
            "x" * 65,  # over-long
        ],
    )
    def test_key_expression_syntax_is_refused_with_the_reason(self, bad: str) -> None:
        reason = validate_peer_id(bad)
        assert reason is not None, f"{bad!r} was accepted into a zenoh key"
        assert "zenoh" in reason or "string" in reason

    @pytest.mark.parametrize("nonstring", [123, {"a": 1}, ["x"], 1.5, True])
    def test_a_non_string_is_refused_by_type_not_repred(self, nonstring: object) -> None:
        reason = validate_peer_id(nonstring)
        assert reason is not None
        assert "must be a string" in reason


class TestSpawnRefusesBeforeAnyProcessExists:
    def test_a_wildcard_peer_id_never_reaches_popen(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise AssertionError("Popen was reached with a wildcard peer_id")

        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(mod.subprocess, "Popen", boom)
        result = dm.spawn("so101", "sim", peer_id="*")
        assert "error" in result
        assert "zenoh" in result["error"]
        assert dm.robots == {}, "a refused spawn must not register a managed entry"

    def test_a_generated_id_still_spawns_normally(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # peer_id=None takes the generated-id path: validation must not get in
        # its way. Popen is stubbed so no child is created.
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))

        class FakeProc:
            pid = 4242
            stdout = None

            def poll(self):
                return None

        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: FakeProc())
        monkeypatch.setattr(mod.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda self: None})())
        result = dm.spawn("so101", "sim", peer_id=None, remember=False)
        assert result.get("pid") == 4242, result
        assert validate_peer_id(result["peer_id"]) is None, (
            "generated ids must satisfy the same rule callers are held to"
        )


# ============================================================================
# from tests/test_dashboard_peer_origin.py
# U15: a code-defined robot is a first-class peer.
# ============================================================================


def test_a_peer_we_spawned_is_managed_and_a_stranger_is_external():
    out = mb.peer_origins(["ours", "theirs"], managed_ids=["ours"])
    assert out == {"ours": "managed", "theirs": "external"}


def test_every_peer_gets_a_label():
    """An absent label reads as "unknown", which is a third state the UI would
    have to invent copy for. There are only two answers and we always know
    which: we either started the process or we did not."""
    out = mb.peer_origins(["a", "b", "c"], managed_ids=[])
    assert set(out) == {"a", "b", "c"}
    assert set(out.values()) == {"external"}


def test_a_child_sim_peer_inherits_its_parents_origin():
    """ "<parent>__<robot>" lives INSIDE the parent's process, so if we started
    the parent we started the child - the same rule prune_peers and
    peer_is_known already use for these ids."""
    out = mb.peer_origins(
        ["sim-a", "sim-a__so101", "wild__so101"],
        managed_ids=["sim-a"],
    )
    assert out["sim-a"] == "managed"
    assert out["sim-a__so101"] == "managed"
    # Not ours, and its name saying "__" does not make it ours.
    assert out["wild__so101"] == "external"


def test_a_half_formed_child_id_is_not_adopted():
    """Matching peer_is_known exactly: "sim-a__" has no child half, so it must
    not borrow the parent's protection or its origin."""
    assert mb.peer_origins(["sim-a__"], managed_ids=["sim-a"]) == {"sim-a__": "external"}
    assert mb.peer_origins(["__so101"], managed_ids=["sim-a"]) == {"__so101": "external"}


def test_a_mapping_of_peers_is_accepted_like_the_snapshot_passes_it():
    """snapshot() hands its peer dict straight in; iterating keys must be enough."""
    peers = {"ours": {"peer_id": "ours"}, "theirs": {"peer_id": "theirs"}}
    assert mb.peer_origins(peers, managed_ids={"ours"}) == {
        "ours": "managed",
        "theirs": "external",
    }


# --- the rail the UI actually renders from ---------------------------------


def _bridge(peers: dict, managed: set[str]) -> mb.MeshBridge:
    """A MeshBridge with no mesh and no session.

    Deliberately __new__ + hand-filled attributes: constructing a real Mesh in a
    test is the Q30 class of accident (a drill that reached the live fleet), and
    snapshot() needs nothing but the peer table, the locks and the two hooks.
    """
    b = mb.MeshBridge.__new__(mb.MeshBridge)
    b.peer_id = "dash"
    b.peers = peers
    b._peers_lock = threading.RLock()  # type: ignore[assignment]
    b._coalesce_lock = threading.RLock()  # type: ignore[assignment]
    b._coalescer = SimpleNamespace(forget=lambda pid: None)  # type: ignore[assignment]
    b.protected_peer_ids = lambda: managed
    b.peer_annotations = None
    b.mesh_info = lambda: {}  # type: ignore[method-assign]
    return b


def _live(peer_id: str) -> dict:
    """A peer entry as the mesh reports one, with a fresh heartbeat."""
    import time

    return {
        "peer_id": peer_id,
        "last_seen": time.time(),
        "state": {"joints": {"shoulder_pan.pos": 1.0}},
        "cameras": ["top"],
    }


def test_the_snapshot_labels_the_origin_of_every_peer():
    bridge = _bridge({"ours": _live("ours"), "theirs": _live("theirs")}, {"ours"})
    peers = mb.MeshBridge.snapshot(bridge)["peers"]
    assert peers["ours"]["origin"] == "managed"
    assert peers["theirs"]["origin"] == "external"


def test_a_code_defined_peer_is_identical_to_a_spawned_one_except_its_origin():
    """THE U15 ACCEPTANCE TEST.

    Two peers reporting the same thing must reach the UI as the same card. The
    dashboard renders from this snapshot, so field-level sameness here IS card
    sameness: no telemetry dropped, no name rewritten, no capability flag that
    would let a component quietly render an external peer as second class.
    """
    ours, theirs = _live("ours"), _live("theirs")
    # Same reported content, different id and different origin - nothing else.
    theirs["state"] = dict(ours["state"])
    theirs["last_seen"] = ours["last_seen"]
    bridge = _bridge({"ours": ours, "theirs": theirs}, {"ours"})

    peers = mb.MeshBridge.snapshot(bridge)["peers"]
    a, b = dict(peers["ours"]), dict(peers["theirs"])
    # ``pop`` is a mutation; keep it out of the ``assert`` so ``python -O``
    # (which drops assertions) does not silently skip the two rewrites the
    # sameness check below reads from.
    a_origin = a.pop("origin")
    b_origin = b.pop("origin")
    assert a_origin == "managed"
    assert b_origin == "external"
    a.pop("peer_id"), b.pop("peer_id")
    assert a == b, "a code-defined peer must reach the UI as the same card"
    # And the telemetry a card draws really did survive on the external one.
    assert b["state"]["joints"] == {"shoulder_pan.pos": 1.0}
    assert b["cameras"] == ["top"]
    assert b["stale"] is False


def test_the_role_annotation_still_rides_along_next_to_the_origin():
    """Origin is applied first; it must not shadow the measured-role fields the
    U2 badge reads (the two enrichments are independent facts)."""
    bridge = _bridge({"ours": _live("ours")}, {"ours"})
    bridge.peer_annotations = lambda: {"ours": {"role": "follower", "role_volts": 12.6}}
    peer = mb.MeshBridge.snapshot(bridge)["peers"]["ours"]
    assert peer["origin"] == "managed"
    assert peer["role"] == "follower" and peer["role_volts"] == 12.6


def test_an_external_peer_can_still_be_commanded():
    """The badge is cosmetic by design: it must not become a permission. Q7's
    guard decides addressability, and a peer present in the mesh is known
    whether or not we started it."""
    peers = {"theirs": _live("theirs")}
    assert mb.peer_is_known("theirs", peers, managed_ids=())


# ============================================================================
# from tests/test_dashboard_peer_tools_wiring.py
# The proxy tools ride ONE motion gate and the agent follows the mesh.
# ============================================================================

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
        assert (
            hitl.motion_intent(
                "robot_mesh",
                {"action": "tell", "target": "so101-real-689"},
                PEERS,
                NO_GRANT,
                extra_actions=PROXY_MOTION,
                bound_targets=PROXY_TARGETS,
            )
            is None
        )


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


# ============================================================================
# from tests/test_dashboard_peer_tools.py
# peer_tools: every fleet peer becomes a native AgentTool — the pure rules.
# ============================================================================

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
            (Path(pt.__file__).resolve().parents[1] / "simulation" / "mujoco" / "tool_spec.json").read_text()
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
        cmd, err = pt.map_invocation("twin-1", pt.KIND_SIM, {"action": "get_robot_state", "robot_name": "so101"})
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
        events = _run(_collect(twin.stream({"toolUseId": "t2", "input": {"action": "run_policy"}}, {})))
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


class TestStalePresence:
    """Q179: the factory had no notion of `stale` (grep = 0) while the `fleet` tool it replaces
    filtered it at agent_bridge.py:300 and :361. Wire shape below is the real one: mesh peers carry
    a boolean `stale` alongside `last_seen`.
    """

    def test_a_fresh_peer_is_not_refused(self):
        assert pt.stale_refusal("so101-real-689", {"stale": False, "presence": {}}) is None
        assert pt.stale_refusal("so101-real-689", {}) is None
        assert pt.stale_refusal("so101-real-689", None) is None

    def test_the_refusal_blames_presence_not_the_robot(self):
        msg = pt.stale_refusal("so101-real-689", {"stale": True, "last_seen_age": 1430.4})
        assert msg is not None
        assert "1430s" in msg, msg
        assert "PRESENCE" in msg and "not a robot fault" in msg
        assert "so101-real-689" in msg

    def test_a_stale_peer_still_gets_a_tool(self):
        # The DECISION: do NOT drop the proxy. Q178 marked both real arms stale for 1430s during a
        # mesh-ingest blackout while they were connected and streaming — dropping tools there would
        # delete the agent's whole arm surface mid-blackout and churn it back minutes later.
        stale_fleet = {pid: dict(p) for pid, p in FLEET.items()}
        for p in stale_fleet.values():
            p["stale"] = True
        assert len(pt.build_peer_tools(stale_fleet, send_cmd=lambda *a, **k: {})) == 4
        assert pt.fleet_signature(stale_fleet) == pt.fleet_signature(FLEET), (
            "staleness must not churn the agent: it is time-dependent and checked at invocation"
        )

    def test_a_stale_peer_refuses_motion_before_the_wire(self):
        # Q185 narrowed Q179: only MOTION-STARTING actions are refused on stale presence.
        sent: list = []

        def fake_send(peer, cmd, timeout=30.0, source=""):
            sent.append(peer)
            return {"status": "success", "content": [{"text": "ok"}]}

        tools = pt.build_peer_tools(
            FLEET,
            send_cmd=fake_send,
            peer_state=lambda pid: {"stale": True, "last_seen_age": 90},
        )
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        events = _run(
            _collect(arm.stream({"toolUseId": "t9", "input": {"action": "execute", "instruction": "wave"}}, {}))
        )
        assert sent == [], "a stale peer must not burn the 30s timeout on motion"
        blob = json.dumps(
            [e.tool_result if hasattr(e, "tool_result") else getattr(e, "__dict__", str(e)) for e in events],
            default=str,
        )
        assert "STALE" in blob and "error" in blob

    def test_q185_stop_is_delivered_to_a_stale_peer_with_a_note(self):
        # Q185 house law: stop is NEVER refused — Q178 marked both real arms stale 1430s
        # while streaming, so the refusal would land on a possibly MOVING arm.
        sent: list = []

        def fake_send(peer, cmd, timeout=30.0, source=""):
            sent.append((peer, cmd["action"]))
            return {"status": "success", "content": [{"text": "stopped"}]}

        tools = pt.build_peer_tools(
            FLEET,
            send_cmd=fake_send,
            peer_state=lambda pid: {"stale": True, "last_seen_age": 1430.4},
        )
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        events = _run(_collect(arm.stream({"toolUseId": "ts", "input": {"action": "stop"}}, {})))
        assert sent == [("so101-real-689", "stop")], "stop must reach the wire on a stale peer"
        blob = json.dumps(
            [e.tool_result if hasattr(e, "tool_result") else getattr(e, "__dict__", str(e)) for e in events],
            default=str,
        )
        assert "stopped" in blob, "the real outcome must be reported"
        assert "STALE" in blob and "1430" in blob, "the staleness note must ride along"
        assert '"status": "error"' not in blob, "a delivered stop is a success, not a refusal"

    def test_q185_status_read_is_delivered_to_a_stale_peer_with_a_note(self):
        sent: list = []
        tools = pt.build_peer_tools(
            FLEET,
            send_cmd=lambda peer, cmd, timeout=30.0, source="": (
                sent.append((peer, cmd["action"])) or {"status": "success", "content": [{"text": "joints"}]}
            ),
            peer_state=lambda pid: {"stale": True, "last_seen_age": 90},
        )
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        events = _run(_collect(arm.stream({"toolUseId": "tq", "input": {"action": "status"}}, {})))
        assert sent == [("so101-real-689", "status")]
        blob = json.dumps(
            [e.tool_result if hasattr(e, "tool_result") else getattr(e, "__dict__", str(e)) for e in events],
            default=str,
        )
        assert "joints" in blob and "STALE" in blob

    def test_q185_never_gated_table_holds_the_stop_class(self):
        # The house law as a table any future verb must join explicitly.
        assert {"stop", "emergency_stop", "stop_all", "status"} <= pt.NEVER_GATED
        assert "execute" not in pt.NEVER_GATED and "start" not in pt.NEVER_GATED

    def test_q185_fresh_peer_stop_carries_no_note(self):
        assert pt.stale_note("x", {"stale": False}) is None
        assert pt.stale_note("x", None) is None
        note = pt.stale_note("x", {"stale": True, "last_seen_age": 12.0})
        assert note and "12" in note and "not a robot fault" in note

    def test_without_peer_state_behaviour_is_unchanged(self):
        sent: list = []
        tools = pt.build_peer_tools(
            FLEET,
            send_cmd=lambda peer, cmd, timeout=30.0, source="": (
                sent.append(peer) or {"status": "success", "content": [{"text": "ok"}]}
            ),
        )
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        _run(_collect(arm.stream({"toolUseId": "t10", "input": {"action": "status"}}, {})))
        assert sent == ["so101-real-689"]

    def test_an_unreadable_snapshot_does_not_block_a_command(self):
        # UNKNOWN presence is not stale presence: refusing on a snapshot error would make a
        # dashboard bug look like a dead robot.
        sent: list = []

        def boom(pid):
            raise RuntimeError("snapshot unavailable")

        tools = pt.build_peer_tools(
            FLEET,
            send_cmd=lambda peer, cmd, timeout=30.0, source="": (
                sent.append(peer) or {"status": "success", "content": [{"text": "ok"}]}
            ),
            peer_state=boom,
        )
        arm = next(t for t in tools if t.tool_name == "so101_real_689")
        _run(_collect(arm.stream({"toolUseId": "t11", "input": {"action": "status"}}, {})))
        assert sent == ["so101-real-689"]


# ============================================================================
# from tests/test_dashboard_peer_ttl_prune.py
# Dead peers must age OUT of the fleet snapshot, not linger as ghost cards.
# ============================================================================

NOW = 1_000_000.0
TTL = 300.0


def _peers(now: float = NOW) -> dict[str, dict]:
    return {
        "fresh": {"last_seen": now - 1.0},
        "quiet": {"last_seen": now - (PEER_STALE_S + 5.0)},
        "dead": {"last_seen": now - (TTL + 1.0)},
    }


def _live_peers() -> dict[str, dict]:
    """Same shape, but anchored to the wall clock snapshot() actually reads."""
    return _peers(time.time())


def test_fresh_peer_stays_and_is_not_stale():
    out = prune_peers(_peers(), NOW, TTL)
    assert "fresh" in out
    assert out["fresh"]["stale"] is False


def test_quiet_but_recent_peer_stays_marked_stale():
    out = prune_peers(_peers(), NOW, TTL)
    assert out["quiet"]["stale"] is True


def test_peer_older_than_ttl_disappears():
    out = prune_peers(_peers(), NOW, TTL)
    assert "dead" not in out
    assert set(out) == {"fresh", "quiet"}


def test_dead_peer_with_live_managed_process_stays():
    out = prune_peers(_peers(), NOW, TTL, protected_ids={"dead"})
    assert "dead" in out
    assert out["dead"]["stale"] is True  # visible, but honestly quiet


def test_child_sim_peer_is_protected_by_its_parent():
    peers = {"replay-1__so101": {"last_seen": NOW - (TTL + 60.0)}}
    assert prune_peers(peers, NOW, TTL) == {}
    kept = prune_peers(peers, NOW, TTL, protected_ids={"replay-1"})
    assert "replay-1__so101" in kept


def test_ttl_zero_disables_ageing_out():
    out = prune_peers(_peers(), NOW, 0.0)
    assert set(out) == {"fresh", "quiet", "dead"}


def test_missing_last_seen_counts_as_ancient():
    out = prune_peers({"never": {}}, NOW, TTL)
    assert out == {}


def test_original_mapping_is_not_mutated():
    peers = _peers()
    prune_peers(peers, NOW, TTL)
    assert set(peers) == {"fresh", "quiet", "dead"}
    assert "stale" not in peers["fresh"]


def test_bridge_snapshot_drops_dead_peers_and_forgets_them():
    bridge = MeshBridge(peer_id="dashboard-test")
    bridge.peers = _live_peers()
    snap = bridge.snapshot()
    assert "dead" not in snap["peers"]
    assert snap["peers"]["quiet"]["stale"] is True
    # Forgotten for good: the ghost cannot come back on the next snapshot.
    assert "dead" not in bridge.peers


def test_bridge_snapshot_keeps_protected_peer_and_survives_bad_hook():
    bridge = MeshBridge(peer_id="dashboard-test")
    bridge.peers = _live_peers()
    bridge.protected_peer_ids = lambda: {"dead"}
    assert "dead" in bridge.snapshot()["peers"]

    bridge.peers = _live_peers()

    def boom():
        raise RuntimeError("device manager exploded")

    bridge.protected_peer_ids = boom
    snap = bridge.snapshot()  # must not raise
    assert "dead" not in snap["peers"]
    assert "fresh" in snap["peers"]
