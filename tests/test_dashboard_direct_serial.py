"""Bus-guarded direct-serial tools (dashboard/direct_serial.py)."""

from __future__ import annotations

from strands_robots.dashboard.direct_serial import (
    POSE_PORT_FREE,
    SERIAL_PORT_FREE,
    build_direct_serial_tools,
    port_refusal,
)

PORT = "/dev/cu.usbmodemFAKE1"


def _deps(holders=None, tracked=None, scan=None, exists=None):
    return {
        "holders": holders or (lambda p: []),
        "tracked": tracked or (lambda: {}),
        "scan": scan or (lambda: [{"device": "/dev/cu.usbmodemA"}, {"device": "/dev/cu.usbmodemB"}]),
        "exists": exists or (lambda p: True),
    }


def _boom(*a, **k):
    raise AssertionError("guard touched I/O for a port-free action")


def test_port_free_actions_never_touch_io():
    for free, act in [(POSE_PORT_FREE, a) for a in POSE_PORT_FREE] + [(SERIAL_PORT_FREE, a) for a in SERIAL_PORT_FREE]:
        assert port_refusal(None, act, port_free=free, holders=_boom, tracked=_boom, scan=_boom, exists=_boom) is None


def test_missing_port_refused_with_scan_choices():
    r = port_refusal(None, "send", port_free=SERIAL_PORT_FREE, **_deps())
    assert r is not None and "no serial port was given" in r
    assert "/dev/cu.usbmodemA" in r and "/dev/cu.usbmodemB" in r


def test_nonexistent_linux_default_port_refused_by_name():
    r = port_refusal("/dev/ttyACM0", "move_motor", port_free=POSE_PORT_FREE, **_deps(exists=lambda p: False))
    assert r is not None and "/dev/ttyACM0" in r and "does not exist" in r and "/dev/cu.usbmodemA" in r


def test_port_held_by_own_child_names_peer_and_pid():
    r = port_refusal(
        PORT,
        "move_motor",
        port_free=POSE_PORT_FREE,
        **_deps(holders=lambda p: [4242], tracked=lambda: {4242: "so101-real-689"}),
    )
    assert r is not None and "so101-real-689" in r and "4242" in r and "despawn" in r.lower()


def test_port_held_by_stranger_names_pid():
    r = port_refusal(PORT, "send", port_free=SERIAL_PORT_FREE, **_deps(holders=lambda p: [777]))
    assert r is not None and "777" in r and "does not manage" in r


def test_free_existing_port_proceeds():
    assert port_refusal(PORT, "move_motor", port_free=POSE_PORT_FREE, **_deps()) is None


def test_unreadable_holder_list_fails_closed():
    def broken(p):
        raise RuntimeError("lsof died")

    r = port_refusal(PORT, "send", port_free=SERIAL_PORT_FREE, **_deps(holders=broken))
    assert r is not None and "refused" in r


def _fake_tools(monkeypatch=None):
    deps = _deps(
        holders=lambda p: [4242], tracked=lambda: {4242: "so101-real-689"}, exists=lambda p: p != "/dev/ttyACM0"
    )
    return build_direct_serial_tools(deps["tracked"], holders=deps["holders"], scan=deps["scan"], exists=deps["exists"])


def test_wrapped_tools_keep_sdk_names_and_schemas():
    tools = _fake_tools()
    names = {t.tool_name for t in tools}
    assert names == {"pose_tool", "serial_tool"}
    for t in tools:
        props = t.tool_spec["inputSchema"]["json"]["properties"]
        assert "action" in props and "port" in props
        assert t.tool_spec["description"]


def test_wrapped_pose_tool_refuses_held_port_before_opening_it():
    pose = next(t for t in _fake_tools() if t.tool_name == "pose_tool")
    out = pose(action="move_motor", port=PORT, motor_name="shoulder_pan", position=10.0)
    assert out["status"] == "error"
    assert "so101-real-689" in out["content"][0]["text"]


def test_wrapped_serial_tool_refuses_missing_port_with_choices():
    ser = next(t for t in _fake_tools() if t.tool_name == "serial_tool")
    out = ser(action="send", data="ping")
    assert out["status"] == "error"
    assert "/dev/cu.usbmodemA" in out["content"][0]["text"]


def test_wrapped_pose_tool_port_free_action_reaches_sdk():
    pose = next(t for t in _fake_tools() if t.tool_name == "pose_tool")
    out = pose(action="list_poses")
    assert out["status"] in ("success", "error")
    text = " ".join(str(c.get("text", "")) for c in out.get("content", []))
    assert "held by" not in text and "does not exist" not in text


class TestMotionGateRows:
    """agent_hitl.MOTION_ACTIONS is the ONLY human gate for these tools (they raise no interrupt)."""

    def _intent(self, tool_name, tool_input):
        from strands_robots.dashboard.agent_hitl import motion_intent

        return motion_intent(tool_name, tool_input, peers={}, env={})

    def test_pose_motion_actions_ask_and_name_the_port(self):
        for act in ("load_pose", "move_motor", "move_multiple", "incremental_move", "reset_to_home"):
            reason = self._intent("pose_tool", {"action": act, "port": PORT})
            assert reason is not None, act
            assert reason["target"] == PORT

    def test_pose_reads_and_stops_are_never_gated(self):
        for act in (
            "read_position",
            "read_all",
            "list_poses",
            "show_pose",
            "delete_pose",
            "connect",
            "emergency_stop",
            "store_pose",
        ):
            assert self._intent("pose_tool", {"action": act, "port": PORT}) is None, act

    def test_serial_writes_ask_and_reads_do_not(self):
        for act in ("send", "send_read", "feetech_position", "feetech_velocity"):
            assert self._intent("serial_tool", {"action": act, "port": PORT}) is not None, act
        for act in ("list_ports", "read", "feetech_ping", "monitor"):
            assert self._intent("serial_tool", {"action": act, "port": PORT}) is None, act

    def test_always_allow_grant_skips_the_gate(self):
        from strands_robots.dashboard.agent_hitl import motion_intent
        from strands_robots.dashboard.agent_motion import MOTION_ENV

        assert (
            motion_intent("pose_tool", {"action": "move_motor", "port": PORT}, peers={}, env={MOTION_ENV: "1"}) is None
        )


class TestAgentWiring:
    """The builder and the not-yet-built badge both carry the new tools."""

    def test_builder_produces_both_guarded_tools_without_devices(self, monkeypatch):
        from strands_robots.dashboard import agent_bridge as ab

        monkeypatch.setattr(ab, "_devices", None)
        names = sorted(t.tool_name for t in ab._direct_serial_tools())
        assert names == ["pose_tool", "serial_tool"]

    def test_tracked_children_is_empty_not_an_error_without_devices(self, monkeypatch):
        from strands_robots.dashboard import agent_bridge as ab

        monkeypatch.setattr(ab, "_devices", None)
        assert ab._tracked_children() == {}

    def test_unbuilt_agent_status_badge_names_the_direct_serial_tools(self, monkeypatch):
        from strands_robots.dashboard import agent_bridge as ab

        monkeypatch.setattr(ab, "_bridge", None)
        monkeypatch.setattr(ab, "_agent", None)
        tools = ab.agent_status()["tools"]
        assert "pose_tool" in tools and "serial_tool" in tools


class TestConfirmDetail:
    """The interrupt's instruction line carries the call's own motion fields."""

    def _intent(self, tool_name, tool_input):
        from strands_robots.dashboard.agent_hitl import motion_intent

        return motion_intent(tool_name, tool_input, peers={}, env={})

    def test_move_motor_confirm_names_motor_and_position(self):
        reason = self._intent(
            "pose_tool", {"action": "move_motor", "port": PORT, "motor_name": "shoulder_pan", "position": 10.5}
        )
        assert reason["instruction"] == "move_motor motor_name=shoulder_pan position=10.5"

    def test_load_pose_confirm_names_the_pose(self):
        reason = self._intent("pose_tool", {"action": "load_pose", "port": PORT, "pose_name": "home_rest"})
        assert "pose_name=home_rest" in reason["instruction"]

    def test_serial_send_confirm_quotes_the_raw_bytes(self):
        reason = self._intent("serial_tool", {"action": "send", "port": PORT, "data": "#5P1500T100"})
        assert "data=#5P1500T100" in reason["instruction"]

    def test_fieldless_gated_call_synthesizes_nothing(self):
        reason = self._intent("pose_tool", {"action": "reset_to_home", "port": PORT})
        assert reason["instruction"] == ""

    def test_explicit_instruction_is_never_overwritten(self):
        reason = self._intent(
            "pose_tool", {"action": "move_motor", "port": PORT, "instruction": "operator words", "motor_name": "elbow"}
        )
        assert reason["instruction"] == "operator words"
