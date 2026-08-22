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
        PORT, "move_motor", port_free=POSE_PORT_FREE,
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
    deps = _deps(holders=lambda p: [4242], tracked=lambda: {4242: "so101-real-689"}, exists=lambda p: p != "/dev/ttyACM0")
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
