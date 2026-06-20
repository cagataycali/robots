"""Behavior tests for the robot_mesh Device Connect dispatch gate.

Covers the two decision points that route a robot_mesh action either through
Device Connect or back to the built-in Zenoh mesh:

  - ``_dc_ensure_connected`` -- the idempotent agent-side connection bring-up,
    including the insecure-transport warning and the connect()-on-miss path.
  - ``_try_device_connect`` -- the gate that returns None (signalling the
    caller to fall back to the mesh) for mesh-only actions, when DC dispatch is
    disabled, when DC raises, or when no devices are discovered; and that
    dispatches when a well-formed device list is present.

These are fully hardware-free: the ``device_connect_agent_tools.connection``
entry points are monkeypatched, so no transport is ever opened.
"""

from __future__ import annotations

import sys
import types

import pytest

import strands_robots.tools.robot_mesh as rm


@pytest.fixture
def fresh_dc_state(monkeypatch):
    """Reset the module-level connection flag so each test starts disconnected.

    ``_dc_state["connected"]`` is process-global; without this a prior test that
    flipped it would mask the connect path here.
    """
    monkeypatch.setitem(rm._dc_state, "connected", False)
    yield


@pytest.fixture
def fake_dc_connection(monkeypatch):
    """Install a fake ``device_connect_agent_tools.connection`` module.

    Returns a record dict tracking how often ``connect`` / ``get_connection``
    were called and lets each test control their behaviour. The module is
    inserted into ``sys.modules`` so robot_mesh's function-local imports resolve
    to the fake.
    """
    record = {"connect_calls": 0, "get_calls": 0, "get_raises": False, "devices": []}

    def _get_connection():
        record["get_calls"] += 1
        if record["get_raises"]:
            raise RuntimeError("no active connection")
        conn = types.SimpleNamespace()
        conn.list_devices = lambda: record["devices"]
        return conn

    def _connect():
        record["connect_calls"] += 1

    pkg = types.ModuleType("device_connect_agent_tools")
    conn_mod = types.ModuleType("device_connect_agent_tools.connection")
    conn_mod.get_connection = _get_connection
    conn_mod.connect = _connect
    pkg.connection = conn_mod
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", conn_mod)
    return record


def test_ensure_connected_short_circuits_when_already_connected(monkeypatch, fake_dc_connection):
    """When the flag is already set, no connection entry point is touched."""
    monkeypatch.setitem(rm._dc_state, "connected", True)
    rm._dc_ensure_connected()
    assert fake_dc_connection["get_calls"] == 0
    assert fake_dc_connection["connect_calls"] == 0


def test_ensure_connected_reuses_existing_connection(fresh_dc_state, fake_dc_connection):
    """A live get_connection() is reused -- connect() is not re-invoked."""
    rm._dc_ensure_connected()
    assert fake_dc_connection["get_calls"] == 1
    assert fake_dc_connection["connect_calls"] == 0
    assert rm._dc_state["connected"] is True


def test_ensure_connected_connects_when_no_live_connection(fresh_dc_state, fake_dc_connection):
    """When get_connection() raises, connect() is called to bring one up."""
    fake_dc_connection["get_raises"] = True
    rm._dc_ensure_connected()
    assert fake_dc_connection["connect_calls"] == 1
    assert rm._dc_state["connected"] is True


def test_ensure_connected_warns_when_insecure_transport_opted_in(
    fresh_dc_state, fake_dc_connection, monkeypatch, caplog
):
    """Opting into insecure transport surfaces a visible warning (not silent)."""
    monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", "true")
    with caplog.at_level("WARNING"):
        rm._dc_ensure_connected()
    assert any("DEVICE_CONNECT_ALLOW_INSECURE" in r.message for r in caplog.records)


@pytest.mark.parametrize("action", ["subscribe", "watch", "inbox", "unsubscribe"])
def test_try_device_connect_returns_none_for_mesh_only_actions(action, monkeypatch):
    """Mesh-only actions never reach Device Connect -- they fall back at once."""
    monkeypatch.delenv("STRANDS_ROBOT_MESH_DC", raising=False)
    out = rm._try_device_connect(action, "t", "", "", "mock", 0, 1.0, 5.0)
    assert out is None


def test_try_device_connect_returns_none_when_dispatch_disabled(monkeypatch):
    """STRANDS_ROBOT_MESH_DC=off disables DC dispatch entirely."""
    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "off")
    out = rm._try_device_connect("peers", "", "", "", "mock", 0, 1.0, 5.0)
    assert out is None


def test_try_device_connect_falls_back_when_connection_raises(fresh_dc_state, fake_dc_connection, monkeypatch):
    """A failure bringing up / querying DC yields None (mesh fallback)."""
    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "on")
    fake_dc_connection["get_raises"] = True

    def _connect_also_raises():
        raise RuntimeError("router down")

    sys.modules["device_connect_agent_tools.connection"].connect = _connect_also_raises
    out = rm._try_device_connect("peers", "", "", "", "mock", 0, 1.0, 5.0)
    assert out is None


def test_try_device_connect_falls_back_when_no_devices_discovered(fresh_dc_state, fake_dc_connection, monkeypatch):
    """An empty device list means DC has nothing to dispatch to -> fall back."""
    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "on")
    fake_dc_connection["devices"] = []
    out = rm._try_device_connect("peers", "", "", "", "mock", 0, 1.0, 5.0)
    assert out is None


def test_try_device_connect_falls_back_when_devices_malformed(fresh_dc_state, fake_dc_connection, monkeypatch):
    """A non-list device payload (malformed/stubbed conn) is not dispatched."""
    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "on")
    fake_dc_connection["devices"] = {"not": "a list"}
    out = rm._try_device_connect("peers", "", "", "", "mock", 0, 1.0, 5.0)
    assert out is None


def test_try_device_connect_dispatches_when_devices_present(fresh_dc_state, fake_dc_connection, monkeypatch):
    """A well-formed, non-empty device list routes through DC dispatch."""
    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "on")
    fake_dc_connection["devices"] = [{"device_id": "robot-1", "device_type": "strands_robot"}]
    sentinel = {"status": "success", "content": [{"text": "dispatched"}]}
    captured = {}

    def _fake_dispatch(action, target, instruction, command, *rest):
        captured["action"] = action
        captured["target"] = target
        return sentinel

    monkeypatch.setattr(rm, "_device_connect_dispatch", _fake_dispatch)
    out = rm._try_device_connect("peers", "robot-1", "", "", "mock", 0, 1.0, 5.0)
    assert out is sentinel
    assert captured == {"action": "peers", "target": "robot-1"}


def test_dcresult_str_renders_text_block():
    """_DCResult.__str__ surfaces the first content text block for the agent."""
    result = rm._DCResult(status="success", content=[{"text": "-> robot-1: done"}])
    assert str(result) == "-> robot-1: done"


def test_dcresult_str_falls_back_to_dict_repr_without_text():
    """With no text content, __str__ degrades to the dict repr (no crash)."""
    result = rm._DCResult(status="error", content=[])
    assert str(result) == dict.__str__(result)
