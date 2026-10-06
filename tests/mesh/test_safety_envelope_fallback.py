"""Regression pin: _publish_safety_envelope fallback must strip body
source_zid so receivers do not hard-reject body-present + wire-absent
envelopes (availability fix)."""

import sys
from unittest.mock import MagicMock

import pytest


def test_estop_fallback_strips_source_zid(monkeypatch):
    from strands_robots.mesh import core

    captured = {}

    def fake_put(key, payload):
        captured["key"] = key
        captured["payload"] = payload

    monkeypatch.setattr(core, "put", fake_put)

    m = core.Mesh(robot=object(), peer_id="t1")
    # Force the fallback path (no publisher available).
    monkeypatch.setattr(m, "_safety_publisher_for", lambda key: None)

    m._publish_safety_envelope(
        "strands/safety/estop",
        {"peer_id": "t1", "t": 1.0, "source_zid": "deadbeef"},
    )

    assert captured["key"] == "strands/safety/estop"
    assert "source_zid" not in captured["payload"]
    # Other fields preserved.
    assert captured["payload"]["peer_id"] == "t1"


def test_strip_wire_zid_noop_when_absent():
    from strands_robots.mesh import core

    m = core.Mesh(robot=object(), peer_id="t2")
    payload = {"peer_id": "t2", "t": 1.0}
    # No source_zid -> returns same object (cheap no-op).
    assert m._strip_wire_zid(payload) is payload


@pytest.mark.parametrize("topic", ["strands/safety/estop", "strands/safety/resume"])
def test_bridge_backend_delivers_safety_envelopes_to_the_iot_leg(monkeypatch, topic):
    """Under ``STRANDS_MESH_BACKEND=bridge`` the envelope must enter the bridge.

    The bridge's Zenoh leg opens the process-wide Zenoh session, so a zid is
    present; the raw ``declare_publisher`` path then bypassed ``put()`` and the
    MQTT/IoT leg never saw an e-stop or a resume.
    """
    import types

    from strands_robots.mesh import core, session
    from strands_robots.mesh.transport import factory
    from strands_robots.mesh.transport.bridge_transport import BridgeTransport

    raw_pub = MagicMock()
    raw_session = MagicMock()
    raw_session.info.zid.return_value = "ab" * 16
    raw_session.declare_publisher.return_value = raw_pub
    fake_zenoh = types.ModuleType("zenoh")
    fake_zenoh.SourceInfo = lambda eid, sn: (eid, sn)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    monkeypatch.setattr(session, "_SESSION", raw_session)

    lan, iot = MagicMock(), MagicMock()
    lan.is_alive.return_value = iot.is_alive.return_value = True
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
    monkeypatch.setattr(factory, "_TRANSPORT", BridgeTransport(zenoh=lan, iot=iot))

    mesh = core.Mesh(robot=object(), peer_id="issuer-1")
    mesh._running = True
    monkeypatch.setattr(mesh, "broadcast", lambda cmd, timeout=5.0: [])
    monkeypatch.setattr(mesh, "publish_safety_event", lambda *a, **k: None)
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret-code-1234567890abcdef")

    mesh.emergency_stop()
    if topic.endswith("resume"):
        iot.put.reset_mock()
        lan.put.reset_mock()
        assert mesh.resume("secret-code-1234567890abcdef") == {"status": "ok"}

    sent = [c.args for c in iot.put.call_args_list if c.args[0] == topic]
    assert len(sent) == 1, iot.put.call_args_list
    assert "source_zid" not in sent[0][1]
    assert [c.args[0] for c in lan.put.call_args_list].count(topic) == 1
    raw_pub.put.assert_not_called()
