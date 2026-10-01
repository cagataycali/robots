"""A robot may opt in to inline camera frames over AWS IoT Core, under the payload cap.

The IoT transport drops every ``strands/<peer>/camera/<cam>`` frame by
default: MQTT caps a publish at 128 KB and a WAN message costs money, so the
designed path is the S3 reference (:mod:`~strands_robots.mesh.iot.camera_offload`).
A robot without a bucket still needs to show its cameras in the dashboard, so
``STRANDS_MESH_IOT_CAMERA_INLINE=1`` lets a frame that fits under the cap
through at QoS 0. A frame over the cap is dropped with one WARNING per topic
naming the size and the cap; the opt-in never lifts the teleop ``input/`` and
``hand/`` drop, and a value that is not a boolean is refused out loud.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def transport(monkeypatch):
    mock_mqtt5 = MagicMock()
    mock_awscrt = MagicMock()
    mock_awscrt.mqtt5 = mock_mqtt5
    monkeypatch.setitem(sys.modules, "awscrt", mock_awscrt)
    monkeypatch.setitem(sys.modules, "awscrt.mqtt5", mock_mqtt5)
    from strands_robots.mesh.transport.iot_transport import IotMqttTransport

    t = IotMqttTransport(thing_name="test-thing", endpoint="x.iot")
    t._client = MagicMock()
    t._connected.set()
    return t


def _packets() -> list[dict]:
    """The keyword arguments of every PublishPacket the transport built."""
    return [call.kwargs for call in sys.modules["awscrt.mqtt5"].PublishPacket.call_args_list]


def _published_topics(t) -> list[str]:
    assert len(t._client.publish.call_args_list) == len(_packets())
    return [pkt["topic"] for pkt in _packets()]


def test_default_still_drops_a_camera_frame(transport, monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_IOT_CAMERA_INLINE", raising=False)
    transport.put("strands/test-thing/camera/front", {"cam": "front", "data": "aGVsbG8="})
    transport._client.publish.assert_not_called()


def test_opt_in_publishes_a_frame_under_the_cap_at_qos0_unretained(transport, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_IOT_CAMERA_INLINE", "1")
    transport.put("strands/test-thing/camera/front", {"cam": "front", "data": "aGVsbG8=", "t": 1.0})
    assert _published_topics(transport) == ["strands/test-thing/camera/front"]
    pkt = _packets()[-1]
    assert pkt["retain"] is False
    assert pkt["qos"] is sys.modules["awscrt.mqtt5"].QoS.AT_MOST_ONCE
    from strands_robots.mesh.transport.iot_transport import _qos_and_retain_for

    assert _qos_and_retain_for("strands/test-thing/camera/front") == (-1, False), "the default table still says DROP"


def test_a_frame_over_the_cap_is_dropped_with_one_warning_per_topic(transport, monkeypatch, caplog):
    from strands_robots.mesh.transport.iot_transport import DIRECT_PAYLOAD_CAP

    monkeypatch.setenv("STRANDS_MESH_IOT_CAMERA_INLINE", "1")
    big = {"cam": "front", "data": "A" * (DIRECT_PAYLOAD_CAP + 1)}
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.transport.iot_transport"):
        transport.put("strands/test-thing/camera/front", big)
        transport.put("strands/test-thing/camera/front", big)
        transport.put("strands/test-thing/camera/wrist", big)
    transport._client.publish.assert_not_called()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "camera" in r.getMessage()]
    assert len(warnings) == 2, [w.getMessage() for w in warnings]
    text = warnings[0].getMessage()
    assert str(DIRECT_PAYLOAD_CAP) in text and "front" in text
    assert "STRANDS_MESH_CAMERA_HZ" in text or "resolution" in text or "S3" in text


def test_opt_in_never_lifts_the_teleop_drop(transport, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_IOT_CAMERA_INLINE", "1")
    transport.put("strands/test-thing/input/leader", {"action": {}})
    transport.put("strands/test-thing/hand/right/state", {"x": 1})
    transport._client.publish.assert_not_called()


def test_a_camera_ref_is_unaffected_by_the_opt_in(transport, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_IOT_CAMERA_INLINE", "0")
    transport.put("strands/test-thing/camera/front/ref", {"cam": "front", "presigned_url": "https://x"})
    assert _published_topics(transport) == ["strands/test-thing/camera/front/ref"]


def test_a_value_that_is_not_a_boolean_is_refused_out_loud(transport, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_IOT_CAMERA_INLINE", "maybe")
    with pytest.raises(ValueError, match="STRANDS_MESH_IOT_CAMERA_INLINE"):
        transport.put("strands/test-thing/camera/front", {"cam": "front", "data": "aGVsbG8="})
