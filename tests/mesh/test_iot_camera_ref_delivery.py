"""Camera S3-reference metadata must reach cloud subscribers over MQTT.

The camera offloader uploads JPEG frames to S3 and publishes a small ``/ref``
message (presigned S3 key + shape) on ``strands/<peer>/camera/<cam>/ref`` so a
cloud subscriber learns where the frame landed. Regression: the transport's
``camera/`` drop rule (meant for the multi-hundred-KB raw frames) also swallowed
the tiny ``/ref`` pointer, so no cloud subscriber ever learned the S3 key -- the
offload cost was paid with nothing on the receiving end.

These tests pin the exact split: raw ``camera/<cam>`` frames stay off MQTT, while
``camera/<cam>/ref`` pointers are delivered on both the pure-iot and bridge
backends.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

# The wire half of this file needs the optional ``awsiotsdk``/``awscrt`` extra:
# IotMqttTransport.put() imports ``awscrt.mqtt5`` to build the PublishPacket, so
# without it the publish is swallowed and the assertion reads "the transport
# dropped the camera ref" when the truth is "an optional dependency is absent".
pytest.importorskip("awscrt")

from strands_robots.mesh.transport.bridge_transport import _should_bridge
from strands_robots.mesh.transport.iot_transport import (
    IotMqttTransport,
    _is_camera_ref,
    _qos_and_retain_for,
    _should_drop,
)

REF_TOPIC = "strands/thor-arm/camera/wrist/ref"
FRAME_TOPIC = "strands/thor-arm/camera/wrist"


class TestCameraRefRoutingHelpers:
    def test_ref_recognised_frame_not(self):
        assert _is_camera_ref(REF_TOPIC) is True
        assert _is_camera_ref(FRAME_TOPIC) is False

    def test_ref_not_dropped_frame_dropped(self):
        assert _should_drop(REF_TOPIC) is False
        assert _should_drop(FRAME_TOPIC) is True

    def test_ref_gets_publishable_qos_frame_is_drop(self):
        qos, retain = _qos_and_retain_for(REF_TOPIC)
        assert qos >= 0  # not the DROP sentinel
        assert retain is False
        # Raw frame remains a DROP (-1).
        assert _qos_and_retain_for(FRAME_TOPIC)[0] < 0

    def test_bridge_forwards_ref_but_not_frame(self):
        suffixes = frozenset({"presence"})  # deliberately excludes camera
        assert _should_bridge(REF_TOPIC, suffixes) is True
        assert _should_bridge(FRAME_TOPIC, suffixes) is False


class _FakeClient:
    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, packet: Any) -> None:
        self.published.append(packet)


class TestCameraRefReachesTheWire:
    """End-to-end at the transport boundary: put() must hand the ref to the
    MQTT client, and never hand it a raw frame."""

    def _connected_transport(self) -> IotMqttTransport:
        t = IotMqttTransport(thing_name="thor-arm", endpoint="x-ats.iot.us-west-2.amazonaws.com")
        t._client = _FakeClient()
        t._connected.set()
        return t

    def test_ref_published_with_s3_key(self):
        t = self._connected_transport()
        ref = {"s3_key": "frames/thor-arm/wrist/123.jpg", "shape": [480, 640, 3]}
        t.put(REF_TOPIC, ref)
        published = t._client.published
        assert len(published) == 1
        assert published[0].topic == REF_TOPIC
        assert json.loads(bytes(published[0].payload)) == ref

    def test_raw_frame_never_published(self):
        t = self._connected_transport()
        t.put(FRAME_TOPIC, {"blob": "x" * 1000})
        assert t._client.published == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
