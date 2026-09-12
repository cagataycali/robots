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

The split is decided from the topic's SHAPE, and both legs decide it the same
way. A camera name is a free string from the robot's ``config.cameras``, so a
substring test for the exemption reads a frame topic as a pointer whenever the
name collides with the tail: a camera named ``ref`` published its whole inline
base64 JPEG on ``strands/<peer>/camera/ref``, straight past the drop rule that
exists to keep that payload off the WAN.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from strands_robots.mesh.transport.bridge_transport import _should_bridge
from strands_robots.mesh.transport.iot_transport import (
    IotMqttTransport,
    _is_camera_ref,
    _qos_and_retain_for,
    _should_drop,
)

REF_TOPIC = "strands/thor-arm/camera/wrist/ref"
FRAME_TOPIC = "strands/thor-arm/camera/wrist"
# The frame topic of a camera named ``ref``: an inline base64 JPEG on a topic
# whose tail is the pointer marker, with no camera name in front of it.
COLLIDING_FRAME_TOPIC = "strands/thor-arm/camera/ref"


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

    def test_a_camera_named_ref_publishes_nothing(self):
        """A frame is a frame whatever the camera is called."""
        t = self._connected_transport()
        # The envelope Mesh._encode_and_publish_frames builds: a base64 JPEG,
        # well past MQTT's 128 KB cap once the transport wraps it.
        frame = {
            "peer_id": "thor-arm",
            "cam": "ref",
            "shape": [480, 640, 3],
            "encoding": "jpeg",
            "data": base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 61_000).decode("ascii"),
        }
        assert len(json.dumps(frame).encode()) > 80_000  # this is a frame, not a pointer
        t.put(COLLIDING_FRAME_TOPIC, frame)
        assert t._client.published == []


class TestOnlyAPointerGetsThePointerExemption:
    """The exemption follows the topic shape, not a substring of it.

    ``camera/<cam>/ref`` is a pointer; a ``ref`` tail reached any other way is
    a payload the drop rule refuses. Delivering one is the cloud-pollution
    shape ``_should_bridge`` already refuses for tail-appended topics.
    """

    @pytest.mark.parametrize(
        "topic, why",
        [
            (COLLIDING_FRAME_TOPIC, "a camera named 'ref' - the frame itself, not a pointer"),
            ("strands/thor-arm/input/camera/ref", "teleop input, LAN-only at 50 Hz"),
            ("strands/thor-arm/hand/camera/ref", "hand control, LAN-only at 50 Hz"),
            ("strands/thor-arm/state/camera/ref", "a state topic that merely names a camera"),
        ],
    )
    def test_a_ref_tail_outside_the_camera_family_is_not_a_pointer(self, topic, why):
        assert _is_camera_ref(topic) is False, why

    @pytest.mark.parametrize(
        "topic",
        [
            REF_TOPIC,
            "strands/thor-arm/camera/front/left/ref",  # multi-segment camera name
        ],
    )
    def test_a_real_pointer_keeps_its_exemption(self, topic):
        assert _is_camera_ref(topic) is True
        assert _should_drop(topic) is False
        assert _qos_and_retain_for(topic)[0] >= 0

    def test_the_colliding_frame_topic_is_dropped_by_both_gates(self):
        assert _should_drop(COLLIDING_FRAME_TOPIC) is True
        assert _qos_and_retain_for(COLLIDING_FRAME_TOPIC)[0] < 0

    @pytest.mark.parametrize("topic", [REF_TOPIC, FRAME_TOPIC, COLLIDING_FRAME_TOPIC])
    def test_the_bridge_leg_and_the_mqtt_leg_agree_on_what_a_pointer_is(self, topic):
        # A filter that excludes camera entirely, so bridging can only come
        # from the pointer exemption.
        suffixes = frozenset({"presence"})
        assert _should_bridge(topic, suffixes) is _is_camera_ref(topic)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
