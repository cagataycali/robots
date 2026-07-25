"""AWS reserved ($aws/...) topics are cloud-plane only on the bridge backend.

The Device Shadow mirror publishes named-shadow updates to
``$aws/things/<thing>/shadow/name/presence/update``. On the bridge backend the
old ``put()`` always wrote to Zenoh and only forwarded to IoT when the topic
suffix matched the bridge filter. ``$aws/...`` is not a ``strands/`` topic, so
its suffix was empty and it never reached IoT (silent shadow no-op) -- while the
Zenoh leg happily leaked the internal ``$aws/...`` key onto the LAN.

Correct routing: reserved topics go to the IoT leg exclusively, LAN never sees
them.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from strands_robots.mesh.iot.shadow import ShadowMirror, shadow_update_topic
from strands_robots.mesh.transport.bridge_transport import BridgeTransport


class _RecordingTransport:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict[str, Any]]] = []

    def connect(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover - not exercised here
        pass

    def is_alive(self) -> bool:
        return True

    def put(self, key: str, data: dict[str, Any]) -> None:
        self.puts.append((key, data))


def _bridge() -> tuple[BridgeTransport, _RecordingTransport, _RecordingTransport]:
    zenoh = _RecordingTransport()
    iot = _RecordingTransport()
    # _RecordingTransport is a structural stub (no declare_subscriber), so cast
    # to Any at the constructor boundary -- matching the convention in
    # test_bridge_dedup.py. BridgeTransport only exercises put/connect here.
    return BridgeTransport(zenoh=cast(Any, zenoh), iot=cast(Any, iot)), zenoh, iot


SHADOW_TOPIC = shadow_update_topic("thor-arm")  # $aws/things/thor-arm/shadow/name/presence/update


class TestReservedTopicRouting:
    def test_shadow_update_goes_to_iot_only(self):
        bridge, zenoh, iot = _bridge()
        bridge.put(SHADOW_TOPIC, {"state": {"reported": {"connected": True}}})
        assert [k for k, _ in iot.puts] == [SHADOW_TOPIC]
        # The reserved key must NEVER leak onto the Zenoh LAN.
        assert zenoh.puts == []

    def test_normal_presence_still_bridges_to_both(self):
        bridge, zenoh, iot = _bridge()
        bridge.put("strands/thor-arm/presence", {"connected": True})
        assert [k for k, _ in zenoh.puts] == ["strands/thor-arm/presence"]
        assert [k for k, _ in iot.puts] == ["strands/thor-arm/presence"]


class TestShadowMirrorOverBridge:
    def test_mirror_update_reaches_iot_leg_only(self):
        bridge, zenoh, iot = _bridge()
        mirror = ShadowMirror(thing_name="thor-arm")
        mirror.update(bridge, {"connected": True, "robot_type": "so101"})
        assert len(iot.puts) == 1
        key, payload = iot.puts[0]
        assert key == SHADOW_TOPIC
        assert payload == {"state": {"reported": {"connected": True, "robot_type": "so101"}}}
        assert zenoh.puts == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
