"""Cross-path deduplication tests for :class:`BridgeTransport`.

In bridge mode the same command can be delivered twice (once over Zenoh,
once over MQTT) because subscriptions fan out on both sides. The
:class:`_CommandDeduplicator` collapses those duplicates by message
identity:

* same envelope nonce → delivered exactly once.
* same content fingerprint (legacy un-enveloped payloads) → delivered once.
* distinct messages → both delivered.
* identity expires after the TTL.
* malformed / no-identity samples bypass dedup and are delivered as-is.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

from strands_robots.mesh.transport.bridge_transport import (
    BridgeTransport,
    _CommandDeduplicator,
)


class _FakeSample:
    """Mimics a zenoh/iot sample: ``sample.payload.to_bytes()`` returns JSON."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.key_expr = "strands/robot-a/cmd"
        encoded = json.dumps(data).encode()
        self.payload = MagicMock()
        self.payload.to_bytes.return_value = encoded


# ─── _CommandDeduplicator unit tests ────────────────────────────────────


class TestCommandDeduplicator:
    def test_first_call_not_duplicate(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"nonce": "abcdef0123456789", "payload": {"sender_id": "a"}}
        assert d.is_duplicate("k", payload) is False

    def test_repeat_payload_is_duplicate(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        d.is_duplicate("k", payload)
        assert d.is_duplicate("k", payload) is True

    def test_different_payloads_not_duplicates(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        a = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        b = {"sender_id": "alice", "turn_id": "t2", "command": {"action": "status"}}
        assert d.is_duplicate("k", a) is False
        assert d.is_duplicate("k", b) is False

    def test_different_keys_isolate_payloads(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        assert d.is_duplicate("k1", payload) is False
        # Same fingerprint on a different topic is NOT a dup -- distinct delivery.
        assert d.is_duplicate("k2", payload) is False

    def test_unsigned_fingerprint_dedup(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        # No nonce → falls back to (sender, turn, command) fingerprint.
        legacy = {
            "sender_id": "alice",
            "turn_id": "t1",
            "command": {"action": "status"},
        }
        assert d.is_duplicate("k", legacy) is False
        assert d.is_duplicate("k", legacy) is True

    def test_unsigned_distinct_turn_ids_not_duplicate(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        a = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        b = {"sender_id": "alice", "turn_id": "t2", "command": {"action": "status"}}
        assert d.is_duplicate("k", a) is False
        assert d.is_duplicate("k", b) is False

    def test_payload_without_dedup_id_passes_through(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"random": "data"}
        assert d.is_duplicate("k", payload) is False
        # Still no dedup id → still passes (does not record, so still False).
        assert d.is_duplicate("k", payload) is False

    def test_ttl_expiry(self):
        d = _CommandDeduplicator(ttl_s=0.05)
        payload = {"nonce": "abcdef0123456789"}
        assert d.is_duplicate("k", payload) is False
        time.sleep(0.1)
        assert d.is_duplicate("k", payload) is False  # expired → re-accepted

    def test_clear(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"nonce": "abcdef0123456789"}
        d.is_duplicate("k", payload)
        d.clear()
        assert d.is_duplicate("k", payload) is False


# ─── BridgeTransport integration ────────────────────────────────────────


class TestBridgeDedupIntegration:
    def _make_bridge(self) -> tuple[BridgeTransport, MagicMock, MagicMock]:
        """Construct a BridgeTransport with mocked Zenoh + IoT siblings."""
        zenoh = MagicMock()
        zenoh.is_alive.return_value = True
        zenoh.connect.return_value = True
        zenoh.declare_subscriber.side_effect = lambda key, handler: ("zenoh", key, handler)

        iot = MagicMock()
        iot.is_alive.return_value = True
        iot.connect.return_value = True
        iot.declare_subscriber.side_effect = lambda key, handler: ("iot", key, handler)

        b = BridgeTransport(zenoh=zenoh, iot=iot)
        return b, zenoh, iot

    def test_subscriber_dedups_across_paths(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []

        def handler(sample):
            delivered.append(sample)

        bridge.declare_subscriber("strands/robot-a/cmd", handler)

        # Pull the dedup-wrapped handlers out of the mocks
        zenoh_handler = zenoh.declare_subscriber.call_args.args[1]
        iot_handler = iot.declare_subscriber.call_args.args[1]

        # Same payload arrives via both paths.
        sample = _FakeSample(
            {
                "sender_id": "alice",
                "turn_id": "t1",
                "command": {"action": "status"},
            }
        )
        zenoh_handler(sample)
        iot_handler(sample)

        assert len(delivered) == 1, "duplicate should be filtered"

    def test_distinct_envelopes_both_delivered(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zh = zenoh.declare_subscriber.call_args.args[1]

        zh(_FakeSample({"sender_id": "a", "turn_id": "t1", "command": {"action": "status"}}))
        zh(_FakeSample({"sender_id": "a", "turn_id": "t2", "command": {"action": "status"}}))
        assert len(delivered) == 2

    def test_legacy_unsigned_dedup_via_fingerprint(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zh = zenoh.declare_subscriber.call_args.args[1]
        ih = iot.declare_subscriber.call_args.args[1]

        legacy = _FakeSample({"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}})
        zh(legacy)
        ih(legacy)
        assert len(delivered) == 1

    def test_malformed_payload_falls_through(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zh = zenoh.declare_subscriber.call_args.args[1]

        broken = MagicMock()
        broken.payload.to_bytes.return_value = b"not json"
        zh(broken)
        # No dedup id → passes through, still calls handler
        assert delivered == [broken]

    def test_dedup_resets_per_topic(self):
        """Same nonce on different topics must NOT be deduplicated together
        (different subscribers, different cache buckets)."""
        bridge, zenoh, iot = self._make_bridge()
        delivered_a: list[Any] = []
        delivered_b: list[Any] = []

        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered_a.append(s))
        # call_args is the LAST call; capture handler now.
        zh_a = zenoh.declare_subscriber.call_args.args[1]

        bridge.declare_subscriber("strands/robot-b/cmd", lambda s: delivered_b.append(s))
        zh_b = zenoh.declare_subscriber.call_args.args[1]

        sample = _FakeSample({"nonce": "abc1234567890def", "payload": {"sender_id": "x"}})
        zh_a(sample)
        zh_b(sample)
        assert len(delivered_a) == 1
        assert len(delivered_b) == 1
