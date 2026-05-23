"""Cross-path deduplication tests for :class:`BridgeTransport`.

In bridge mode the same command can be delivered twice (once over Zenoh,
once over MQTT) because subscriptions fan out on both sides. The
:class:`_CommandDeduplicator` collapses those duplicates by message
identity:

* same envelope nonce -> delivered exactly once.
* same content fingerprint (legacy un-enveloped payloads) -> delivered once.
* distinct messages -> both delivered.
* identity expires after the TTL.
* malformed / no-identity samples bypass dedup and are delivered as-is.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
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


# --- _CommandDeduplicator unit tests ------------------------------------


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
        # No nonce -> falls back to (sender, turn, command) fingerprint.
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
        # Still no dedup id -> still passes (does not record, so still False).
        assert d.is_duplicate("k", payload) is False

    def test_ttl_expiry(self):
        d = _CommandDeduplicator(ttl_s=0.05)
        payload = {"nonce": "abcdef0123456789"}
        assert d.is_duplicate("k", payload) is False
        time.sleep(0.1)
        assert d.is_duplicate("k", payload) is False  # expired -> re-accepted

    def test_clear(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"nonce": "abcdef0123456789"}
        d.is_duplicate("k", payload)
        d.clear()
        assert d.is_duplicate("k", payload) is False


# --- BridgeTransport integration ----------------------------------------


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
        # No dedup id -> passes through, still calls handler
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


class TestMonotonicClockR12:
    """R12 pin test - bridge dedup TTL math uses time.monotonic, not time.time.

    Pre-R12: time.time() was used for the now/cutoff math in
    is_duplicate(). When the wall clock moves backwards (NTP step, manual
    'date -s', VM resume from snapshot) the TTL window math is wrong and
    cached entries either survive forever or all get evicted at once.

    Post-R12: time.monotonic() is used; the cache survives wall-clock jumps.
    """

    def test_dedup_uses_monotonic_clock(self):
        """is_duplicate() must use time.monotonic, not time.time."""
        from strands_robots.mesh.transport import bridge_transport

        src = Path(bridge_transport.__file__).read_text()
        # The is_duplicate() implementation must read monotonic.
        assert "time.monotonic()" in src, (
            "R12 regression: bridge_transport must use time.monotonic() for TTL math. "
            "time.time() can move backwards (NTP step, snapshot resume) and break TTL semantics."
        )

    def test_no_time_dot_time_in_dedup_path(self):
        """R12 regression pin: no time.time() in the is_duplicate body."""
        from strands_robots.mesh.transport import bridge_transport

        src = Path(bridge_transport.__file__).read_text()
        # Locate the is_duplicate function body via string search (no regex).
        marker = "def is_duplicate("
        start = src.find(marker)
        assert start >= 0, "is_duplicate not found in bridge_transport source"
        # Body ends at the next 'def ' at the same indentation OR end of class
        end_marker = "\n    def "
        body = src[start:]
        next_def = body.find(end_marker, len(marker))
        if next_def > 0:
            body = body[:next_def]
        assert "time.time()" not in body, (
            "R12 regression: time.time() found inside is_duplicate body. "
            "Use time.monotonic() for TTL math (NTP-safe, snapshot-resume-safe)."
        )


class TestStrictDedupModeR15:
    """R15 pin tests — opt-in strict mode dedups payloads with no canonical fields.

    Default mode (strict=False): payloads without (sender_id, turn_id, command)
    pass through (preserves heartbeat-style semantics where the same payload
    legitimately recurs).

    Strict mode (strict=True): falls back to a full-payload SHA-256 hash so
    bridge cross-transport path can dedup ANY payload, not just canonical ones.
    """

    def test_default_mode_passes_through_no_canonical_payload(self):
        """Pre-R15 default behaviour preserved."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"random": "data"}
        assert d.is_duplicate("k", payload) is False
        assert d.is_duplicate("k", payload) is False  # still passes through

    def test_strict_mode_dedups_no_canonical_payload(self):
        """R15: strict mode must dedup payloads with no canonical triple."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        payload = {"heartbeat": "ping"}
        assert d.is_duplicate("k", payload) is False
        assert d.is_duplicate("k", payload) is True  # second copy = duplicate

    def test_strict_mode_distinguishes_different_payloads(self):
        """Different non-canonical payloads must NOT alias under strict mode."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        a = {"value": 1}
        b = {"value": 2}
        assert d.is_duplicate("k", a) is False
        assert d.is_duplicate("k", b) is False  # different payload, not a duplicate

    def test_strict_mode_canonical_payloads_unchanged(self):
        """Canonical payloads still use the canonical dedup id under strict mode."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        a = {"sender_id": "x", "turn_id": "1", "command": "stop", "extra": "noise"}
        b = {"sender_id": "x", "turn_id": "1", "command": "stop", "extra": "different_noise"}
        assert d.is_duplicate("k", a) is False
        # b has same canonical triple as a -> still a duplicate even though "extra" differs.
        assert d.is_duplicate("k", b) is True
