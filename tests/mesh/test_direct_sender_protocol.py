#!/usr/bin/env python3
"""The optional ``DirectSender`` capability next to ``MeshTransport``.

AWS IoT Core Direct Messaging delivers one MQTT PUBLISH to ONE connected
client with no subscription on its side. ``strands_robots.mesh.transport.base``
models that as a second, optional protocol so the Zenoh and Bridge backends,
which have no such primitive, stay exactly what they were. This module pins:

  - the ``DirectResult`` vocabulary: ``reason`` is one of ``DIRECT_REASONS`` and
    ``delivered`` agrees with it (an empty reason means delivered, and only
    then), so a caller can branch on the string alone;
  - which transports satisfy the protocol: anything exposing ``send_direct``
    does, the Zenoh transport does not;
  - the two optional attributes on the MQTT sample wrapper,
    ``response_topic`` and ``correlation_data``, which a handler reads with
    ``getattr(sample, name, None)`` because a ``zenoh.Sample`` has neither;
  - the normalisation of the MQTT5 properties off an inbound packet: bytes
    become text, undecodable bytes become ``None`` rather than a raise, and
    the values reach the subscriber's handler through ``_on_publish_received``.

The AWS IoT SDK is never reached over the network: the session module's
``_FakeMqtt5Client`` stands in for it, exactly as the other transport tests do.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from strands_robots.mesh.transport.base import (
    DIRECT_REASONS,
    DirectResult,
    DirectSender,
    MeshTransport,
)
from strands_robots.mesh.transport.iot_transport import (
    _mqtt5_reply_properties,
    _MqttSample,
)


class TestDirectResultVocabulary:
    """``reason`` is closed and ``delivered`` is derived from it."""

    def test_delivered_result_has_empty_reason(self):
        r = DirectResult(delivered=True, reason="", latency_ms=81.0)
        assert r.delivered is True
        assert r.reason == ""
        assert r.trace_id == ""
        assert r.detail == ""

    @pytest.mark.parametrize("reason", [r for r in DIRECT_REASONS if r])
    def test_every_failure_reason_is_constructible(self, reason):
        r = DirectResult(delivered=False, reason=reason, latency_ms=1.0, trace_id="t", detail="d")
        assert r.delivered is False
        assert r.reason == reason

    def test_unknown_reason_is_refused(self):
        with pytest.raises(ValueError, match="reason must be one of"):
            DirectResult(delivered=False, reason="timeout", latency_ms=1.0)

    def test_delivered_with_a_failure_reason_is_refused(self):
        with pytest.raises(ValueError, match="disagrees"):
            DirectResult(delivered=True, reason="offline", latency_ms=1.0)

    def test_not_delivered_with_empty_reason_is_refused(self):
        with pytest.raises(ValueError, match="disagrees"):
            DirectResult(delivered=False, reason="", latency_ms=1.0)

    def test_result_is_immutable(self):
        r = DirectResult(delivered=True, reason="", latency_ms=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.delivered = False  # type: ignore[misc]

    def test_vocabulary_names_every_api_outcome_once(self):
        # 200, 404, 403, 429, 413, 504, other 5xx, and the sender itself unusable.
        assert DIRECT_REASONS == (
            "",
            "offline",
            "forbidden",
            "throttled",
            "too_large",
            "unconfirmed",
            "error",
            "unavailable",
        )
        assert len(set(DIRECT_REASONS)) == len(DIRECT_REASONS)


class _AddressedFake:
    """Minimal object that satisfies both protocols."""

    def put(self, key: str, data: dict[str, Any]) -> None:
        pass

    def declare_subscriber(self, key_expr: str, handler: Any) -> Any:
        return None

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def send_direct(
        self,
        peer_id: str,
        key: str,
        data: dict[str, Any],
        *,
        confirm: bool = False,
        timeout: float = 5.0,
        response_key: str | None = None,
        correlation: str | None = None,
    ) -> DirectResult:
        return DirectResult(delivered=True, reason="", latency_ms=0.0)


class _PubSubOnlyFake:
    def put(self, key: str, data: dict[str, Any]) -> None:
        pass

    def declare_subscriber(self, key_expr: str, handler: Any) -> Any:
        return None

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        pass


class TestWhoIsADirectSender:
    """The capability is structural and optional."""

    def test_an_object_with_send_direct_is_a_direct_sender(self):
        fake = _AddressedFake()
        assert isinstance(fake, MeshTransport)
        assert isinstance(fake, DirectSender)

    def test_a_pub_sub_only_transport_is_not(self):
        fake = _PubSubOnlyFake()
        assert isinstance(fake, MeshTransport)
        assert not isinstance(fake, DirectSender)

    def test_the_zenoh_transport_is_not_a_direct_sender(self):
        from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

        assert not isinstance(ZenohTransport(), DirectSender)

    def test_mesh_transport_protocol_is_unchanged(self):
        # The base protocol gained nothing: a backend is not obliged to address peers.
        assert not hasattr(MeshTransport, "send_direct")


class TestMqttSampleReplyProperties:
    """``_MqttSample`` exposes the two MQTT5 properties, ``None`` when absent."""

    def test_defaults_are_none(self):
        s = _MqttSample("strands/a/cmd", b"{}")
        assert s.key_expr == "strands/a/cmd"
        assert s.payload.to_bytes() == b"{}"
        assert s.response_topic is None
        assert s.correlation_data is None

    def test_values_are_carried(self):
        s = _MqttSample("strands/a/cmd", b"{}", "strands/op/response/a/" + "0" * 32, "0" * 32)
        assert s.response_topic == "strands/op/response/a/" + "0" * 32
        assert s.correlation_data == "0" * 32

    def test_getattr_default_is_the_zenoh_contract(self):
        # A zenoh.Sample has neither attribute; handlers must not care which they got.
        class _ZenohShaped:
            key_expr = "strands/a/cmd"

        assert getattr(_ZenohShaped(), "response_topic", None) is None
        assert getattr(_MqttSample("k", b""), "response_topic", None) is None


class _Pkt:
    def __init__(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)


class TestReplyPropertyNormalisation:
    """``_mqtt5_reply_properties`` hands the handlers one shape."""

    def test_absent_properties_are_none(self):
        assert _mqtt5_reply_properties(_Pkt(topic="t", payload=b"")) == (None, None)
        assert _mqtt5_reply_properties(_Pkt(response_topic=None, correlation_data=None)) == (None, None)

    def test_bytes_correlation_becomes_text(self):
        assert _mqtt5_reply_properties(_Pkt(response_topic="r", correlation_data=b"abc")) == ("r", "abc")
        assert _mqtt5_reply_properties(_Pkt(correlation_data=bytearray(b"xy"))) == (None, "xy")

    def test_text_correlation_passes_through(self):
        assert _mqtt5_reply_properties(_Pkt(correlation_data="abc")) == (None, "abc")

    def test_undecodable_correlation_is_dropped_not_raised(self):
        assert _mqtt5_reply_properties(_Pkt(correlation_data=b"\xff\xfe")) == (None, None)

    def test_non_text_response_topic_is_dropped(self):
        assert _mqtt5_reply_properties(_Pkt(response_topic=b"bytes-topic")) == (None, None)
        assert _mqtt5_reply_properties(_Pkt(correlation_data=42)) == (None, None)


class TestInboundDeliveryCarriesTheProperties:
    """A subscribed handler sees the Response Topic of a direct message."""

    def test_handler_receives_response_topic_and_correlation(self, tmp_path):
        pytest.importorskip("awsiot")
        import awsiot.mqtt5_client_builder as builder

        from .test_iot_transport_session import _connect, _FakeMqtt5Client

        holder: dict[str, Any] = {"client": None, "auto_connack": True, "subscribe_exc": None}
        original = builder.mtls_from_path

        def fake_mtls_from_path(**kwargs):
            client = _FakeMqtt5Client(holder=holder, **kwargs)
            holder["client"] = client
            return client

        builder.mtls_from_path = fake_mtls_from_path
        try:
            t = _connect(tmp_path, thing="thor-arm")
            assert t.connect() is True
            seen: list[Any] = []
            t.declare_subscriber("strands/thor-arm/cmd", seen.append)

            class _Data:
                def __init__(self) -> None:
                    self.publish_packet = _Pkt(
                        topic="strands/thor-arm/cmd",
                        payload=b'{"a":1}',
                        response_topic="strands/op/response/thor-arm/" + "f" * 32,
                        correlation_data=b"f" * 32,
                    )

            holder["client"]._kwargs["on_publish_received"](_Data())
        finally:
            builder.mtls_from_path = original
            t.close()

        assert len(seen) == 1
        sample = seen[0]
        assert sample.key_expr == "strands/thor-arm/cmd"
        assert sample.payload.to_bytes() == b'{"a":1}'
        assert sample.response_topic == "strands/op/response/thor-arm/" + "f" * 32
        assert sample.correlation_data == "f" * 32

    def test_plain_publish_yields_none_properties(self, tmp_path):
        pytest.importorskip("awsiot")
        import awsiot.mqtt5_client_builder as builder

        from .test_iot_transport_session import _connect, _FakeMqtt5Client

        holder: dict[str, Any] = {"client": None, "auto_connack": True, "subscribe_exc": None}
        original = builder.mtls_from_path

        def fake_mtls_from_path(**kwargs):
            client = _FakeMqtt5Client(holder=holder, **kwargs)
            holder["client"] = client
            return client

        builder.mtls_from_path = fake_mtls_from_path
        try:
            t = _connect(tmp_path, thing="thor-arm")
            assert t.connect() is True
            seen: list[Any] = []
            t.declare_subscriber("strands/thor-arm/cmd", seen.append)
            holder["client"].fire_inbound("strands/thor-arm/cmd", b"{}")
        finally:
            builder.mtls_from_path = original
            t.close()

        assert len(seen) == 1
        assert seen[0].response_topic is None
        assert seen[0].correlation_data is None
