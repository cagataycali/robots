#!/usr/bin/env python3
"""Session-lifecycle contracts for :class:`IotMqttTransport`.

``tests/mesh/test_transport.py`` pins the pure wire-format helpers and the
config-validation guards (missing thing/endpoint/cert -> ``connect() is False``).
This module covers the *live-session* path that runs once the mTLS client is
built: CONNACK handling, the QoS/retain/drop publish contract, subscriber
fan-out and rollback, inbound routing through ``_on_publish_received``, and
idempotent teardown.

The AWS IoT SDK is never reached over the network. ``mtls_from_path`` is
patched to hand back a :class:`_FakeMqtt5Client` that captures the lifecycle
callbacks the transport registers and records every publish/subscribe packet,
so each assertion is on observable transport behaviour, not internal state.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from strands_robots.mesh.transport.iot_transport import IotMqttTransport


class _Future:
    """Minimal stand-in for the ``concurrent.futures``-like object the SDK
    returns from ``subscribe``/``unsubscribe`` - only ``.result(timeout=)`` is
    used by the transport."""

    def __init__(self, raise_exc: Exception | None = None) -> None:
        self._raise_exc = raise_exc

    def result(self, timeout: float | None = None) -> Any:
        if self._raise_exc is not None:
            raise self._raise_exc
        return object()


class _FakeMqtt5Client:
    """Records packets and fires the lifecycle callbacks the transport wires up.

    ``mtls_from_path`` passes the transport's ``on_lifecycle_*`` and
    ``on_publish_received`` callbacks as kwargs; we stash them so a test can
    drive a CONNACK or an inbound message exactly as the real IO thread would.
    """

    def __init__(self, *, holder: dict[str, Any], **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._holder = holder
        self.started = False
        self.stopped = False
        self.published: list[Any] = []
        self.subscribed: list[Any] = []
        self.unsubscribed: list[Any] = []

    def start(self) -> None:
        self.started = True
        if self._holder["auto_connack"]:
            self._kwargs["on_lifecycle_connection_success"](object())

    def stop(self) -> None:
        self.stopped = True

    def publish(self, packet: Any) -> _Future:
        self.published.append(packet)
        return _Future()

    def subscribe(self, packet: Any) -> _Future:
        self.subscribed.append(packet)
        return _Future(raise_exc=self._holder["subscribe_exc"])

    def unsubscribe(self, packet: Any) -> _Future:
        self.unsubscribed.append(packet)
        return _Future()

    # Test helpers that mimic the broker's IO thread -------------------------

    def fire_inbound(self, topic: str, payload: bytes) -> None:
        """Invoke the transport's ``on_publish_received`` callback."""

        class _Pkt:
            def __init__(self, t: str, p: bytes) -> None:
                self.topic = t
                self.payload = p

        class _Data:
            def __init__(self, t: str, p: bytes) -> None:
                self.publish_packet = _Pkt(t, p)

        self._kwargs["on_publish_received"](_Data(topic, payload))

    def fire_disconnect(self) -> None:
        class _D:
            pass

        self._kwargs["on_lifecycle_disconnection"](_D())


def _make_certs(tmp_path: Any, thing: str = "thor-arm") -> Any:
    """Create the three files ``connect()`` requires under a cert dir."""
    cert_dir = tmp_path / "iot"
    cert_dir.mkdir()
    (cert_dir / f"{thing}.cert.pem").write_text("cert")
    (cert_dir / f"{thing}.private.key").write_text("key")
    (cert_dir / "AmazonRootCA1.pem").write_text("ca")
    return cert_dir


@pytest.fixture
def patched_builder(monkeypatch):
    """Patch ``mtls_from_path`` to return a captured ``_FakeMqtt5Client``.

    Yields a one-element list whose slot is filled with the client instance on
    the first ``connect()`` so tests can inspect recorded packets.
    """
    import awsiot.mqtt5_client_builder as builder

    holder: dict[str, Any] = {"client": None, "auto_connack": True, "subscribe_exc": None}

    def fake_mtls_from_path(**kwargs):
        client = _FakeMqtt5Client(holder=holder, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(builder, "mtls_from_path", fake_mtls_from_path)
    return holder


def _connect(tmp_path, thing="thor-arm", timeout=5.0) -> IotMqttTransport:
    cert_dir = _make_certs(tmp_path, thing)
    return IotMqttTransport(
        thing_name=thing,
        endpoint="x-ats.iot.us-west-2.amazonaws.com",
        cert_dir=str(cert_dir),
        connect_timeout=timeout,
    )


class TestConnectLifecycle:
    """``connect()`` returns True only after a CONNACK and is idempotent."""

    def test_connack_marks_connected_and_alive(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        assert t.connect() is True
        assert t.is_alive() is True
        assert patched_builder["client"].started is True

    def test_second_connect_is_noop_when_already_connected(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        assert t.connect() is True
        first_client = patched_builder["client"]
        # Re-connecting must not rebuild the client.
        assert t.connect() is True
        assert patched_builder["client"] is first_client

    def test_connect_timeout_tears_down_client(self, tmp_path, patched_builder):
        patched_builder["auto_connack"] = False  # no CONNACK -> wait() times out
        t = _connect(tmp_path, timeout=0.05)
        assert t.connect() is False
        assert t.is_alive() is False
        assert patched_builder["client"].stopped is True

    def test_disconnection_callback_clears_alive(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        assert t.connect() is True
        patched_builder["client"].fire_disconnect()
        assert t.is_alive() is False


class TestPublishContract:
    """``put()`` honours the per-topic QoS/retain map and never-bridge drops."""

    def test_state_publishes_qos0_no_retain(self, tmp_path, patched_builder):
        from awscrt import mqtt5

        t = _connect(tmp_path)
        t.connect()
        t.put("strands/thor-arm/state", {"j": [1, 2, 3]})
        pkt = patched_builder["client"].published[-1]
        assert pkt.topic == "strands/thor-arm/state"
        assert pkt.qos == mqtt5.QoS.AT_MOST_ONCE
        assert pkt.retain is False
        assert json.loads(bytes(pkt.payload)) == {"j": [1, 2, 3]}

    def test_presence_publishes_qos1_retained(self, tmp_path, patched_builder):
        from awscrt import mqtt5

        t = _connect(tmp_path)
        t.connect()
        t.put("strands/thor-arm/presence", {"up": True})
        pkt = patched_builder["client"].published[-1]
        assert pkt.qos == mqtt5.QoS.AT_LEAST_ONCE
        assert pkt.retain is True

    def test_camera_topic_is_dropped(self, tmp_path, patched_builder):
        # camera/ is a never-bridge prefix AND a DROP policy: nothing published.
        t = _connect(tmp_path)
        t.connect()
        t.put("strands/thor-arm/camera/frame", {"jpeg": "..."})
        assert patched_builder["client"].published == []

    def test_put_before_connect_is_silent_noop(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        # No connect() -> no client; must not raise.
        t.put("strands/thor-arm/state", {"x": 1})
        assert patched_builder["client"] is None


class TestSubscribeAndRouting:
    """Subscriber registration, fan-out, rollback, and inbound delivery."""

    def test_declare_subscriber_requires_connection(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        with pytest.raises(RuntimeError):
            t.declare_subscriber("strands/+/state", lambda s: None)

    def test_single_broker_subscribe_for_duplicate_filter(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        t.declare_subscriber("strands/+/state", lambda s: None)
        t.declare_subscriber("strands/+/state", lambda s: None)
        # Two handlers on the same filter -> exactly one broker SUBSCRIBE.
        assert len(patched_builder["client"].subscribed) == 1

    def test_inbound_message_routed_to_matching_handlers(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        received: list[tuple[str, bytes]] = []
        t.declare_subscriber("strands/+/state", lambda s: received.append((s.key_expr, s.payload.to_bytes())))
        patched_builder["client"].fire_inbound("strands/thor-arm/state", b'{"v":1}')
        assert received == [("strands/thor-arm/state", b'{"v":1}')]

    def test_non_matching_inbound_is_ignored(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        received: list[Any] = []
        t.declare_subscriber("strands/+/cmd", lambda s: received.append(s))
        patched_builder["client"].fire_inbound("strands/thor-arm/state", b"{}")
        assert received == []

    def test_handler_exception_does_not_break_delivery(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        delivered: list[str] = []

        def boom(_s):
            raise ValueError("handler bug")

        t.declare_subscriber("strands/+/state", boom)
        t.declare_subscriber("strands/+/state", lambda s: delivered.append("ok"))
        # A raising handler must not stop the sibling handler from running.
        patched_builder["client"].fire_inbound("strands/thor-arm/state", b"{}")
        assert delivered == ["ok"]

    def test_subscribe_failure_rolls_back_handler(self, tmp_path, patched_builder):
        patched_builder["subscribe_exc"] = RuntimeError("SUBACK denied")
        t = _connect(tmp_path)
        t.connect()
        with pytest.raises(RuntimeError, match="failed"):
            t.declare_subscriber("strands/+/state", lambda s: None)
        # After rollback a fresh subscribe is attempted (handler not stuck).
        patched_builder["subscribe_exc"] = None
        t.declare_subscriber("strands/+/state", lambda s: None)
        assert len(patched_builder["client"].subscribed) == 2

    def test_undeclare_last_handler_unsubscribes_at_broker(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        handle = t.declare_subscriber("strands/+/state", lambda s: None)
        handle.undeclare()
        assert len(patched_builder["client"].unsubscribed) == 1
        # Idempotent: a second undeclare is a no-op.
        handle.undeclare()
        assert len(patched_builder["client"].unsubscribed) == 1

    def test_undeclare_keeps_broker_sub_while_peers_remain(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        h1 = t.declare_subscriber("strands/+/state", lambda s: None)
        t.declare_subscriber("strands/+/state", lambda s: None)
        h1.undeclare()
        # One handler still active -> no broker UNSUBSCRIBE yet.
        assert patched_builder["client"].unsubscribed == []


class TestClose:
    """``close()`` stops the client, clears handlers, and is idempotent."""

    def test_close_stops_client_and_drops_routing(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        received: list[Any] = []
        t.declare_subscriber("strands/+/state", lambda s: received.append(s))
        client = patched_builder["client"]
        t.close()
        assert client.stopped is True
        assert t.is_alive() is False
        # Handlers cleared: a late inbound delivery routes to nobody.
        client.fire_inbound("strands/thor-arm/state", b"{}")
        assert received == []

    def test_close_twice_is_safe(self, tmp_path, patched_builder):
        t = _connect(tmp_path)
        t.connect()
        t.close()
        t.close()  # must not raise
        assert t.is_alive() is False
