"""IotMqttTransport reconnect + construction-failure contracts.

Two failure modes on the connect path:

* After a broker drop ``_connected`` is clear while the stale client object
  lingers (its IO thread + socket still open). A second ``connect()`` must stop
  the old client before building a new one, or every retry leaks a client and
  the broker delivers each inbound message twice.
* ``mtls_from_path`` (corrupt PEM -> AwsCrtError) or ``start()`` can raise
  synchronously. ``connect()`` must contain that and return ``False`` -- the
  mesh stays OFF rather than crashing the host (and, in bridge mode, stranding
  the already-acquired Zenoh session).
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh.transport.iot_transport import IotMqttTransport


class _FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        # Fire CONNACK immediately, as the real IO thread would.
        self._kwargs["on_lifecycle_connection_success"](object())

    def stop(self) -> None:
        self.stopped = True


def _make_certs(tmp_path: Any, thing: str = "thor-arm") -> Any:
    cert_dir = tmp_path / "iot"
    cert_dir.mkdir()
    (cert_dir / f"{thing}.cert.pem").write_text("cert")
    (cert_dir / f"{thing}.private.key").write_text("key")
    (cert_dir / "AmazonRootCA1.pem").write_text("ca")
    return cert_dir


def _transport(tmp_path, thing="thor-arm") -> IotMqttTransport:
    return IotMqttTransport(
        thing_name=thing,
        endpoint="x-ats.iot.us-west-2.amazonaws.com",
        cert_dir=str(_make_certs(tmp_path, thing)),
        connect_timeout=1.0,
    )


class TestReconnectStopsStaleClient:
    def test_reconnect_after_drop_stops_old_client(self, tmp_path, monkeypatch):
        import awsiot.mqtt5_client_builder as builder

        built: list[_FakeClient] = []

        def fake_mtls_from_path(**kwargs):
            c = _FakeClient(**kwargs)
            built.append(c)
            return c

        monkeypatch.setattr(builder, "mtls_from_path", fake_mtls_from_path)

        t = _transport(tmp_path)
        assert t.connect() is True
        first = built[0]

        # Simulate a broker-side drop: connection lost, client object lingers.
        t._connected.clear()

        assert t.connect() is True
        # A new client was built AND the stale one was explicitly stopped.
        assert len(built) == 2
        assert built[1] is not first
        assert first.stopped is True
        assert t._client is built[1]


class TestConnectContainsConstructionFailure:
    def test_mtls_raise_returns_false_not_exception(self, tmp_path, monkeypatch):
        import awsiot.mqtt5_client_builder as builder

        class _AwsCrtError(Exception):
            pass

        def boom(**kwargs):
            raise _AwsCrtError("corrupt PEM")

        monkeypatch.setattr(builder, "mtls_from_path", boom)

        t = _transport(tmp_path)
        # Must NOT propagate -- returns False, leaves no dangling client.
        assert t.connect() is False
        assert t.is_alive() is False
        assert t._client is None

    def test_start_raise_returns_false_and_stops_client(self, tmp_path, monkeypatch):
        import awsiot.mqtt5_client_builder as builder

        class _Client(_FakeClient):
            def start(self) -> None:
                raise RuntimeError("io thread failed to launch")

        built: list[_Client] = []

        def fake_mtls_from_path(**kwargs):
            c = _Client(**kwargs)
            built.append(c)
            return c

        monkeypatch.setattr(builder, "mtls_from_path", fake_mtls_from_path)

        t = _transport(tmp_path)
        assert t.connect() is False
        assert t._client is None
        assert built[0].stopped is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
