#!/usr/bin/env python3
"""``IotMqttTransport.send_direct``: the wire it produces and the outcomes it maps.

No AWS, no network. The persistent HTTPS client is replaced by a recorder that
answers with a scripted status, so each test pins one observable:

  - the request line: URL-encoded client id and topic, ``confirmation`` and a
    whole-second ``timeout`` of at least 1, ``responseTopic``, ``contentType``;
  - the headers: JSON content type, ``UTF8_DATA`` format indicator, the one
    user property as base64 JSON, base64 correlation data only when given;
  - the status map: 200 delivered, 404 offline, 403 forbidden (and the per-peer
    memo it leaves), 429 throttled with exactly one retry, 413 too_large, 504
    unconfirmed, 500 error with one retry, a socket failure error;
  - the guards that never reach the wire: an unencodable payload, a payload
    over the 128 KB cap, an invalid client id, no endpoint;
  - the SigV4 path building the boto3 parameter set and mapping ClientError;
  - the two knobs, ``STRANDS_MESH_IOT_DIRECT`` and ``STRANDS_IOT_DIRECT_AUTH``,
    and the inbound safety net that routes an unsubscribed direct message to
    this thing's ``cmd`` / ``response/#`` handlers while counting the rest.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.parse
from typing import Any

import pytest

from strands_robots.mesh.transport.base import DirectSender
from strands_robots.mesh.transport.iot_transport import (
    DIRECT_AUTH_ENV_VAR,
    DIRECT_ENV_VAR,
    DIRECT_PAYLOAD_CAP,
    IotMqttTransport,
    _SigV4DirectClient,
    _X509DirectClient,
    direct_auth_mode,
    direct_messaging_enabled,
)

_EP = "x-ats.iot.us-west-2.amazonaws.com"


class _Recorder(_X509DirectClient):
    """Stands in for ``_X509DirectClient``: records posts, answers a script."""

    def __init__(self, script: list[Any]) -> None:  # noqa: D107 - no socket is opened
        self.script = list(script)
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []
        self.deadlines: list[float] = []
        self.closed = False

    def post(self, path: str, body: bytes, headers: dict[str, str], *, deadline: float = 0.0) -> tuple[int, bytes]:
        self.calls.append((path, body, headers))
        self.deadlines.append(deadline)
        nxt = self.script.pop(0) if self.script else (200, b"")
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    def close(self) -> None:
        self.closed = True


def _transport(tmp_path, monkeypatch, script: list[Any], thing: str = "thor-arm") -> tuple[IotMqttTransport, _Recorder]:
    cert_dir = tmp_path / "iot"
    cert_dir.mkdir(exist_ok=True)
    (cert_dir / f"{thing}.cert.pem").write_text("cert")
    (cert_dir / f"{thing}.private.key").write_text("key")
    (cert_dir / "AmazonRootCA1.pem").write_text("ca")
    t = IotMqttTransport(thing_name=thing, endpoint=_EP, cert_dir=str(cert_dir))
    rec = _Recorder(script)
    t._direct_client = rec
    monkeypatch.setattr("strands_robots.mesh.transport.iot_transport.time.sleep", lambda s: None)
    return t, rec


def _query(path: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(path.split("?", 1)[1], keep_blank_values=True))


class TestRequestShape:
    def test_client_id_and_topic_are_url_encoded(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        r = t.send_direct("so 101/a", "strands/so 101/a/cmd", {"x": 1})
        assert r.delivered and r.reason == ""
        path, body, headers = rec.calls[0]
        assert path.startswith("/connections/so%20101%2Fa/messages?")
        q = _query(path)
        assert q["topic"] == "strands/so 101/a/cmd"
        assert "strands%2Fso%20101%2Fa%2Fcmd" in path
        assert q["contentType"] == "application/json"
        assert q["confirmation"] == "false"
        assert "timeout" not in q
        assert "responseTopic" not in q
        assert json.loads(body) == {"x": 1}

    def test_confirmation_carries_a_whole_second_timeout_of_at_least_one(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b""), (200, b""), (200, b"")])
        t.send_direct("p", "strands/p/cmd", {}, confirm=True, timeout=7.9)
        t.send_direct("p", "strands/p/cmd", {}, confirm=True, timeout=0.2)
        t.send_direct("p", "strands/p/cmd", {}, confirm=False, timeout=7.9)
        assert _query(rec.calls[0][0])["timeout"] == "7"
        assert _query(rec.calls[0][0])["confirmation"] == "true"
        assert _query(rec.calls[1][0])["timeout"] == "1"
        assert "timeout" not in _query(rec.calls[2][0])

    def test_response_topic_and_correlation_travel_as_mqtt5_properties(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        turn = "a" * 32
        t.send_direct("p", "strands/p/cmd", {}, response_key=f"strands/thor-arm/response/p/{turn}", correlation=turn)
        path, _body, headers = rec.calls[0]
        assert _query(path)["responseTopic"] == f"strands/thor-arm/response/p/{turn}"
        assert base64.b64decode(headers["x-amz-mqtt5-correlation-data"]).decode() == turn

    def test_fixed_headers(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        t.send_direct("p", "strands/p/cmd", {})
        headers = rec.calls[0][2]
        assert headers["Content-Type"] == "application/json"
        assert headers["x-amz-mqtt5-payload-format-indicator"] == "UTF8_DATA"
        # A JSON array of ONE-KEY objects: the API refuses a name/value shape.
        assert json.loads(base64.b64decode(headers["x-amz-mqtt5-user-properties"])) == [{"strands-mesh": "1"}]
        assert "x-amz-mqtt5-correlation-data" not in headers


class TestStatusMap:
    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (404, "offline"),
            (403, "forbidden"),
            (413, "too_large"),
            (504, "unconfirmed"),
            (400, "error"),
            (401, "forbidden"),
        ],
    )
    def test_terminal_statuses_are_not_retried(self, tmp_path, monkeypatch, status, reason):
        body = json.dumps({"message": "why", "traceId": "trace-1"}).encode()
        t, rec = _transport(tmp_path, monkeypatch, [(status, body)])
        r = t.send_direct("p", "strands/p/cmd", {})
        assert (r.delivered, r.reason, r.trace_id, r.detail) == (False, reason, "trace-1", "why")
        assert len(rec.calls) == 1

    def test_throttled_retries_once_then_reports(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(429, b"{}"), (429, b"{}")])
        r = t.send_direct("p", "strands/p/cmd", {})
        assert (r.delivered, r.reason) == (False, "throttled")
        assert len(rec.calls) == 2

    def test_throttled_then_ok_is_delivered(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(429, b"{}"), (200, b"")])
        r = t.send_direct("p", "strands/p/cmd", {})
        assert r.delivered
        assert len(rec.calls) == 2

    def test_server_error_retries_once(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(500, b'{"traceId":"t9"}'), (503, b"not json")])
        r = t.send_direct("p", "strands/p/cmd", {})
        assert (r.delivered, r.reason) == (False, "error")
        assert r.detail == "not json"
        assert len(rec.calls) == 2

    def test_socket_failure_retries_once_then_is_error(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [OSError("boom"), OSError("boom again")])
        r = t.send_direct("p", "strands/p/cmd", {})
        assert (r.delivered, r.reason) == (False, "error")
        assert "OSError" in r.detail
        assert len(rec.calls) == 2

    def test_forbidden_is_memoised_per_peer_and_cleared_on_connect(self, tmp_path, monkeypatch):
        t, _rec = _transport(tmp_path, monkeypatch, [(403, b"{}"), (200, b"")])
        assert not t.direct_forbidden("p")
        t.send_direct("p", "strands/p/cmd", {})
        assert t.direct_forbidden("p")
        assert not t.direct_forbidden("q")
        t.send_direct("q", "strands/q/cmd", {})
        assert not t.direct_forbidden("q")
        t._on_connection_success(object())
        assert not t.direct_forbidden("p")

    def test_stats_count_every_call(self, tmp_path, monkeypatch):
        t, _rec = _transport(tmp_path, monkeypatch, [(200, b""), (404, b"{}")])
        t.send_direct("p", "strands/p/cmd", {})
        t.send_direct("p", "strands/p/cmd", {})
        assert t.direct_stats == {"sent": 2, "delivered": 1, "failed": 1}

    def test_latency_is_measured(self, tmp_path, monkeypatch):
        t, _rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        assert t.send_direct("p", "strands/p/cmd", {}).latency_ms >= 0.0


class TestGuardsBeforeTheWire:
    def test_unencodable_payload_never_posts(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        r = t.send_direct("p", "strands/p/cmd", {"bad": object()})
        assert (r.delivered, r.reason) == (False, "error")
        assert "not JSON encodable" in r.detail
        assert rec.calls == []

    def test_payload_over_cap_is_too_large_without_a_round_trip(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        r = t.send_direct("p", "strands/p/cmd", {"blob": "x" * DIRECT_PAYLOAD_CAP})
        assert (r.delivered, r.reason) == (False, "too_large")
        assert rec.calls == []

    @pytest.mark.parametrize("peer", ["", "$aws", "x" * 129, None, 3])
    def test_invalid_client_id_is_refused_locally(self, tmp_path, monkeypatch, peer):
        t, rec = _transport(tmp_path, monkeypatch, [(200, b"")])
        r = t.send_direct(peer, "strands/p/cmd", {})
        assert (r.delivered, r.reason) == (False, "error")
        assert rec.calls == []

    def test_no_endpoint_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("STRANDS_IOT_ENDPOINT", raising=False)
        t = IotMqttTransport(thing_name="thor-arm", endpoint="", cert_dir=str(tmp_path))
        r = t.send_direct("p", "strands/p/cmd", {})
        assert (r.delivered, r.reason) == (False, "unavailable")

    def test_x509_requested_without_certificate_is_unavailable(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv(DIRECT_AUTH_ENV_VAR, "x509")
        t = IotMqttTransport(thing_name="thor-arm", endpoint=_EP, cert_dir=str(tmp_path / "empty"))
        with caplog.at_level(logging.WARNING):
            r = t.send_direct("p", "strands/p/cmd", {})
        assert r.reason == "unavailable"
        assert "no certificate" in caplog.text

    def test_close_drops_the_direct_client_and_the_memo(self, tmp_path, monkeypatch):
        t, rec = _transport(tmp_path, monkeypatch, [(403, b"{}")])
        t.send_direct("p", "strands/p/cmd", {})
        assert t.direct_forbidden("p")
        t.close()
        assert rec.closed
        assert not t.direct_forbidden("p")
        assert t._direct_client is None


class TestSigV4Path:
    def test_parameters_and_success(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []

        class _Boto:
            def send_direct_message(self, **kw: Any) -> dict[str, Any]:
                calls.append(kw)
                return {"traceId": "tr-1"}

        monkeypatch.setenv(DIRECT_AUTH_ENV_VAR, "sigv4")
        t = IotMqttTransport(thing_name="agent", endpoint=_EP, cert_dir=str(tmp_path / "none"))
        sender = t._direct_sender()
        assert isinstance(sender, _SigV4DirectClient)
        sender._client = _Boto()
        turn = "b" * 32
        r = t.send_direct(
            "p",
            "strands/p/cmd",
            {"a": 1},
            confirm=True,
            timeout=4.6,
            response_key="strands/agent/response/p/" + turn,
            correlation=turn,
        )
        assert r.delivered and r.trace_id == "tr-1"
        kw = calls[0]
        assert kw["clientId"] == "p"
        assert kw["topic"] == "strands/p/cmd"
        assert kw["contentType"] == "application/json"
        assert kw["payloadFormatIndicator"] == "UTF8_DATA"
        assert kw["userProperties"] == [{"strands-mesh": "1"}]
        assert kw["confirmation"] is True
        assert kw["timeout"] == 4
        assert kw["responseTopic"] == "strands/agent/response/p/" + turn
        assert base64.b64decode(kw["correlationData"]).decode() == turn
        assert json.loads(kw["payload"]) == {"a": 1}

    @pytest.mark.parametrize(
        ("code", "status", "reason"),
        [
            ("ResourceNotFoundException", 404, "offline"),
            ("ForbiddenException", 403, "forbidden"),
            ("ThrottlingException", 429, "throttled"),
            ("RequestEntityTooLargeException", 413, "too_large"),
            ("GatewayTimeoutException", 504, "unconfirmed"),
            ("InternalFailureException", 500, "error"),
        ],
    )
    def test_client_error_maps_to_reason(self, tmp_path, monkeypatch, code, status, reason):
        pytest.importorskip("botocore")
        from botocore.exceptions import ClientError

        class _Boto:
            def send_direct_message(self, **kw: Any) -> dict[str, Any]:
                raise ClientError(
                    {"Error": {"Code": code, "Message": "m"}, "ResponseMetadata": {"HTTPStatusCode": status}},
                    "SendDirectMessage",
                )

        monkeypatch.setenv(DIRECT_AUTH_ENV_VAR, "sigv4")
        monkeypatch.setattr("strands_robots.mesh.transport.iot_transport.time.sleep", lambda s: None)
        t = IotMqttTransport(thing_name="agent", endpoint=_EP, cert_dir=str(tmp_path / "none"))
        sender = t._direct_sender()
        sender._client = _Boto()  # type: ignore[union-attr]
        r = t.send_direct("p", "strands/p/cmd", {})
        assert (r.delivered, r.reason, r.detail) == (False, reason, "m")

    def test_default_is_sigv4_without_certificate_and_x509_with(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DIRECT_AUTH_ENV_VAR, raising=False)
        assert direct_auth_mode(cert_present=False) == "sigv4"
        assert direct_auth_mode(cert_present=True) == "x509"
        t = IotMqttTransport(thing_name="agent", endpoint=_EP, cert_dir=str(tmp_path / "none"))
        assert isinstance(t._direct_sender(), _SigV4DirectClient)


class TestKnobs:
    def test_direct_enabled_domain(self, monkeypatch, caplog):
        monkeypatch.delenv(DIRECT_ENV_VAR, raising=False)
        assert direct_messaging_enabled() is True
        monkeypatch.setenv(DIRECT_ENV_VAR, "1")
        assert direct_messaging_enabled() is True
        monkeypatch.setenv(DIRECT_ENV_VAR, "0")
        assert direct_messaging_enabled() is False
        monkeypatch.setenv(DIRECT_ENV_VAR, " 0 ")
        assert direct_messaging_enabled() is False

    @pytest.mark.parametrize("raw", ["true", "false", "yes", "off", "2", ""])
    def test_direct_enabled_refuses_other_spellings_loudly_and_stays_on(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(DIRECT_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING):
            assert direct_messaging_enabled() is True
        assert DIRECT_ENV_VAR in caplog.text
        assert "'0' or '1'" in caplog.text

    def test_auth_mode_domain(self, monkeypatch, caplog):
        monkeypatch.setenv(DIRECT_AUTH_ENV_VAR, "SigV4")
        assert direct_auth_mode(True) == "sigv4"
        monkeypatch.setenv(DIRECT_AUTH_ENV_VAR, "x509")
        assert direct_auth_mode(False) == "x509"
        monkeypatch.setenv(DIRECT_AUTH_ENV_VAR, "iam")
        with caplog.at_level(logging.WARNING):
            assert direct_auth_mode(True) == "x509"
        assert DIRECT_AUTH_ENV_VAR in caplog.text and "x509, sigv4" in caplog.text


class TestInboundSafetyNet:
    """A direct message needs no subscription, so unmatched topics are routed or counted."""

    def _connected(self, tmp_path):
        pytest.importorskip("awsiot")
        import awsiot.mqtt5_client_builder as builder

        from .test_iot_transport_session import _connect, _FakeMqtt5Client

        holder: dict[str, Any] = {"client": None, "auto_connack": True, "subscribe_exc": None}
        original = builder.mtls_from_path

        def fake(**kwargs):
            client = _FakeMqtt5Client(holder=holder, **kwargs)
            holder["client"] = client
            return client

        builder.mtls_from_path = fake
        try:
            t = _connect(tmp_path, thing="thor-arm")
            assert t.connect() is True
        finally:
            builder.mtls_from_path = original
        return t, holder["client"]

    def test_unsubscribed_cmd_and_response_reach_handlers_registered_for_those_keys(self, tmp_path):
        t, client = self._connected(tmp_path)
        cmd_seen: list[Any] = []
        resp_seen: list[Any] = []
        # Handlers registered under the direct keys, no broker subscription behind them.
        t._handlers["strands/thor-arm/cmd"] = [cmd_seen.append]
        t._handlers["strands/thor-arm/response/#"] = [resp_seen.append]
        client.fire_inbound("strands/thor-arm/cmd", b'{"a":1}')
        client.fire_inbound("strands/thor-arm/response/op/" + "0" * 32, b'{"b":2}')
        assert [s.key_expr for s in cmd_seen] == ["strands/thor-arm/cmd"]
        assert [s.key_expr for s in resp_seen] == ["strands/thor-arm/response/op/" + "0" * 32]
        assert t.unmatched_inbound == 0
        t.close()

    def test_other_unmatched_topics_are_counted_and_dropped(self, tmp_path, caplog):
        t, client = self._connected(tmp_path)
        t._handlers["strands/thor-arm/cmd"] = [lambda s: None]
        with caplog.at_level(logging.DEBUG, logger="strands_robots.mesh.transport.iot_transport"):
            client.fire_inbound("strands/other/state", b"{}")
            client.fire_inbound("strands/other/cmd", b"{}")  # another thing's cmd is not ours
        assert t.unmatched_inbound == 2
        assert "matched no subscription" in caplog.text
        t.close()

    def test_a_subscribed_topic_is_not_double_delivered(self, tmp_path):
        t, client = self._connected(tmp_path)
        seen: list[Any] = []
        t.declare_subscriber("strands/thor-arm/cmd", seen.append)
        client.fire_inbound("strands/thor-arm/cmd", b"{}")
        assert len(seen) == 1
        t.close()

    def test_the_transport_is_a_direct_sender(self, tmp_path):
        t = IotMqttTransport(thing_name="thor-arm", endpoint=_EP, cert_dir=str(tmp_path))
        assert isinstance(t, DirectSender)
