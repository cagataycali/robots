#!/usr/bin/env python3
"""The direct send honours the caller's budget and never serialises behind one peer.

Five critics measured the previous tip: a ``timeout=0.5`` send took 1.51 s
(confirmation window plus a fresh wait), the kept HTTPS connection carried a
fixed 15 s socket timeout and was reused half-open after a socket timeout, and
one lock across request and PUBACK wait made three concurrent posts finish at
2, 4 and 6 s. This module pins the fixes with a fake ``HTTPSConnection``:

  - the whole call fits in ``timeout``: the confirmation window is derived from
    the remaining budget and clamped to [1, 10], the socket timeout is the
    remaining budget, a retry is skipped when less than a quarter second is left;
  - a connection that failed in any way (timeout, TLS, reset) is closed and never
    handed out again; a stale idle connection is replaced and the request
    repeated once;
  - two posts in flight at once run concurrently, and up to four idle
    connections are kept.
"""

from __future__ import annotations

import http.client
import threading
import time
from typing import Any

import pytest

from strands_robots.mesh.transport import iot_transport as mod
from strands_robots.mesh.transport.iot_transport import (
    _POOL_SIZE,
    IotMqttTransport,
    _X509DirectClient,
)

_EP = "x-ats.iot.us-west-2.amazonaws.com"


class _FakeResp:
    def __init__(self, status: int, body: bytes = b"", will_close: bool = False) -> None:
        self.status = status
        self._body = body
        self.will_close = will_close

    def read(self) -> bytes:
        return self._body


class _FakeSock:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeConn:
    """Stands in for ``http.client.HTTPSConnection``; behaviour scripted per instance."""

    made: list[_FakeConn] = []
    script: list[Any] = []  # per connection creation: callable(conn) -> response, or exception
    barrier: threading.Barrier | None = None

    def __init__(self, host: str, port: int, *, context: Any = None) -> None:
        self.host, self.port = host, port
        self.timeout: float | None = None
        self.sock: Any = None
        self.closed = False
        self.requests: list[tuple[str, str]] = []
        self.behaviour = _FakeConn.script.pop(0) if _FakeConn.script else (lambda c: _FakeResp(200))
        _FakeConn.made.append(self)

    def request(self, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.requests.append((method, path))
        if self.sock is None:
            self.sock = _FakeSock()  # "connected" from now on
        b = self.behaviour
        if isinstance(b, BaseException):
            raise b
        self._pending = b

    def getresponse(self) -> _FakeResp:
        if _FakeConn.barrier is not None:
            _FakeConn.barrier.wait(timeout=5)
        r = self._pending(self)
        if isinstance(r, BaseException):
            raise r
        return r

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_conn(monkeypatch):
    _FakeConn.made = []
    _FakeConn.script = []
    _FakeConn.barrier = None
    monkeypatch.setattr(mod.http.client, "HTTPSConnection", _FakeConn)
    monkeypatch.setattr(mod.ssl, "create_default_context", lambda cafile=None: _Ctx())
    monkeypatch.setattr("strands_robots.mesh.transport.iot_transport.time.sleep", lambda s: None)
    return _FakeConn


class _Ctx:
    minimum_version = None

    def load_cert_chain(self, cert: str, key: str) -> None:
        pass


def _client() -> _X509DirectClient:
    return _X509DirectClient(_EP, "c.pem", "k.pem", "ca.pem")


class TestBudget:
    def test_socket_timeout_is_the_remaining_budget_not_a_constant(self, fake_conn):
        c = _client()
        deadline = time.monotonic() + 2.0
        c.post("/p", b"{}", {}, deadline=deadline)
        conn = fake_conn.made[0]
        assert conn.timeout is not None and 1.5 < conn.timeout <= 2.0

    def test_a_spent_budget_makes_no_request(self, fake_conn):
        c = _client()
        with pytest.raises(TimeoutError):
            c.post("/p", b"{}", {}, deadline=time.monotonic() - 0.01)
        assert fake_conn.made == []

    def test_send_direct_fits_in_its_timeout_when_the_broker_hangs(self, fake_conn, tmp_path):
        # The broker never answers: the socket timeout fires at the budget.
        def _hang(conn: _FakeConn) -> Any:
            time.sleep(max(0.0, conn.timeout or 0))
            return TimeoutError("timed out")

        fake_conn.script = [_hang, _hang]
        t = _transport(tmp_path)
        t0 = time.monotonic()
        r = t.send_direct("p", "strands/p/cmd", {}, confirm=True, timeout=0.5)
        elapsed = time.monotonic() - t0
        assert not r.delivered and r.reason == "error" and "Timeout" in r.detail
        assert elapsed <= 0.5 + 0.15, elapsed
        # No second attempt: after the timeout nothing of the budget was left.
        assert len(fake_conn.made) == 1

    def test_confirmation_window_is_the_remaining_budget_clamped(self, fake_conn, tmp_path):
        t = _transport(tmp_path)
        for timeout, expected in ((0.4, "1"), (3.7, "3"), (30.0, "10")):
            fake_conn.made.clear()
            t.send_direct("p", "strands/p/cmd", {}, confirm=True, timeout=timeout)
            path = fake_conn.made[-1].requests[0][1] if fake_conn.made else _last_path(t)
            assert f"timeout={expected}" in path, (timeout, path)

    def test_a_retry_is_skipped_when_a_quarter_second_is_not_left(self, fake_conn, tmp_path):
        fake_conn.script = [lambda c: _FakeResp(429, b"{}"), lambda c: _FakeResp(200)]
        t = _transport(tmp_path)
        r = t.send_direct("p", "strands/p/cmd", {}, timeout=0.2)
        assert r.reason == "throttled"
        assert sum(len(c.requests) for c in fake_conn.made) == 1


def _last_path(t: IotMqttTransport) -> str:
    # The pool reuses one connection; read its latest request.
    client = t._direct_client
    assert isinstance(client, _X509DirectClient)
    conns = client._idle
    return conns[-1].requests[-1][1]  # type: ignore[attr-defined]


def _transport(tmp_path) -> IotMqttTransport:
    cert_dir = tmp_path / "iot"
    cert_dir.mkdir(exist_ok=True)
    for name in ("thor-arm.cert.pem", "thor-arm.private.key", "AmazonRootCA1.pem"):
        (cert_dir / name).write_text("x")
    return IotMqttTransport(thing_name="thor-arm", endpoint=_EP, cert_dir=str(cert_dir))


class TestConnectionHygiene:
    def test_a_timed_out_connection_is_closed_and_not_reused(self, fake_conn):
        fake_conn.script = [lambda c: TimeoutError("read timed out"), lambda c: _FakeResp(200)]
        c = _client()
        with pytest.raises(TimeoutError):
            c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)
        assert fake_conn.made[0].closed
        c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)
        assert len(fake_conn.made) == 2  # a fresh connection, never the half-open one

    def test_a_tls_error_drops_the_connection(self, fake_conn):
        import ssl

        fake_conn.script = [lambda c: ssl.SSLError("bad record mac")]
        c = _client()
        with pytest.raises(ssl.SSLError):
            c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)
        assert fake_conn.made[0].closed
        assert c._idle == []

    def test_a_stale_idle_connection_is_replaced_and_the_request_repeated_once(self, fake_conn):
        c = _client()
        c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)  # now idle
        first = fake_conn.made[0]
        first.behaviour = lambda conn: http.client.RemoteDisconnected("closed by broker")
        fake_conn.script = [lambda conn: _FakeResp(200, b"ok")]
        status, body = c.post("/p2", b"{}", {}, deadline=time.monotonic() + 1)
        assert (status, body) == (200, b"ok")
        assert first.closed
        assert len(fake_conn.made) == 2
        # The reused connection had its live socket re-timed to the new budget.
        assert first.sock.timeouts and 0.5 < first.sock.timeouts[-1] <= 1.0

    def test_a_reset_on_a_fresh_connection_is_not_retried_here(self, fake_conn):
        fake_conn.script = [ConnectionResetError("reset")]
        c = _client()
        with pytest.raises(ConnectionResetError):
            c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)
        assert len(fake_conn.made) == 1

    def test_a_response_that_closes_is_not_pooled(self, fake_conn):
        fake_conn.script = [lambda c: _FakeResp(200, will_close=True)]
        c = _client()
        c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)
        assert c._idle == [] and fake_conn.made[0].closed

    def test_close_empties_the_pool(self, fake_conn):
        c = _client()
        c.post("/p", b"{}", {}, deadline=time.monotonic() + 1)
        assert len(c._idle) == 1
        c.close()
        assert c._idle == [] and fake_conn.made[0].closed


class TestNoSerialisation:
    def test_two_posts_are_in_flight_at_once(self, fake_conn):
        # Both threads must reach getresponse before either returns: a lock
        # held across the request would deadlock this barrier (timeout=5).
        fake_conn.barrier = threading.Barrier(2)
        c = _client()
        results: list[Any] = []

        def _go() -> None:
            results.append(c.post("/p", b"{}", {}, deadline=time.monotonic() + 5)[0])

        threads = [threading.Thread(target=_go) for _ in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=6)
        assert results == [200, 200]
        assert len(c._idle) == 2

    def test_at_most_pool_size_idle_connections_are_kept(self, fake_conn):
        fake_conn.barrier = threading.Barrier(_POOL_SIZE + 2)
        c = _client()
        threads = [
            threading.Thread(target=lambda: c.post("/p", b"{}", {}, deadline=time.monotonic() + 5))
            for _ in range(_POOL_SIZE + 2)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=6)
        assert len(c._idle) == _POOL_SIZE
        assert sum(1 for conn in fake_conn.made if conn.closed) == 2
