"""``IotMqttTransport`` records every MQTT5 client teardown it could not finish.

The transport calls ``self._client.stop()`` from four places, all of them a
teardown of a client it is about to stop referencing:

* the reconnect path, stopping a stale client left by a broker drop;
* the construction-failure path in :meth:`~IotMqttTransport.connect`;
* the connect-timeout path in the same method;
* :meth:`~IotMqttTransport.close`.

``stop()`` tears down an IO thread and a socket, so it is exactly the call that
fails when the thing it is tearing down is already broken - which is the state
each of these four paths is reacting to. The first two contain that and log at
debug, and the construction-failure one states the policy in a comment: the
connect has already failed, so *"a stop() error here ... must not mask the
original failure. Log at debug and move on."*

The other two did not follow it. The timeout path called ``stop()`` unwrapped,
so a raising teardown left :meth:`connect` - documented to return ``False`` for
a broker that is unreachable within ``connect_timeout`` - raising instead, with
``self._client`` still set and its IO thread still running. ``close()`` swallowed
the same failure into a bare ``pass`` and then logged *"IoT mesh session
closed"*, which is the only line an operator gets and says the opposite of what
happened; because ``close()`` drops the reference either way, nothing can reach
that client afterwards to retry.

The sibling module :mod:`tests.mesh.test_iot_reconnect_client_lifecycle` already
pins the two synchronous raise sources on the connect path - ``mtls_from_path``
and ``start()`` - with the same reasoning ("the mesh stays OFF rather than
crashing the host"). ``stop()`` is the third, and it was the one left out.
"""

from __future__ import annotations

import ast
import inspect
import logging
from typing import Any

import pytest

import strands_robots.mesh.transport.iot_transport as iot_transport
from strands_robots.mesh.transport.iot_transport import IotMqttTransport

from .test_iot_reconnect_client_lifecycle import _FakeClient, _make_certs

LOGGER_NAME = "strands_robots.mesh.transport.iot_transport"
STOP_FAILURE = "io thread already gone"


class _StopRaises(_FakeClient):
    """A client whose teardown fails - the state each path is reacting to."""

    def stop(self) -> None:
        self.stopped = True  # the attempt happened; the teardown did not finish
        raise RuntimeError(STOP_FAILURE)


class _NoConnack(_FakeClient):
    """Starts without ever firing CONNACK, so ``connect()`` times out."""

    def start(self) -> None:
        self.started = True


class _NoConnackStopRaises(_NoConnack):
    def stop(self) -> None:
        self.stopped = True
        raise RuntimeError(STOP_FAILURE)


def _transport(tmp_path: Any, thing: str = "thor-arm") -> IotMqttTransport:
    return IotMqttTransport(
        thing_name=thing,
        endpoint="x-ats.iot.us-west-2.amazonaws.com",
        cert_dir=str(_make_certs(tmp_path, thing)),
        connect_timeout=0.05,
    )


def _install(monkeypatch: pytest.MonkeyPatch, cls: type[_FakeClient]) -> list[_FakeClient]:
    """Make ``mtls_from_path`` build *cls*; return the list of built clients."""
    import awsiot.mqtt5_client_builder as builder

    built: list[_FakeClient] = []

    def fake_mtls_from_path(**kwargs: Any) -> _FakeClient:
        client = cls(**kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(builder, "mtls_from_path", fake_mtls_from_path)
    return built


class TestConnectTimeoutContainsTheTeardownFailure:
    """A timeout whose teardown fails is still reported as a timeout."""

    def test_a_stop_failure_on_the_timeout_path_still_returns_false(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = _install(monkeypatch, _NoConnackStopRaises)
        transport = _transport(tmp_path)

        assert transport.connect() is False
        assert built[0].stopped is True, "the teardown was not attempted"

    def test_the_client_reference_is_dropped_when_the_timeout_teardown_fails(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, _NoConnackStopRaises)
        transport = _transport(tmp_path)

        transport.connect()

        # Left set, the next connect() would reach the reconnect path with a
        # client this one already failed to stop.
        assert transport._client is None
        assert transport.is_alive() is False

    def test_the_timeout_teardown_failure_is_recorded_at_debug(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _install(monkeypatch, _NoConnackStopRaises)
        transport = _transport(tmp_path)

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            transport.connect()

        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(STOP_FAILURE in m for m in debug), f"teardown failure unrecorded: {debug}"
        # The timeout itself is still the headline report.
        assert any("timed out" in r.getMessage() for r in caplog.records)

    def test_a_clean_timeout_records_no_teardown_failure(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        built = _install(monkeypatch, _NoConnack)
        transport = _transport(tmp_path)

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert transport.connect() is False

        assert built[0].stopped is True
        assert not [r for r in caplog.records if STOP_FAILURE in r.getMessage()]


class TestCloseRecordsTheTeardownFailure:
    """``close()`` is the only teardown whose visible report is a success."""

    def test_a_stop_failure_during_close_is_recorded(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _install(monkeypatch, _StopRaises)
        transport = _transport(tmp_path)
        assert transport.connect() is True

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            transport.close()

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a failed client teardown left no trace at all"
        assert any(STOP_FAILURE in m and "thor-arm" in m for m in warnings), warnings

    def test_close_completes_when_stop_fails(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, _StopRaises)
        transport = _transport(tmp_path)
        assert transport.connect() is True
        transport._handlers["strands/*/state"] = [lambda _sample: None]

        transport.close()  # must not raise: teardown is best-effort by contract

        assert transport._client is None
        assert transport._handlers == {}
        assert transport.is_alive() is False

    def test_a_clean_close_warns_about_nothing(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        built = _install(monkeypatch, _FakeClient)
        transport = _transport(tmp_path)
        assert transport.connect() is True

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            transport.close()

        assert built[0].stopped is True
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestThePreExistingTeardownsStillRecord:
    """The two connect-path teardowns that already followed the policy."""

    def test_the_reconnect_stale_stop_failure_is_recorded(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        built = _install(monkeypatch, _StopRaises)
        transport = _transport(tmp_path)
        assert transport.connect() is True
        transport._connected.clear()  # broker-side drop; the client object lingers

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert transport.connect() is True

        assert len(built) == 2, "the reconnect did not build a fresh client"
        assert built[0].stopped is True
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(STOP_FAILURE in m for m in debug), debug

    def test_the_construction_failure_teardown_is_recorded(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _StartRaisesStopRaises(_StopRaises):
            def start(self) -> None:
                raise RuntimeError("io thread failed to launch")

        built = _install(monkeypatch, _StartRaisesStopRaises)
        transport = _transport(tmp_path)

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert transport.connect() is False

        assert transport._client is None
        assert built[0].stopped is True
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(STOP_FAILURE in m for m in debug), debug


def _teardown_handlers(source: str) -> list[tuple[int, bool]]:
    """Every ``try`` whose body stops the client, with whether it logs.

    Returns ``(lineno, logs)`` per handler so a site that tolerates a teardown
    failure without recording it is visible as ``logs=False``.
    """
    tree = ast.parse(source)
    found: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = "\n".join(ast.get_source_segment(source, stmt) or "" for stmt in node.body)
        if "_client.stop()" not in body:
            continue
        for handler in node.handlers:
            text = "\n".join(ast.get_source_segment(source, stmt) or "" for stmt in handler.body)
            found.append((handler.lineno, "logger." in text))
    return sorted(found)


class TestEveryClientTeardownRecordsItsFailure:
    """One rule for all four sites, so a fifth cannot ship without it."""

    def test_every_client_teardown_handler_logs(self) -> None:
        source = inspect.getsource(iot_transport)
        silent = [lineno for lineno, logs in _teardown_handlers(source) if not logs]
        assert not silent, f"client teardown failure tolerated without a record at line(s) {silent}"

    def test_the_module_still_has_the_four_teardown_sites(self) -> None:
        source = inspect.getsource(iot_transport)
        assert len(_teardown_handlers(source)) == 4, _teardown_handlers(source)
        assert source.count("_client.stop()") == 4

    def test_the_scanner_sees_a_planted_silent_swallow(self) -> None:
        planted = (
            "class T:\n"
            "    def close(self):\n"
            "        try:\n"
            "            self._client.stop()\n"
            "        except Exception:\n"
            "            pass\n"
        )
        assert _teardown_handlers(planted) == [(5, False)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
