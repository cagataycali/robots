"""After an awscrt reconnect the broker holds no subscriptions (clean session): the transport must re-issue them.

Reproduced live 2026-09-30 (account 947951559549, Thing crit-r1): 3/3 e-stops
received before a broker DISCONNECT, awscrt reconnected in 1.35 s, 0/3 after,
while ``_handlers`` still listed ``strands/safety/estop`` and ``is_alive()``
was True. A robot that dropped its MQTT session once was deaf to every
pub/sub topic (safety/estop, safety/resume, broadcast, presence) for the rest
of its life, with nothing above DEBUG to say so.
"""

from __future__ import annotations

import logging
import threading
import types
from typing import Any

import pytest

from strands_robots.mesh.transport.iot_transport import IotMqttTransport

LOGGER = "strands_robots.mesh.transport.iot_transport"


class _Suback:
    """What awscrt resolves the subscribe future with: a SubackPacket, never a raise for a refusal."""

    def __init__(self, code: int) -> None:
        self.reason_codes = [code]


class _Future:
    def __init__(self, suback: _Suback) -> None:
        self._suback = suback

    def result(self, timeout: float | None = None) -> _Suback:
        return self._suback


class _Client:
    def __init__(self, fail: bool = False) -> None:
        self.subscribed: list[list[str]] = []
        self.fail = fail
        self.seen = threading.Event()

    def subscribe(self, packet: Any) -> _Future:
        self.subscribed.append([s.topic_filter for s in packet.subscriptions])
        self.seen.set()
        # 135 = not authorized, the code a policy gap produces (measured live, critic probe c09).
        return _Future(_Suback(135 if self.fail else 1))


def _connack(session_present: bool | None) -> Any:
    if session_present is None:
        return types.SimpleNamespace()  # no connack at all: treat as a fresh session
    return types.SimpleNamespace(connack_packet=types.SimpleNamespace(session_present=session_present))


@pytest.fixture
def transport() -> IotMqttTransport:
    pytest.importorskip("awscrt")
    t = IotMqttTransport(thing_name="ac-arm-01", endpoint="x-ats.iot.us-west-2.amazonaws.com")
    t._client = _Client()
    t._connected.set()
    return t


def _client_of(transport: IotMqttTransport) -> _Client:
    client = transport._client
    assert isinstance(client, _Client)
    return client


def _subscribed(transport: IotMqttTransport, *keys: str) -> None:
    for key in keys:
        transport.declare_subscriber(key, lambda sample: None)
    _client_of(transport).subscribed.clear()
    _client_of(transport).seen.clear()


def _wait_for_resubscribe(transport: IotMqttTransport) -> None:
    assert _client_of(transport).seen.wait(timeout=2.0), "no subscribe was re-issued after the reconnect"
    transport.wait_for_resubscribe(timeout=2.0)


class TestAReconnectWithoutASessionReissuesEveryFilter:
    def test_every_registered_filter_is_subscribed_again(self, transport, caplog):
        _subscribed(
            transport, "strands/safety/estop", "strands/safety/resume", "strands/*/presence", "strands/ac-arm-01/cmd"
        )
        with caplog.at_level(logging.INFO, logger=LOGGER):
            transport._on_connection_success(_connack(session_present=False))
            _wait_for_resubscribe(transport)
        reissued = sorted(f for batch in _client_of(transport).subscribed for f in batch)
        assert reissued == sorted(
            ["strands/safety/estop", "strands/safety/resume", "strands/+/presence", "strands/ac-arm-01/cmd"]
        )
        (w,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert "re-subscribed 4 topic filters" in w and "strands/safety/estop" in w and "(thing=ac-arm-01)" in w

    def test_a_connack_without_the_flag_is_treated_as_a_fresh_session(self, transport):
        _subscribed(transport, "strands/safety/estop")
        transport._on_connection_success(_connack(session_present=None))
        _wait_for_resubscribe(transport)
        assert _client_of(transport).subscribed == [["strands/safety/estop"]]

    def test_the_first_connect_has_nothing_to_reissue_and_stays_quiet(self, transport, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            transport._on_connection_success(_connack(session_present=False))
            transport.wait_for_resubscribe(timeout=1.0)
        assert _client_of(transport).subscribed == []
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_rejoined_session_keeps_its_subscriptions_and_is_not_touched(self, transport, caplog):
        _subscribed(transport, "strands/safety/estop")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            transport._on_connection_success(_connack(session_present=True))
            transport.wait_for_resubscribe(timeout=1.0)
        assert _client_of(transport).subscribed == []
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_refused_resubscribe_is_an_error_naming_the_filter_not_a_silent_success(self, transport, caplog):
        _subscribed(transport, "strands/safety/estop")
        _client_of(transport).fail = True
        with caplog.at_level(logging.INFO, logger=LOGGER):
            transport._on_connection_success(_connack(session_present=False))
            _wait_for_resubscribe(transport)
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and "strands/safety/estop" in errors[0] and "SUBACK reason code 135" in errors[0]
        assert "this peer will not hear it" in errors[0]
        assert not [r for r in caplog.records if r.levelno == logging.WARNING], (
            "a refused filter is not 're-subscribed'"
        )

    def test_declare_subscriber_reads_the_suback_verdict_too(self, transport):
        _client_of(transport).fail = True
        with pytest.raises(RuntimeError, match="SUBACK reason code 135"):
            transport.declare_subscriber("strands/safety/estop", lambda sample: None)
        assert "strands/safety/estop" not in transport._handlers, "a refused filter leaves no handler behind"

    def test_the_reissue_runs_off_the_lifecycle_callback_thread(self, transport):
        # awscrt runs lifecycle callbacks on its event-loop thread; a blocking
        # subscribe().result() there deadlocks the client. The reissue must
        # therefore not happen synchronously inside the callback.
        _subscribed(transport, "strands/safety/estop")
        client = _client_of(transport)
        caller = threading.current_thread()
        seen_on: list[threading.Thread] = []
        original = client.subscribe

        def subscribe(packet: Any) -> _Future:
            seen_on.append(threading.current_thread())
            return original(packet)

        client.subscribe = subscribe  # type: ignore[method-assign]
        transport._on_connection_success(_connack(session_present=False))
        _wait_for_resubscribe(transport)
        assert seen_on and all(t is not caller for t in seen_on)
