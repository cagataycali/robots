"""An unrecognised ``STRANDS_MESH_BACKEND`` is reported, once, with its value.

:func:`strands_robots.mesh.transport.factory._select_backend` falls back to
``zenoh`` for a value it does not recognise. The fallback is correct - a peer
should still come up - but it used to be reported in a form an operator could
not act on: the message did not name the accepted set, and it quoted the value
after normalisation rather than as it was typed, so it was not greppable in the
configuration that produced it.

That matters because the fallback is silent in every other channel: a peer
configured for a cloud transport through a typo comes up on the LAN default and
``init_mesh`` reports a live mesh, so the log line is the only signal that the
requested backend was not the one built.

One environment variable has two readers, and the reporting one was not the one
on the caller's path. :func:`strands_robots.mesh.session._backend_choice` gates
``get_session`` / ``put`` / ``session_alive``, so a typo answers there first and
the peer takes the legacy Zenoh route without the factory being consulted at
all -- which is why the factory's existing warning could not fire for it. Both
readers now build one message through
:func:`~strands_robots.mesh.transport.factory.unknown_backend_message`.

The value-resolution assertions for this reader live in
``tests/mesh/test_transport_factory.py``; this module pins the reporting
contract - what the message contains, its level, and that it is emitted once per
distinct value rather than on every resolution.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import pytest

from strands_robots.mesh.transport import factory
from strands_robots.mesh.transport.base import MeshTransport

_ENV = "STRANDS_MESH_BACKEND"
_LOGGER = "strands_robots.mesh.transport.factory"
_VALID = ("zenoh", "iot", "bridge")


@pytest.fixture(autouse=True)
def _clear_report_latch() -> Any:
    """Give each case an unreported process.

    The latch is module state, so without this a case's report depends on
    whichever case ran first.
    """
    factory._REPORTED_UNKNOWN_BACKENDS.clear()
    yield
    factory._REPORTED_UNKNOWN_BACKENDS.clear()


def _reports(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Records this reader emitted, newest last."""
    return [r for r in caplog.records if r.name == _LOGGER]


class TestTheFallbackIsReported:
    """An unrecognised value produces an actionable record."""

    def test_the_report_names_the_variable_the_value_and_the_accepted_set(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All three are needed to act on it without reading the source."""
        monkeypatch.setenv(_ENV, "iot-direct")
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert factory._select_backend() == "zenoh"

        reported = _reports(caplog)
        assert reported, "the fallback left no record naming the rejected value"
        message = reported[0].getMessage()
        assert _ENV in message
        assert "iot-direct" in message
        for name in _VALID:
            assert name in message, f"the accepted set omits {name!r}: {message}"

    def test_the_value_is_quoted_as_it_was_typed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A normalised echo is not greppable in the configuration that set it."""
        monkeypatch.setenv(_ENV, "  IoT-Direct  ")
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            factory._select_backend()

        reported = _reports(caplog)
        assert reported, "no record to inspect"
        assert "IoT-Direct" in reported[0].getMessage()

    def test_the_report_is_at_warning(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """Every other channel reports success, so this must not be debug."""
        monkeypatch.setenv(_ENV, "nope")
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            factory._select_backend()

        reported = _reports(caplog)
        assert reported, "no record to inspect"
        assert reported[0].levelno >= logging.WARNING, logging.getLevelName(reported[0].levelno)

    def test_a_recognised_backend_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reporting an honoured value would make the record meaningless."""
        monkeypatch.setenv(_ENV, "zenoh")
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            assert factory._select_backend() == "zenoh"
        assert _reports(caplog) == []

    def test_an_unset_variable_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The documented default is not a fallback from a caller's mistake."""
        monkeypatch.delenv(_ENV, raising=False)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            assert factory._select_backend() == "zenoh"
        assert _reports(caplog) == []


class TestTheReportIsOncePerValue:
    """The reader runs per construction, so an unlatched report is a flood."""

    def test_repeated_resolution_reports_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One misconfiguration is one operator action, not one per call."""
        monkeypatch.setenv(_ENV, "iot-direct")
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            for _ in range(25):
                assert factory._select_backend() == "zenoh"
        assert len(_reports(caplog)) == 1

    def test_a_second_distinct_value_reports_again(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Latching on the value, not on the process, keeps a later typo visible."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            monkeypatch.setenv(_ENV, "iot-direct")
            factory._select_backend()
            monkeypatch.setenv(_ENV, "zenho")
            factory._select_backend()

        messages = [r.getMessage() for r in _reports(caplog)]
        assert len(messages) == 2, messages
        assert any("iot-direct" in m for m in messages)
        assert any("zenho" in m for m in messages)

    def test_a_failing_log_handler_does_not_re_arm_the_report(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The latch is the record of having reported, not of having logged.

        A handler that raises must not turn one misconfiguration into a report on
        every later resolution.
        """

        class _Failing(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                raise RuntimeError("handler is down")

        logger = logging.getLogger(_LOGGER)
        handler = _Failing()
        logger.addHandler(handler)
        try:
            monkeypatch.setenv(_ENV, "iot-direct")
            with contextlib.suppress(Exception):
                factory._select_backend()
            assert factory._REPORTED_UNKNOWN_BACKENDS, "the report was not recorded as made"
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                for _ in range(5):
                    with contextlib.suppress(Exception):
                        factory._select_backend()
            assert _reports(caplog) == []
        finally:
            logger.removeHandler(handler)


class TestTheAcceptedSetTracksTheConstructor:
    """The reported set is the set this module can actually build."""

    def test_the_accepted_set_is_exactly_the_documented_backends(self) -> None:
        """A backend added to the constructor and not here fails this."""
        assert factory._VALID_BACKENDS == _VALID

    def test_every_accepted_backend_builds_a_distinct_transport(self) -> None:
        """A name accepted but mapped to the same transport is not a backend.

        A backend whose optional dependency is absent raises ``ImportError``,
        which is a different and honest outcome, so this runs on a minimal
        install too.
        """
        built: dict[str, type] = {}
        for name in factory._VALID_BACKENDS:
            try:
                transport = factory._construct(name)
            except ImportError:
                continue  # optional dependency absent on this install
            try:
                assert isinstance(transport, MeshTransport), name
                built[name] = type(transport)
            finally:
                with contextlib.suppress(Exception):
                    transport.close()

        assert built, "no accepted backend was constructible on this install"
        counts = list(built.values())
        collisions = {n: c.__name__ for n, c in built.items() if counts.count(c) > 1}
        assert not collisions, f"backends sharing one transport class: {collisions}"
