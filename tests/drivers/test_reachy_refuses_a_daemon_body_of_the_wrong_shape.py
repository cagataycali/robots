"""A Reachy daemon body is read as the JSON shape it arrived as, or refused.

:func:`~strands_robots.device_connect.reachy_transport.api` hands the decoded
body back unreshaped - ``json.loads`` decodes any JSON value, not only an
object - so what a caller receives is whatever the daemon, or an interposed
proxy, answered with. The rule for that is written down in this transport's
other consumer,
:meth:`~strands_robots.device_connect.reachy_mini_driver.ReachyMiniDriver._transport_failure`:
"the catalogue endpoint answers with a JSON array, so only the callers that
require an object judge that shape". Its own ``getDaemonStatus`` honours it.

The native driver requires an object at six sites and judged it at none. Every
one of them opens with ``result.get("error")``, so an array, a string, a number,
``null`` or ``true`` raised ``AttributeError: 'list' object has no attribute
'get'`` out of ``connect_eagerly``, ``wake_up``, ``goto_sleep``, ``stop_task``,
``play_move`` and ``stop`` - methods documented to report a reason and leave the
driver usable, and whose docstrings say ``api`` reports every failure as
``{"error": ...}`` "which is why no call here needs a ``try``".

The catalogue door had the mirror defect: ``list_moves`` graded dict-vs-not, the
wrong axis, so a scalar body came back as ``{"status": "success", "moves":
"ok"}`` - a catalogue that is not one.

Two doubles stand in. The premises class uses a **real** ``http.server`` on an
ephemeral port, because the table below is only meaningful if the transport
really does hand these shapes back; a double asked to return a list proves
nothing about ``api``. The door table then uses the daemon double this suite's
sibling module uses, since what is under test there is the driver's reading of a
decoded body rather than the decode.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.device_connect import reachy_transport
from strands_robots.drivers.reachy import ReachyDriver

#: A Lite status body: the variant flag the driver reads. A Lite needs no Zenoh
#: transport, so it is the cheapest variant to bring up.
_LITE_STATUS: dict[str, Any] = {"wireless_version": False}

#: Every JSON shape that is not an object, with the word a refusal must name it
#: by and the preview it must quote. ``true`` is carried separately from a number
#: because ``bool`` is a subclass of ``int``: a refusal that called it a number
#: would name the wrong shape.
_NOT_AN_OBJECT: tuple[tuple[Any, str, str], ...] = (
    (["wake_up", "nod"], "array", "['wake_up', 'nod']"),
    ("ok", "string", "'ok'"),
    (5, "number", "5"),
    (None, "null", "None"),
    (True, "boolean", "True"),
)


#: Sentinel for "leave the status body healthy". ``None`` cannot serve: it is
#: itself one of the shapes under test.
_KEEP_LITE: Any = object()


class _Link:
    """The minimum link the driver needs to consider itself connected."""

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Accept the sensor callbacks and start nothing."""

    async def stop(self) -> None:
        """Tear down nothing."""

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        """Discard one command; no cell here reads the wire."""


class _Daemon:
    """Answers every REST call with one configured body, recording each call.

    Attributes:
        calls: ``(path, method)`` for every call, in order.
    """

    def __init__(self, body: Any, *, status: Any = _KEEP_LITE) -> None:
        self._body = body
        self._status = _LITE_STATUS if status is _KEEP_LITE else status
        self.calls: list[tuple[str, str]] = []

    def __call__(self, host: str, port: int, path: str, method: str = "GET", data: Any = None) -> Any:
        """Answer one REST call: the status body for the probe, else the body."""
        del host, port, data
        self.calls.append((path, method))
        return self._status if path == reachy_mod._PATH_STATUS else self._body


def _install(monkeypatch: pytest.MonkeyPatch, daemon: _Daemon) -> ReachyDriver:
    """Build an unconnected driver wired to ``daemon`` and a no-op link."""
    monkeypatch.setattr("strands_robots.device_connect.reachy_transport.api", daemon)
    monkeypatch.setattr(ReachyDriver, "_build_link", lambda self, *, is_lite: _Link())
    return ReachyDriver(tool_name="reachy_mini", port="reachy-a.local")


def _connected(monkeypatch: pytest.MonkeyPatch, daemon: _Daemon) -> ReachyDriver:
    """Build a driver and connect it, asserting the probe succeeded."""
    driver = _install(monkeypatch, daemon)
    assert driver.connect_eagerly() is None
    return driver


def _text(value: Any) -> str:
    """Reduce a driver verdict - reason string or envelope - to its text."""
    if isinstance(value, str):
        return value
    assert isinstance(value, dict), f"expected an envelope or a reason, got {value!r}"
    return " ".join(block.get("text", "") for block in value.get("content", []))


#: The doors that require an object, as ``(label, method, path, call)``. ``call``
#: drives the door on a connected driver and hands back its verdict.
_OBJECT_DOORS: tuple[tuple[str, str, str, Any], ...] = (
    ("wake_up", "POST", reachy_mod._PATH_WAKE, lambda d: d.wake_up()),
    ("goto_sleep", "POST", reachy_mod._PATH_SLEEP, lambda d: d.goto_sleep()),
    ("stop_task", "POST", reachy_mod._PATH_STOP, lambda d: d.stop_task()),
    (
        "play_move",
        "POST",
        reachy_mod._PATH_MOVE_PLAY.format(dataset=reachy_mod._MOVE_LIBRARIES["emotions"], move="happy"),
        lambda d: d.play_move("happy"),
    ),
)


class TestThePremiseTheTransportHandsTheBodyBackUnreshaped:
    """What ``api`` really answers with, measured against a real daemon socket."""

    @pytest.fixture
    def daemon_port(self, request: pytest.FixtureRequest) -> Iterator[int]:
        """Serve one JSON body, taken from the requesting cell's parameter."""
        body: bytes = request.param

        class _Handler(BaseHTTPRequestHandler):
            def _reply(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            do_GET = _reply
            do_POST = _reply

            def log_message(self, *args: Any) -> None:
                """Keep the test output clean."""

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            server.server_close()

    @pytest.mark.parametrize(
        ("daemon_port", "decoded"),
        [pytest.param(json.dumps(body).encode(), body, id=kind) for body, kind, _ in _NOT_AN_OBJECT],
        indirect=["daemon_port"],
    )
    def test_a_non_object_body_arrives_at_the_caller_unreshaped(self, daemon_port: int, decoded: Any) -> None:
        assert reachy_transport.api("127.0.0.1", daemon_port, reachy_mod._PATH_STATUS) == decoded

    @pytest.mark.parametrize("daemon_port", [b"<html>502 Bad Gateway</html>"], indirect=True)
    def test_a_body_that_is_not_json_at_all_is_still_the_error_envelope(self, daemon_port: int) -> None:
        # The shape guards are additional to this path, not a replacement for it.
        result = reachy_transport.api("127.0.0.1", daemon_port, reachy_mod._PATH_STATUS)
        assert isinstance(result, dict) and "error" in result


class TestACallerThatNeedsAnObjectRefusesEveryOtherShape:
    """One reason per door, naming the call, the shape and the body."""

    @pytest.mark.parametrize(("body", "kind", "preview"), _NOT_AN_OBJECT)
    def test_the_probe_refuses_a_status_that_is_not_an_object(
        self, monkeypatch: pytest.MonkeyPatch, body: Any, kind: str, preview: str
    ) -> None:
        driver = _install(monkeypatch, _Daemon(body, status=body))

        reason = driver.connect_eagerly()

        assert reason is not None
        assert f"GET {reachy_mod._PATH_STATUS}" in reason
        assert f"a JSON {kind}, not an object" in reason
        assert preview in reason
        # The no-raise contract: a driver that refused a probe stays usable.
        assert driver._connected is False
        reported = asyncio.run(driver.get_status())["content"][0]["json"]
        assert reported["connected"] is False
        assert reported["connect_error"] == reason

    @pytest.mark.parametrize(
        ("label", "method", "path", "call"), _OBJECT_DOORS, ids=lambda v: v if isinstance(v, str) else ""
    )
    @pytest.mark.parametrize(("body", "kind", "preview"), _NOT_AN_OBJECT)
    def test_every_verb_that_reads_an_object_refuses_the_other_shapes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        method: str,
        path: str,
        call: Any,
        body: Any,
        kind: str,
        preview: str,
    ) -> None:
        driver = _connected(monkeypatch, _Daemon(body))

        text = _text(call(driver))

        assert text.startswith(f"{label}: "), "a caller with several verbs in flight has to know which one refused"
        assert f"{method} {path}" in text
        assert f"a JSON {kind}, not an object" in text
        assert preview in text

    @pytest.mark.parametrize(("body", "kind", "preview"), _NOT_AN_OBJECT)
    def test_the_mesh_stop_reports_the_shape_and_does_not_claim_a_stop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, body: Any, kind: str, preview: str
    ) -> None:
        # ``stop`` answers ``None`` either way, so the log is the only place its
        # refusal can be read - and a stop it never got must not be recorded.
        driver = _connected(monkeypatch, _Daemon(body))

        with caplog.at_level("WARNING"):
            assert asyncio.run(driver.stop()) is None

        assert f"POST {reachy_mod._PATH_STOP}" in caplog.text
        assert f"a JSON {kind}, not an object" in caplog.text
        assert preview in caplog.text
        assert driver._stopped is False

    def test_a_long_body_is_previewed_rather_than_quoted_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _connected(monkeypatch, _Daemon(list(range(500))))

        text = _text(driver.wake_up())

        assert "a JSON array, not an object" in text
        assert text.endswith("...")
        assert len(text) < 200, "a refusal ends up on one log line"


class TestOnlyAJsonArrayIsReadAsACatalogue:
    """The catalogue door grades the axis that decides what it returns."""

    @pytest.mark.parametrize(
        ("body", "preview"),
        [pytest.param(body, preview, id=kind) for body, kind, preview in _NOT_AN_OBJECT if kind != "array"],
    )
    def test_a_body_that_is_not_an_array_is_never_returned_as_the_catalogue(
        self, monkeypatch: pytest.MonkeyPatch, body: Any, preview: str
    ) -> None:
        driver = _connected(monkeypatch, _Daemon(body))

        result = driver.list_moves("emotions")

        assert result["status"] == "error"
        assert "not an array" in _text(result)
        assert preview in _text(result)

    def test_an_array_body_is_the_catalogue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _connected(monkeypatch, _Daemon(["happy", "sad"]))

        result = driver.list_moves("emotions")

        assert result["status"] == "success"
        assert result["content"][0]["json"]["moves"] == ["happy", "sad"]

    def test_the_transports_own_failure_still_reaches_the_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The shape guard must not stand in front of the reason the daemon was
        # not reached: that cause is the one an operator can act on.
        driver = _connected(monkeypatch, _Daemon({"error": "Connection refused"}))

        text = _text(driver.list_moves("emotions"))

        assert "Connection refused" in text
        assert "not an array" not in text, "a daemon that was never reached did not answer with a shape"


class TestTheHealthyPathIsUnchanged:
    """An object body still reads as an object, at every door."""

    def test_the_probe_accepts_an_object_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _Daemon({"ok": True}))

        assert driver.connect_eagerly() is None
        assert driver._connected is True

    @pytest.mark.parametrize(
        ("label", "method", "path", "call"), _OBJECT_DOORS, ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_every_verb_still_succeeds_on_an_object_body(
        self, monkeypatch: pytest.MonkeyPatch, label: str, method: str, path: str, call: Any
    ) -> None:
        del label, method, path
        driver = _connected(monkeypatch, _Daemon({"ok": True}))

        assert call(driver)["status"] == "success"
