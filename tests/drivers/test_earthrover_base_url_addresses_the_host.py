"""The EarthRover base URL has to address the host the caller wrote.

Every endpoint the driver speaks is built from one string, so the host that
string resolves to is the rover that moves. Two spellings resolved to a
*different* host than the one they name, and neither is refused by the
transport -- it reports only the host it ended up with:

* ``bot.local@10.0.0.9:8001`` -- everything before the ``@`` is userinfo, so
  ``10.0.0.9`` is dialled while the address still reads as ``bot.local``.
* ``ws://10.0.0.9:8001`` -- no lowercase ``http`` prefix to recognise, so it was
  prefixed into ``http://ws://10.0.0.9:8001``, whose authority is ``ws:``: the
  request went to the host ``ws`` on port 80 and the configured port was
  discarded.

Unlike its sibling module these tests use a **real** ``http.server`` on an
ephemeral port rather than a ``requests`` double, because the property under
test is which host the transport actually connects to -- a double would answer
whatever it was asked and could not tell "dialled the host we named" from
"dialled a different one".
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from strands_robots.drivers.earthrover import (
    DEFAULT_SDK_URL,
    EarthRoverDriver,
    base_url_error,
)
from strands_robots.utils import dial_host_error

requests = pytest.importorskip("requests")

_TELEMETRY = {"battery": 88, "signal_level": 3, "lamp": 0}


class _Recorder(BaseHTTPRequestHandler):
    """Answer the two endpoints a connect-and-drive needs, recording each hit."""

    received: list[tuple[str, str]] = []

    def log_message(self, *args: Any) -> None:
        """Silence the stdlib access log."""

    def _answer(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
        """Record and serve ``GET /data``."""
        type(self).received.append(("GET", self.path))
        self._answer(dict(_TELEMETRY))

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming
        """Record and acknowledge ``POST /control``."""
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).received.append(("POST", self.path))
        self._answer({})


@pytest.fixture
def sdk() -> Iterator[tuple[int, list[tuple[str, str]]]]:
    """A real SDK-shaped server on ``127.0.0.1``; yields its port and its log.

    It stands in for "something is listening", which is the condition that turns
    a misaddressed base URL from a failed connection into a drive command
    delivered to the wrong rover.
    """
    _Recorder.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1], _Recorder.received
    finally:
        server.shutdown()
        server.server_close()


# Spellings whose authority names one host and dials another. Each is
# (value template, the fragment the refusal must name).
_MISADDRESSED: tuple[tuple[str, str], ...] = (
    ("bot.local@127.0.0.1:{port}", "userinfo"),
    ("http://bot.local@127.0.0.1:{port}", "userinfo"),
    ("http://user:pw@127.0.0.1:{port}", "userinfo"),
    ("ws://127.0.0.1:{port}", "'ws'"),
    ("wss://127.0.0.1:{port}", "'wss'"),
)


class TestTheAuthorityMustNameTheHostThatIsDialled:
    @pytest.mark.parametrize(("template", "needle"), _MISADDRESSED)
    def test_a_misaddressed_base_is_refused_by_name(self, template: str, needle: str) -> None:
        value = template.format(port=8001)
        reason = base_url_error(value, "port", "EarthRoverDriver")
        assert reason is not None, f"{value!r} names one host and dials another, so it is not a base URL"
        assert needle in reason, reason
        assert value in reason, reason

    @pytest.mark.parametrize(("template", "needle"), _MISADDRESSED)
    def test_the_constructor_refuses_before_a_session_exists(self, template: str, needle: str) -> None:
        with pytest.raises(ValueError, match="port"):
            EarthRoverDriver(port=template.format(port=8001))

    @pytest.mark.parametrize(("template", "needle"), _MISADDRESSED)
    def test_no_request_reaches_the_listening_server(
        self, template: str, needle: str, sdk: tuple[int, list[tuple[str, str]]]
    ) -> None:
        """The refusal lands before the wire, so nothing is driven by mistake."""
        port, received = sdk
        with pytest.raises(ValueError):
            EarthRoverDriver(port=template.format(port=port), timeout_s=0.5)
        assert received == [], f"a refused address still reached the server: {received}"


class TestThePremiseIsWhatTheTransportDoes:
    """These hold on any tree: they measure ``requests``, not the driver."""

    def test_userinfo_makes_requests_dial_the_host_after_the_at_sign(
        self, sdk: tuple[int, list[tuple[str, str]]]
    ) -> None:
        port, received = sdk
        response = requests.get(f"http://bot.local@127.0.0.1:{port}/data", timeout=5)
        assert response.status_code == 200
        assert received == [("GET", "/data")], (
            "the request named bot.local and was served by 127.0.0.1, which is why "
            "the authority has to be graded before it is dialled"
        )

    def test_a_double_scheme_makes_requests_dial_the_scheme_token(self) -> None:
        with pytest.raises(requests.exceptions.ConnectionError) as caught:
            requests.get("http://ws://127.0.0.1:8001/data", timeout=0.5)
        assert "host='ws'" in str(caught.value), str(caught.value)
        assert "port=80" in str(caught.value), "the configured port is discarded, not just the host"


class TestTheAcceptedShapesAreUntouched:
    @pytest.mark.parametrize(
        ("port", "base"),
        [
            (None, DEFAULT_SDK_URL),
            ("http://10.0.0.9:8001", "http://10.0.0.9:8001"),
            ("10.0.0.9:8001", "http://10.0.0.9:8001"),
            ("http://10.0.0.9:8001/", "http://10.0.0.9:8001"),
            ("https://rover.local:8001", "https://rover.local:8001"),
            ("10.0.0.9:8001/rover-7", "http://10.0.0.9:8001/rover-7"),
            ("http://[::1]:8001", "http://[::1]:8001"),
            ("0.0.0.0:8001", "http://0.0.0.0:8001"),
            ("http://rover.local", "http://rover.local"),
            ("HTTP://10.0.0.9:8001", "HTTP://10.0.0.9:8001"),
        ],
        ids=[
            "default",
            "http",
            "bare-host-port",
            "trailing-slash",
            "https",
            "path-prefix",
            "ipv6-literal",
            "every-interface",
            "no-port",
            "uppercase-scheme",
        ],
    )
    def test_a_usable_base_still_constructs(self, port: str | None, base: str) -> None:
        assert EarthRoverDriver(port=port)._base == base


class TestTheSharedHostDomainIsTheWrongToolHere:
    """Why ``dial_host_error`` is not asked, pinned rather than asserted in prose."""

    def test_it_refuses_the_host_inside_a_legitimate_ipv6_base(self) -> None:
        assert dial_host_error("::1", "port", "EarthRoverDriver") is not None, (
            "if this ever becomes None the shared domain could be asked here directly"
        )

    def test_and_that_base_url_is_accepted(self) -> None:
        assert EarthRoverDriver(port="http://[::1]:8001")._base == "http://[::1]:8001"


class TestTheTransportKeepsTheCasesItAlreadyNames:
    @pytest.mark.parametrize(
        "value",
        ["http://", "http://localhost:99999", "http://localhost:8001 x"],
        ids=["scheme-only", "port-out-of-range", "embedded-space"],
    )
    def test_an_unusable_url_the_transport_reports_is_left_to_it(self, value: str) -> None:
        assert base_url_error(value, "port", "EarthRoverDriver") is None
        driver = EarthRoverDriver(port=value, timeout_s=0.5)
        reason = driver.connect_eagerly()
        assert reason is not None, f"{value!r} is unusable, so connect must report a reason"
        assert "Invalid URL" in reason or "Failed to parse" in reason, reason
        assert driver._base in reason, "the reason names the URL the driver actually tried"


class TestTheUppercaseSchemeReachesTheHostItNames:
    def test_it_connects_and_drives(self, sdk: tuple[int, list[tuple[str, str]]]) -> None:
        """``HTTP://`` used to be prefixed into a URL whose host was ``http``."""
        port, received = sdk
        driver = EarthRoverDriver(port=f"HTTP://127.0.0.1:{port}", timeout_s=5.0)
        assert driver.connect_eagerly() is None
        assert driver.send_action({"linear": 0.5})["status"] == "success"
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["connected"] is True
        assert status["battery_pct"] == _TELEMETRY["battery"]
        driver.cleanup()
        assert ("GET", "/data") in received
        assert ("POST", "/control") in received


class TestTheSchemeOwnerIsShared:
    def test_the_accepted_schemes_are_the_ones_the_validator_admits(self) -> None:
        # Imported here rather than at module scope so this module still
        # collects against a tree without the constant, which is what lets the
        # cells above report the defect instead of a collection error.
        from strands_robots.drivers.earthrover import ACCEPTED_SCHEMES  # noqa: PLC0415

        assert ACCEPTED_SCHEMES == ("http", "https")
        for scheme in ACCEPTED_SCHEMES:
            assert base_url_error(f"{scheme}://rover.local:8001", "port", "t") is None
