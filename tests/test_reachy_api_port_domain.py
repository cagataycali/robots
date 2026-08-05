"""``ReachyMiniDriver``'s ``api_port`` is the shared TCP port domain.

The driver is a public export documented with a constructor example, and its
``api_port`` is the only thing that addresses the Reachy Mini daemon. It is
interpolated verbatim into two targets the module builds itself - the REST URL
in :func:`strands_robots.device_connect.reachy_transport.api` and the Lite
WebSocket target in ``WebSocketLink`` - and nothing downstream refuses it:

* ``api`` reports every failure as an ``{"error": ...}`` result rather than
  raising, so a port outside ``[1, 65535]`` is reported as an unreachable
  daemon. The message is byte-identical to the one a reachable port produces
  with the daemon down, so the two cannot be told apart.
* ``connect()`` derives the Wireless-vs-Lite variant from that result with
  ``not status.get("wireless_version", True)``. A result carrying only
  ``error`` therefore reads as Wireless, so an unusable port silently selects
  the Zenoh link and ``connect()`` logs a successful connection.

:func:`strands_robots.utils.tcp_port_error` is the shared domain for every
caller-supplied port number, and its docstring gives the reason this surface
needs it: a lazily-connecting transport makes the range load-bearing at the
boundary rather than at the socket. The constructor is where the caller names
the port, so it is the only point a caller can act on.

``TestWhyTheConstructorOwnsTheDomain`` pins the premises above rather than
asserting them in prose - they pass on both trees, and they are what makes the
constructor, and not a downstream check, the right owner. The fail-safe that
treats an unreachable daemon as Wireless is deliberately unchanged: a genuinely
down daemon is not a caller mistake, and a test here pins that it still holds.
"""

from __future__ import annotations

import ast
import asyncio
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import strands_robots
from strands_robots.utils import tcp_port_error
from tests.test_reachy_mini_driver import _force_real_device_connect_edge

# Values that cannot address a TCP port. ``True`` is included because it is an
# ``int`` subclass: a bare range test reads it as a silent port 1.
UNUSABLE_PORTS: list[Any] = [
    0,
    -1,
    65536,
    99999,
    True,
    False,
    2.7,
    8000.0,
    "8000",
    float("nan"),
    float("inf"),
    None,
    [8000],
]

USABLE_PORTS = [1, 8000, 9001, 65535]


@pytest.fixture
def rmd():
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    _force_real_device_connect_edge()
    import strands_robots.device_connect.reachy_mini_driver as module

    return module


def _connect_capturing_urls(rmd, monkeypatch, api_port: Any) -> tuple[list[str], list[str]]:
    """Construct a driver, ``connect()`` it, and report what it reached for.

    Returns the URLs that reached ``urllib`` and the hardware links that were
    constructed. Both hardware links are replaced so nothing dials, and the
    daemon probe is refused so the auto-detection takes its error branch -
    which is the branch an unusable port lands on.
    """
    urls: list[str] = []
    links: list[str] = []

    def spy(req, body=None, timeout=None, **kwargs):
        urls.append(req.full_url)
        raise urllib.error.URLError("test: never dialed")

    class _FakeZenoh:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            links.append("zenoh")

        async def start(self, **kwargs: Any) -> None:
            return None

    class _FakeWebSocket:
        def __init__(self, host: str, port: Any) -> None:
            links.append(f"websocket:{port!r}")

        async def start(self, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", spy)
    monkeypatch.setattr(rmd, "ZenohLink", _FakeZenoh)
    monkeypatch.setattr(rmd, "WebSocketLink", _FakeWebSocket)

    driver = rmd.ReachyMiniDriver(host="bot.local", api_port=api_port)
    asyncio.run(driver.connect())
    return urls, links


class TestApiPortDomain:
    """The constructor accepts exactly the ports the shared domain accepts."""

    @pytest.mark.parametrize("port", UNUSABLE_PORTS, ids=repr)
    def test_an_unusable_api_port_is_refused_at_construction(self, rmd, port):
        """A value that cannot address a port is refused where it is named."""
        with pytest.raises(ValueError, match="api_port"):
            rmd.ReachyMiniDriver(host="bot.local", api_port=port)

    @pytest.mark.parametrize("port", USABLE_PORTS)
    def test_a_usable_api_port_is_accepted_and_stored(self, rmd, port):
        """Every port in range still constructs and is carried verbatim."""
        assert rmd.ReachyMiniDriver(host="bot.local", api_port=port)._api_port == port

    def test_the_default_api_port_is_usable(self, rmd):
        """The documented default must satisfy the domain it now enforces."""
        assert rmd.ReachyMiniDriver(host="bot.local")._api_port == 8000

    def test_the_refusal_names_the_parameter_and_the_class(self, rmd):
        """The message says which parameter, on which class, and the range."""
        with pytest.raises(ValueError) as excinfo:
            rmd.ReachyMiniDriver(host="bot.local", api_port=99999)
        assert str(excinfo.value) == "ReachyMiniDriver: invalid api_port: 99999 (expected 1-65535)"

    def test_a_boolean_is_refused_rather_than_read_as_port_one(self, rmd):
        """``True`` is an ``int``, so a bare range test would accept it as 1."""
        with pytest.raises(ValueError, match=r"invalid api_port: True"):
            rmd.ReachyMiniDriver(host="bot.local", api_port=True)

    @pytest.mark.parametrize("port", UNUSABLE_PORTS + USABLE_PORTS, ids=repr)
    def test_the_accepted_domain_is_the_shared_tcp_port_domain(self, rmd, port):
        """The driver refuses a port iff the shared domain refuses it.

        Asserted as an equivalence so the two cannot drift: the same port must
        not be refused by one surface that addresses a service and accepted by
        the next.
        """
        shared_refuses = tcp_port_error(port, "api_port", "ReachyMiniDriver") is not None
        try:
            rmd.ReachyMiniDriver(host="bot.local", api_port=port)
            driver_refuses = False
        except ValueError:
            driver_refuses = True
        assert driver_refuses is shared_refuses


class TestWhyTheConstructorOwnsTheDomain:
    """Premises for the guard's placement. These hold on either tree."""

    @pytest.mark.parametrize("port", [99999, True, 2.7, None, float("nan")], ids=repr)
    def test_the_daemon_url_interpolates_the_port_verbatim(self, monkeypatch, port):
        """``api`` builds ``http://host:<port>/path`` with no coercion."""
        from strands_robots.device_connect import reachy_transport

        captured: list[str] = []

        def spy(req, body=None, timeout=None, **kwargs):
            captured.append(req.full_url)
            raise urllib.error.URLError("test: never dialed")

        monkeypatch.setattr(urllib.request, "urlopen", spy)
        reachy_transport.api("bot.local", port, "/api/daemon/status")
        assert captured == [f"http://bot.local:{port}/api/daemon/status"]

    @pytest.mark.parametrize("port", [99999, True, None], ids=repr)
    def test_the_websocket_target_interpolates_the_port_verbatim(self, port):
        """The Lite link carries the same value into its ``ws://`` target."""
        from strands_robots.device_connect import reachy_transport

        link = reachy_transport.WebSocketLink("bot.local", port)
        assert link._port is port

    def test_api_reports_a_failure_as_a_result_rather_than_raising(self, monkeypatch):
        """So no exception from an unusable port can reach the caller."""
        from strands_robots.device_connect import reachy_transport

        def refuse(req, body=None, timeout=None, **kwargs):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        result = reachy_transport.api("bot.local", 99999, "/api/daemon/status")
        assert "error" in result

    def test_an_unusable_port_reads_as_wireless_in_the_auto_detection(self):
        """``not {"error": ...}.get("wireless_version", True)`` is ``False``.

        This is the expression ``connect()`` derives the variant from, so a
        probe that could not succeed selects the Wireless branch. It is why an
        unusable port had to be refused before it reached the probe.
        """
        is_lite = not {"error": "connection refused"}.get("wireless_version", True)
        assert is_lite is False


class TestTheRefusalPrecedesAnyState:
    """A refused port allocates nothing and reaches nothing."""

    def test_a_refused_port_allocates_no_base_driver_state(self, rmd, monkeypatch):
        """The guard runs before ``DeviceDriver.__init__``."""
        calls: list[int] = []
        original = rmd.DeviceDriver.__init__

        def recording_init(self, *args: Any, **kwargs: Any) -> None:
            calls.append(1)
            original(self, *args, **kwargs)

        monkeypatch.setattr(rmd.DeviceDriver, "__init__", recording_init)

        with pytest.raises(ValueError):
            rmd.ReachyMiniDriver(host="bot.local", api_port=99999)
        assert calls == []

        rmd.ReachyMiniDriver(host="bot.local", api_port=8000)
        assert calls == [1]

    def test_a_refused_port_never_reaches_the_daemon(self, rmd, monkeypatch):
        """No URL is built, because the driver cannot be constructed."""
        with pytest.raises(ValueError, match="api_port"):
            _connect_capturing_urls(rmd, monkeypatch, 99999)

    def test_a_usable_port_still_probes_the_daemon(self, rmd, monkeypatch):
        """Control: the probe the guard protects still runs unchanged."""
        urls, _ = _connect_capturing_urls(rmd, monkeypatch, 9001)
        assert urls == ["http://bot.local:9001/api/daemon/status"]

    def test_an_unreachable_daemon_still_falls_back_to_wireless(self, rmd, monkeypatch):
        """The documented fail-safe is unchanged for a port that names one.

        A daemon that is down is not a caller mistake, so this guard must not
        turn it into one.
        """
        _, links = _connect_capturing_urls(rmd, monkeypatch, 8000)
        assert links == ["zenoh"]


def _exported_names(package_init: Path) -> list[str]:
    """The names listed in ``__all__`` of a package's ``__init__``."""
    for node in ast.parse(package_init.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        if not isinstance(node.value, ast.List | ast.Tuple):
            continue
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    raise AssertionError(f"no __all__ in {package_init}")


def _exported_port_constructors(source: str, exported: list[str]) -> dict[str, list[str]]:
    """Map each exported class to the port-ish ``__init__`` params it declares.

    Scoped to classes the package exports, because that is the surface a caller
    constructs: the hardware links are built only from an already-validated
    ``api_port``, so requiring them to re-check it would institutionalize a
    second copy of the rule.
    """
    found: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name not in exported:
            continue
        for member in node.body:
            if not isinstance(member, ast.FunctionDef) or member.name != "__init__":
                continue
            ports = [
                arg.arg
                for arg in member.args.args + member.args.kwonlyargs
                if arg.arg == "port" or arg.arg.endswith("_port")
            ]
            if ports:
                found[node.name] = ports
    return found


def _validates_port(source: str, class_name: str) -> bool:
    """True when the class's ``__init__`` calls the shared port domain."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    return any(
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "tcp_port_error"
                        for call in ast.walk(member)
                    )
    return False


class TestNoExportedDeviceConnectPortSurfaceDrifts:
    """Every exported driver that takes a port routes it through one domain."""

    @staticmethod
    def _package_dir() -> Path:
        return Path(strands_robots.__file__).parent / "device_connect"

    def _surfaces(self) -> dict[str, tuple[Path, list[str]]]:
        package = self._package_dir()
        exported = _exported_names(package / "__init__.py")
        surfaces: dict[str, tuple[Path, list[str]]] = {}
        for module in sorted(package.rglob("*.py")):
            source = module.read_text(encoding="utf-8")
            for class_name, ports in _exported_port_constructors(source, exported).items():
                surfaces[class_name] = (module, ports)
        return surfaces

    def test_the_scan_finds_the_known_port_surface(self):
        """Non-vacuity: a scan resolving elsewhere would report nothing."""
        assert {name: ports for name, (_, ports) in self._surfaces().items()} == {"ReachyMiniDriver": ["api_port"]}

    def test_every_exported_port_constructor_validates_it(self):
        """A future exported driver cannot address a daemon unvalidated."""
        adrift = {
            name: ports
            for name, (module, ports) in self._surfaces().items()
            if not _validates_port(module.read_text(encoding="utf-8"), name)
        }
        assert adrift == {}, f"exported constructors taking a port without the shared domain: {adrift}"

    def test_the_scan_detects_a_planted_unguarded_port(self):
        """Meta: an empty result must mean clean sources, not a dead scanner."""
        planted = "class Planted:\n    def __init__(self, api_port: int = 8000):\n        self._p = api_port\n"
        assert _exported_port_constructors(planted, ["Planted"]) == {"Planted": ["api_port"]}
        assert not _validates_port(planted, "Planted")
