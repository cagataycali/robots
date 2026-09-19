"""``Robot("reachy_mini", mode="real")`` with no address finds its daemon.

Before this, ``port=None`` meant ``localhost`` and nothing else, so a Wireless
Mini on the desk needed ``port="reachy-mini.local:8000"`` spelled out - and a
Mac that also runs a desktop Lite daemon on ``:8000`` (in ``state: error``
because it saw unrelated USB-serial devices) connected to THAT and reported it
unreachable. Measured 2026-09-18 on the owner's Mac against daemon 1.10.0.

What is graded here, all off the network - the transport is a double installed
through :func:`strands_robots.drivers.reachy._resolve_transport`:

* the discovery order and the "first usable daemon wins" rule, including the
  skip of a daemon that answers with its own error;
* ``REACHY_HOST``/``REACHY_PORT`` pinning a single address;
* an explicit ``port=`` never discovering;
* the reason a failed discovery returns naming every candidate;
* ``probe_hardware()`` and the factory's ``mode="auto"`` consulting it;
* the agent path connecting lazily on the first verb that needs the daemon,
  and ``status`` never connecting.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers.reachy import ReachyDriver

_RUNNING_WIRELESS: dict[str, Any] = {"state": "running", "wireless_version": True, "version": "1.10.0", "error": None}
_RUNNING_LITE: dict[str, Any] = {"state": "running", "wireless_version": False, "error": None}
_DESKTOP_LITE_IN_ERROR: dict[str, Any] = {
    "state": "error",
    "wireless_version": False,
    "desktop_app_daemon": True,
    "error": "Multiple Reachy Mini serial ports found ['/dev/cu.usbmodemA', '/dev/cu.usbmodemB'].",
}
_REFUSED: dict[str, Any] = {"error": "<urlopen error [Errno 61] Connection refused>"}


class _HostTable:
    """A transport double that answers ``/api/daemon/status`` per host and records every call."""

    def __init__(self, by_host: dict[str, Any]) -> None:
        self.by_host = by_host
        self.calls: list[tuple[str, int, str, str]] = []

    def api(self, host: str, port: int, path: str, method: str = "GET", data: Any = None) -> Any:
        del data
        self.calls.append((host, port, path, method))
        if path == reachy_mod._PATH_STATUS:
            return dict(self.by_host.get(host, _REFUSED))
        if "/recorded-move-datasets/list/" in path:
            return ["cheerful1", "curious1"]
        return {"ok": True}

    @staticmethod
    def rpy_to_pose(*args: Any) -> list[list[float]]:
        del args
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class _Link:
    """A link that starts and stops and does nothing else."""

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Accept the driver's callbacks and never call them."""

    async def stop(self) -> None:
        return None

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        """Swallow a command; this link grades discovery, not the wire."""


def _wire(monkeypatch: pytest.MonkeyPatch, by_host: dict[str, Any]) -> _HostTable:
    """Install the host table as the transport and a no-op link, clear the address environment."""
    table = _HostTable(by_host)
    monkeypatch.setattr(reachy_mod, "_resolve_transport", lambda: table)
    monkeypatch.setattr(ReachyDriver, "_build_link", lambda self, *, is_lite: _Link())
    monkeypatch.delenv("REACHY_HOST", raising=False)
    monkeypatch.delenv("REACHY_PORT", raising=False)
    return table


def _probed_hosts(table: _HostTable) -> list[str]:
    return [host for host, _, path, _ in table.calls if path == reachy_mod._PATH_STATUS]


class TestAZeroArgumentDriverDiscoversItsDaemon:
    """``port=None`` with no environment probes localhost, then reachy-mini.local."""

    def test_the_wireless_on_the_lan_is_found_behind_a_dead_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver()
        assert driver._discover_host is True
        assert driver.connect_eagerly() is None
        assert (driver._host, driver._api_port) == ("reachy-mini.local", 8000)
        assert driver._variant == "wireless"
        assert _probed_hosts(table) == ["localhost", "reachy-mini.local"]

    def test_a_running_localhost_daemon_wins_without_a_second_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"localhost": _RUNNING_LITE, "reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver()
        assert driver.connect_eagerly() is None
        assert driver._host == "localhost"
        assert driver._variant == "lite"
        assert _probed_hosts(table) == ["localhost"]

    def test_a_desktop_daemon_in_error_is_skipped_not_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The measured Mac case: a Lite desktop daemon on :8000 that found no
        # robot. It answers, but it serves nothing - the Wireless behind it does.
        _wire(monkeypatch, {"localhost": _DESKTOP_LITE_IN_ERROR, "reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver()
        assert driver.connect_eagerly() is None
        assert driver._host == "reachy-mini.local"

    def test_no_daemon_anywhere_names_every_candidate_and_the_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {"localhost": _DESKTOP_LITE_IN_ERROR})
        driver = ReachyDriver()
        reason = driver.connect_eagerly()
        assert reason is not None
        assert reason.startswith("daemon unreachable (localhost:8000, reachy-mini.local:8000)")
        assert "Multiple Reachy Mini serial ports" in reason
        assert "Connection refused" in reason
        assert 'port="host[:port]"' in reason and "REACHY_HOST" in reason
        assert driver._connected is False
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["connect_error"] == reason
        assert status["discovery"] is True

    def test_the_api_port_keyword_reprices_every_candidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver(api_port=9100)
        assert driver.connect_eagerly() is None
        assert [(h, p) for h, p, path, _ in table.calls if path == reachy_mod._PATH_STATUS] == [
            ("localhost", 9100),
            ("reachy-mini.local", 9100),
        ]
        assert driver._api_port == 9100

    def test_a_second_connect_does_not_discover_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver()
        assert driver.connect_eagerly() is None
        probes = len(_probed_hosts(table))
        assert driver.connect_eagerly() is None
        assert len(_probed_hosts(table)) == probes


class TestTheEnvironmentPinsOneAddress:
    """``REACHY_HOST``/``REACHY_PORT`` read as if the caller had passed ``port=``."""

    def test_reachy_host_disables_discovery_and_is_the_only_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-b.local": _RUNNING_WIRELESS, "localhost": _RUNNING_LITE})
        monkeypatch.setenv("REACHY_HOST", "reachy-b.local")
        driver = ReachyDriver()
        assert driver._discover_host is False
        assert (driver._host, driver._api_port) == ("reachy-b.local", 8000)
        assert driver.connect_eagerly() is None
        assert _probed_hosts(table) == ["reachy-b.local"]

    def test_reachy_port_supplies_the_port_for_a_bare_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {})
        monkeypatch.setenv("REACHY_HOST", "10.0.0.7")
        monkeypatch.setenv("REACHY_PORT", "9000")
        assert (ReachyDriver()._host, ReachyDriver()._api_port) == ("10.0.0.7", 9000)

    def test_a_suffix_in_reachy_host_beats_reachy_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {})
        monkeypatch.setenv("REACHY_HOST", "10.0.0.7:8001")
        monkeypatch.setenv("REACHY_PORT", "9000")
        assert ReachyDriver()._api_port == 8001

    def test_an_unusable_reachy_port_is_refused_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {})
        monkeypatch.setenv("REACHY_HOST", "10.0.0.7")
        monkeypatch.setenv("REACHY_PORT", "70000")
        with pytest.raises(ValueError, match="REACHY_PORT"):
            ReachyDriver()

    def test_an_explicit_port_argument_beats_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {})
        monkeypatch.setenv("REACHY_HOST", "reachy-b.local")
        driver = ReachyDriver(port="reachy-a.local:8000")
        assert driver._host == "reachy-a.local"
        assert driver._discover_host is False

    def test_an_unreachable_explicit_address_is_named_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver(port="reachy-a.local:8000")
        reason = driver.connect_eagerly()
        assert reason is not None
        assert reason.startswith("daemon unreachable (reachy-a.local:8000)")
        assert "reachy-mini.local" not in reason
        assert _probed_hosts(table) == ["reachy-a.local"]


class TestTheFactoryAutoModeAsksTheDriver:
    """``probe_hardware()`` and ``Robot(..., mode="auto")``."""

    def test_probe_hardware_is_true_when_a_candidate_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        assert ReachyDriver.probe_hardware() is True
        assert _probed_hosts(table) == ["localhost", "reachy-mini.local"]

    def test_probe_hardware_is_false_for_a_daemon_in_error_and_for_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wire(monkeypatch, {"localhost": _DESKTOP_LITE_IN_ERROR})
        assert ReachyDriver.probe_hardware() is False
        _wire(monkeypatch, {})
        assert ReachyDriver.probe_hardware() is False

    def test_probe_hardware_only_reads_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"localhost": _RUNNING_LITE})
        ReachyDriver.probe_hardware()
        assert {(path, method) for _, _, path, method in table.calls} == {(reachy_mod._PATH_STATUS, "GET")}

    def test_probe_hardware_is_false_when_the_transport_cannot_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reachy_mod, "_resolve_transport", lambda: "cannot import transport")
        assert ReachyDriver.probe_hardware() is False

    def test_auto_mode_becomes_real_when_the_daemon_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots import robot as robot_mod

        _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr(robot_mod, "scan_serial_devices", list)
        assert robot_mod._auto_detect_mode("reachy_mini") == "real"

    def test_auto_mode_stays_sim_when_nothing_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots import robot as robot_mod

        _wire(monkeypatch, {"localhost": _DESKTOP_LITE_IN_ERROR})
        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr(robot_mod, "scan_serial_devices", list)
        assert robot_mod._auto_detect_mode("reachy_mini") == "sim"

    def test_a_probe_that_raises_is_no_hardware_not_a_failed_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots import robot as robot_mod

        def _boom() -> bool:
            raise RuntimeError("probe exploded")

        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr(robot_mod, "scan_serial_devices", list)
        monkeypatch.setattr(ReachyDriver, "probe_hardware", staticmethod(_boom))
        assert robot_mod._auto_detect_mode("reachy_mini") == "sim"

    def test_a_driver_without_a_probe_is_not_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots import robot as robot_mod

        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr(robot_mod, "scan_serial_devices", list)
        monkeypatch.setattr(robot_mod, "get_native_driver_class", lambda canonical: object)
        assert robot_mod._auto_detect_mode("reachy_mini") == "sim"

    def test_a_daemon_that_does_not_answer_is_asked_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One detection is one probe: a second ask doubles the wait for a silent daemon."""
        from strands_robots import robot as robot_mod

        calls = 0

        def _counting() -> bool:
            nonlocal calls
            calls += 1
            return False

        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr(robot_mod, "scan_serial_devices", list)
        monkeypatch.setattr(ReachyDriver, "probe_hardware", staticmethod(_counting))
        assert robot_mod._auto_detect_mode("reachy_mini") == "sim"
        assert calls == 1


def _run_tool(driver: ReachyDriver, **params: Any) -> dict[str, Any]:
    async def _drive() -> dict[str, Any]:
        results = [
            event async for event in driver.stream({"name": driver.tool_name, "toolUseId": "t1", "input": params}, {})
        ]
        assert len(results) == 1
        return results[0]

    return asyncio.run(_drive())


class TestTheAgentPathConnectsLazily:
    """``Agent(tools=[Robot("reachy_mini", mode="real")])`` needs no connect_eagerly line."""

    def test_the_first_verb_connects_and_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver()
        assert driver._connected is False
        result = _run_tool(driver, action="list_moves")
        assert result["status"] == "success"
        assert driver._connected is True

    def test_status_never_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        driver = ReachyDriver()
        result = _run_tool(driver, action="status")
        assert result["status"] == "success"
        # The verb nests the whole ``get_status`` envelope in one json block.
        assert result["content"][0]["json"]["content"][0]["json"]["connected"] is False
        assert driver._connected is False
        assert table.calls == []

    def test_a_failed_lazy_connect_is_the_verbs_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire(monkeypatch, {})
        driver = ReachyDriver()
        result = _run_tool(driver, action="sensors")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith("sensors: daemon unreachable")
        assert driver._connected is False

    def test_an_undeclared_verb_is_refused_before_any_connect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _wire(monkeypatch, {"reachy-mini.local": _RUNNING_WIRELESS})
        result = _run_tool(ReachyDriver(), action="teleport")
        assert result["status"] == "error"
        assert table.calls == []
