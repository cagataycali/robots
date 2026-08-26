"""Each Reachy transport guard is graded where it is the only thing protecting.

The driver resolves its transport through
:func:`~strands_robots.drivers.reachy._resolve_transport` at four sites, and
``test_reachy_driver.py`` pins the refusal through ``connect_eagerly``,
``send_action`` and ``stop``. That leaves two of the four ungraded, because
``connect_eagerly`` resolves the transport *before* it probes the daemon: with the
pre-check in place neither ``_daemon_get`` nor ``_build_link`` is ever reached with
the extra absent, so a test that drives ``connect_eagerly`` cannot tell whether
those two guards are there at all. Measured against that suite, removing either
one fires nothing.

That matters because the arrangement is defence in depth and the depth is what is
unpinned. ``_build_link`` is documented as an overridable method, and both daemon
helpers are ordinary callables a subclass or a bring-up script can reach without
going through ``connect_eagerly`` first. Grading them here means the pre-check is
not a single point of failure: whichever of the two layers a later change removes,
something fails.

These cells reach each guard directly rather than through the connect path, and the
rest of the file pins what nothing else does - the premise that the extra is a
packaging accident, that the reason reaches a mesh peer, and that a working install
is unaffected.

The absence is simulated rather than installed, since the suite runs where the
extra is present: :func:`_hide_extra` makes ``device_connect_edge`` unimportable and
evicts the already-imported ``strands_robots.device_connect`` package so the next
import re-executes the ``__init__`` that fails. That is the same failure a stock
install produces, at the same place, and ``monkeypatch`` restores both halves.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers.reachy import ReachyDriver

#: The dependency the ``[device-connect]`` extra supplies, and the only reason the
#: transport package needs the extra at all.
_EXTRA_DEP = "device_connect_edge"

#: A daemon status body shaped like the real one. ``wireless_version=False`` is a
#: Lite, the variant that needs no Zenoh transport and so is simplest to bring up.
_LITE_STATUS: dict[str, Any] = {"wireless_version": False, "motors": "on"}


def _hide_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ``[device-connect]`` extra unimportable for the current test.

    Args:
        monkeypatch: pytest's patcher, which restores every change made here.
    """
    for name in [n for n in list(sys.modules) if n.startswith("strands_robots.device_connect")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, _EXTRA_DEP, None)


def _names_the_extra(text: str) -> bool:
    """Whether ``text`` names the missing extra and how to install it.

    Args:
        text: A refusal reason.

    Returns:
        True when the reason carries both the extra's name and a pip remedy.
    """
    return "strands-robots[device-connect]" in text and "pip install" in text


class _StubLink:
    """A ``HardwareLink`` that records the commands it is given."""

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Accept the driver's callbacks."""
        del on_joints, on_imu

    async def stop(self) -> None:
        """Accept teardown."""

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        """Record one command verbatim."""
        self.commands.append(cmd)


def _connected_driver(monkeypatch: pytest.MonkeyPatch) -> tuple[ReachyDriver, _StubLink]:
    """Return a driver brought up with the extra present, and its link.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        ``(driver, link)`` for a connected driver, so a test can hide the extra
        afterwards and exercise a command path only a connected driver reaches.
    """
    link = _StubLink()
    monkeypatch.setattr(
        "strands_robots.device_connect.reachy_transport.api",
        lambda *a, **k: dict(_LITE_STATUS),
    )
    monkeypatch.setattr(ReachyDriver, "_build_link", lambda self, *, is_lite: link)
    driver = ReachyDriver(host="reachy.local")
    assert driver.connect_eagerly() is None, "premise: the driver connects while the extra is present"
    return driver, link


class TestTheExtraIsAPackagingAccident:
    """The premise the named refusal rests on.

    If the transport genuinely needed ``device_connect_edge``, requiring the extra
    would be correct and a refusal would be the wrong answer - the driver would
    simply not be a core-install driver. These pin that the leaf needs nothing from
    it and that the parent package is what pulls it in.
    """

    def test_the_transport_module_never_mentions_the_extras_dependency(self) -> None:
        """The transport source carries no reference to the extra's dependency."""
        package = Path(reachy_mod.__file__).parent.parent / "device_connect"
        assert _EXTRA_DEP not in (package / "reachy_transport.py").read_text(encoding="utf-8")

    def test_the_package_init_is_what_pulls_the_extra_in(self) -> None:
        """The package ``__init__`` imports the dependency at module scope."""
        package = Path(reachy_mod.__file__).parent.parent / "device_connect"
        assert f"from {_EXTRA_DEP} import" in (package / "__init__.py").read_text(encoding="utf-8")

    def test_the_resolver_answers_with_the_module_when_the_extra_is_present(self) -> None:
        """The control: nothing here refuses in an install that has the extra."""
        resolved = reachy_mod._resolve_transport()
        assert not isinstance(resolved, str)
        assert hasattr(resolved, "api")


class TestEachGuardIsReachedOnItsOwn:
    """The two guards ``connect_eagerly``'s pre-check hides, reached directly."""

    def test_the_daemon_probe_reports_an_error_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_daemon_get`` answers in the ``{"error": ...}`` shape its callers read."""
        _hide_extra(monkeypatch)
        driver = ReachyDriver(host="reachy.local")
        assert _names_the_extra(driver._daemon_get(reachy_mod._PATH_STATUS)["error"])

    def test_the_link_builder_returns_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_build_link`` answers in the reason contract it already documents."""
        _hide_extra(monkeypatch)
        driver = ReachyDriver(host="reachy.local")
        assert _names_the_extra(driver._build_link(is_lite=True))


class TestTheReasonReachesAMeshPeer:
    """A Mini on an install without the extra is still a describable peer."""

    def test_get_status_answers_and_carries_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``get_status`` succeeds, reporting the reason as ``connect_error``."""
        _hide_extra(monkeypatch)
        driver = ReachyDriver(host="reachy.local")
        driver.connect_eagerly()
        status = asyncio.run(driver.get_status())
        assert status["status"] == "success"
        assert _names_the_extra(status["content"][0]["json"]["connect_error"])


class TestSendActionRefusesInsteadOfMisdiagnosing:
    """What the command path must not say, and must not do.

    ``_wire_commands`` could report a missing transport by returning no commands,
    which ``send_action`` renders as "nothing to send - none of [...] names a Reachy
    Mini axis". That blames the caller's axis names for a packaging problem, so the
    wording is pinned as well as the refusal.
    """

    def test_the_refusal_does_not_blame_the_callers_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid axis name must not be reported as an unrecognised one."""
        driver, _link = _connected_driver(monkeypatch)
        try:
            _hide_extra(monkeypatch)
            assert "nothing to send" not in driver.send_action({"body_yaw": 10.0})["content"][0]["text"]
        finally:
            driver.cleanup()

    def test_nothing_is_put_on_the_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused action sends no partial command."""
        driver, link = _connected_driver(monkeypatch)
        try:
            _hide_extra(monkeypatch)
            driver.send_action({"body_yaw": 10.0})
            assert link.commands == []
        finally:
            driver.cleanup()


class TestAWorkingInstallIsUnaffected:
    """The over-reach control: resolving through a helper changed no behaviour."""

    def test_a_connected_driver_still_sends_the_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the extra present the same action reaches the link as before."""
        driver, link = _connected_driver(monkeypatch)
        try:
            result = driver.send_action({"body_yaw": 10.0})
            assert result["status"] == "success"
            assert [sorted(c) for c in link.commands] == [["body_yaw"]]
        finally:
            driver.cleanup()
