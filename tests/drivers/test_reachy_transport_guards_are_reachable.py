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
transport is importable. :func:`_block_transport` blocks the one module
:func:`~strands_robots.drivers.reachy._resolve_transport` imports, and ``monkeypatch``
restores it. Blocking that module is what ties the simulation to the resolver rather
than to a packaging detail upstream of it: the resolver reports on this import and on
nothing before it, so an absence staged anywhere else is only a refusal by
coincidence. Staging it by making ``device_connect_edge`` unimportable and evicting
the ``strands_robots.device_connect`` subtree - so the next import re-executes an
``__init__`` that fails - stops simulating anything as soon as that ``__init__``
stops needing the extra, and the driver then reaches the real daemon instead of
refusing.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.abc
import sys
from pathlib import Path
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers.reachy import ReachyDriver

#: The dependency the ``[device-connect]`` extra supplies. The transport leaf needs
#: nothing from it; only the drivers its parent package ships do.
_EXTRA_DEP = "device_connect_edge"

#: The module :func:`~strands_robots.drivers.reachy._resolve_transport` imports. A
#: refusal-by-name has to be measured against this one being unimportable, because
#: this import is the only thing the resolver reports on.
_TRANSPORT_MODULE = reachy_mod._TRANSPORT_MODULE

#: A daemon status body shaped like the real one. ``wireless_version=False`` is a
#: Lite, the variant that needs no Zenoh transport and so is simplest to bring up.
_LITE_STATUS: dict[str, Any] = {"wireless_version": False, "motors": "on"}


def _block_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the transport module unimportable for the current test.

    Blocks that one entry and leaves the rest of ``sys.modules`` alone. Evicting the
    ``strands_robots.device_connect`` subtree instead lets the next import build a
    second module object for a name a later test patches through the first, so a
    double installed on one is not read through the other.

    Args:
        monkeypatch: pytest's patcher, which restores the entry after the test.
    """
    monkeypatch.setitem(sys.modules, _TRANSPORT_MODULE, None)


def _reports_without_prescribing(text: str) -> bool:
    """Whether ``text`` names the module that failed and prescribes no install.

    The resolver reports on one import and on nothing before it, so the module's
    name is what makes the reason actionable. A ``pip install`` line is the part
    it must not carry: the leaf imports nothing outside the standard library, so
    no install supplies a module whose absence reaches that branch, and a remedy
    that cannot help is offered with the confidence of a diagnosis.

    Args:
        text: A refusal reason.

    Returns:
        True when the reason names the module and prescribes no pip remedy.
    """
    return _TRANSPORT_MODULE in text and "pip install" not in text


class _StubLink:
    """A ``HardwareLink`` that records the commands it is given."""

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Accept the driver's callbacks.

        The two callbacks are unused on this stub - the harness only
        checks the *shape* of the invocation - and they are dropped by
        assigning to ``_`` rather than ``del``d (CodeQL's
        ``py/unnecessary-delete`` fires for a bare delete on a local
        that is about to leave scope).
        """
        _ = (on_joints, on_imu)

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
    it, and that nothing on the load path pulls it in either - which is why the
    absence a refusal is measured against has to be staged at the leaf itself.
    """

    def test_the_transport_module_never_mentions_the_extras_dependency(self) -> None:
        """The transport source carries no reference to the extra's dependency."""
        package = Path(reachy_mod.__file__).parent.parent / "device_connect"
        assert _EXTRA_DEP not in (package / "reachy_transport.py").read_text(encoding="utf-8")

    def test_the_package_init_does_not_pull_the_extra_in_at_runtime(self) -> None:
        """No runtime module-scope import of the extra, so blocking it stages nothing.

        Read off the statements that actually execute. A ``TYPE_CHECKING`` block
        carries the same text and never runs, so a source scan that does not exclude
        it reports an eager import where there is none - and a simulation built on
        that reading refuses nothing.
        """
        package = Path(reachy_mod.__file__).parent.parent / "device_connect"
        module = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        pending: list[ast.AST] = [
            node for node in module.body if not (isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test))
        ]
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                pending.extend(ast.iter_child_nodes(node))
        assert not [name for name in imported if name.split(".")[0] == _EXTRA_DEP], (
            f"{_EXTRA_DEP} is imported where the package load will execute it: {sorted(imported)}"
        )

    def test_the_resolver_answers_with_the_module_when_the_extra_is_present(self) -> None:
        """The control: nothing here refuses in an install that has the extra."""
        resolved = reachy_mod._resolve_transport()
        assert not isinstance(resolved, str)
        assert hasattr(resolved, "api")


class TestEachGuardIsReachedOnItsOwn:
    """The two guards ``connect_eagerly``'s pre-check hides, reached directly."""

    def test_the_daemon_probe_reports_an_error_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_daemon_get`` answers in the ``{"error": ...}`` shape its callers read."""
        _block_transport(monkeypatch)
        driver = ReachyDriver(host="reachy.local")
        assert _reports_without_prescribing(driver._daemon_get(reachy_mod._PATH_STATUS)["error"])

    def test_the_link_builder_returns_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_build_link`` answers in the reason contract it already documents."""
        _block_transport(monkeypatch)
        driver = ReachyDriver(host="reachy.local")
        assert _reports_without_prescribing(driver._build_link(is_lite=True))


class TestTheReasonReachesAMeshPeer:
    """A Mini on an install without the extra is still a describable peer."""

    def test_get_status_answers_and_carries_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``get_status`` succeeds, reporting the reason as ``connect_error``."""
        _block_transport(monkeypatch)
        driver = ReachyDriver(host="reachy.local")
        driver.connect_eagerly()
        status = asyncio.run(driver.get_status())
        assert status["status"] == "success"
        assert _reports_without_prescribing(status["content"][0]["json"]["connect_error"])


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
            _block_transport(monkeypatch)
            assert "nothing to send" not in driver.send_action({"body_yaw": 10.0})["content"][0]["text"]
        finally:
            driver.cleanup()

    def test_nothing_is_put_on_the_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused action sends no partial command."""
        driver, link = _connected_driver(monkeypatch)
        try:
            _block_transport(monkeypatch)
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


class TestTheReasonPrescribesNothingItCannotEstablish:
    """The refusal reports the failure; it does not diagnose a cause.

    ``[device-connect]`` was once the cause: the parent package imported
    ``device_connect_edge`` at module scope, so the extra's absence really did make
    the leaf unimportable, and naming it was the one useful hint available. That
    import is now lazy, which leaves the remedy attached to a cause it can no longer
    establish - and a ``pip install`` that cannot supply the module is advice offered
    with the confidence of a diagnosis, which is the failure mode this whole file
    exists to prevent, pointed the wrong way.

    The rule is the one the shared optional-dependency helper already applies. Where
    :func:`~strands_robots.utils.require_optional` is given ``system_install`` it
    prints no pip line at all, "because a pip command for such a module is a remedy
    the caller can follow to no effect: it either installs something that leaves the
    module exactly as missing, or fails outright". A stdlib-only leaf is that
    position exactly.
    """

    def test_every_module_scope_import_of_the_leaf_is_in_the_standard_library(self) -> None:
        """The premise the whole class rests on, derived rather than asserted.

        If the leaf ever grows a third-party import this stops being true, and the
        question of whether an install remedy is establishable reopens. Deriving it
        means that shows up here rather than in a reason nobody re-read.
        """
        package = Path(reachy_mod.__file__).parent.parent / "device_connect"
        module = ast.parse((package / "reachy_transport.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        pending: list[ast.AST] = [
            node for node in module.body if not (isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test))
        ]
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                pending.extend(ast.iter_child_nodes(node))
        assert imported, "no module-scope imports were found, so the derivation read nothing"
        outside = sorted(name for name in imported if name.split(".")[0] not in sys.stdlib_module_names)
        assert not outside, f"the leaf imports outside the standard library: {outside}"

    def test_the_reason_prescribes_no_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No install command, because none of them supplies this module."""
        _block_transport(monkeypatch)
        reason = reachy_mod._resolve_transport()
        assert isinstance(reason, str)
        assert "pip install" not in reason, f"prescribes an install: {reason}"

    def test_the_reason_names_no_optional_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nor the extra itself, which is the cause the remedy asserted."""
        _block_transport(monkeypatch)
        reason = reachy_mod._resolve_transport()
        assert isinstance(reason, str)
        assert _EXTRA_DEP not in reason, f"names the extra's dependency: {reason}"
        assert "device-connect" not in reason, f"names the extra: {reason}"

    def test_the_reason_still_names_the_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The first of two facts dropping the remedy must not cost.

        Checked as a prefix rather than by containment. For a blocked module the
        ``ImportError`` text happens to repeat the dotted name, so a containment
        check passes even when the reason's own prose has stopped naming it - and the
        reason has to stand on its own, since the two are separated everywhere this
        string is read back.
        """
        _block_transport(monkeypatch)
        reason = reachy_mod._resolve_transport()
        assert isinstance(reason, str)
        assert reason.startswith(f"cannot import {_TRANSPORT_MODULE}:"), f"does not name the module: {reason}"

    def test_the_reason_still_carries_the_underlying_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second: with no remedy to offer, the cause is all a caller gets."""
        _block_transport(monkeypatch)
        reason = reachy_mod._resolve_transport()
        assert isinstance(reason, str)
        assert "halted" in reason, f"does not carry the underlying ImportError: {reason}"

    def test_the_reason_is_the_shape_its_docstring_claims_parity_with(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The declared sibling is the whole specification for this string.

        :func:`~strands_robots.drivers.reachy._resolve_transport` documents itself as
        "the same shape as" :func:`~strands_robots.drivers.g1._resolve_message_class`.
        Handed the same blocked module, the two must therefore answer identically -
        which is the check a re-added remedy fails, whatever wording it arrives in.
        """
        from strands_robots.drivers import g1 as g1_mod

        _block_transport(monkeypatch)
        mine = reachy_mod._resolve_transport()
        sibling = g1_mod._resolve_message_class((_TRANSPORT_MODULE, "api"))
        assert isinstance(mine, str) and isinstance(sibling, str)
        assert mine == sibling, f"diverges from the shape it claims parity with:\n  {mine!r}\n  {sibling!r}"


class TestTheDocumentedReasonIsTheRealOne:
    """The reference page quotes this reason, so the quote is part of the contract.

    ``docs/getting-started/robot-factory.md`` shows the refusal as the output of
    ``Robot("reachy_mini", mode="real").connect_eagerly()``. A quoted output rots the
    moment the surface it quotes changes, and nothing read that block: the page's own
    suite grades other sections, so the quote carried a remedy the code had stopped
    offering. Deriving the expected text from the resolver means the page cannot drift
    from it again without failing here.
    """

    @staticmethod
    def _quoted_reason() -> str:
        """The reason the reference page quotes, unwrapped to a single line."""
        page = Path(reachy_mod.__file__).parents[2] / "docs" / "getting-started" / "robot-factory.md"
        text = page.read_text(encoding="utf-8")
        blocks = [chunk for chunk in text.split("```") if "connect_eagerly()" in chunk]
        assert len(blocks) == 1, f"expected exactly one connect_eagerly block, found {len(blocks)}"
        after = blocks[0].split("connect_eagerly()", 1)[1]
        return " ".join(after.replace('"', " ").split())

    @staticmethod
    def _reason_for_an_absent_module() -> str:
        """The reason the resolver gives when the leaf is genuinely not installed.

        ``sys.modules[name] = None`` produces a different ``ImportError`` ("halted"),
        which is the right stand-in for the other cells but the wrong one to document:
        a reader meets the absent-module wording. A finder that refuses the name
        reproduces it without touching the installed tree.
        """

        class _Absent(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
                if fullname == _TRANSPORT_MODULE or fullname.startswith(f"{_TRANSPORT_MODULE}."):
                    raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                return None

        saved = sys.modules.pop(_TRANSPORT_MODULE, None)
        sys.meta_path.insert(0, _Absent())
        try:
            reason = reachy_mod._resolve_transport()
        finally:
            sys.meta_path.pop(0)
            if saved is not None:
                sys.modules[_TRANSPORT_MODULE] = saved
        assert isinstance(reason, str), "the resolver returned a module for a name it cannot import"
        return reason

    def test_the_page_quotes_the_reason_the_resolver_returns(self) -> None:
        """What the page shows is what the code says, word for word."""
        assert self._quoted_reason() == self._reason_for_an_absent_module()

    def test_the_page_prescribes_no_install_for_this_import(self) -> None:
        """A reader must not be sent to an install that cannot supply the module."""
        quoted = self._quoted_reason()
        assert "pip install" not in quoted, f"the quoted reason prescribes an install: {quoted}"


class TestTheSiteWhereTheExtraIsStillTheCauseKeepsIt:
    """The over-reach control: this is not a sweep of the extra's name.

    ``strands_robots.robot`` names the same extra when Device Connect bring-up
    fails, and there it is correct: that path imports ``init_device_connect_sync``,
    which resolves through modules that do import ``device_connect_edge`` at module
    scope, so the extra's absence really is a cause and installing it really is the
    remedy. Dropping the remedy wherever the string appears would take that with it.
    """

    def test_the_bring_up_path_still_names_the_extra(self) -> None:
        """The remedy survives where a module on the path needs the extra."""
        robot_source = (Path(reachy_mod.__file__).parent.parent / "robot.py").read_text(encoding="utf-8")
        assert "strands-robots[device-connect]" in robot_source

    def test_a_module_on_that_path_really_imports_the_extras_dependency(self) -> None:
        """And it is a cause there, unlike at the leaf: the import is module-scope."""
        package = Path(reachy_mod.__file__).parent.parent / "device_connect"
        eager = [
            path.name
            for path in sorted(package.glob("*.py"))
            if any(
                line.startswith((f"from {_EXTRA_DEP}", f"import {_EXTRA_DEP}"))
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        ]
        assert eager, f"no module in {package.name} imports {_EXTRA_DEP} at module scope"
