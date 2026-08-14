"""Optional-dependency degradation in the ``robot_mesh`` agent tool.

Two of the tool's optional-dependency arms answer through OPPOSITE channels and
neither was driven:

* :func:`strands_robots.tools.robot_mesh._audit_tool_action` DEGRADES.  Its 55
  call sites cover every safety-significant action the tool exposes, and its
  docstring records why the swallow carries a breadcrumb -- "a swallowed
  exception with no log line means a broken audit path silently disappears",
  emitted at DEBUG so an operator asking "why don't I see my LLM tool actions in
  the audit log?" has something to find.  Nothing verified the breadcrumb is
  emitted, so the forensic record of every LLM tool action could go quiet with
  the suite green.
* The mesh-module arm REFUSES with a structured error instead of degrading,
  because an action that needs the fleet cannot be served without it.

``Mesh._on_cmd`` -- the pattern ``_audit_tool_action``'s docstring says it
matches -- already has a module pinning the mesh side of that fail-soft
contract.  These cases pin the tool side, and one asserts the two channels side
by side so the asymmetry reads as deliberate rather than as an inconsistency.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import Any

import pytest

import strands_robots.tools.robot_mesh as rmt

_AUDIT_MODULE = "strands_robots.mesh.audit"
_MESH_MODULE = "strands_robots.mesh"


@pytest.fixture(autouse=True)
def _isolate_rate_limits() -> Iterator[None]:
    """Match the convention of the sibling ``robot_mesh`` contract modules."""
    rmt._reset_rate_limits()
    yield
    rmt._reset_rate_limits()


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    """Collect the tool module's own log records at DEBUG.

    ``setLevel`` rather than assigning ``logger.level``: :class:`logging.Logger`
    caches ``isEnabledFor`` verdicts, and only ``setLevel`` clears that cache.
    Assigning the attribute leaves a cached "DEBUG is disabled" answer in place
    and the breadcrumb is never emitted, which reads as an absent record.
    """
    collected: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    handler = _Collect()
    previous = rmt.logger.level
    rmt.logger.addHandler(handler)
    rmt.logger.setLevel(logging.DEBUG)
    try:
        yield collected
    finally:
        rmt.logger.removeHandler(handler)
        rmt.logger.setLevel(previous)


def _break_audit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the deferred ``mesh.audit`` import fail the way an absent extra does."""
    monkeypatch.setitem(sys.modules, _AUDIT_MODULE, None)


def _break_mesh_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the deferred ``mesh`` import fail the way an absent extra does."""
    monkeypatch.setitem(sys.modules, _MESH_MODULE, None)


def _breadcrumbs(collected: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in collected if "audit log unavailable" in r.getMessage()]


def _answer(result: dict[str, Any]) -> tuple[str, str]:
    """Reduce a tool result to what a caller reads: its status and its text."""
    return str(result.get("status", "")), str((result.get("content") or [{}])[0].get("text", ""))


class TestABrokenAuditPathDegradesWithABreadcrumb:
    """The audit path is best-effort, and the breadcrumb is what keeps it visible."""

    def test_the_audited_action_is_not_disturbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unavailable audit path must not raise into the action it records."""
        _break_audit_path(monkeypatch)
        rmt._audit_tool_action("emergency_stop", "arm-1", True, "halted the fleet")

    def test_the_unavailable_audit_path_leaves_a_breadcrumb(
        self, monkeypatch: pytest.MonkeyPatch, records: list[logging.LogRecord]
    ) -> None:
        """Without this record a broken audit path disappears silently."""
        _break_audit_path(monkeypatch)
        rmt._audit_tool_action("emergency_stop", "arm-1", True, "halted the fleet")

        reported = _breadcrumbs(records)
        assert reported, "a broken audit path left no record that it was unavailable"
        message = reported[0].getMessage()
        assert "robot_mesh" in message, message
        assert _AUDIT_MODULE in message, message

    def test_the_breadcrumb_is_debug_not_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, records: list[logging.LogRecord]
    ) -> None:
        """DEBUG is the documented level: the audited action reports normally itself.

        Pinning the level is what catches a downgrade to one an operator
        investigating a quiet audit log would never enable.
        """
        _break_audit_path(monkeypatch)
        rmt._audit_tool_action("resume", "arm-1", True, "resumed")

        reported = _breadcrumbs(records)
        assert reported, "a broken audit path left no record that it was unavailable"
        assert reported[0].levelno == logging.DEBUG, logging.getLevelName(reported[0].levelno)

    def test_a_tool_calls_answer_is_unchanged_when_the_audit_path_is_broken(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: the degradation costs the record, never the answer.

        Comparing the two answers rather than asserting one of them keeps this
        independent of whether this process happens to have joined a mesh.
        """
        intact = _answer(rmt.robot_mesh(action="list"))

        with monkeypatch.context() as broken:
            _break_audit_path(broken)
            degraded = _answer(rmt.robot_mesh(action="list"))

        assert degraded == intact, f"{intact!r} became {degraded!r}"


class TestABrokenMeshModuleRefuses:
    """An action that needs the fleet cannot be served without the mesh module."""

    def test_the_refusal_is_a_structured_error_naming_the_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tool answers through its envelope rather than raising past it."""
        _break_mesh_module(monkeypatch)
        status, text = _answer(rmt.robot_mesh(action="list"))

        assert status == "error", text
        assert _MESH_MODULE in text, text

    def test_the_refusal_carries_the_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reason is what tells the caller the extra is missing, not the fleet."""
        _break_mesh_module(monkeypatch)
        _, text = _answer(rmt.robot_mesh(action="list"))

        assert "unavailable" in text, text
        assert "None in sys.modules" in text, text


class TestTheTwoChannelsAreOpposite:
    """One optional dependency degrades and the other refuses, on purpose."""

    def test_a_broken_audit_path_is_invisible_while_a_broken_mesh_module_is_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserting both together states why the two arms differ.

        The audit record is a forensic side effect, so losing it must not change
        the answer.  The mesh module IS the action's subject, so an action that
        proceeded without it would report a fleet it never reached.
        """
        intact = _answer(rmt.robot_mesh(action="list"))

        with monkeypatch.context() as broken_audit:
            _break_audit_path(broken_audit)
            audit_broken = _answer(rmt.robot_mesh(action="list"))

        with monkeypatch.context() as broken_mesh:
            _break_mesh_module(broken_mesh)
            mesh_broken = _answer(rmt.robot_mesh(action="list"))

        assert audit_broken == intact, f"the audit path changed the answer: {audit_broken!r}"
        assert mesh_broken != intact, "a broken mesh module answered as though the fleet were reachable"
        assert _MESH_MODULE in mesh_broken[1], mesh_broken[1]
