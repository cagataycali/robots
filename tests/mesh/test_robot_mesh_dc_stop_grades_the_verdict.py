"""``robot_mesh(action="stop")`` over Device Connect reads the target's answer.

A device answers its ``stop`` RPC with an envelope: an authorization refusal
(``{"status": "error", "reason": "caller not authorized for 'stop'"}``) or a
``stop_policy`` that could not halt a rollout both arrive as a *delivered*
reply, not as a raised ``invoke``. The single-target branch used to count
delivery as a stop, so a refused stop was returned under ``status="success"``
and its audit row recorded ``ok=True`` - the raw reply was echoed in the prose,
but the tool-level verdict and the audit verdict, which is the first thing an
incident reads, both said the robot stopped. The fleet-wide ``emergency_stop``
branch of the same dispatcher grades each reply through
:func:`~strands_robots.mesh.core._reports_failure_to_stop`; this pins the
single-target branch to the same rule.

Hardware-free: ``device_connect_agent_tools.connection`` is a fake module.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

import strands_robots.tools.robot_mesh as rm

REFUSAL = {"status": "error", "reason": "caller not authorized for 'stop'", "caller": "my-agent"}
NOT_HALTED = {"ok": False, "error": "no rollout to stop"}
HALTED = {"status": "success", "content": [{"text": "Halted 1 rollout(s): so100"}]}
TARGET = "so100-lab-1"


@pytest.fixture
def dc_device(monkeypatch):
    """One reachable device whose ``stop`` answer is scripted; audit rows are captured."""
    state: dict[str, Any] = {"answer": None, "audit": []}
    conn = types.SimpleNamespace()
    conn.list_devices = lambda: [{"device_id": TARGET}]
    conn.invoke = lambda device_id, func, params, timeout: {"jsonrpc": "2.0", "result": state["answer"]}
    module = types.ModuleType("device_connect_agent_tools.connection")
    module.get_connection = lambda: conn
    pkg = types.ModuleType("device_connect_agent_tools")
    pkg.connection = module
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", module)
    monkeypatch.setattr(rm, "_audit_tool_action", lambda *args, **kwargs: state["audit"].append(args))
    return state


def _stop() -> dict[str, Any]:
    result = rm._device_connect_dispatch("stop", TARGET, "", "", "mock", 0, 30.0, 30.0)
    assert result is not None
    return result


@pytest.mark.parametrize("answer", [REFUSAL, NOT_HALTED], ids=["authz-refusal", "ok-false"])
def test_a_delivered_refusal_is_an_error_naming_the_target_and_its_answer(dc_device, answer):
    dc_device["answer"] = answer

    result = _stop()

    text = result["content"][0]["text"]
    assert result["status"] == "error", text
    assert TARGET in text
    assert "did NOT stop" in text
    for value in answer.values():
        if isinstance(value, str):
            assert value in text


@pytest.mark.parametrize("answer", [REFUSAL, NOT_HALTED], ids=["authz-refusal", "ok-false"])
def test_a_delivered_refusal_is_audited_as_not_ok(dc_device, answer):
    dc_device["answer"] = answer

    _stop()

    (row,) = dc_device["audit"]
    action, target, ok, detail = row
    assert (action, target, ok) == ("stop", TARGET, False)
    assert "did NOT stop" in detail


def test_a_delivered_refusal_is_logged_at_critical(dc_device, caplog):
    dc_device["answer"] = REFUSAL

    with caplog.at_level(logging.CRITICAL, logger=rm.logger.name):
        _stop()

    records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert TARGET in records[0].getMessage()
    assert "caller not authorized" in records[0].getMessage()


def test_a_halted_device_is_a_success_audited_as_ok(dc_device):
    dc_device["answer"] = HALTED

    result = _stop()

    assert result["status"] == "success"
    assert result["content"][0]["text"].startswith(f"Stop {TARGET}: ")
    assert "Halted 1 rollout(s)" in result["content"][0]["text"]
    (row,) = dc_device["audit"]
    assert row[:3] == ("stop", TARGET, True)


def test_an_answer_that_is_not_an_envelope_is_still_a_success(dc_device):
    """A reply carrying neither key is not a failure report - the rule stays conservative."""
    dc_device["answer"] = "stopped"

    result = _stop()

    assert result["status"] == "success"
    assert result["content"][0]["text"] == f'Stop {TARGET}: "stopped"'
