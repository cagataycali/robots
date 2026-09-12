"""A stop over Device Connect reads the device's answer instead of counting delivery.

A device answers its ``stop`` RPC with an envelope: an authorization refusal
(``{"status": "error", "reason": "caller not authorized for 'stop'"}``) or a
``stop_policy`` that could not halt a rollout both arrive as a *delivered*
reply, not as a raised ``invoke``. Counting delivery as a stop reported a
fleet whose every device refused as ``E-STOP: N/N devices stopped``, and a
single refused target under ``status="success"`` with an audit row saying
``ok=True`` -- so a caller branching on the envelope, and the audit row an
incident reads first, both said the robot had halted while the robot had just
said it had not.

Both branches of :func:`~strands_robots.tools.robot_mesh._device_connect_dispatch`
that stop a robot are pinned here, because they answer one question and read one
rule: :func:`~strands_robots.mesh.core._reports_failure_to_stop`, whose whole
purpose is that its callers do not drift. Keeping the fleet-wide and
single-target cells in one module is what makes a drift between them visible.

Hardware-free: ``device_connect_agent_tools.connection`` is a fake module.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import strands_robots.tools.robot_mesh as rm

REFUSAL = {"status": "error", "reason": "caller not authorized for 'stop'", "caller": "my-agent"}
#: The other spelling ``_reports_failure_to_stop`` owns: ``stop_task`` and the
#: dispatch's own branches answer ``{"ok": ...}`` where ``stop_policy`` answers
#: the agent-tool ``{"status": ...}``. A verdict read for one spelling only
#: reports the other as a halt.
COULD_NOT_HALT = {"ok": False, "reason": "stop_policy: rollout still in flight"}
#: The envelope a real halt answers with, as ``Simulation.stop_policy`` emits it
#: (measured on a live MuJoCo rollout: a ``text`` block naming the robot and a
#: ``json`` block carrying the ``was_running`` verdict its own readers look for).
HALTED = {
    "status": "success",
    "content": [{"text": "Stopped on 'so100'"}, {"json": {"robot": "so100", "was_running": True}}],
}


@pytest.fixture
def dc_target(monkeypatch):
    """One addressed device whose ``stop`` answer is scripted, plus its audit rows.

    Returns a ``(answer, audit)`` pair: assign ``answer["reply"]`` and read the
    recorded ``(target, ok, detail)`` tuples. The audit verdict is asserted
    rather than discarded, because it is the surface an incident reads first.
    """
    answer: dict[str, Any] = {}
    audit: list[tuple[str, bool, str]] = []
    conn = types.SimpleNamespace()
    conn.invoke = lambda target, func, params, timeout: {"jsonrpc": "2.0", "result": answer["reply"]}
    module = types.ModuleType("device_connect_agent_tools.connection")
    module.get_connection = lambda: conn
    pkg = types.ModuleType("device_connect_agent_tools")
    pkg.connection = module
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", module)
    monkeypatch.setattr(rm, "_audit_tool_action", lambda a, t, ok, detail: audit.append((t, ok, detail)))
    return answer, audit


def _stop(target: str = "so100-lab-1") -> dict[str, Any]:
    result = rm._device_connect_dispatch("stop", target, "", "", "mock", 0, 30.0, 30.0)
    assert result is not None
    return result


@pytest.mark.parametrize("reply", [REFUSAL, COULD_NOT_HALT], ids=["authz_refusal", "could_not_halt"])
def test_a_target_that_reported_it_did_not_stop_is_an_error(dc_target, reply):
    answer, audit = dc_target
    answer["reply"] = reply

    result = _stop()

    text = result["content"][0]["text"]
    assert result["status"] == "error", text
    assert "did NOT stop" in text
    assert "so100-lab-1" in text
    assert reply["reason"] in text
    assert [(target, ok) for target, ok, _ in audit] == [("so100-lab-1", False)]


def test_a_target_that_halted_is_a_success_carrying_its_answer(dc_target):
    answer, audit = dc_target
    answer["reply"] = HALTED

    result = _stop()

    assert result["status"] == "success"
    assert "was_running" in result["content"][0]["text"]
    assert [(target, ok) for target, ok, _ in audit] == [("so100-lab-1", True)]


def test_an_answer_that_reports_no_verdict_is_not_read_as_a_refusal(dc_target):
    """A reply carrying neither key is not a failure report.

    The conservative direction ``_reports_failure_to_stop`` documents: a device
    whose reply says nothing either way must not be reported as having refused,
    or the verdict invents a refusal the device never sent.
    """
    answer, audit = dc_target
    answer["reply"] = {"detail": "acknowledged"}

    result = _stop()

    assert result["status"] == "success", result["content"][0]["text"]
    assert [ok for _, ok, _ in audit] == [True]


@pytest.fixture
def dc_fleet(monkeypatch):
    """Two discovered devices whose ``stop`` answers are scripted per device id."""
    answers: dict[str, dict[str, Any]] = {}
    conn = types.SimpleNamespace()
    conn.list_devices = lambda: [{"device_id": device_id} for device_id in answers]
    conn.invoke = lambda device_id, func, params, timeout: {"jsonrpc": "2.0", "result": answers[device_id]}
    module = types.ModuleType("device_connect_agent_tools.connection")
    module.get_connection = lambda: conn
    pkg = types.ModuleType("device_connect_agent_tools")
    pkg.connection = module
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", module)
    monkeypatch.setattr(rm, "_audit_tool_action", lambda *a, **k: None)
    return answers


def _estop() -> dict[str, Any]:
    result = rm._device_connect_dispatch("emergency_stop", "", "", "", "mock", 0, 30.0, 30.0)
    assert result is not None
    return result


def test_a_device_that_refused_its_stop_is_reported_as_not_stopped(dc_fleet):
    dc_fleet["so100-lab-1"] = REFUSAL
    dc_fleet["aloha-lab-2"] = HALTED

    result = _estop()

    text = result["content"][0]["text"]
    assert result["status"] == "error", text
    assert "1/2 devices stopped" in text
    assert "so100-lab-1" in text
    assert "caller not authorized" in text


def test_a_fleet_that_halted_everywhere_is_a_success(dc_fleet):
    dc_fleet["so100-lab-1"] = HALTED
    dc_fleet["aloha-lab-2"] = HALTED

    result = _estop()

    assert result["status"] == "success"
    assert result["content"][0]["text"] == "E-STOP: 2/2 devices stopped"
