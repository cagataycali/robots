"""Regression for GH #4156: ``robot_mesh``'s headless refusal names the remedy and honours the bypass.

``agents.md`` documents one operator gate for every path from the model to an
actuator: an allowlist variable, then ``BYPASS_TOOL_CONSENT=true`` with a
WARNING, then a refusal that names the variable and value which pre-approve
the call. ``robot_mesh`` keeps its own gate (its interrupt reason carries the
fleet scope and the validated command, which the shared gate has no slot for),
and that gate did neither: a headless ``tell`` answered ``requires a
human-in-the-loop interrupt, but no tool_context is available`` naming nothing,
and ``BYPASS_TOOL_CONSENT=true`` changed nothing.

Pinned here: the headless refusal names ``STRANDS_MESH_HITL_ACTIONS`` with the
value that pre-approves this one action and ``BYPASS_TOOL_CONSENT``; the bypass
lets the action proceed with a WARNING and an audit row; without the bypass the
gate still fails closed; and the docs sentence agrees with the docs table.
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

rmt = importlib.import_module("strands_robots.tools.robot_mesh")

_AGENTS_MD = Path(__file__).resolve().parents[2] / "docs" / "learn" / "agents.md"


def _call(*, ctx: Any = None, **kwargs: Any) -> dict[str, Any]:
    fn = getattr(rmt.robot_mesh, "original", rmt.robot_mesh)
    return fn(tool_context=ctx, **kwargs)


def _text(out: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in out["content"])


@pytest.fixture
def fake_mesh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_MESH_HITL_ACTIONS", raising=False)
    rmt._reset_interrupt_actions_cache()
    fake = MagicMock(name="LocalMesh")
    fake.peer_id = "local-a"
    fake.peer_type = "sim"
    fake.inbox = {}
    fake.tell.return_value = {"type": "response", "result": {"status": "success"}}
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={"local-a": fake}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        yield fake
    rmt._reset_interrupt_actions_cache()


def test_the_headless_refusal_names_the_variable_and_the_value_that_pre_approve_tell(fake_mesh) -> None:
    out = _call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    text = _text(out)
    assert "human-in-the-loop interrupt" in text
    assert "STRANDS_MESH_HITL_ACTIONS=" in text
    # The value keeps every other gated action asking; only ``tell`` is pre-approved.
    named = re.search(r"STRANDS_MESH_HITL_ACTIONS=([a-z_,]+)", text)
    assert named is not None, text
    assert set(named.group(1).split(",")) == set(rmt._DEFAULT_INTERRUPT_ACTIONS) - {"tell"}
    assert "BYPASS_TOOL_CONSENT=true" in text
    fake_mesh.tell.assert_not_called()


def test_pre_approving_the_last_gated_action_names_none(fake_mesh, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "tell")
    rmt._reset_interrupt_actions_cache()
    out = _call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    assert "STRANDS_MESH_HITL_ACTIONS=none" in _text(out)


def test_bypass_tool_consent_lets_the_action_proceed_with_a_warning_and_an_audit_row(
    fake_mesh, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
    with patch.object(rmt, "_audit_tool_action") as audit, caplog.at_level(logging.WARNING):
        out = _call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "success", out
    fake_mesh.tell.assert_called_once()
    assert any("BYPASS_TOOL_CONSENT" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records)
    assert any("BYPASS_TOOL_CONSENT" in str(call.args) for call in audit.call_args_list)


@pytest.mark.parametrize("value", ["", "false", "1", "yes"])
def test_only_the_literal_true_bypasses(fake_mesh, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """The same spelling rule as the shared gate: anything but ``true`` keeps the gate closed."""
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", value)
    out = _call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    fake_mesh.tell.assert_not_called()


def test_an_interrupt_the_host_refuses_names_the_same_remedy(fake_mesh) -> None:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.side_effect = RuntimeError("interrupts not supported here")
    out = _call(ctx=ctx, action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    text = _text(out)
    assert "STRANDS_MESH_HITL_ACTIONS=" in text and "BYPASS_TOOL_CONSENT=true" in text
    fake_mesh.tell.assert_not_called()


def test_the_docs_sentence_agrees_with_the_docs_table() -> None:
    """``agents.md`` gates ``stop`` and ``emergency_stop`` for ``robot_mesh`` in its table; its prose must not say otherwise."""
    page = _AGENTS_MD.read_text(encoding="utf-8")
    assert "Reading and stopping are never gated." not in page
    assert "Reading is never gated; `robot_mesh` gates `stop`." in page
    assert re.search(r"\| `robot_mesh` \| `emergency_stop`, `broadcast`, `tell`, `send`, `stop`, `rpc` \|", page)
