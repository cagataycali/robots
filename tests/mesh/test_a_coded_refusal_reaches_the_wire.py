"""Regression for GH #4173: a peer's continuable refusal answers with its code, not ``dispatch error``.

``execute`` with ``lerobot_local`` on a peer that has not set
``STRANDS_TRUST_REMOTE_CODE=1`` raised
:class:`~strands_robots.policies.factory.UntrustedRemoteCodeError` inside
``Mesh._dispatch``. The generic handler in ``_exec_cmd`` answered the fixed
``{"type": "error", "error": "dispatch error"}`` and the remedy stayed in the
peer's own stderr; the operator at the other end had nothing to act on.

:mod:`strands_robots.refusal_codes` already gives such a refusal a stable code
and the operator grant that lifts it. Pinned here: an exception carrying a code
from that vocabulary answers with the code, the grant and the subject the caller
supplied, in one remedy sentence built from the vocabulary and never from the
exception text; an exception without a code still answers the fixed string, so
internal detail never reaches a remote prober.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots import refusal_codes
from strands_robots.mesh import Mesh
from strands_robots.mesh import core as mesh_core
from strands_robots.policies.factory import UntrustedRemoteCodeError


class _FakeRobot:
    tool_name_str = "fakebot"

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}


@pytest.fixture
def captured_puts() -> Iterator[list[tuple[str, dict[str, Any]]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    with patch.object(mesh_core, "put", side_effect=lambda key, data: seen.append((key, data))):
        yield seen


def _reply(puts: list[tuple[str, dict[str, Any]]], turn: str) -> dict[str, Any]:
    return next(d for k, d in puts if k == f"strands/alice/response/me/{turn}")


def _command(turn: str) -> dict[str, Any]:
    return {
        "sender_id": "alice",
        "turn_id": turn,
        "command": {
            "action": "execute",
            "instruction": "pick up the red cube",
            "policy_provider": "lerobot_local",
            "pretrained_name_or_path": "lerobot/smolvla_base",
        },
    }


def test_an_untrusted_remote_code_refusal_answers_with_its_code_and_grant(captured_puts) -> None:
    m = Mesh(_FakeRobot(), peer_id="me")
    exc = UntrustedRemoteCodeError(
        "Policy provider 'lerobot_local' loads HuggingFace models with trust_remote_code=True ...",
        code=refusal_codes.TRUST_REMOTE_CODE_REQUIRED,
        subject="lerobot_local",
    )
    with patch.object(mesh_core, "log_safety_event"), patch.object(m, "_dispatch", side_effect=exc):
        m._exec_cmd(_command("t1"))
    reply = _reply(captured_puts, "t1")
    assert reply["type"] == "error"
    assert reply["code"] == "TRUST_REMOTE_CODE_REQUIRED"
    assert reply["grant"] == "STRANDS_TRUST_REMOTE_CODE"
    assert reply["subject"] == "lerobot_local"
    assert "STRANDS_TRUST_REMOTE_CODE" in reply["error"]
    assert reply["error"] != "dispatch error"
    # The sentence is built from the vocabulary, never from the exception text.
    assert "HuggingFace" not in reply["error"]


def test_the_audit_record_names_the_code(captured_puts) -> None:
    m = Mesh(_FakeRobot(), peer_id="me")
    exc = UntrustedRemoteCodeError("x", code=refusal_codes.TRUST_REMOTE_CODE_REQUIRED, subject="lerobot_local")
    with (
        patch.object(mesh_core, "log_safety_event") as audit,
        patch.object(m, "_dispatch", side_effect=exc),
    ):
        m._exec_cmd(_command("t2"))
    assert audit.call_args.args[0] == "command_rejected"
    payload = audit.call_args.args[2]
    assert payload["code"] == "TRUST_REMOTE_CODE_REQUIRED"
    assert payload["reason"] == "dispatch error"


def test_an_exception_without_a_code_still_answers_the_fixed_string(captured_puts) -> None:
    m = Mesh(_FakeRobot(), peer_id="me")
    with (
        patch.object(mesh_core, "log_safety_event"),
        patch.object(m, "_dispatch", side_effect=RuntimeError("boom-with-internal-detail")),
    ):
        m._exec_cmd(_command("t3"))
    reply = _reply(captured_puts, "t3")
    assert reply["error"] == "dispatch error"
    assert "code" not in reply
    assert "boom-with-internal-detail" not in str(reply)


def test_a_code_outside_the_vocabulary_is_not_trusted(captured_puts) -> None:
    """A stray ``code`` attribute on an adapter exception is not a refusal contract."""
    m = Mesh(_FakeRobot(), peer_id="me")
    exc = RuntimeError("boom")
    exc.code = "SOMETHING_ELSE"  # type: ignore[attr-defined]
    with patch.object(mesh_core, "log_safety_event"), patch.object(m, "_dispatch", side_effect=exc):
        m._exec_cmd(_command("t4"))
    reply = _reply(captured_puts, "t4")
    assert reply["error"] == "dispatch error"
    assert "code" not in reply


def test_the_real_gate_is_what_the_wire_reports(captured_puts, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not a stand in: ``create_policy`` itself refuses, and the refusal crosses the wire with its code."""
    from strands_robots.policies import factory

    monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)

    class _Sim:
        """A sim shaped peer (the real robot rail is gated before the policy is built)."""

        tool_name_str = "sim"
        _world: Any = object()

        def list_robots(self) -> list[str]:
            return ["so101"]

        def run_policy(self, robot_name: str, **kwargs: Any) -> dict[str, Any]:
            factory._check_trust_remote_code(str(kwargs["policy_provider"]))
            raise AssertionError("the gate must refuse before this line")

    m = Mesh(_Sim(), peer_id="me")
    with patch.object(mesh_core, "log_safety_event"):
        m._exec_cmd(_command("t5"))
    reply = _reply(captured_puts, "t5")
    assert reply["code"] == refusal_codes.TRUST_REMOTE_CODE_REQUIRED
    assert reply["grant"] == refusal_codes.REFUSAL_GRANTS[refusal_codes.TRUST_REMOTE_CODE_REQUIRED]
