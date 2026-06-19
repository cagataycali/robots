"""Error- and validation-path behavior for ``_device_connect_dispatch``.

The Device Connect dispatcher in ``robot_mesh`` renders ``send``, ``rpc`` and
``stop`` actions onto an agent-side connection. Existing tests exercise the
happy paths (``peers``/``tell``/``emergency_stop``/``status``); these pin the
guard clauses an autonomous caller hits when it supplies a malformed request:

* required-parameter guards (missing ``target``/``function``/``command``),
* JSON decode + shape validation for the ``send``/``rpc`` payloads,
* the security-layer rejection surface (unknown action, bad RPC function name),
* the dispatcher's top-level catch that converts any backend failure into a
  tool-error dict instead of letting it propagate out of the tool.

All paths are hardware-free: the agent-side connection is a stub injected via
``device_connect_agent_tools.connection.get_connection``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import strands_robots.tools.robot_mesh as rm


class _StubConnection:
    """Minimal agent-side Device Connect connection.

    ``invoke`` returns a fixed envelope, or raises when ``invoke_error`` is set,
    so the dispatcher's success and backend-failure branches can both be driven.
    """

    def __init__(self, invoke_error: Exception | None = None) -> None:
        self.invoke_error = invoke_error
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def list_devices(self, device_type: str | None = None) -> list[dict[str, Any]]:
        return [{"device_id": "dev-1"}]

    def invoke(
        self, device_id: str, function: str, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        self.calls.append((device_id, function, params))
        if self.invoke_error is not None:
            raise self.invoke_error
        return {"result": {"function": function, "ok": True}}


def _dispatch(conn: _StubConnection, action: str, **kwargs: Any) -> dict[str, Any]:
    """Call ``_device_connect_dispatch``; fail if it defers to the mesh path."""
    result = _raw_dispatch(conn, action, **kwargs)
    assert result is not None, f"{action} unexpectedly deferred to the mesh path"
    return result


def _raw_dispatch(conn: _StubConnection, action: str, **kwargs: Any) -> dict[str, Any] | None:
    """Call the dispatcher and return its raw result (``None`` means mesh-only)."""
    defaults: dict[str, Any] = {
        "target": "",
        "instruction": "",
        "command": "",
        "policy_provider": "mock",
        "policy_port": 0,
        "duration": 30.0,
        "timeout": 5.0,
        "function": "",
        "validated_command": None,
    }
    defaults.update(kwargs)
    with patch("device_connect_agent_tools.connection.get_connection", return_value=conn):
        return rm._device_connect_dispatch(
            action,
            defaults["target"],
            defaults["instruction"],
            defaults["command"],
            defaults["policy_provider"],
            defaults["policy_port"],
            defaults["duration"],
            defaults["timeout"],
            defaults["function"],
            defaults["validated_command"],
        )


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


# --- send ----------------------------------------------------------------


def test_send_requires_target() -> None:
    result = _dispatch(_StubConnection(), "send", command='{"action": "status"}')
    assert result["status"] == "error"
    assert "send requires target" in _text(result)


def test_send_requires_command() -> None:
    result = _dispatch(_StubConnection(), "send", target="dev-1")
    assert result["status"] == "error"
    assert "send requires command" in _text(result)


def test_send_rejects_non_json_command() -> None:
    result = _dispatch(_StubConnection(), "send", target="dev-1", command="not json")
    assert result["status"] == "error"
    assert "not valid JSON" in _text(result)


def test_send_rejects_non_object_command() -> None:
    result = _dispatch(_StubConnection(), "send", target="dev-1", command="[1, 2, 3]")
    assert result["status"] == "error"
    assert "must decode to a JSON object" in _text(result)


def test_send_rejects_command_failing_security_validation() -> None:
    # An action outside the mesh allowlist is rejected before it reaches invoke.
    conn = _StubConnection()
    result = _dispatch(conn, "send", target="dev-1", command='{"action": "rm -rf /"}')
    assert result["status"] == "error"
    assert "send rejected" in _text(result)
    assert conn.calls == []  # never dispatched to the device


def test_send_invokes_validated_action_on_success() -> None:
    conn = _StubConnection()
    result = _dispatch(conn, "send", target="dev-1", command='{"action": "status"}')
    assert result["status"] == "success"
    assert conn.calls == [("dev-1", "status", {})]
    assert "dev-1" in _text(result)


# --- rpc -----------------------------------------------------------------


def test_rpc_requires_target() -> None:
    result = _dispatch(_StubConnection(), "rpc", function="nod")
    assert result["status"] == "error"
    assert "rpc requires target" in _text(result)


def test_rpc_requires_function() -> None:
    result = _dispatch(_StubConnection(), "rpc", target="dev-1")
    assert result["status"] == "error"
    assert "rpc requires function" in _text(result)


def test_rpc_rejects_non_json_params() -> None:
    result = _dispatch(_StubConnection(), "rpc", target="dev-1", function="nod", command="not json")
    assert result["status"] == "error"
    assert "not valid JSON" in _text(result)


def test_rpc_rejects_non_object_params() -> None:
    result = _dispatch(_StubConnection(), "rpc", target="dev-1", function="nod", command='"hello"')
    assert result["status"] == "error"
    assert "must decode to a JSON object" in _text(result)


def test_rpc_rejects_function_name_with_illegal_charset() -> None:
    conn = _StubConnection()
    result = _dispatch(conn, "rpc", target="dev-1", function="bad-name!")
    assert result["status"] == "error"
    assert "rpc rejected" in _text(result)
    assert conn.calls == []


def test_rpc_invokes_device_native_function_on_success() -> None:
    conn = _StubConnection()
    result = _dispatch(conn, "rpc", target="dev-1", function="nod", command='{"angle": 10}')
    assert result["status"] == "success"
    assert conn.calls and conn.calls[0][1] == "nod"
    body = _text(result)
    assert "dev-1.nod(" in body


def test_rpc_with_no_params_defaults_to_empty_dict() -> None:
    conn = _StubConnection()
    result = _dispatch(conn, "rpc", target="dev-1", function="getStatus")
    assert result["status"] == "success"
    assert conn.calls and conn.calls[0][1] == "getStatus"


# --- stop ----------------------------------------------------------------


def test_stop_requires_target() -> None:
    result = _dispatch(_StubConnection(), "stop")
    assert result["status"] == "error"
    assert "stop requires target" in _text(result)


def test_stop_invokes_stop_on_success() -> None:
    conn = _StubConnection()
    result = _dispatch(conn, "stop", target="dev-1")
    assert result["status"] == "success"
    assert conn.calls == [("dev-1", "stop", {})]
    assert "Stop dev-1" in _text(result)


# --- top-level catch -----------------------------------------------------


def test_dispatch_converts_backend_failure_to_error_dict() -> None:
    # A backend exception must surface as a tool-error dict, never propagate.
    conn = _StubConnection(invoke_error=RuntimeError("device offline"))
    result = _dispatch(conn, "stop", target="dev-1")
    assert result["status"] == "error"
    body = _text(result)
    assert "[stop] Device Connect error" in body
    assert "device offline" in body


def test_dispatch_returns_none_for_mesh_only_actions() -> None:
    # subscribe/watch/inbox/unsubscribe are mesh-only; the DC dispatcher defers.
    assert _raw_dispatch(_StubConnection(), "subscribe", target="topic") is None


@pytest.mark.parametrize("action", ["send", "rpc", "stop"])
def test_dispatch_error_messages_are_ascii(action: str) -> None:
    # The no-emoji rule applies to every user-facing tool string.
    result = _dispatch(_StubConnection(), action)
    body = _text(result)
    offenders = {hex(ord(c)) for c in body if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"
