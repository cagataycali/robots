"""Pin test: underscore-prefixed SDK methods (_Call, _CallNoReply, ...)
must be refused before dispatch.

The base ``unitree_sdk2py.rpc.client.Client`` exposes ``_Call(apiId,
parameter)`` and siblings that bypass the prefix-based mutative
classifier: they are classified neither read-only (no ``READONLY_WHITELIST``
entry, no ``Get``/``Check`` prefix) nor mutative (no ``MUTATIVE_PREFIXES``
match), so the gate at ``high_danger or mutative`` would not fire and
``_execute``'s ``getattr(client, operation_name)`` would dispatch them
happily -- delivering the same physical command the named method carries,
with no operator prompt, no allowlist check and no audit row.

The fix refuses any ``operation_name`` starting with ``_`` before the
classifier runs, matching the ``list_operations`` filter that already
makes them undiscoverable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real SDK, no real mesh, no real audit trail."""
    monkeypatch.delenv("STRANDS_UNITREE_COMMAND_ALLOW", raising=False)
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "false")


_PRIVATE_METHODS = ["_Call", "_CallNoReply", "_CallBinary", "_send"]


@pytest.mark.parametrize("private_name", _PRIVATE_METHODS)
def test_private_sdk_method_is_refused_before_dispatch(
    private_name: str,
) -> None:
    """A ``_``-prefixed operation name is refused with ``dispatched: False``
    and ``_execute`` is never called."""
    import strands_robots.tools.g1.use_unitree as uu

    with patch.object(uu, "_execute") as mock_exec:
        result = uu.use_unitree("loco", private_name, {})

    assert result["status"] == "error", f"expected error, got {result}"
    assert result.get("dispatched") is False, "must report dispatched=False"
    assert "private SDK methods" in result["message"]
    assert "list_operations" in result["message"]
    mock_exec.assert_not_called()


def test_public_mutative_method_is_not_blocked_by_private_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public method that IS mutative still reaches _execute (the guard
    does not over-refuse)."""
    import strands_robots.tools.g1.use_unitree as uu

    monkeypatch.setenv("STRANDS_UNITREE_COMMAND_ALLOW", "*")

    with patch.object(
        uu,
        "_execute",
        return_value={"ok": True, "result": {"api_id": 7105}},
    ) as mock_exec:
        result = uu.use_unitree("loco", "SetVelocity", {"vx": 0.1})

    # SetVelocity must have reached _execute
    mock_exec.assert_called_once()
    # It was NOT refused by the private guard
    assert "private SDK" not in result.get("message", "")


def test_dunder_method_is_also_refused() -> None:
    """Dunder names (``__init__``, ``__repr__``) start with ``_`` and are
    refused the same way."""
    import strands_robots.tools.g1.use_unitree as uu

    with patch.object(uu, "_execute") as mock_exec:
        result = uu.use_unitree("loco", "__init__", {})

    assert result["status"] == "error"
    assert result.get("dispatched") is False
    mock_exec.assert_not_called()
