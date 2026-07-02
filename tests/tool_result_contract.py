"""Strands tool-result contract helper and enforcement primitives.

A Strands tool result is a dict with exactly two author-controlled top-level
keys, ``status`` and ``content`` (the runtime may additionally inject
``toolUseId``). Structured telemetry belongs inside ``content`` as a
``{"json": {...}}`` block alongside the human-readable ``{"text": ...}``
block -- never as extra top-level keys. Extra top-level keys are silently
dropped by the agent runtime and never surface in ``agent.messages``, so any
data placed there is invisible to the agent and to downstream consumers
(mesh, dashboards, evals).

See ``docs/contracts.md`` for the canonical shape and rationale.
"""

from __future__ import annotations

from typing import Any

# ``toolUseId`` is injected by the Strands runtime when a method result is
# wrapped into a ToolResultEvent; it is a legitimate top-level key.
VALID_TOP_LEVEL_KEYS = frozenset({"status", "content", "toolUseId"})

# Author-selectable statuses. ``degraded`` reports partial success (e.g. a
# teleop loop that ran but logged some per-frame errors).
VALID_STATUSES = frozenset({"success", "error", "degraded"})

# The value-bearing content-block variants this contract recognises.
VALID_CONTENT_KEYS = frozenset({"text", "json", "image"})


def assert_strands_tool_result(result: Any) -> None:
    """Assert ``result`` conforms to the Strands tool-result contract.

    Enforces:
      * ``result`` is a dict whose keys are a subset of
        ``{"status", "content", "toolUseId"}`` (no other top-level extras).
      * ``status`` is present and one of ``{"success", "error", "degraded"}``.
      * ``content`` is a non-empty list of dicts, each carrying exactly one of
        ``{"text", "json", "image"}``.

    Args:
        result: The value returned by a tool method / ``@tool`` function.

    Raises:
        AssertionError: If any part of the contract is violated, with a message
            naming the offending keys so the fix is obvious.
    """
    assert isinstance(result, dict), f"tool result must be a dict, got {type(result).__name__}"

    extra = set(result) - VALID_TOP_LEVEL_KEYS
    assert not extra, (
        f"tool result has extra top-level keys {sorted(extra)}; move telemetry "
        f'into content as a {{"json": {{...}}}} block'
    )

    assert "status" in result, "tool result missing 'status'"
    assert result["status"] in VALID_STATUSES, f"status {result['status']!r} not in {sorted(VALID_STATUSES)}"

    assert "content" in result, "tool result missing 'content'"
    content = result["content"]
    assert isinstance(content, list) and content, "content must be a non-empty list"
    for i, block in enumerate(content):
        assert isinstance(block, dict), f"content[{i}] must be a dict, got {type(block).__name__}"
        present = set(block) & VALID_CONTENT_KEYS
        assert len(present) == 1, (
            f"content[{i}] must carry exactly one of {sorted(VALID_CONTENT_KEYS)}, got keys {sorted(block)}"
        )


def tool_json(result: Any) -> dict[str, Any]:
    """Merge every ``{"json": {...}}`` content block of a tool result into one dict.

    Telemetry that used to live as extra top-level keys now lives inside
    ``content`` json blocks. Tests read it back through this helper so they
    assert on the agent-visible payload rather than on dropped top-level keys.

    Args:
        result: A Strands tool result dict.

    Returns:
        The union of all json content blocks (later blocks win on key clash).
    """
    merged: dict[str, Any] = {}
    for block in result.get("content", []):
        if isinstance(block, dict) and isinstance(block.get("json"), dict):
            merged.update(block["json"])
    return merged
