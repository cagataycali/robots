"""Pin: robot_mesh subscribe is allowlist-scoped and inbox reads are audited.

Defence in depth for the cross-peer telemetry-leak surface. Even on a mesh
running the permissive default ACL, the tool layer must:

* allow subscribing only to low-impact shared topic classes by default,
* reject subscribing to another peer's cmd / state / camera streams,
* let operators extend the allowlist via STRANDS_MESH_SUBSCRIBE_ALLOW,
* audit every inbox read (which sub, how many frames).

These fail on pre-fix code (subscribe accepted any key expr; inbox reads
were never audited).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt


@pytest.fixture(autouse=True)
def _reset():
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    rmt._reset_subscribe_allowlist_cache()
    yield
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    rmt._reset_subscribe_allowlist_cache()


def _make_ctx(response: str = "y") -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


def _call(action, *, ctx=None, **kw):
    fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
    return fn(action=action, tool_context=ctx or _make_ctx(), **kw)


def _stub_mesh() -> MagicMock:
    m = MagicMock()
    m.subscribe.return_value = "sub-name"
    m.inbox = {}
    return m


# --- matcher unit tests -------------------------------------------------


def test_ke_matches_exact():
    assert rmt._ke_matches("**/presence", "**/presence") is True


def test_ke_matches_trailing_doublestar():
    assert rmt._ke_matches("**/safety/**", "**/safety/event") is True
    assert rmt._ke_matches("**/safety/**", "**/safety/estop") is True
    assert rmt._ke_matches("**/safety/**", "**/safety") is True


def test_ke_matches_rejects_unrelated():
    assert rmt._ke_matches("**/presence", "reachy/cmd") is False
    assert rmt._ke_matches("**/safety/**", "**/state/x") is False


def test_default_allowlist_blocks_cmd_and_state():
    assert rmt._is_allowed_subscribe_target("reachy/cmd") is False
    assert rmt._is_allowed_subscribe_target("peer-b/state/joints") is False
    assert rmt._is_allowed_subscribe_target("peer-b/camera/rgb") is False


def test_default_allowlist_permits_shared_classes():
    assert rmt._is_allowed_subscribe_target("**/presence") is True
    assert rmt._is_allowed_subscribe_target("**/health") is True
    assert rmt._is_allowed_subscribe_target("**/safety/event") is True


# --- dispatcher integration --------------------------------------------


def test_subscribe_allows_presence():
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="**/presence", name="p")
    assert r["status"] == "success"
    m.subscribe.assert_called_once()


def test_subscribe_blocks_cmd_stream():
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="victim/cmd", name="x")
    assert r["status"] == "error"
    assert "allowed topic set" in r["content"][0]["text"]
    m.subscribe.assert_not_called()


def test_subscribe_env_extends_allowlist(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "**/state/**")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="**/state/joints", name="s")
    assert r["status"] == "success"
    m.subscribe.assert_called_once()


def test_inbox_read_is_audited():
    m = _stub_mesh()
    m.inbox = {"sub-a": [("topic", {"x": 1}), ("topic", {"x": 2})]}
    with (
        patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m),
        patch("strands_robots.tools.robot_mesh._audit_tool_action") as audit,
    ):
        r = _call("inbox", name="sub-a")
    assert r["status"] == "success"
    # An inbox read must emit exactly one audit event recording the count.
    inbox_audits = [c for c in audit.call_args_list if c.args and c.args[0] == "inbox"]
    assert inbox_audits, "inbox read was not audited"
    # detail string carries the read count
    assert any("read=2" in (c.args[3] if len(c.args) > 3 else "") for c in inbox_audits)
