"""Mesh.start() must emit a loud WARNING when multicast scouting
is enabled via STRANDS_MESH_MULTICAST=true.

Multicast scouting operates below the mTLS/ACL layer: any device on the LAN can
attract the entire fleet without credentials. The default is
gossip-only (MULTICAST=false), which is safe and silent. When an operator opts
into the dangerous posture we want an explicit, logged signal.

These tests drive the full Mesh.start() flow with stubbed session/loops (same
harness as test_acl_snapshot_toctou) and assert on caplog.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh import core as mesh_core

_MULTICAST_MARKER = "STRANDS_MESH_MULTICAST=true"


class _StubDecl:
    def undeclare(self) -> None:
        pass


class _StubSession:
    def declare_subscriber(self, *args, **kwargs):
        return _StubDecl()


def _make_mesh(peer_id: str) -> Mesh:
    robot = SimpleNamespace(
        tool_name_str=peer_id,
        robot=SimpleNamespace(
            is_connected=True,
            name=f"{peer_id}_test",
            config=SimpleNamespace(cameras={}),
            get_observation=MagicMock(return_value={}),
        ),
    )
    return Mesh(robot, peer_id=peer_id, peer_type="robot")


def _run_start(mesh: Mesh, caplog: pytest.LogCaptureFixture) -> None:
    with patch.object(mesh_core, "get_session", return_value=_StubSession()):
        with patch.object(mesh_core, "release_session"):
            with patch.object(mesh, "_heartbeat_loop"), patch.object(mesh, "_state_loop"):
                with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
                    mesh.start()
                mesh.stop()


def test_multicast_enabled_emits_warning(monkeypatch, caplog):
    """STRANDS_MESH_MULTICAST=true -> a WARNING naming the flag is emitted."""
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
    # Keep the H-1 override-code warning out of the way so we assert only ours.
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")

    mesh = _make_mesh("mc-on")
    _run_start(mesh, caplog)

    multicast_warnings = [r.getMessage() for r in caplog.records if _MULTICAST_MARKER in r.getMessage()]
    assert multicast_warnings, f"expected a multicast WARNING when STRANDS_MESH_MULTICAST=true, got: {caplog.messages}"
    msg = multicast_warnings[0]
    assert "[safety]" in msg
    assert "224.0.0.224:7446" in msg
    assert "mc-on" in msg  # peer_id is interpolated


def test_multicast_default_is_silent(monkeypatch, caplog):
    """Default (flag unset) -> no multicast warning (safe posture stays quiet)."""
    monkeypatch.delenv("STRANDS_MESH_MULTICAST", raising=False)
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")

    mesh = _make_mesh("mc-default")
    _run_start(mesh, caplog)

    assert not any(_MULTICAST_MARKER in m for m in caplog.messages), (
        f"multicast warning fired with the flag unset (default should be silent): {caplog.messages}"
    )


def test_multicast_false_is_silent(monkeypatch, caplog):
    """Explicit STRANDS_MESH_MULTICAST=false -> no warning."""
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "false")
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")

    mesh = _make_mesh("mc-off")
    _run_start(mesh, caplog)

    assert not any(_MULTICAST_MARKER in m for m in caplog.messages), (
        f"multicast warning fired with STRANDS_MESH_MULTICAST=false: {caplog.messages}"
    )
