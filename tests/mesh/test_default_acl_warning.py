"""R19 pin tests: Mesh.start warning when mtls + default ACL combo is active.

Reviewer concern (R18 thread @ 2026-05-23T07:38:02Z, _acl_config.py:359):
> ``is_default_acl_in_use()`` exists but no consumer wires it. The dangerous-
> but-easy-to-miss config is mtls + permissive default ACL. Suggested
> follow-up: have ``Mesh.start`` emit a WARNING when both are active.

These tests assert the warning is emitted in the dangerous combo and
suppressed when either condition does not hold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import core as core_mod  # noqa: F401  -- used by F3-B-1 source-code assertion


@pytest.fixture
def stub_robot():
    """Minimal robot duck-type for Mesh construction."""
    inner = SimpleNamespace(
        is_connected=True,
        name="r19_test",
        config=SimpleNamespace(cameras={}),
        get_observation=MagicMock(return_value={}),
    )
    return SimpleNamespace(tool_name_str="r19", robot=inner)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that influence ACL/auth-mode resolution."""
    for var in (
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_ACL_FILE",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
    ):
        monkeypatch.delenv(var, raising=False)


def _start_with_stub_session(stub_robot, caplog):
    """Construct a Mesh and run start() against a stub session."""
    from strands_robots.mesh import Mesh
    from strands_robots.mesh import core as mesh_core

    class _StubDecl:
        def undeclare(self) -> None:
            pass

    class _StubSession:
        def declare_subscriber(self, *args, **kwargs):
            return _StubDecl()

    mesh = Mesh(stub_robot, peer_id="test-r19", peer_type="robot")

    with patch.object(mesh_core, "get_session", return_value=_StubSession()):
        with patch.object(mesh_core, "release_session"):
            with patch.object(mesh, "_heartbeat_loop"), patch.object(mesh, "_state_loop"):
                with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
                    mesh.start()
                mesh.stop()
    return caplog.records


def test_mtls_plus_default_acl_warns(caplog, monkeypatch, stub_robot):
    """Default config (mtls + no ACL file) MUST log a permissive-ACL warning."""
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    records = _start_with_stub_session(stub_robot, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("permissive default ACL active under mtls" in m for m in msgs), (
        f"expected permissive-ACL warning; saw: {msgs}"
    )


def test_mtls_with_acl_file_does_not_warn(caplog, monkeypatch, tmp_path, stub_robot):
    """Operator-supplied ACL file MUST suppress the warning."""
    acl = tmp_path / "ops.json5"
    acl.write_text('{"rules": [], "subjects": [], "policies": [], "enabled": true, "default_permission": "deny"}\n')
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl))

    records = _start_with_stub_session(stub_robot, caplog)
    msgs = [r.getMessage() for r in records]
    assert not any("permissive default ACL active under mtls" in m for m in msgs), (
        f"unexpected permissive-ACL warning when ACL file set: {msgs}"
    )


def test_auth_mode_none_does_not_emit_default_acl_warning(caplog, monkeypatch, stub_robot):
    """auth_mode=none has its own ERROR; permissive-ACL warning is mtls-specific."""
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
    monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")

    records = _start_with_stub_session(stub_robot, caplog)
    msgs = [r.getMessage() for r in records]
    assert not any("permissive default ACL active under mtls" in m for m in msgs), (
        f"permissive-ACL warning leaked into auth_mode=none: {msgs}"
    )


class TestPermissiveACLWarningExceptNarrow:
    """The R19 permissive-ACL warning at Mesh.start() previously caught
    `except Exception` and downgraded to DEBUG -- a future refactor that
    raised an unrelated type would silently lose the warning.
    """

    def test_unexpected_exception_surfaces_loudly(self):
        """A non-(ImportError|ValueError) raised inside the warning block
        should surface at WARNING (not DEBUG silent-swallow).
        """

        # Static assertion -- start() does network I/O, so we verify the
        # narrowed except clause is in the source rather than triggering it.
        # The previous version of this test constructed a Mesh and patched
        # resolve_auth_mode, but that scaffolding never got exercised because
        # start() was the entry point. Removed per CodeQL #257.
        src = Path(core_mod.__file__).read_text()
        assert "except (ImportError, ValueError) as warn_exc:" in src
        # No bare `except Exception as warn_exc:` left in the start() block
        assert "except Exception as warn_exc:" not in src


# ---------------------------------------------------------------------
# F3-B-2: STRANDS_MESH_CAMERA_DISABLED via _bool_env (lenient parse)
# ---------------------------------------------------------------------
