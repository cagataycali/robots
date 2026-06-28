"""Mesh.start() fail-closed and partial-failure cleanup behavior.

Two robustness invariants on the mesh bring-up path that have operator-visible
safety consequences:

* ACL-gate fail-closed: when ACL/auth-mode resolution raises ``ValueError``
  (bad ``STRANDS_MESH_AUTH_MODE`` or an unloadable ACL file), the pre-session
  gate treats the posture as permissive-under-mtls and REFUSES to start rather
  than bringing an un-vetted wire up. A config error must never silently open
  the mesh.
* Subscriber-declare partial failure: if declaring the core subscribers fails
  part-way, every already-declared subscriber is undeclared, the session
  reference is released, and ``start()`` aborts without the node going live.
  An undeclare that itself fails during cleanup must not mask the original
  failure or crash the cleanup loop.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stub_robot() -> SimpleNamespace:
    """Minimal robot duck-type sufficient for Mesh construction."""
    inner = SimpleNamespace(
        is_connected=True,
        name="failclosed_test",
        config=SimpleNamespace(cameras={}),
        get_observation=MagicMock(return_value={}),
    )
    return SimpleNamespace(tool_name_str="fc", robot=inner)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that influence ACL / auth-mode resolution."""
    for var in (
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_ACL_FILE",
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_acl_gate_fail_closed_on_bad_acl_config(
    monkeypatch: pytest.MonkeyPatch, stub_robot: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """A ValueError from ACL resolution fail-closes: refuse to start, no snapshot.

    The gate must not let a malformed ACL/auth-mode configuration through as if
    it were a vetted posture. On ValueError it forces ``auth_mode='mtls'`` +
    ``is_permissive=True`` (so the refuse-to-start branch fires), stashes a
    ``None`` snapshot, and logs an actionable WARNING.
    """
    from strands_robots.mesh import _acl_config
    from strands_robots.mesh import core as mesh_core

    def _raise(_namespace: str) -> tuple[bool, dict]:
        raise ValueError("unloadable ACL file")

    monkeypatch.setattr(_acl_config, "snapshot_acl", _raise)

    m = mesh_core.Mesh(stub_robot, peer_id="test-failclosed-acl", peer_type="robot")
    try:
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            refused = m._refuse_under_permissive_default_acl()

        assert refused is True, "bad ACL config must fail-closed (refuse to start)"
        assert m._acl_snapshot is None, "fail-closed path stashes a None snapshot"
        assert any("ACL gate evaluation failed" in r.getMessage() for r in caplog.records), (
            "operator must see the fail-closed WARNING breadcrumb"
        )
    finally:
        _acl_config._clear_thread_snapshot()


def test_subscriber_declare_failure_cleans_up_and_aborts(
    monkeypatch: pytest.MonkeyPatch, stub_robot: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """A declare_subscriber failure undeclares prior subs, releases, and aborts.

    Drives the partial-failure cleanup branch: the third subscriber declaration
    raises, so the two already-declared subscribers must be undeclared (one of
    which itself fails to undeclare, exercising the best-effort DEBUG path), the
    session reference is released, and the node does not go live.
    """
    from strands_robots.mesh import core as mesh_core

    undeclared: list[str] = []

    class _GoodDecl:
        def undeclare(self) -> None:
            undeclared.append("good")

    class _BadUndeclare:
        def undeclare(self) -> None:
            undeclared.append("bad-attempt")
            raise OSError("undeclare transport error")

    class _StubSession:
        def __init__(self) -> None:
            self._n = 0

        def declare_subscriber(self, *_args, **_kwargs):
            self._n += 1
            if self._n == 1:
                return _BadUndeclare()
            if self._n == 2:
                return _GoodDecl()
            raise RuntimeError("zenoh declare failed")

    released: list[bool] = []
    monkeypatch.setattr(mesh_core, "get_session", lambda: _StubSession())
    monkeypatch.setattr(mesh_core, "release_session", lambda: released.append(True))
    # Force the gate open so start() proceeds to the declare path.
    monkeypatch.setattr(mesh_core.Mesh, "_refuse_under_permissive_default_acl", lambda self: False)

    m = mesh_core.Mesh(stub_robot, peer_id="test-failclosed-declare", peer_type="robot")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
        m.start()

    assert m._running is False, "declare failure must abort start()"
    assert m._has_session_ref is False, "session reference must be cleared on cleanup"
    assert released == [True], "the session must be released exactly once on cleanup"
    assert undeclared == ["bad-attempt", "good"], "all already-declared subs are undeclared (best-effort)"
    assert any("failed to declare subscribers" in r.getMessage() for r in caplog.records), (
        "operator must see the declare-failure WARNING"
    )
