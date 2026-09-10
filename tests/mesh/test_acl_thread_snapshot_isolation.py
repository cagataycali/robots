"""A mesh test leaves no ACL snapshot behind for the test that runs next.

:meth:`strands_robots.mesh.core.Mesh._refuse_under_permissive_default_acl`
stashes a thread-local ``(acl, auth_mode)`` snapshot before it decides whether
to refuse, and only ``Mesh.start``'s ``finally`` clears it. A test that calls
the gate directly never reaches that ``finally``, so ``auth_mode="mtls"``
survives into later tests on the same thread, where every
``session._build_config()`` reads mTLS whatever the environment says.

The autouse fixture in ``tests/mesh/conftest.py`` clears the snapshot around
every mesh test. These two cases pin that: the first deliberately leaks the way
the ACL tests do, the second asserts the suite still starts clean. They run in
definition order, so the second observes whatever the first left behind -- drop
the fixture and it fails.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import _acl_config


@pytest.fixture
def stub_robot():
    """Minimal robot duck-type for Mesh construction."""
    inner = SimpleNamespace(
        is_connected=True,
        name="acl_snapshot_isolation",
        config=SimpleNamespace(cameras={}),
        get_observation=MagicMock(return_value={}),
    )
    return SimpleNamespace(tool_name_str="snap", robot=inner)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the env vars that influence ACL / auth-mode resolution."""
    for var in (
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_ACL_FILE",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestTheSnapshotDoesNotOutliveTheTestThatStashedIt:
    """Ordered pair: one test leaks, the next one must not see it."""

    def test_calling_the_gate_directly_stashes_the_snapshot(self, monkeypatch, stub_robot) -> None:
        """The gate stashes mTLS on the thread and returns the refusal."""
        from strands_robots.mesh import core as mesh_core

        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")

        mesh = mesh_core.Mesh(stub_robot, peer_id="snapshot-isolation", peer_type="robot")
        assert mesh._refuse_under_permissive_default_acl() is True
        # Observed inside the test: this is the state the next test must not inherit.
        assert _acl_config._get_thread_auth_mode() == "mtls"

    def test_the_next_test_starts_with_no_snapshot(self) -> None:
        """Nothing from the sibling above is still on the thread."""
        assert _acl_config._get_thread_auth_mode() is None
        assert _acl_config._get_thread_snapshot() is None
