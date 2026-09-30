"""The permissive-ACL start gate reads the Zenoh ACL; the pure ``iot`` backend has none.

On ``STRANDS_MESH_BACKEND=iot`` every topic the peer may publish or receive is
bounded by the AWS IoT policy attached to its certificate (the ``strands-robot``
and ``strands-operator`` documents in :mod:`strands_robots.mesh.iot.provision`).
There is no Zenoh session to protect, so the gate that refuses a permissive
Zenoh ACL under mtls has nothing to decide there. Measured on the account
(iot-deep lane, D1): with the four ``provision_robot`` export lines and nothing
else, ``Mesh.start()`` logged "Mesh did NOT start" and ``running`` stayed False
until ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1`` was set, the very variable the
security docs tell operators never to set in production.

The ``bridge`` backend keeps the gate: it has a Zenoh leg.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def stub_robot():
    from types import SimpleNamespace

    return SimpleNamespace(
        name="stub",
        joint_names=["j1"],
        cameras={},
        config=SimpleNamespace(cameras={}, robot_type="stub", mesh=None),
        _hub_sim=None,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_ACL_FILE",
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
        "STRANDS_MESH_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)


def test_pure_iot_backend_does_not_refuse_under_permissive_acl(monkeypatch, stub_robot, caplog):
    """mtls + permissive default ACL + backend=iot: the gate proceeds and says why at INFO."""
    from strands_robots.mesh import core as mesh_core

    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")

    m = mesh_core.Mesh(stub_robot, peer_id="test-iot-gate", peer_type="robot")
    with caplog.at_level(logging.INFO, logger="strands_robots.mesh.core"):
        assert m._refuse_under_permissive_default_acl() is False
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("IoT policy" in msg and "iot" in msg for msg in infos), infos
    assert not any("Mesh did NOT start" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("backend", ["zenoh", "bridge"])
def test_backends_with_a_zenoh_leg_keep_the_gate(monkeypatch, stub_robot, backend):
    """The same permissive shape still refuses wherever a Zenoh session would open."""
    from strands_robots.mesh import core as mesh_core

    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_BACKEND", backend)

    m = mesh_core.Mesh(stub_robot, peer_id=f"test-{backend}-gate", peer_type="robot")
    assert m._refuse_under_permissive_default_acl() is True


def test_iot_gate_skip_still_publishes_the_acl_snapshot(monkeypatch, stub_robot):
    """Skipping the decision must not skip the thread-local snapshot the session builder reads."""
    from strands_robots.mesh import _acl_config
    from strands_robots.mesh import core as mesh_core

    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")

    m = mesh_core.Mesh(stub_robot, peer_id="test-iot-snap", peer_type="robot")
    try:
        assert m._refuse_under_permissive_default_acl() is False
        assert _acl_config._get_thread_snapshot() is not None or m._acl_snapshot is not None
    finally:
        _acl_config._clear_thread_snapshot()
