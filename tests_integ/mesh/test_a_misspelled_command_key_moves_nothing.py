"""Integration: a misspelled mesh command key is refused before the peer moves.

Two local Zenoh peers (``STRANDS_MESH_LOCAL_DEV``). Before the unit fix in
``tests/mesh/test_a_command_key_the_peer_does_not_read_is_refused.py`` this send
answered ``success`` after ``30.0s | 1500 steps``: ``durration`` was dropped and
the 30 s default ran.

Requires: eclipse-zenoh, mujoco, MUJOCO_GL=egl (headless)
"""

from __future__ import annotations

import time

import pytest

zenoh = pytest.importorskip("zenoh", reason="mesh integ tests require eclipse-zenoh")
mujoco = pytest.importorskip("mujoco", reason="sim mesh tests require mujoco")

_EXECUTE = {"action": "execute", "instruction": "wave", "policy_provider": "mock"}


@pytest.fixture(autouse=True)
def _mesh_local_dev(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.delenv("STRANDS_MESH", raising=False)


def test_a_misspelled_key_over_the_mesh_is_refused_before_anything_moves():
    """The two-peer repro, end to end: the typo comes back as an error in well under the 30 s."""
    from strands_robots import Robot

    a = Robot("so101", mesh=True, peer_id="qa-keys-a")
    b = Robot("so101", mesh=True, peer_id="qa-keys-b", tool_name="so101_b")
    try:
        time.sleep(1.5)
        start = time.monotonic()
        reply = a.mesh.send("qa-keys-b", {**_EXECUTE, "durration": 0.5}, timeout=10.0)
        elapsed = time.monotonic() - start

        assert reply["status"] == "error", reply
        assert "did you mean 'duration'" in reply["error"]
        assert elapsed < 5.0
        assert b._world.step_count == 0, "the peer ran a rollout it was never asked for"
    finally:
        a.mesh.stop()
        b.mesh.stop()
        a.cleanup()
        b.cleanup()
