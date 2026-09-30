"""An e-stop acknowledged by a locked peer must not paint that peer clear (f033).

The dashboard stamps ``_lockout_proof`` for a peer whenever it answers an
action :func:`safety_state.proves_clear` admits, on the reasoning that a
locked-out peer would have refused it. ``Mesh._dispatch`` admits four actions
while its lockout is engaged: ``status``, ``resume``, ``stop`` and ``ping``.
The dashboard's exempt set named only the first two, so an operator pressing
STOP ALL against a fleet that was already locked collected an ``ok`` from
every peer, stamped proof for each, and the next snapshot annotated every
still-locked robot ``state="clear"``: the exact false reassurance the Q43
contract exists to prevent, produced by the safety button itself.

Both sides now read one set, :data:`strands_robots.mesh.security.LOCKOUT_ADMITTED_ACTIONS`,
and the peer's behaviour is graded against it here so the two cannot drift apart again.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from strands_robots.dashboard import safety_state
from strands_robots.mesh import security
from strands_robots.mesh.core import Mesh


class _StoppableRobot:
    def stop_task(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "stopped"}]}

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "idle"}]}


def _locked_mesh() -> Mesh:
    mesh = Mesh(_StoppableRobot(), peer_id="arm-1")
    mesh._estop_lockout.set()
    return mesh


class TestStopProvesNothing:
    def test_stop_is_not_proof_of_a_cleared_lockout(self) -> None:
        # A locked peer de-energises on a second stop; answering it says nothing about the lockout.
        assert safety_state.proves_clear("stop") is False

    def test_ping_is_not_proof_either(self) -> None:
        # The mesh layer answers ping without reaching the robot, locked or not.
        assert safety_state.proves_clear("ping") is False

    def test_motion_actions_still_prove_clear(self) -> None:
        for action in ("execute", "start", "step", "set_joints", "teleop_receive", "task"):
            assert safety_state.proves_clear(action) is True, action


class TestOneSetOnBothSides:
    def test_dashboard_exempt_set_is_the_peer_admitted_set(self) -> None:
        assert safety_state.LOCKOUT_EXEMPT_ACTIONS == security.LOCKOUT_ADMITTED_ACTIONS

    @pytest.mark.parametrize("action", sorted(security.LOCKOUT_ADMITTED_ACTIONS - {"resume"}))
    def test_a_locked_peer_answers_every_admitted_action(self, action: str) -> None:
        # ``resume`` needs an override code and is graded by its own module.
        out = _locked_mesh()._dispatch({"action": action})
        assert "error" not in out or out.get("ok") is not False, out

    @pytest.mark.parametrize("action", sorted(security.ALLOWED_ACTIONS - security.LOCKOUT_ADMITTED_ACTIONS))
    def test_a_locked_peer_refuses_everything_else(self, action: str) -> None:
        with pytest.raises(security.LockoutError):
            _locked_mesh()._dispatch({"action": action})

    def test_an_admitted_action_never_stamps_proof(self) -> None:
        # The dashboard rule, read against the peer's rule: no admitted action may prove clear.
        for action in security.LOCKOUT_ADMITTED_ACTIONS:
            assert safety_state.proves_clear(action) is False, action


class TestTheCardStaysLocked:
    def test_a_stop_answered_after_the_estop_leaves_the_verdict_locked(self) -> None:
        # What the fleet snapshot computes for a peer that acknowledged STOP ALL while locked.
        fleet = safety_state.apply_event(safety_state.Lockout(), kind="estop", data={"source": "op"}, now=10.0)
        stop_answered_at = 12.0
        proof_at = stop_answered_at if safety_state.proves_clear("stop") else None
        verdict = safety_state.resolve_peer(fleet, first_seen=1.0, proof_at=proof_at)
        assert verdict.state == "locked", verdict

    def test_the_bridge_does_not_stamp_proof_for_a_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.dashboard.mesh_bridge import MeshBridge

        bridge = MeshBridge.__new__(MeshBridge)
        bridge._running = True
        bridge._lockout_proof = {}
        import threading

        bridge._peers_lock = threading.Lock()

        class _Mesh:
            def send(self, target: str, cmd: dict[str, Any], timeout: float = 0.0) -> dict[str, Any]:
                return {"status": "success", "responder_id": target, "result": {"ok": True}}

        monkeypatch.setattr(bridge, "_safety_mesh", lambda: _Mesh())
        monkeypatch.setattr(bridge, "record_activity", lambda *a, **k: None)
        bridge.send_cmd("arm-1", {"action": "stop"}, source="test", timeout=1.0)
        assert "arm-1" not in bridge._lockout_proof, "a stop acknowledged by a locked peer is not proof"
        bridge.send_cmd("arm-1", {"action": "execute"}, source="test", timeout=1.0)
        assert bridge._lockout_proof["arm-1"] <= time.time()
