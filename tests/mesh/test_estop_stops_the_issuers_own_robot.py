"""``Mesh.emergency_stop`` must halt the robot in the issuing process too.

Measured on main with a real local-dev session (STRANDS_MESH_LOCAL_DEV=true):
a ``Mesh(robot, "issuer-1")`` whose ``robot.stop_task`` records calls answered
``emergency_stop()`` after 3.02 s with the remote peers' replies, lockout
engaged -- and ``robot.stop_task`` had been called zero times. ``broadcast``
publishes on the command topic and ``_on_cmd`` drops every envelope whose
``sender_id`` is our own, so the one robot the operator is standing next to was
the one robot the e-stop never reached.

The second half: ``_dispatch`` refused every action but ``status``/``resume``
while the lockout was engaged, ``stop`` included -- so a second e-stop arriving
at an already locked-out peer was "command rejected" instead of a halt.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh.core import Mesh, _peers_that_did_not_stop, _security


class _Arm:
    def __init__(self, *, fail: bool = False) -> None:
        self.stop_calls = 0
        self.fail = fail

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "running"}

    def stop_task(self) -> dict[str, Any]:
        self.stop_calls += 1
        if self.fail:
            raise RuntimeError("bus went away")
        return {"status": "success", "content": [{"text": "stopped"}]}


def _quiet(mesh: Mesh, monkeypatch: pytest.MonkeyPatch, remote: list[dict[str, Any]]) -> None:
    """Give *mesh* a running wire that answers with *remote* and publishes nothing.

    ``emergency_stop`` refuses outright on a mesh that never started (an e-stop
    that reached no peer must not read as "asked, nobody answered"); that
    contract is pinned in ``tests/mesh/test_mesh_rpc.py``. Here the wire is up
    and the question is what the ISSUER's own robot does.
    """
    mesh._running = True
    monkeypatch.setattr(mesh, "broadcast", lambda cmd, timeout=5.0: list(remote))
    monkeypatch.setattr(mesh, "_publish_safety_envelope", lambda *a, **k: None)
    monkeypatch.setattr(mesh, "publish_safety_event", lambda *a, **k: None)


def test_the_local_robot_is_stopped_and_answers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _Arm()
    mesh = Mesh(arm, peer_id="issuer-1")
    remote = [{"type": "response", "responder_id": "peer-2", "result": {"status": "success"}}]
    _quiet(mesh, monkeypatch, remote)

    responses = mesh.emergency_stop()

    assert arm.stop_calls == 1
    assert responses[0]["responder_id"] == "issuer-1"
    assert responses[0]["result"] == {"status": "success", "content": [{"text": "stopped"}]}
    assert responses[1:] == remote
    assert mesh._estop_lockout.is_set()


def test_a_local_stop_that_raises_is_counted_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _Arm(fail=True)
    mesh = Mesh(arm, peer_id="issuer-1")
    _quiet(mesh, monkeypatch, [])

    responses = mesh.emergency_stop()

    assert arm.stop_calls == 1
    assert _peers_that_did_not_stop(responses) == {"issuer-1"}


def test_a_local_dispatch_that_raises_answers_instead_of_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop must answer, not raise -- including the issuer's own.

    ``_dispatch`` catches a failing ``stop_task`` itself, so this drives the
    outer guard: whatever else the local stop path raises must become an
    ``ok=False`` answer the accounting can read, never an exception that
    abandons the broadcast and the lockout envelope.
    """
    arm = _Arm()
    mesh = Mesh(arm, peer_id="issuer-1")
    _quiet(mesh, monkeypatch, [])
    monkeypatch.setattr(mesh, "_dispatch", lambda cmd: (_ for _ in ()).throw(RuntimeError("engine gone")))

    responses = mesh.emergency_stop()

    assert _peers_that_did_not_stop(responses) == {"issuer-1"}
    assert "engine gone" in responses[0]["result"]["error"]


def test_a_peer_without_a_robot_adds_no_local_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(None, peer_id="gateway-1", peer_type="gateway")
    _quiet(mesh, monkeypatch, [])
    assert mesh.emergency_stop() == []


def test_stop_is_admitted_while_the_lockout_is_engaged() -> None:
    arm = _Arm()
    mesh = Mesh(arm, peer_id="peer-2")
    mesh._estop_lockout.set()

    assert mesh._dispatch({"action": "stop"})["status"] == "success"
    assert arm.stop_calls == 1
    with pytest.raises(_security.LockoutError):
        mesh._dispatch({"action": "tell", "text": "wave"})
