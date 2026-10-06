"""Regression tests: every wire verb that moves metal is gated, and the approval names its sender.

Two gaps in the receiving side's operator gate (``Mesh._wire_motion_refusal``):

* ``reset`` (every joint to the home pose at once) and ``step`` were dispatched
  with no approval on a hardware peer. The dashboard already lists ``reset``
  among the actions that start motion; the wire did not, so the two surfaces
  disagreed on what needs a yes. One set now owns the answer
  (:data:`strands_robots._command_gate.PHYSICAL_MOTION_ACTIONS`) and both read it.
* The approval was not bound to who asked. ``STRANDS_ROBOT_COMMAND_ALLOW`` and a
  dashboard grant approved a verb for every publisher on the mesh, and the
  envelope's ``sender_id`` is whatever the publisher wrote. A hardware peer now
  attributes a motion command to a sender first: the envelope must name one, the
  sample must carry the publisher's TLS-bound session id, and that id must be the
  one the sender announced its presence from (``Mesh.peer_wire_zid``). A command
  that cannot be attributed is refused before any approval is consulted.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots import _motion_grants
from strands_robots.mesh.core import Mesh, WireSource

LEADER_ZID = "a1b2c3d4e5f60718"
OTHER_ZID = "0f0e0d0c0b0a0908"


class _HardwareArm:
    """Duck-typed real-hardware host that also answers ``reset`` and ``step``."""

    tool_name_str = "so101"

    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []
        self.resets = 0
        self.steps: list[int] = []
        self.following: list[tuple[str, str]] = []

    def _execute_task_sync(self, instruction: str, **kw: Any) -> dict[str, Any]:
        self.executed.append({"instruction": instruction, **kw})
        return {"status": "success", "content": [{"text": "ran"}]}

    def reset(self) -> dict[str, Any]:
        self.resets += 1
        return {"status": "success", "reset": True}

    def step(self, n: int = 1) -> dict[str, Any]:
        self.steps.append(n)
        return {"status": "success", "steps": n}

    def start_teleop_receive(self, source: str, dev: str = "leader") -> dict[str, Any]:
        self.following.append((source, dev))
        return {"status": "success"}

    def stop_teleop(self, dev: str | None = None) -> dict[str, Any]:
        return {"status": "success"}


class _SimWorld(_HardwareArm):
    tool_name_str = "sim"
    _world = object()

    def list_robots(self) -> list[str]:
        return ["so101"]

    def run_policy(self, *a: Any, **kw: Any) -> dict[str, Any]:
        return {"status": "success"}


class _Zid:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def _sample(payload: dict[str, Any], *, zid: str | None, key: str = "strands/so101-1/cmd") -> Any:
    body = SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    source_info = None if zid is None else SimpleNamespace(source_id=SimpleNamespace(zid=_Zid(zid)))
    return SimpleNamespace(payload=body, source_info=source_info, key_expr=key)


def _presence(peer: str, *, zid: str | None) -> Any:
    return _sample({"robot_id": peer, "robot_type": "operator", "timestamp": time.time()}, zid=zid)


def _envelope(cmd: dict[str, Any], *, sender: str | None = "leader-1") -> dict[str, Any]:
    data: dict[str, Any] = {"turn_id": "turn-1", "command": cmd, "timestamp": time.time()}
    if sender is not None:
        data["sender_id"] = sender
    return data


def _bound(sender: str = "leader-1", zid: str | None = LEADER_ZID) -> WireSource:
    return WireSource(sender_id=sender, wire_zid=zid, leg="lan")


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()


@pytest.fixture
def arm() -> _HardwareArm:
    return _HardwareArm()


@pytest.fixture
def mesh(arm: _HardwareArm) -> Mesh:
    m = Mesh(arm, peer_id="so101-1", peer_type="robot")
    # The leader announced itself from LEADER_ZID; its commands must arrive from there.
    m._on_presence(_presence("leader-1", zid=LEADER_ZID))
    return m


@pytest.fixture
def audits(monkeypatch: pytest.MonkeyPatch, mesh: Mesh) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(mesh, "_audit_local", lambda event, payload: seen.append((event, payload)))
    monkeypatch.setattr(mesh, "_reply", lambda *a, **k: None)
    return seen


class TestResetAndStepAreGatedLikeEveryOtherMotionVerb:
    @pytest.mark.parametrize("action", ["reset", "step"])
    def test_reset_and_step_over_the_wire_are_refused_without_operator_approval(
        self, mesh: Mesh, arm: _HardwareArm, audits: list, action: str
    ) -> None:
        out = mesh._dispatch({"action": action}, source=_bound())

        assert "error" in out, out
        assert "operator approval" in out["error"]
        assert arm.resets == 0 and arm.steps == []
        assert [e for e, _ in audits] == ["wire_motion_refused"]
        assert audits[0][1]["action"] == action

    def test_an_allowlisted_reset_from_a_bound_sender_proceeds(
        self, mesh: Mesh, arm: _HardwareArm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset")

        assert mesh._dispatch({"action": "reset"}, source=_bound())["status"] == "success"
        assert arm.resets == 1
        assert "error" in mesh._dispatch({"action": "step", "steps": 2}, source=_bound())
        assert arm.steps == []

    def test_a_simulation_peer_resets_and_steps_without_approval(self) -> None:
        sim = _SimWorld()
        mesh = Mesh(sim, peer_id="sim-1", peer_type="simulation")

        assert mesh._dispatch({"action": "reset"})["status"] == "success"
        assert mesh._dispatch({"action": "step", "steps": 3})["status"] == "success"
        assert sim.resets == 1 and sim.steps == [3]


class TestApprovalIsBoundToTheAuthenticatedSender:
    """With the verb allowlisted on the robot host, only an attributable command proceeds."""

    @pytest.fixture(autouse=True)
    def _allow_execute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "execute")

    def _run(self, mesh: Mesh, data: dict[str, Any], *, zid: str | None) -> None:
        mesh._exec_cmd(data, wire_zid=zid, leg="lan")

    def test_a_command_from_the_senders_announced_session_proceeds(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        self._run(
            mesh, _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"}), zid=LEADER_ZID
        )

        assert [e["instruction"] for e in arm.executed] == ["wave"]
        assert "wire_motion_refused" not in [e for e, _ in audits]

    def test_a_command_from_another_session_than_the_sender_announced_is_refused(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        self._run(
            mesh, _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"}), zid=OTHER_ZID
        )

        assert arm.executed == []
        refused = [p for e, p in audits if e == "wire_motion_refused"]
        assert len(refused) == 1
        assert refused[0]["sender"] == "leader-1"
        assert refused[0]["wire_zid"] == OTHER_ZID
        assert "session" in refused[0]["reason"]

    def test_a_command_whose_sample_carries_no_publisher_session_is_refused(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        self._run(mesh, _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"}), zid=None)

        assert arm.executed == []
        assert [e for e, _ in audits if e == "wire_motion_refused"] == ["wire_motion_refused"]

    def test_a_sender_that_never_announced_presence_is_refused(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        self._run(
            mesh,
            _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"}, sender="stranger"),
            zid=OTHER_ZID,
        )

        assert arm.executed == []
        refused = [p for e, p in audits if e == "wire_motion_refused"]
        assert refused and refused[0]["sender"] == "stranger"

    def test_a_motion_command_that_names_no_sender_is_refused(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        self._run(
            mesh,
            _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"}, sender=None),
            zid=LEADER_ZID,
        )

        assert arm.executed == []
        assert [e for e, _ in audits if e == "wire_motion_refused"] == ["wire_motion_refused"]

    def test_a_command_on_a_transport_without_publisher_identity_is_refused(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        mesh._exec_cmd(
            _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"}), wire_zid=None, leg="iot"
        )

        assert arm.executed == []
        refused = [p for e, p in audits if e == "wire_motion_refused"]
        assert refused and "iot" in refused[0]["reason"]

    def test_a_local_dispatch_with_no_wire_source_does_not_move_hardware(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        out = mesh._dispatch({"action": "execute", "instruction": "wave"})

        assert "error" in out and arm.executed == []

    def test_the_wire_source_is_read_from_the_sample_not_the_body(
        self, mesh: Mesh, arm: _HardwareArm, audits: list
    ) -> None:
        """A publisher cannot vouch for itself by writing a session id into the envelope."""
        data = _envelope({"action": "execute", "instruction": "wave", "policy_provider": "mock"})
        data["wire_zid"] = LEADER_ZID
        data["source_zid"] = LEADER_ZID

        mesh._on_cmd(_sample(data, zid=OTHER_ZID))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(e == "wire_motion_refused" for e, _ in audits):
            time.sleep(0.02)

        assert arm.executed == []
        assert any(e == "wire_motion_refused" for e, _ in audits)

    def test_a_dashboard_grant_is_only_spent_by_an_attributed_command(self, mesh: Mesh, arm: _HardwareArm) -> None:
        call = {"action": "teleop_receive", "source_peer_id": "leader-1", "device_name": "leader"}
        _motion_grants.deposit_grant("so101", call)

        assert "error" in mesh._dispatch(dict(call), source=_bound(zid=OTHER_ZID))
        assert arm.following == []
        assert mesh._dispatch(dict(call), source=_bound())["status"] == "success"
        assert arm.following == [("leader-1", "leader")]

    def test_stopping_needs_no_source(self, mesh: Mesh) -> None:
        assert mesh._dispatch({"action": "teleop_stop", "device_name": "leader"}) == {"status": "success"}
