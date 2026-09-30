"""Regression tests: a wire command that moves REAL hardware passes the operator gate.

The agent tool gates ``execute`` / ``start`` on a hardware ``Robot`` through
:func:`strands_robots._command_gate.gate_motion`, and the sending ``robot_mesh``
tool asks its own operator before it publishes. The receiving peer trusted both:
``Mesh._dispatch`` handed a wire ``execute`` / ``start`` straight to
``_execute_task_sync`` / ``start_task``, and ``teleop_receive`` opened an
unbounded input stream onto the motor bus with no approval at all, so any peer
admitted to the mesh could drive a physical arm with nobody asked.

The wire is the trust boundary. A hardware peer now runs the same allowlist ->
bypass -> refuse path on the receiving side (``STRANDS_ROBOT_COMMAND_ALLOW``
names the verbs an operator pre-approves on the robot host, a dashboard grant for
the exact call is spent), and with none of those it refuses with a reason that
says how to approve. Simulation peers are never gated: they move no metal.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots import _motion_grants
from strands_robots.mesh.core import Mesh


class _HardwareArm:
    """Duck-typed real-hardware host: the attributes ``Mesh._dispatch`` reaches for."""

    tool_name_str = "so101"

    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []
        self.started: list[dict[str, Any]] = []
        self.following: list[tuple[str, str]] = []

    def _execute_task_sync(self, instruction: str, **kw: Any) -> dict[str, Any]:
        self.executed.append({"instruction": instruction, **kw})
        return {"status": "success", "content": [{"text": "ran"}]}

    def start_task(self, instruction: str, **kw: Any) -> dict[str, Any]:
        self.started.append({"instruction": instruction, **kw})
        return {"status": "success", "content": [{"text": "started"}]}

    def start_teleop_receive(self, source: str, dev: str = "leader") -> dict[str, Any]:
        self.following.append((source, dev))
        return {"status": "success", "content": [{"text": f"following {source}/{dev}"}]}

    def stop_teleop(self, dev: str | None = None) -> dict[str, Any]:
        return {"status": "success"}


class _SimWorld(_HardwareArm):
    """A simulation host: ``run_policy`` + ``_world`` + ``list_robots`` is the sim duck."""

    tool_name_str = "sim"
    _world = object()

    def list_robots(self) -> list[str]:
        return ["so101"]

    def run_policy(self, *a: Any, **kw: Any) -> dict[str, Any]:
        return {"status": "success"}


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
    return Mesh(arm, peer_id="so101-1", peer_type="robot")


@pytest.fixture
def audits(monkeypatch: pytest.MonkeyPatch, mesh: Mesh) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(mesh, "_audit_local", lambda event, payload: seen.append((event, payload)))
    return seen


class TestHardwareRefusesUnapprovedMotion:
    @pytest.mark.parametrize("action", ["execute", "start"])
    def test_policy_rollout_over_the_wire_is_refused_with_a_remedy(
        self, mesh: Mesh, arm: _HardwareArm, audits: list, action: str
    ) -> None:
        out = mesh._dispatch({"action": action, "instruction": "wave", "policy_provider": "mock"})

        assert "error" in out, out
        assert "operator approval" in out["error"]
        assert "STRANDS_ROBOT_COMMAND_ALLOW" in out["error"]
        assert arm.executed == [] and arm.started == []
        assert [e for e, _ in audits] == ["wire_motion_refused"]
        assert audits[0][1]["action"] == action

    def test_teleop_receive_over_the_wire_is_refused_before_any_stream_opens(
        self, mesh: Mesh, arm: _HardwareArm, audits: list, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            out = mesh._dispatch({"action": "teleop_receive", "source_peer_id": "evil-leader", "device_name": "leader"})

        assert "error" in out, out
        assert "operator approval" in out["error"]
        assert arm.following == []
        assert [e for e, _ in audits] == ["wire_motion_refused"]
        assert audits[0][1]["source_peer_id"] == "evil-leader"
        assert any("evil-leader" in rec.message for rec in caplog.records)

    def test_teleop_stop_is_never_gated(self, mesh: Mesh) -> None:
        """Stopping must not get harder."""
        assert mesh._dispatch({"action": "teleop_stop", "device_name": "leader"}) == {"status": "success"}


class TestOperatorApprovalIsHonoured:
    def test_allowlisted_verb_on_the_robot_host_proceeds(
        self, mesh: Mesh, arm: _HardwareArm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "execute")

        out = mesh._dispatch({"action": "execute", "instruction": "wave", "policy_provider": "mock"})

        assert out["status"] == "success"
        assert len(arm.executed) == 1

    def test_allowlist_for_one_verb_does_not_cover_another(
        self, mesh: Mesh, arm: _HardwareArm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "execute")

        out = mesh._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1"})

        assert "error" in out
        assert arm.following == []

    def test_star_allowlists_every_motion_verb(
        self, mesh: Mesh, arm: _HardwareArm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "*")

        assert mesh._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1"})["status"] == "success"
        assert arm.following == [("leader-1", "leader")]

    def test_a_dashboard_grant_for_the_exact_call_is_spent_once(self, mesh: Mesh, arm: _HardwareArm) -> None:
        call = {"action": "teleop_receive", "source_peer_id": "leader-1", "device_name": "leader"}
        _motion_grants.deposit_grant("so101", call)

        assert mesh._dispatch(dict(call))["status"] == "success"
        assert "error" in mesh._dispatch(dict(call))
        assert arm.following == [("leader-1", "leader")]

    def test_a_grant_for_one_leader_is_not_spendable_by_another(self, mesh: Mesh, arm: _HardwareArm) -> None:
        _motion_grants.deposit_grant("so101", {"action": "teleop_receive", "source_peer_id": "leader-1"})

        out = mesh._dispatch({"action": "teleop_receive", "source_peer_id": "leader-2"})

        assert "error" in out
        assert arm.following == []

    def test_bypass_tool_consent_lifts_the_gate_with_a_warning(
        self, mesh: Mesh, arm: _HardwareArm, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with caplog.at_level(logging.WARNING):
            out = mesh._dispatch({"action": "start", "instruction": "wave", "policy_provider": "mock"})

        assert out["status"] == "success"
        assert any("BYPASS_TOOL_CONSENT" in rec.message for rec in caplog.records)


class TestSimulationIsNotGated:
    def test_sim_world_teleop_receive_needs_no_approval(self) -> None:
        sim = _SimWorld()
        mesh = Mesh(sim, peer_id="sim-1", peer_type="simulation")

        out = mesh._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1"})

        assert out["status"] == "success"
        assert sim.following == [("leader-1", "leader")]

    def test_sim_child_peer_teleop_receive_needs_no_approval(self) -> None:
        child = _HardwareArm()
        child._sim_parent = _SimWorld()  # type: ignore[attr-defined]
        mesh = Mesh(child, peer_id="sim-1__so101", peer_type="sim_robot")

        out = mesh._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1"})

        assert out["status"] == "success"


def test_teleop_leader_fields_are_part_of_the_grant_identity() -> None:
    """A yes shown for one leader and device must not be spendable by another."""
    a = _motion_grants.grant_key("so101", {"action": "teleop_receive", "source_peer_id": "l1", "device_name": "d"})
    b = _motion_grants.grant_key("so101", {"action": "teleop_receive", "source_peer_id": "l2", "device_name": "d"})
    c = _motion_grants.grant_key("so101", {"action": "teleop_receive", "source_peer_id": "l1", "device_name": "e"})
    assert len({a, b, c}) == 3


def test_the_wire_gate_reads_the_same_allowlist_variable_as_the_agent_tool() -> None:
    """One operator setting on the robot host approves a verb whichever path it arrives by."""
    from strands_robots.hardware_robot import COMMAND_ALLOW_ENV
    from strands_robots.mesh.core import WIRE_MOTION_ALLOW_ENV

    assert WIRE_MOTION_ALLOW_ENV == COMMAND_ALLOW_ENV
