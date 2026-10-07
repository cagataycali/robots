"""Regression: a motion approval is for one verified actor, and a name cannot be taken by another certificate.

Path B of the finding, step by step. The receiving robot used to attribute a
wire command to the session id its publisher attached (copyable) and then spend
an approval that named a verb but no sender: ``STRANDS_ROBOT_COMMAND_ALLOW=reset``
admitted every attributed peer, a dashboard grant for ``leader-1``'s pending
command was spent by any peer sending the same shape, and after ten seconds of
silence any peer could announce itself as ``leader-1``.

With signed identity required, the actor is the peer id the signing
certificate speaks for (the id a dashboard deposits its grants for, so a
``<cn>__<suffix>`` child spends its own approvals), and every approval names
its actor: a bare allowlist verb still admits
only a verified signer, ``reset@leader-1`` refuses a correctly signed
``attacker``, a grant deposited for ``leader-1`` is not spent by ``attacker``'s
identical command, and a presence for ``leader-1`` signed by ``attacker``'s
certificate is dropped and audited, silence or not.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from strands_robots import _motion_grants  # noqa: E402
from strands_robots.mesh import core as mesh_core  # noqa: E402
from strands_robots.mesh import session as mesh_session  # noqa: E402
from strands_robots.mesh.core import Mesh  # noqa: E402
from tests._wire_identity import (  # noqa: E402
    arm_receiver,
    identity_for,
    signed_cmd_sample,
    signed_presence_sample,
)
from tests.mesh._pki import EphemeralCA  # noqa: E402

LEADER_ZID = "a1b2c3d4e5f60718a1b2c3d4e5f60718"
ATTACKER_ZID = "0f0e0d0c0b0a09080f0e0d0c0b0a0908"


class _HardwareArm:
    tool_name_str = "so101"

    def __init__(self) -> None:
        self.resets = 0
        self.executed: list[str] = []

    def reset(self) -> dict[str, Any]:
        self.resets += 1
        return {"status": "success", "reset": True}

    def _execute_task_sync(self, instruction: str, **kw: Any) -> dict[str, Any]:
        self.executed.append(instruction)
        return {"status": "success", "content": [{"text": "ran"}]}

    def stop_task(self) -> dict[str, Any]:
        return {"status": "success"}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()
    yield
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()


@pytest.fixture
def world(require_signatures: EphemeralCA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A hardware robot requiring signatures; ``leader-1`` and ``attacker`` hold leaves from its CA."""
    arm = _HardwareArm()
    robot = Mesh(arm, peer_id="so101-1", peer_type="robot")
    arm_receiver(robot, require_signatures)
    audits: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(robot, "_audit_local", lambda event, payload: audits.append((event, payload)))
    monkeypatch.setattr(robot, "_reply", lambda *a, **k: None)
    leader = identity_for(require_signatures, "leader-1", tmp_path / "leaves")
    attacker = identity_for(require_signatures, "attacker", tmp_path / "leaves")
    robot._on_presence(signed_presence_sample(leader, "leader-1", zid=LEADER_ZID))
    robot._on_presence(signed_presence_sample(attacker, "attacker", zid=ATTACKER_ZID))
    return {
        "arm": arm,
        "robot": robot,
        "audits": audits,
        "leader": leader,
        "attacker": attacker,
        "ca": require_signatures,
    }


def _deliver(robot: Mesh, sample: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the command the way the wire does: ``_on_cmd`` verifies, ``_exec_cmd`` dispatches, synchronously."""
    started: list[Any] = []

    class _Inline:
        def __init__(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any], **_: Any) -> None:
            started.append((target, args, kwargs))

        def start(self) -> None:
            target, args, kwargs = started[-1]
            target(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mesh_core.threading, "Thread", _Inline)
        robot._on_cmd(sample)


RESET = {"action": "reset"}
EXECUTE = {"action": "execute", "instruction": "wave", "policy_provider": "mock"}
#: The call as the robot's gate sees it after ``validate_command`` fills the
#: defaults, which is also the shape the dashboard shows the operator (the
#: proxy tool carries every field it sends).
EXECUTE_SHOWN = {**EXECUTE, "policy_host": "localhost", "duration": 30.0}


class TestStep1TheBareAllowlistAdmitsOnlyAVerifiedSigner:
    def test_an_unsigned_reset_is_refused_even_with_the_verb_allowlisted(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset")

        _deliver(world["robot"], signed_cmd_sample(None, "attacker", "so101-1", RESET, zid=ATTACKER_ZID), monkeypatch)

        assert world["arm"].resets == 0
        refused = [p for e, p in world["audits"] if e == "wire_motion_refused"]
        assert len(refused) == 1
        assert refused[0]["reason"] == "the command carries no verifiable signature"
        assert refused[0]["signer"] is None

    def test_a_copied_session_id_does_not_make_the_attacker_the_leader(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@leader-1")

        # Signed by the attacker, body claims leader-1, sample carries leader-1's copied zid.
        _deliver(
            world["robot"],
            signed_cmd_sample(world["attacker"], "leader-1", "so101-1", RESET, zid=LEADER_ZID),
            monkeypatch,
        )

        assert world["arm"].resets == 0
        refused = [p for e, p in world["audits"] if e == "wire_motion_refused"]
        assert refused[0]["signer"] == "attacker"
        assert "does not speak for" in refused[0]["reason"]

    def test_a_signed_reset_from_any_verified_peer_proceeds_on_a_bare_verb_with_a_warning(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset")
        mesh_core._bare_allow_warned.discard("reset")

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            _deliver(world["robot"], signed_cmd_sample(world["attacker"], "attacker", "so101-1", RESET), monkeypatch)

        assert world["arm"].resets == 1
        assert any("reset@<peer>" in r.getMessage() for r in caplog.records)


class TestStep1ScopedAllowlistEntries:
    def test_reset_at_leader_refuses_a_correctly_signed_attacker(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@leader-1")

        _deliver(world["robot"], signed_cmd_sample(world["attacker"], "attacker", "so101-1", RESET), monkeypatch)
        assert world["arm"].resets == 0
        refused = [p for e, p in world["audits"] if e == "wire_motion_refused"]
        assert refused[0]["reason"] == "no operator approval"
        assert refused[0]["signer"] == "attacker"

        _deliver(world["robot"], signed_cmd_sample(world["leader"], "leader-1", "so101-1", RESET), monkeypatch)
        assert world["arm"].resets == 1

    def test_star_at_peer_admits_every_verb_for_that_peer_only(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "*@leader-1")

        _deliver(world["robot"], signed_cmd_sample(world["leader"], "leader-1", "so101-1", EXECUTE), monkeypatch)
        _deliver(world["robot"], signed_cmd_sample(world["attacker"], "attacker", "so101-1", EXECUTE), monkeypatch)

        assert world["arm"].executed == ["wave"]

    def test_the_refusal_names_the_scoped_spelling(self, world: dict[str, Any]) -> None:
        out = world["robot"]._dispatch(
            dict(RESET),
            source=mesh_core.WireSource(sender_id="leader-1", wire_zid=None, leg="lan", signer="leader-1"),
        )

        assert "STRANDS_ROBOT_COMMAND_ALLOW=reset@leader-1" in out["error"]


class TestStep2AGrantIsSpentOnlyByItsActor:
    def test_the_leaders_grant_is_not_spent_by_the_attackers_identical_command(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The dashboard approved leader-1's pending execute: a yes FOR leader-1.
        _motion_grants.deposit_grant("so101", EXECUTE_SHOWN, actor="leader-1")

        _deliver(world["robot"], signed_cmd_sample(world["attacker"], "attacker", "so101-1", EXECUTE), monkeypatch)
        assert world["arm"].executed == []
        assert [p["reason"] for e, p in world["audits"] if e == "wire_motion_refused"] == ["no operator approval"]
        # The grant is still there for the one it was given to.
        assert [g["actor"] for g in _motion_grants.pending_grants()] == ["leader-1"]

        _deliver(world["robot"], signed_cmd_sample(world["leader"], "leader-1", "so101-1", EXECUTE), monkeypatch)
        assert world["arm"].executed == ["wave"]
        assert _motion_grants.pending_grants() == []

    def test_a_dashboard_whose_peer_id_is_a_child_of_its_cn_spends_the_grant_it_deposited(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The spend side, the deposit side and STRANDS_DASHBOARD_PEER_ID agree on one actor: the peer id.

        Two dashboards sharing a certificate CN run as ``<cn>__<suffix>``
        children (the only shape STRANDS_DASHBOARD_PEER_ID offers them). The
        dashboard deposits its grants for its peer id, so the robot must
        spend them for the peer id its signer speaks for, not for the CN,
        or every operator yes is deposited unspendable.
        """
        operator = identity_for(world["ca"], "lab-op", tmp_path / "op")
        dashboard_id = "lab-op__dash-2"
        world["robot"]._on_presence(signed_presence_sample(operator, dashboard_id, zid="c0ffee" * 5 + "c0"))
        # The dashboard approved its own pending execute: a yes FOR its peer id (agent_console deposits bridge.peer_id).
        _motion_grants.deposit_grant("so101", EXECUTE_SHOWN, actor=dashboard_id)

        _deliver(world["robot"], signed_cmd_sample(operator, dashboard_id, "so101-1", EXECUTE), monkeypatch)

        assert world["arm"].executed == ["wave"]
        assert _motion_grants.pending_grants() == []
        assert [p for e, p in world["audits"] if e == "wire_motion_refused"] == []

    def test_verb_at_peer_names_the_peer_id_not_the_certificate_cn(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``reset@lab-op__dash-2`` admits that child; ``reset@lab-op`` (the CN) does not, so the rule is one-way."""
        operator = identity_for(world["ca"], "lab-op", tmp_path / "op")
        dashboard_id = "lab-op__dash-2"
        world["robot"]._on_presence(signed_presence_sample(operator, dashboard_id, zid="c0ffee" * 5 + "c0"))

        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@lab-op")
        _deliver(world["robot"], signed_cmd_sample(operator, dashboard_id, "so101-1", RESET), monkeypatch)
        assert world["arm"].resets == 0
        refused = [p for e, p in world["audits"] if e == "wire_motion_refused"]
        assert refused[0]["reason"] == "no operator approval"
        assert refused[0]["sender"] == dashboard_id and refused[0]["signer"] == "lab-op"

        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", f"reset@{dashboard_id}")
        _deliver(world["robot"], signed_cmd_sample(operator, dashboard_id, "so101-1", RESET), monkeypatch)
        assert world["arm"].resets == 1

    def test_an_in_process_grant_is_not_spendable_over_the_wire(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _motion_grants.deposit_grant("so101", EXECUTE_SHOWN)  # actor None: a yes for this process

        _deliver(world["robot"], signed_cmd_sample(world["leader"], "leader-1", "so101-1", EXECUTE), monkeypatch)

        assert world["arm"].executed == []
        assert _motion_grants.consume_grant("so101", EXECUTE_SHOWN) is True


class TestStep3ANameCannotBeTakenOverByAnotherCertificate:
    def test_a_presence_for_the_leader_signed_by_the_attacker_is_dropped_while_the_leader_is_live(
        self, world: dict[str, Any]
    ) -> None:
        robot = world["robot"]
        before = robot.peer_cert("leader-1")

        robot._on_presence(signed_presence_sample(world["attacker"], "leader-1", zid=ATTACKER_ZID))

        assert robot.peer_cert("leader-1") == before
        assert [e for e, _ in world["audits"]] == ["presence_identity_rejected"]
        assert world["audits"][0][1]["cn"] == "attacker"

    def test_silence_does_not_open_the_name_either(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        robot = world["robot"]
        before = robot.peer_cert("leader-1")
        later = time.monotonic() + mesh_session.PEER_TIMEOUT + 1.0
        monkeypatch.setattr(time, "monotonic", lambda: later)

        robot._on_presence(signed_presence_sample(world["attacker"], "leader-1", zid=ATTACKER_ZID))

        assert robot.peer_cert("leader-1") == before
        assert [e for e, _ in world["audits"]] == ["presence_identity_rejected"]

    def test_a_second_certificate_for_the_same_name_from_the_trust_root_is_refused_while_live(
        self, world: dict[str, Any], tmp_path: Path, require_signatures: EphemeralCA
    ) -> None:
        """A reissued certificate (same CN) may only take over once the first one has gone quiet."""
        robot = world["robot"]
        reissued = identity_for(require_signatures, "leader-1", tmp_path / "reissued")

        robot._on_presence(signed_presence_sample(reissued, "leader-1"))
        assert robot.peer_cert("leader-1") == (world["leader"].cert_sha256, "leader-1")
        assert [e for e, _ in world["audits"]] == ["presence_identity_conflict"]

    def test_a_reissued_certificate_takes_over_after_silence(
        self, world: dict[str, Any], tmp_path: Path, require_signatures: EphemeralCA, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        robot = world["robot"]
        reissued = identity_for(require_signatures, "leader-1", tmp_path / "reissued")
        later = time.monotonic() + mesh_session.PEER_TIMEOUT + 1.0
        monkeypatch.setattr(time, "monotonic", lambda: later)

        robot._on_presence(signed_presence_sample(reissued, "leader-1"))

        assert robot.peer_cert("leader-1") == (reissued.cert_sha256, "leader-1")
        assert world["audits"] == []


class TestStep4ASignedCommandIsForOneTarget:
    """``target_id`` is inside the signed body, so a captured command cannot be pointed at another robot."""

    def test_the_leaders_reset_for_another_robot_replayed_here_is_refused_before_the_gate(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # so101-1 pre-approves the leader's reset: exactly the approval a replay would spend.
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@leader-1")

        # The leader's genuine reset for so101-2, captured off its topic and republished on ours unchanged.
        replayed = signed_cmd_sample(world["leader"], "leader-1", "so101-1", RESET, signed_for="so101-2")
        _deliver(world["robot"], replayed, monkeypatch)

        assert world["arm"].resets == 0
        refused = [p for e, p in world["audits"] if e == "command_refused"]
        assert len(refused) == 1
        assert refused[0]["reason"] == "signed_for_another_peer"
        assert refused[0]["signer"] == "leader-1" and refused[0]["target_id"] == "so101-2"
        assert refused[0]["action"] == "reset" and refused[0]["topic"] == "strands/so101-1/cmd"
        # Never reached the motion gate, so the approval was neither consulted nor spent.
        assert [e for e, _ in world["audits"] if e == "wire_motion_refused"] == []

        # The same leader's reset FOR so101-1 proceeds on that approval.
        _deliver(world["robot"], signed_cmd_sample(world["leader"], "leader-1", "so101-1", RESET), monkeypatch)
        assert world["arm"].resets == 1

    def test_a_signed_command_that_names_no_target_is_refused(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@leader-1")

        unbound = signed_cmd_sample(world["leader"], "leader-1", "so101-1", RESET, name_target=False)
        _deliver(world["robot"], unbound, monkeypatch)

        assert world["arm"].resets == 0
        refused = [p for e, p in world["audits"] if e == "command_refused"]
        assert [p["reason"] for p in refused] == ["signed_for_another_peer"]
        assert refused[0]["target_id"] is None

    def test_a_broadcast_is_signed_for_every_peer_and_admitted(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@leader-1")

        fleet_wide = signed_cmd_sample(
            world["leader"], "leader-1", "so101-1", RESET, signed_for=mesh_core.BROADCAST_RESPONDER
        )
        _deliver(world["robot"], fleet_wide, monkeypatch)

        assert world["arm"].resets == 1
        assert [e for e, _ in world["audits"] if e == "command_refused"] == []

    def test_an_unsigned_command_is_judged_by_the_gate_not_by_a_target_it_cannot_name(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy envelopes carry no target; they are refused for lacking a signature, with that reason."""
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@leader-1")

        _deliver(world["robot"], signed_cmd_sample(None, "leader-1", "so101-1", RESET, name_target=False), monkeypatch)

        assert world["arm"].resets == 0
        assert [p["reason"] for e, p in world["audits"] if e == "wire_motion_refused"] == [
            "the command carries no verifiable signature"
        ]
        assert "signed_for_another_peer" not in [p.get("reason") for e, p in world["audits"] if e == "command_refused"]

    def test_send_and_broadcast_name_their_target_inside_the_signed_body(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        leader = Mesh(_HardwareArm(), peer_id="leader-1", peer_type="operator")
        leader._wire_identity = world["leader"]
        leader._running = True
        published: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(leader, "publish", lambda key, payload: published.append((key, payload)))
        monkeypatch.setattr(leader, "_pace_cmd_publish", lambda: None)

        leader.send("so101-1", RESET, timeout=0.05)
        leader.broadcast(RESET, timeout=0.05)

        assert [k for k, _ in published] == ["strands/so101-1/cmd", "strands/broadcast"]
        sent, fleet_wide = (p for _, p in published)
        assert sent["target_id"] == "so101-1" and fleet_wide["target_id"] == mesh_core.BROADCAST_RESPONDER
        # Both are signed over the target: altering it breaks the signature.
        roots = world["robot"]._trust_roots
        assert not isinstance(mesh_core._wire_identity.verify(roots, sent), str)
        assert isinstance(mesh_core._wire_identity.verify(roots, {**sent, "target_id": "so101-2"}), str)
