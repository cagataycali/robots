"""Regression: a reply carrying a copied or replayed identity is refused; the genuine one still counts.

The e-stop acknowledgement path used to attribute a reply by the ``SourceInfo``
the publisher attached to its own sample. That label is copyable: an admitted
peer that read the victim's session id off its heartbeat could publish
``{"ok": true}`` in the victim's name, the operator saw the victim as stopped,
and the victim's genuine ``{"ok": false}`` was then discarded as a duplicate
and the discard blamed on the victim.

With signed identity required, a reply is attributed to the certificate that
signed it. These cells replay the exact sequence from the finding against a
receiver with a trust root: the attacker's copy of the victim's session id buys
nothing, the attacker's own valid certificate cannot answer in the victim's
name, a captured genuine reply cannot be replayed on a second turn, and the
victim's real answer is recorded, not deduplicated away.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from strands_robots.mesh import session as mesh_session  # noqa: E402
from strands_robots.mesh.core import BROADCAST_RESPONDER, Mesh  # noqa: E402
from tests._wire_identity import (  # noqa: E402
    arm_receiver,
    identity_for,
    signed_presence_sample,
    signed_reply_sample,
)
from tests.mesh._pki import EphemeralCA  # noqa: E402

VICTIM_ZID = "0123456789abcdef0123456789abcdef"
TURN = "a" * 32
TURN_2 = "b" * 32


class _Operator:
    tool_name_str = "console"

    def stop_task(self) -> dict[str, Any]:
        return {"status": "success"}


def _register(m: Mesh, turn: str, expected: str) -> threading.Event:
    event = threading.Event()
    with m._rpc_lock:
        m._pending[turn] = event
        m._responses[turn] = []
        m._expected_responders[turn] = expected
    return event


def _recorded(m: Mesh, turn: str) -> list[dict[str, Any]]:
    with m._rpc_lock:
        return list(m._responses.get(turn, []))


@pytest.fixture(autouse=True)
def _clean_roster() -> Any:
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()
    yield
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()


@pytest.fixture
def fleet(require_signatures: EphemeralCA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An operator that requires signatures, a victim robot and an attacker, all issued by one CA."""
    op = Mesh(_Operator(), peer_id="op", peer_type="operator")
    arm_receiver(op, require_signatures)
    audits: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(op, "_audit_local", lambda event, payload: audits.append((event, payload)))
    victim = identity_for(require_signatures, "victim", tmp_path / "leaves")
    attacker = identity_for(require_signatures, "attacker", tmp_path / "leaves")
    # The victim announces itself (signed) and, as on the wire, with its session id attached.
    op._on_presence(signed_presence_sample(victim, "victim", zid=VICTIM_ZID))
    assert op.peer_cert("victim") == (victim.cert_sha256, "victim")
    return {"op": op, "audits": audits, "victim": victim, "attacker": attacker, "ca": require_signatures}


STOPPED = {"ok": True, "stopped": ["victim"]}
DID_NOT_STOP = {"ok": False, "error": "motor bus busy"}


class TestPathAFromTheFinding:
    def test_a_copied_session_id_with_no_signature_buys_nothing(self, fleet: dict[str, Any]) -> None:
        op, audits = fleet["op"], fleet["audits"]
        _register(op, TURN, "victim")

        # Step 3 of the finding: the attacker attaches the victim's captured zid.
        op._on_response(signed_reply_sample(None, "op", "victim", TURN, STOPPED, zid=VICTIM_ZID))
        # Step 5: the victim's genuine, signed answer arrives next.
        op._on_response(signed_reply_sample(fleet["victim"], "op", "victim", TURN, DID_NOT_STOP, zid=VICTIM_ZID))

        assert [r["result"] for r in _recorded(op, TURN)] == [DID_NOT_STOP]
        assert [e for e, _ in audits] == ["response_hijack_rejected"]
        assert audits[0][1]["reason"] == "message carries no signature envelope"
        assert "response_duplicate_rejected" not in [e for e, _ in audits]

    def test_the_attackers_own_valid_certificate_cannot_answer_in_the_victims_name(self, fleet: dict[str, Any]) -> None:
        op, audits = fleet["op"], fleet["audits"]
        _register(op, TURN, "victim")

        op._on_response(signed_reply_sample(fleet["attacker"], "op", "victim", TURN, STOPPED, zid=VICTIM_ZID))
        op._on_response(signed_reply_sample(fleet["victim"], "op", "victim", TURN, DID_NOT_STOP, zid=VICTIM_ZID))

        assert [r["result"] for r in _recorded(op, TURN)] == [DID_NOT_STOP]
        refused = [p for e, p in audits if e == "response_hijack_rejected"]
        assert len(refused) == 1
        assert refused[0]["signer"] == "attacker"
        assert refused[0]["responder_id"] == "victim"
        assert "does not speak for" in refused[0]["reason"]

    def test_a_captured_genuine_reply_cannot_be_replayed_on_a_later_turn(self, fleet: dict[str, Any]) -> None:
        op, audits = fleet["op"], fleet["audits"]
        _register(op, TURN, "victim")
        genuine = signed_reply_sample(fleet["victim"], "op", "victim", TURN, STOPPED, zid=VICTIM_ZID)
        op._on_response(genuine)
        assert [r["result"] for r in _recorded(op, TURN)] == [STOPPED]

        # The attacker re-publishes the captured bytes on the next turn's key.
        _register(op, TURN_2, "victim")
        replayed = signed_reply_sample(None, "op", "victim", TURN_2, STOPPED, zid=VICTIM_ZID)
        replayed.payload = genuine.payload  # the captured, still-valid signed body
        op._on_response(replayed)
        # And on the same turn again (a second copy of the same bytes).
        op._on_response(genuine)

        assert _recorded(op, TURN_2) == []
        reasons = [p["reason"] for e, p in audits if e == "response_hijack_rejected"]
        assert len(reasons) == 2
        # Different turn: the body says TURN, the key says TURN_2, and the
        # signature is over the body, so the nonce is what refuses it.
        assert all("replayed" in r for r in reasons)

    def test_a_broadcast_turn_accepts_each_signer_once(self, fleet: dict[str, Any], tmp_path: Path) -> None:
        op, audits = fleet["op"], fleet["audits"]
        other = identity_for(fleet["ca"], "other", tmp_path / "leaves")
        op._on_presence(signed_presence_sample(other, "other"))
        event = _register(op, TURN, BROADCAST_RESPONDER)

        op._on_response(signed_reply_sample(fleet["victim"], "op", "victim", TURN, STOPPED))
        op._on_response(signed_reply_sample(other, "op", "other", TURN, STOPPED))
        op._on_response(signed_reply_sample(fleet["attacker"], "op", "victim", TURN, STOPPED))
        op._on_response(signed_reply_sample(fleet["victim"], "op", "victim", TURN, STOPPED))

        assert sorted(r["responder_id"] for r in _recorded(op, TURN)) == ["other", "victim"]
        assert event.is_set()
        assert [e for e, _ in audits] == ["response_hijack_rejected", "response_duplicate_rejected"]
        assert audits[1][1]["source"].startswith("cert:")


class TestAChildSpeaksFromItsParentsCertificate:
    def test_a_robot_child_announced_by_its_parent_answers_with_the_parent_leaf(self, fleet: dict[str, Any]) -> None:
        """``Robot(mesh=True)`` announces ``<peer>__<robot>`` from one session and one certificate."""
        op = fleet["op"]
        op._on_presence(signed_presence_sample(fleet["victim"], "victim__so101"))
        assert op.peer_cert("victim__so101") == (fleet["victim"].cert_sha256, "victim")
        _register(op, TURN, "victim__so101")

        op._on_response(signed_reply_sample(fleet["victim"], "op", "victim__so101", TURN, STOPPED))

        assert [r["responder_id"] for r in _recorded(op, TURN)] == ["victim__so101"]

    def test_a_similar_name_is_not_a_child(self, fleet: dict[str, Any]) -> None:
        op, audits = fleet["op"], fleet["audits"]

        op._on_presence(signed_presence_sample(fleet["attacker"], "victim2"))
        op._on_presence(signed_presence_sample(fleet["attacker"], "attacker_victim"))

        assert op.peer_cert("victim2") is None and op.peer_cert("attacker_victim") is None
        assert [e for e, _ in audits] == ["presence_identity_rejected"] * 2
