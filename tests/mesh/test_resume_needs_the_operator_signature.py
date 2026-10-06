"""Clearing an e-stop lockout needs the operator's signature, bound to one lockout and its robots.

A peer is configured with the operator's PUBLIC key only, so nothing a peer
holds or sees (its environment, the command topic, a relayed resume) lets it
clear a robot. Each refusal below leaves the lockout engaged.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from strands_robots.mesh import resume_authority
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.security import ValidationError, validate_command

from ._resume import lock, resume_sample, sign_for, trust_new_key


def _mesh(peer_id: str) -> Any:
    m: Any = Mesh(MagicMock(), peer_id)
    m.publish_safety_event = MagicMock()
    m._publish_safety_envelope = MagicMock()
    m._audit_local = MagicMock()
    return m


def _denials(m: Any) -> list[str]:
    return [c.args[1]["reason"] for c in m._audit_local.call_args_list if c.args[0] == "resume_denied"]


def test_a_plain_code_on_the_command_topic_is_refused_and_audited(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "Correct-Horse-Battery-9")
    trust_new_key(monkeypatch)
    robot = _mesh("robot-a")
    lock(robot)
    cmd = {"action": "resume", "override_code": "Correct-Horse-Battery-9"}

    with pytest.raises(ValidationError, match="plain override code is not accepted"):
        validate_command(cmd)
    assert robot._dispatch(cmd) == {"status": "error", "error": "resume rejected"}
    assert robot._estop_lockout.is_set()
    assert any("plain override_code" in r for r in _denials(robot))


def test_the_operator_signature_clears_every_named_peer_through_the_relay(monkeypatch):
    key = trust_new_key(monkeypatch)
    a, b = _mesh("robot-a"), _mesh("robot-b")
    epoch = lock(a)
    lock(b, epoch)  # one fleet e-stop, one epoch
    assertion = sign_for(key, a, b)

    assert a._dispatch(validate_command({"action": "resume", "assertion": assertion})) == {"status": "ok"}
    relayed = a._publish_safety_envelope.call_args.args[1]
    assert "override_code" not in json.dumps(relayed) and relayed["assertion"] == assertion
    b._on_safety_resume(resume_sample(relayed["assertion"], peer_id="robot-a"))

    assert not a._estop_lockout.is_set() and not b._estop_lockout.is_set()
    assert a._lockout_epoch is None and b._lockout_epoch is None


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("signed by a key no peer trusts", "signature does not verify"),
        ("names another robot", "does not name this peer"),
        ("names an earlier lockout", "different lockout"),
        ("signature over altered targets", "signature does not verify"),
        ("stale", "old"),
        ("already used", "already used"),
        ("no key configured on the verifier", "no resume verification key"),
        ("missing", "no signed assertion"),
        ("extra field", "unexpected or missing fields"),
    ],
)
def test_a_resume_the_verifier_cannot_check_leaves_the_lockout_engaged(monkeypatch, case, reason):
    key = trust_new_key(monkeypatch)
    target, other = _mesh("robot-b"), _mesh("robot-a")
    lock(target)
    lock(other)
    assertion: object = sign_for(key, target)
    if case == "signed by a key no peer trusts":
        assertion = sign_for(Ed25519PrivateKey.generate(), target)
    elif case == "names another robot":
        assertion = sign_for(key, other, epoch=target._lockout_epoch)
    elif case == "names an earlier lockout":
        old = sign_for(key, target)
        lock(target)  # cleared out of band and locked again: a new epoch
        assertion = old
    elif case == "signature over altered targets":
        assertion = {**sign_for(key, other, epoch=target._lockout_epoch), "targets": ["robot-a", "robot-b"]}
    elif case == "stale":
        assertion = sign_for(key, target, t=time.time() - 3600)
    elif case == "already used":
        target._on_safety_resume(resume_sample(assertion))
        assert not target._estop_lockout.is_set()
        lock(target, assertion["epoch"])  # type: ignore[index]
    elif case == "no key configured on the verifier":
        monkeypatch.delenv(resume_authority.PUBLIC_KEY_ENV)
    elif case == "missing":
        assertion = None
    elif case == "extra field":
        assertion = {**assertion, "override_proof": "00"}  # type: ignore[dict-item]

    target._on_safety_resume(resume_sample(assertion))

    assert target._estop_lockout.is_set(), case
    assert any(reason in r for r in _denials(target)), _denials(target)


def test_one_fleet_estop_gives_every_peer_the_same_lockout_epoch(monkeypatch):
    issuer, peer = _mesh("console"), _mesh("robot-a")
    issuer._running = True
    issuer.broadcast = MagicMock(return_value=[])
    issuer.emergency_stop()
    envelope = issuer._publish_safety_envelope.call_args.args[1]
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(envelope).encode()
    sample.source_info = None
    sample.retain = False

    peer._on_safety_estop(sample)

    assert resume_authority.is_epoch(envelope["estop_id"])
    assert peer._estop_lockout.is_set() and peer._lockout_epoch == issuer._lockout_epoch == envelope["estop_id"]


def test_mesh_resume_signs_for_its_own_lockout_and_names_the_roster(monkeypatch):
    key = trust_new_key(monkeypatch)
    console = _mesh("console")
    lock(console)
    monkeypatch.setattr(Mesh, "peers", property(lambda self: [{"peer_id": "robot-a"}, {"peer_id": "robot-b"}]))

    assert console.resume(key) == {"status": "ok"}
    assert console._publish_safety_envelope.call_args.args[1]["assertion"]["targets"] == [
        "console",
        "robot-a",
        "robot-b",
    ]


@pytest.mark.parametrize(
    ("passphrase", "weak"),
    [
        ("aaaaaaaaaaaaaaaa", True),
        ("short-1A", True),
        ("abcdefghijklmnopqrst", True),  # one character class
        ("Correct-Horse-Battery-9", False),
    ],
)
def test_a_weak_passphrase_cannot_protect_a_signing_key(tmp_path, passphrase, weak):
    path = tmp_path / "resume_key.pem"
    if weak:
        with pytest.raises(ValueError, match="passphrase"):
            resume_authority.generate_signing_key(path, passphrase)
        assert not path.exists()
        return
    public = resume_authority.generate_signing_key(path, passphrase)
    assert path.stat().st_mode & 0o777 == 0o600
    key = resume_authority.load_signing_key(path, passphrase)
    assert resume_authority.public_key_text(key.public_key()) == public
    with pytest.raises(ValueError, match="wrong passphrase"):
        resume_authority.load_signing_key(path, passphrase + "x")
