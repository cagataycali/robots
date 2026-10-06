"""The resume override code never travels, and a resume proof clears one lockout.

The code that clears a fleet e-stop lockout is typed on the operator's own peer
(:meth:`Mesh.resume`) and never sent: a ``resume`` command on the command
topic is refused, so no peer the ACL lets read that topic learns the code. The
fleet envelope carries ``override_proof``, an HMAC under an scrypt key salted
with the fleet namespace, over the published fields plus two the receiver
supplies itself: the epoch of the e-stop its lockout holds and the fleet. A
proof minted with the raw code, in another fleet or for another lockout is
refused and counted against the same throttle as a wrong code. A code that is
too short or too repetitive is treated as unset on both sides.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import OVERRIDE_CODE_MIN_LEN, Mesh, resume_proof, resume_proof_key

CODE = "operator-code-1234567890abcdef"
EPOCH = "a" * 32


class _Robot:
    tool_name_str = "arm"


def _sample(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode()), source_info=None)


def _envelope(code: str = CODE, *, epoch: str = EPOCH, fleet: str | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "peer_id": "op-1",
        "t": time.time(),
        "lockout_elapsed_s": 1.0,
        "proof_nonce": uuid.uuid4().hex,
    }
    fleet = mesh_core._fleet_namespace() if fleet is None else fleet
    return {**fields, "override_proof": resume_proof(code, fleet=fleet, lockout_epoch=epoch, **fields)}


@pytest.fixture
def mesh(monkeypatch: pytest.MonkeyPatch) -> Mesh:
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)
    m = Mesh(_Robot(), peer_id="arm-1")
    m._estop_lockout.set(EPOCH)
    m._last_estop_mono = time.monotonic()
    return m


@pytest.fixture(autouse=True)
def _quiet_posture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mesh_core, "_POSTURE_WARNINGS_EMITTED", set())


class TestTheCodeNeverRidesTheCommandTopic:
    @pytest.mark.parametrize("cmd", [{"action": "resume", "override_code": CODE}, {"action": "resume"}])
    def test_a_resume_command_is_refused_even_with_the_right_code(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch, cmd: dict[str, Any]
    ) -> None:
        audits: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(mesh, "_audit_local", lambda e, p: audits.append((e, p)))
        monkeypatch.setattr(mesh, "_audit", lambda **kw: None)

        assert mesh._dispatch(cmd) == {"status": "error", "error": "resume rejected"}

        assert mesh._estop_lockout.is_set()
        assert [e for e, _ in audits] == ["resume_denied"]
        assert "command topic" in audits[0][1]["reason"] and CODE not in audits[0][1]["reason"]

    def test_the_operator_resumes_on_its_own_peer_and_publishes_no_code_and_no_epoch(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(mesh, "_publish_safety_envelope", lambda topic, env: published.append(env))
        monkeypatch.setattr(mesh, "publish_safety_event", lambda **kw: None)

        assert mesh.resume(CODE) == {"status": "ok"}

        env = published[0]
        assert set(env) == {"peer_id", "t", "lockout_elapsed_s", "proof_nonce", "override_proof"}
        fields = {k: env[k] for k in ("peer_id", "t", "lockout_elapsed_s", "proof_nonce")}
        fleet = mesh_core._fleet_namespace()
        assert env["override_proof"] == resume_proof(CODE, fleet=fleet, lockout_epoch=EPOCH, **fields)
        assert CODE not in json.dumps(env)


class TestAProofClearsOnlyTheLockoutItWasMintedFor:
    def test_a_proof_for_the_held_epoch_in_this_fleet_clears(self, mesh: Mesh) -> None:
        mesh._on_safety_resume(_sample(_envelope()))

        assert not mesh._estop_lockout.is_set()

    @pytest.mark.parametrize(
        "envelope",
        [
            pytest.param(lambda: _envelope(epoch="b" * 32), id="another-lockout"),
            pytest.param(lambda: _envelope(fleet="another-fleet"), id="another-fleet"),
            pytest.param(lambda: _envelope("not-the-code-1234567890abcdef"), id="wrong-code"),
            pytest.param(
                lambda: {
                    **(e := _envelope()),
                    "override_proof": hmac.new(
                        CODE.encode(),
                        json.dumps(
                            {k: e[k] for k in ("peer_id", "t", "lockout_elapsed_s", "proof_nonce")},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode(),
                        "sha256",
                    ).hexdigest(),
                },
                id="raw-code-key",
            ),
        ],
    )
    def test_any_other_proof_is_refused_and_counted(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch, envelope: Any
    ) -> None:
        audits: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(mesh, "_audit_local", lambda e, p: audits.append((e, p)))
        monkeypatch.setattr(mesh, "_audit", lambda **kw: None)

        mesh._on_safety_resume(_sample(envelope()))

        assert mesh._estop_lockout.is_set()
        assert mesh._resume_fail_count == 1
        assert [e for e, _ in audits] == ["resume_denied"]

    def test_the_epoch_travels_on_the_estop_and_a_later_lockout_gets_a_new_one(self, mesh: Mesh) -> None:
        receiver = Mesh(_Robot(), peer_id="arm-2")
        estop = {"peer_id": "op-1", "t": time.time(), "lockout_epoch": "c" * 32}

        receiver._on_safety_estop(_sample(estop))
        held = receiver._estop_lockout.epochs
        receiver._on_safety_resume(_sample(_envelope(epoch="c" * 32)))
        cleared = not receiver._estop_lockout.is_set()
        receiver._estop_lockout.set()

        assert held == ("c" * 32,) and cleared
        assert receiver._estop_lockout.epochs not in ((), held)

    def test_the_key_depends_on_the_fleet(self) -> None:
        assert resume_proof_key(CODE, "fleet-a") == resume_proof_key(CODE, "fleet-a")
        assert resume_proof_key(CODE, "fleet-a") != resume_proof_key(CODE, "fleet-b")
        assert len(resume_proof_key(CODE, "fleet-a")) == 32


class TestAWeakCodeIsUnset:
    @pytest.mark.parametrize(
        "weak", ["1234", "secret", "a" * (OVERRIDE_CODE_MIN_LEN - 1), "a" * 40, "ab" * 20, "passwordpassword"]
    )
    def test_both_sides_refuse_under_a_weak_code(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, weak: str
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", weak)

        with caplog.at_level(logging.WARNING):
            mesh._on_safety_resume(_sample(_envelope(weak)))
            assert mesh.resume(weak) == {"status": "error", "error": "resume rejected"}

        assert mesh._estop_lockout.is_set()
        assert any("OVERRIDE_CODE" in r.message and "secrets.token_urlsafe" in r.message for r in caplog.records)

    def test_a_generated_code_is_usable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import secrets

        code = secrets.token_urlsafe(32)
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", code)

        assert mesh_core.override_code() == code


class TestTheBroadcastPathIsThrottled:
    def test_proof_mismatches_engage_the_cooldown_that_holds_a_correct_proof(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_RESUME_MAX_FAILS", "3")
        monkeypatch.setattr(mesh, "_audit", lambda **kw: None)

        for _ in range(3):
            mesh._on_safety_resume(_sample(_envelope(epoch="b" * 32)))

        assert mesh._resume_locked_until_mono > time.monotonic()
        mesh._on_safety_resume(_sample(_envelope()))
        assert mesh._estop_lockout.is_set()
