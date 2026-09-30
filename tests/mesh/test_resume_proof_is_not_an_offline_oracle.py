"""Regression tests: the resume override proof is not a cheap offline oracle.

A fleet resume envelope carries ``override_proof = HMAC(code, fields)`` next
to every field the MAC covers, so one captured envelope let anyone test
candidate codes offline at one SHA-256 per guess, and nothing stopped a
four-character code. The broadcast handler that actually clears a lockout also
had no brute-force throttle; only the RPC ``resume`` action did.

Now the MAC key is derived from the code with scrypt (memory-hard, one
derivation per process), a code shorter than ``OVERRIDE_CODE_MIN_LEN`` is
treated as unset on both sides (fail closed, with the reason logged at start),
a proof minted with the raw code is refused, and the broadcast handler counts
proof mismatches against the same throttle as the RPC path and records them.
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
from strands_robots.mesh.core import OVERRIDE_CODE_MIN_LEN, Mesh, resume_proof_key

CODE = "operator-code-1234567890abcdef"


class _Robot:
    tool_name_str = "arm"


def _sample(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode()), source_info=None)


def _envelope(key: bytes, *, peer_id: str = "op-1", nonce: str | None = None) -> dict[str, Any]:
    fields = {"peer_id": peer_id, "t": time.time(), "lockout_elapsed_s": 1.0, "proof_nonce": nonce or uuid.uuid4().hex}
    mac_input = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return {**fields, "override_proof": hmac.new(key, mac_input, "sha256").hexdigest()}


@pytest.fixture
def mesh() -> Mesh:
    m = Mesh(_Robot(), peer_id="arm-1")
    m._estop_lockout.set()
    m._last_estop_mono = time.monotonic()
    return m


@pytest.fixture(autouse=True)
def _quiet_posture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mesh_core, "_POSTURE_WARNINGS_EMITTED", set())


class TestTheKeyIsDerivedNotTheCode:
    def test_a_proof_keyed_with_the_raw_code_is_refused(self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)

        mesh._on_safety_resume(_sample(_envelope(CODE.encode())))

        assert mesh._estop_lockout.is_set()

    def test_a_proof_keyed_with_the_derived_key_clears_the_lockout(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)

        mesh._on_safety_resume(_sample(_envelope(resume_proof_key(CODE))))

        assert not mesh._estop_lockout.is_set()

    def test_the_issuer_signs_with_the_derived_key(self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(mesh, "_publish_safety_envelope", lambda topic, env: published.append(env))
        monkeypatch.setattr(mesh, "publish_safety_event", lambda **kw: None)

        assert mesh._resume_lockout(CODE) == {"status": "ok"}

        env = published[0]
        fields = {k: env[k] for k in ("peer_id", "t", "lockout_elapsed_s", "proof_nonce")}
        mac_input = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        assert env["override_proof"] == hmac.new(resume_proof_key(CODE), mac_input, "sha256").hexdigest()
        assert env["override_proof"] != hmac.new(CODE.encode(), mac_input, "sha256").hexdigest()

    def test_the_derivation_is_memory_hard_and_deterministic(self) -> None:
        assert resume_proof_key(CODE) == resume_proof_key(CODE)
        assert resume_proof_key(CODE) != resume_proof_key(CODE + "x")
        assert len(resume_proof_key(CODE)) == 32


class TestAShortCodeIsUnset:
    @pytest.mark.parametrize("short", ["1234", "secret", "a" * (OVERRIDE_CODE_MIN_LEN - 1)])
    def test_receiver_refuses_a_resume_under_a_short_code(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, short: str
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", short)

        with caplog.at_level(logging.WARNING):
            mesh._on_safety_resume(_sample(_envelope(resume_proof_key(short))))

        assert mesh._estop_lockout.is_set()
        assert any("OVERRIDE_CODE" in r.message and "short" in r.message for r in caplog.records)

    def test_issuer_refuses_to_resume_under_a_short_code(self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")

        assert mesh._resume_lockout("1234") == {"status": "error", "error": "resume rejected"}
        assert mesh._estop_lockout.is_set()

    def test_the_floor_is_at_least_twelve_characters(self) -> None:
        assert OVERRIDE_CODE_MIN_LEN >= 12

    def test_startup_names_the_short_code_next_to_the_unset_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")

        with caplog.at_level(logging.WARNING):
            assert mesh_core.override_code() is None

        assert any("STRANDS_MESH_OVERRIDE_CODE" in r.message and "short" in r.message for r in caplog.records)
        assert any("secrets.token_urlsafe" in r.message for r in caplog.records)


class TestTheBroadcastPathIsThrottled:
    def test_proof_mismatches_count_against_the_throttle_and_are_audited(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)
        monkeypatch.setenv("STRANDS_MESH_RESUME_MAX_FAILS", "3")
        audits: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(mesh, "_audit_local", lambda e, p: audits.append((e, p)))
        monkeypatch.setattr(mesh, "_audit", lambda **kw: None)
        wrong = resume_proof_key("not-the-code-1234567890abcdef")

        for _ in range(3):
            mesh._on_safety_resume(_sample(_envelope(wrong)))

        assert [e for e, _ in audits] == ["resume_denied"] * 3
        assert all("mismatch" in p["reason"] for _, p in audits)
        assert mesh._resume_locked_until_mono > time.monotonic()
        # The cooldown holds even a correct proof, exactly as the RPC path does.
        mesh._on_safety_resume(_sample(_envelope(resume_proof_key(CODE))))
        assert mesh._estop_lockout.is_set()

    def test_a_correct_proof_before_the_threshold_still_clears(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)
        monkeypatch.setattr(mesh, "_audit", lambda **kw: None)

        mesh._on_safety_resume(_sample(_envelope(resume_proof_key("wrong-code-1234567890abcdef"))))
        mesh._on_safety_resume(_sample(_envelope(resume_proof_key(CODE))))

        assert not mesh._estop_lockout.is_set()
