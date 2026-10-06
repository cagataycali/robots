"""Unit tests for :mod:`strands_robots.mesh.wire_identity`: what a signature proves and what is refused.

A mesh message used to be attributed by a ``SourceInfo`` label the publisher
chose; any admitted peer could copy another peer's label off its heartbeat.
These tests pin the replacement: a signature by a certificate chained to the
receiver's trust root, read for the common name that may speak for the peer id,
fresh, single use, and refused on every malformation without raising.
"""

from __future__ import annotations

import base64
import datetime as _dt
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, ed25519  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from strands_robots.mesh import wire_identity as wi  # noqa: E402
from tests._wire_identity import identity_for, roots_for  # noqa: E402
from tests.mesh._pki import EphemeralCA, make_test_ca  # noqa: E402


def _issue_with_key(ca: EphemeralCA, cn: str, key: Any, out_dir: Path, *, hours: float = 1.0) -> wi.WireIdentity:
    """A leaf for *cn* over *key* (EC or Ed25519), signed by *ca*."""
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = _dt.datetime.now(_dt.UTC)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(now + _dt.timedelta(hours=hours))
        .sign(ca.key, hashes.SHA256())
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / f"{cn}.crt"
    key_path = out_dir / f"{cn}.key"
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return wi.WireIdentity._from_files(cert_path, key_path, "test")


@pytest.fixture
def ca(tmp_path: Path) -> EphemeralCA:
    return make_test_ca(tmp_path / "ca")


@pytest.fixture
def roots(ca: EphemeralCA) -> wi.TrustRoots:
    return roots_for(ca)


BODY = {"robot_id": "arm-1", "robot_type": "robot", "timestamp": 1760000000.0}


class TestRoundTrip:
    def test_rsa_leaf_signs_and_verifies(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))

        got = wi.verify(roots, signed)

        assert isinstance(got, wi.Verified), got
        assert got.cn == "arm-1"
        assert got.alg == wi.ALG_RSA
        assert got.cert_sha256 == ident.cert_sha256
        assert {k: v for k, v in signed.items() if k != "sig"} == BODY

    def test_ec_leaf_signs_and_verifies(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = _issue_with_key(ca, "arm-ec", ec.generate_private_key(ec.SECP256R1()), tmp_path / "ec")

        got = wi.verify(roots, wi.sign(ident, dict(BODY)))

        assert isinstance(got, wi.Verified) and got.alg == wi.ALG_EC and got.cn == "arm-ec"

    def test_ed25519_leaf_signs_and_verifies(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = _issue_with_key(ca, "arm-ed", ed25519.Ed25519PrivateKey.generate(), tmp_path / "ed")

        got = wi.verify(roots, wi.sign(ident, dict(BODY)))

        assert isinstance(got, wi.Verified) and got.alg == wi.ALG_ED25519

    def test_signing_a_signed_body_is_refused(self, ca: EphemeralCA, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))

        with pytest.raises(ValueError, match="already carries"):
            wi.sign(ident, signed)

    def test_the_envelope_fits_a_command_budget(self, ca: EphemeralCA, tmp_path: Path) -> None:
        """An RSA-2048 envelope is about 1.5 KB: well inside the 16 KiB command cap."""
        import json

        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, {"sender_id": "op", "turn_id": "t" * 32, "command": {"action": "status"}})

        assert len(json.dumps(signed)) < 2048


class TestRefusals:
    """Every refusal is a one-line string; nothing raises on wire input."""

    def test_unknown_root(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        rogue = make_test_ca(tmp_path / "rogue")
        ident = identity_for(rogue, "arm-1", tmp_path / "rogue-leaf")

        assert wi.verify(roots, wi.sign(ident, dict(BODY))) == (
            "signature certificate is not issued by a configured trust root"
        )

    def test_no_roots(self, ca: EphemeralCA, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)

        got = wi.verify(None, wi.sign(ident, dict(BODY)))

        assert isinstance(got, str) and "no trust roots" in got

    def test_expired_leaf(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = _issue_with_key(ca, "arm-ec", ec.generate_private_key(ec.SECP256R1()), tmp_path / "ec", hours=1.0)
        signed = wi.sign(ident, dict(BODY), t=time.time() + 7200)

        got = wi.verify(roots, signed, now=time.time() + 7200)

        assert got == "signature certificate is outside its validity window"

    def test_tampered_body(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))
        signed["robot_id"] = "arm-2"

        assert wi.verify(roots, signed) == "signature does not verify over the message"

    def test_tampered_time(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY), t=time.time() - 3600)
        signed["sig"]["t"] = time.time()  # make a stale signature look fresh

        assert wi.verify(roots, signed) == "signature does not verify over the message"

    def test_stale_and_future_times(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)

        assert wi.verify(roots, wi.sign(ident, dict(BODY), t=time.time() - 120)) == "signature is stale"
        assert wi.verify(roots, wi.sign(ident, dict(BODY), t=time.time() + 60)) == "signature time is in the future"
        assert isinstance(
            wi.verify(roots, wi.sign(ident, dict(BODY), t=time.time() - 120), freshness_s=300.0), wi.Verified
        )

    def test_bad_alg(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))
        signed["sig"]["alg"] = "hmac-sha256"

        assert wi.verify(roots, signed) == "signature algorithm not supported"

    def test_alg_that_does_not_match_the_key(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))
        signed["sig"]["alg"] = wi.ALG_EC

        assert wi.verify(roots, signed) == "signature algorithm does not match the certificate's key"

    def test_oversized_cert(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))
        signed["sig"]["cert"] = base64.b64encode(b"\x30" * (wi.MAX_CERT_DER_BYTES + 1)).decode()

        assert wi.verify(roots, signed) == "signature certificate is oversized or not base64"

    def test_cert_that_does_not_parse(self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))
        signed["sig"]["cert"] = base64.b64encode(b"\x30\x03\x02\x01\x01").decode()

        assert wi.verify(roots, signed) == "signature certificate does not parse"

    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda e: e.pop("nonce"), "signature envelope has unexpected or missing fields"),
            (lambda e: e.update(extra=1), "signature envelope has unexpected or missing fields"),
            (lambda e: e.update(v=2), "signature envelope version not supported"),
            (lambda e: e.update(t="now"), "signature time is not a finite number"),
            (lambda e: e.update(t=float("nan")), "signature time is not a finite number"),
            (lambda e: e.update(nonce="short"), "signature nonce is malformed"),
            (lambda e: e.update(sig="not base64!"), "signature value is malformed"),
        ],
    )
    def test_malformed_envelopes(
        self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path, mutate: Any, expected: str
    ) -> None:
        ident = identity_for(ca, "arm-1", tmp_path)
        signed = wi.sign(ident, dict(BODY))
        mutate(signed["sig"])

        assert wi.verify(roots, signed) == expected

    def test_unsigned_and_non_dict_envelopes(self, roots: wi.TrustRoots) -> None:
        assert wi.verify(roots, dict(BODY)) == "message carries no signature envelope"
        assert wi.verify(roots, {**BODY, "sig": "yes"}) == "message carries no signature envelope"


class TestWhoACertificateSpeaksFor:
    @pytest.mark.parametrize(
        "cn, peer, expected",
        [
            ("arm-1", "arm-1", True),
            ("lab-op", "lab-op__so101", True),
            ("lab-op", "lab-op2", False),
            ("lab-op", "lab-op_so101", False),
            ("lab-op", "other__lab-op", False),
            ("", "x", False),
            ("x", "", False),
        ],
    )
    def test_cn_speaks_for(self, cn: str, peer: str, expected: bool) -> None:
        assert wi.cn_speaks_for(cn, peer) is expected

    def test_a_cn_that_is_not_a_peer_id_is_refused_at_verify(
        self, ca: EphemeralCA, roots: wi.TrustRoots, tmp_path: Path
    ) -> None:
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "has space/slash")])
        now = _dt.datetime.now(_dt.UTC)
        leaf = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca.cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(minutes=1))
            .not_valid_after(now + _dt.timedelta(hours=1))
            .sign(ca.key, hashes.SHA256())
        )
        with pytest.raises(ValueError, match="common name"):
            wi.WireIdentity._from_files(
                _write(tmp_path / "bad.crt", leaf.public_bytes(serialization.Encoding.PEM)),
                _write(
                    tmp_path / "bad.key",
                    key.private_bytes(
                        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
                    ),
                ),
                "test",
            )


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


class TestReplayGuard:
    def test_a_nonce_is_good_once_per_certificate(self) -> None:
        guard = wi.ReplayGuard(ttl_s=60.0)

        assert guard.seen_before("fp-a", "n1", now_mono=0.0) is False
        assert guard.seen_before("fp-a", "n1", now_mono=1.0) is True
        assert guard.seen_before("fp-b", "n1", now_mono=1.0) is False

    def test_an_entry_expires_with_the_window(self) -> None:
        guard = wi.ReplayGuard(ttl_s=10.0)
        guard.seen_before("fp", "n", now_mono=0.0)

        assert guard.seen_before("fp", "n", now_mono=11.0) is False

    def test_the_table_is_bounded(self) -> None:
        guard = wi.ReplayGuard(ttl_s=1000.0, max_size=3)
        for i in range(5):
            guard.seen_before("fp", f"n{i}", now_mono=float(i))

        assert len(guard._seen) == 3
        assert guard.seen_before("fp", "n0", now_mono=5.0) is False  # evicted oldest


class TestTheKnob:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("STRANDS_MESH_AUTH_MODE", "STRANDS_MESH_TLS_CA", "STRANDS_MESH_LOCAL_DEV", wi.REQUIRE_ENV):
            monkeypatch.delenv(name, raising=False)

    def test_one_and_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(wi.REQUIRE_ENV, "1")
        assert wi.signing_required() is True
        monkeypatch.setenv(wi.REQUIRE_ENV, "0")
        assert wi.signing_required() is False

    def test_auto_is_off_without_mtls_or_roots(self, monkeypatch: pytest.MonkeyPatch, ca: EphemeralCA) -> None:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca.cert_path))
        assert wi.signing_required() is False

        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.delenv("STRANDS_MESH_TLS_CA")
        assert wi.signing_required() is False

    def test_auto_is_on_under_mtls_with_a_root(self, monkeypatch: pytest.MonkeyPatch, ca: EphemeralCA) -> None:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca.cert_path))

        assert wi.signing_required() is True
        assert wi.signing_required(roots_for(ca)) is True

    def test_a_typo_fails_closed_and_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(wi.REQUIRE_ENV, "yes")
        wi._required_warned.discard("yes")
        with caplog.at_level("WARNING", logger="strands_robots.mesh.wire_identity"):
            assert wi.signing_required() is True
            assert wi.signing_required() is True

        assert sum("not one of 1, 0, auto" in r.getMessage() for r in caplog.records) == 1


class TestLoadingAnIdentity:
    def test_mtls_pair_is_the_identity(self, monkeypatch: pytest.MonkeyPatch, ca: EphemeralCA, tmp_path: Path) -> None:
        cert_path, key_path = ca.issue("arm-1", tmp_path / "arm")
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca.cert_path))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert_path))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key_path))

        ident = wi.WireIdentity.load()

        assert ident is not None and ident.cn == "arm-1" and ident.source == "STRANDS_MESH_TLS_CERT"
        assert wi.TrustRoots.load() is not None

    def test_no_pair_configured_is_none_not_a_problem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
        monkeypatch.delenv("STRANDS_IOT_THING_NAME", raising=False)

        assert wi.WireIdentity.load_or_problem() is None

    def test_a_configured_pair_that_is_missing_is_a_problem(
        self, monkeypatch: pytest.MonkeyPatch, ca: EphemeralCA, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca.cert_path))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(tmp_path / "nope.crt"))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(tmp_path / "nope.key"))

        got = wi.WireIdentity.load_or_problem()

        assert isinstance(got, str) and got.startswith("wire identity:")
        assert wi.WireIdentity.load() is None

    def test_a_mismatched_pair_is_a_problem(
        self, monkeypatch: pytest.MonkeyPatch, ca: EphemeralCA, tmp_path: Path
    ) -> None:
        cert_path, _ = ca.issue("arm-1", tmp_path / "arm")
        _, other_key = ca.issue("arm-2", tmp_path / "other")
        rsa_ident = identity_for(ca, "arm-3", tmp_path / "rsa")
        ec_ident = _issue_with_key(ca, "arm-ec", ec.generate_private_key(ec.SECP256R1()), tmp_path / "ec")
        del rsa_ident, other_key
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca.cert_path))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert_path))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(tmp_path / "ec" / "arm-ec.key"))
        (tmp_path / "ec" / "arm-ec.key").chmod(0o600)

        got = wi.WireIdentity.load_or_problem()

        assert isinstance(got, str) and "does not match the key type" in got
        del ec_ident

    def test_an_unreadable_root_bundle_loads_no_roots(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(tmp_path / "missing.pem"))
        assert wi.TrustRoots.load() is None
        empty = tmp_path / "empty.pem"
        empty.write_bytes(b"not a certificate\n")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(empty))
        assert wi.TrustRoots.load() is None
