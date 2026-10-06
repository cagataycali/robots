"""Signed wire identity: a mesh message is attributed to the certificate that signed it.

The mesh used to identify the sender of a presence, a reply or a command by
the ``SourceInfo`` the sender attached to its own Zenoh sample. That field is a
label the publisher chooses; ``zenoh.SourceInfo(entity_id, sn)`` accepts any id
copied off another peer's heartbeat, so an admitted peer could answer a fleet
emergency stop in another robot's name, or spend an approval that was granted
to someone else. Encrypting the link did not help: the forgeable element sat
inside the authenticated channel.

This module signs the message instead of labelling it. A peer that holds a
certificate and its private key (the same pair the mTLS link uses, or the AWS
IoT device certificate) attaches to every presence, reply and command::

    {"...body...", "sig": {"v": 1, "alg": "rsa-pss-sha256", "cert": "<base64 DER leaf>",
                           "t": 1760000000.0, "nonce": "<32 hex>", "sig": "<base64url>"}}

The receiver chains the leaf to a trust root (``STRANDS_MESH_TLS_CA``), checks
the leaf's validity window and the freshness of ``t``, verifies the signature
with the leaf's own public key, and reads the sender's name from the
certificate's common name. :func:`cn_speaks_for` is the one rule that maps a
certificate to the peer ids it may announce or answer for: its CN, and the
``<cn>__<robot>`` children a ``Robot(mesh=True)`` announces from the same
session. Nothing a peer writes into a body can widen that.

Fail closed: :func:`verify` returns either a :class:`Verified` record or a
one-line refusal string the caller logs and audits; it never raises on wire
input. ``cryptography`` is imported where a key is used so importing this module
stays cheap, the way :mod:`~strands_robots.mesh.resume_authority` does it.

Whether a receiver REQUIRES a signature is :func:`signing_required`:
``STRANDS_MESH_REQUIRE_SIGNED_IDENTITY`` is ``1`` (always), ``0`` (never, the
legacy session-id path) or ``auto`` (the default: required exactly when the
mesh runs mTLS and a trust root is configured). Under ``auto`` a development
mesh with ``STRANDS_MESH_AUTH_MODE=none`` behaves as before; a production mesh
with certificates gets signed identity without a new setting.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.mesh._backend_select import select_backend
from strands_robots.utils import refusal_str

if TYPE_CHECKING:  # cryptography is imported where a key is used, so importing this module stays cheap
    from cryptography import x509

__all__ = [
    "ALGORITHMS",
    "MAX_CERT_DER_BYTES",
    "REQUIRE_ENV",
    "SIG_FIELD",
    "ReplayGuard",
    "TrustRoots",
    "Verified",
    "WireIdentity",
    "cn_speaks_for",
    "sign",
    "signing_required",
    "verify",
]

logger = logging.getLogger(__name__)

#: The knob: ``1`` requires a verifiable signature on every presence, reply and
#: motion command this peer receives; ``0`` keeps the legacy session-id path;
#: ``auto`` (default) requires one exactly when the mesh runs mTLS and
#: ``STRANDS_MESH_TLS_CA`` loads as a trust root.
REQUIRE_ENV = "STRANDS_MESH_REQUIRE_SIGNED_IDENTITY"

#: The body key the signature envelope travels under.
SIG_FIELD = "sig"

#: Wire version of the envelope; a peer refuses any other.
ENVELOPE_VERSION = 1

#: Signature algorithm per key type. The name travels on the wire and the
#: verifier picks the key type it implies; a leaf whose key does not match the
#: named algorithm is refused.
ALG_RSA = "rsa-pss-sha256"
ALG_EC = "ecdsa-sha256"
ALG_ED25519 = "ed25519"
ALGORITHMS: frozenset[str] = frozenset({ALG_RSA, ALG_EC, ALG_ED25519})

#: Largest DER leaf accepted on the wire. An RSA-2048 leaf is about 900 bytes
#: and an RSA-4096 one about 1.4 KB; the cap keeps a hostile envelope from
#: carrying a kilobyte-heavy chain into every command's 16 KiB budget.
MAX_CERT_DER_BYTES = 4096

#: Salt length for RSA-PSS, the SHA-256 digest size.
_PSS_SALT_LEN = 32

_NONCE_RE = re.compile(r"^[0-9a-f]{32}\Z")
_B64URL_RE = re.compile(r"^[A-Za-z0-9_\-]+={0,2}\Z")
_B64_RE = re.compile(r"^[A-Za-z0-9+/_\-]+={0,2}\Z")
#: A common name is a peer id: the same charset the mesh's identifier
#: validator admits, at most 128 characters, never starting with ``.`` or ``-``.
_CN_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}\Z")

#: The envelope's exact key set; anything else is malformed.
_SIG_KEYS = frozenset({"v", "alg", "cert", "t", "nonce", "sig"})

#: Default freshness bounds for ``t`` when the caller passes none, matching
#: the mesh's resume/presence defaults (60 s back, 5 s forward).
DEFAULT_FRESHNESS_S = 60.0
DEFAULT_FORWARD_SKEW_S = 5.0


@dataclass(frozen=True)
class Verified:
    """What a verified signature says about the message it covers."""

    #: The signing certificate's common name: the peer id the message speaks for.
    cn: str
    #: Lowercase hex SHA-256 of the DER leaf; pins one certificate, not a name.
    cert_sha256: str
    alg: str
    t: float
    nonce: str


@dataclass(frozen=True)
class WireIdentity:
    """This peer's signing certificate and private key."""

    cn: str
    cert_der: bytes
    cert_sha256: str
    alg: str
    _private_key: Any
    source: str

    @property
    def cert_b64(self) -> str:
        """The DER leaf as standard base64, the way it travels on the wire."""
        return base64.b64encode(self.cert_der).decode()

    @classmethod
    def load(cls) -> WireIdentity | None:
        """This peer's identity from its certificate files, or ``None`` when it has none.

        In order: the mTLS pair (``STRANDS_MESH_TLS_CERT`` / ``STRANDS_MESH_TLS_KEY``)
        when the auth mode is ``mtls``; else, on the ``iot`` and ``bridge``
        backends, ``STRANDS_IOT_CERT_DIR/<thing>.cert.pem`` and
        ``<thing>.private.key``; else nothing. A pair that exists but cannot be
        used is reported by :meth:`load_or_problem`; this method only answers
        "is there an identity to sign with".
        """
        loaded = cls.load_or_problem()
        return loaded if isinstance(loaded, WireIdentity) else None

    @classmethod
    def load_or_problem(cls) -> WireIdentity | str | None:
        """Like :meth:`load`, but a pair that was configured and failed is a one-line reason.

        ``None`` means no pair is configured anywhere, which is not a problem:
        the peer publishes unsigned and the receivers decide what that means.
        """
        try:
            located = _locate_pair()
        except (ValueError, OSError) as exc:
            return f"wire identity: {exc}"
        if located is None:
            return None
        cert_path, key_path, source = located
        try:
            return cls._from_files(cert_path, key_path, source)
        except (ValueError, TypeError, OSError) as exc:
            return f"wire identity from {source}: {exc}"

    @classmethod
    def _from_files(cls, cert_path: Path, key_path: Path, source: str) -> WireIdentity:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        alg = _alg_for_key(key)
        if alg is None:
            raise ValueError(f"{key_path} is not an RSA, EC or Ed25519 private key")
        if _alg_for_key(cert.public_key()) != alg:
            raise ValueError(f"{cert_path} does not match the key type of {key_path}")
        cn = _common_name(cert)
        if cn is None:
            raise ValueError(f"{cert_path} has no single common name usable as a peer id")
        der = cert.public_bytes(serialization.Encoding.DER)
        if len(der) > MAX_CERT_DER_BYTES:
            raise ValueError(f"{cert_path} is {len(der)} bytes DER; the wire cap is {MAX_CERT_DER_BYTES}")
        return cls(
            cn=cn,
            cert_der=der,
            cert_sha256=hashlib.sha256(der).hexdigest(),
            alg=alg,
            _private_key=key,
            source=source,
        )


def _locate_pair() -> tuple[Path, Path, str] | None:
    """Where this peer's certificate and key are, or ``None`` when nowhere is configured."""
    from strands_robots.mesh._zenoh_config import _resolve_tls_paths, resolve_auth_mode

    backend = select_backend()
    if backend != "iot":
        # The Zenoh leg's auth mode; ``none`` without its acknowledgement is a
        # mode the session refuses to open under, so there is nothing to sign for.
        try:
            mode = resolve_auth_mode()
        except ValueError:
            mode = None
        tls_set = all(
            os.getenv(name, "").strip()
            for name in ("STRANDS_MESH_TLS_CA", "STRANDS_MESH_TLS_CERT", "STRANDS_MESH_TLS_KEY")
        )
        if mode == "mtls" and tls_set:
            _ca, cert, key = _resolve_tls_paths()
            return cert, key, "STRANDS_MESH_TLS_CERT"
    if backend in ("iot", "bridge"):
        thing = os.getenv("STRANDS_IOT_THING_NAME", "").strip()
        if not thing:
            return None
        cert_dir = Path(os.getenv("STRANDS_IOT_CERT_DIR") or Path.home() / ".strands_robots" / "iot")
        cert = cert_dir / f"{thing}.cert.pem"
        key = cert_dir / f"{thing}.private.key"
        if cert.is_symlink() or key.is_symlink():
            raise ValueError("IoT certificate or key is a symlink; refusing")
        if not cert.is_file() or not key.is_file():
            return None
        return cert, key, "STRANDS_IOT_CERT_DIR"
    return None


@dataclass(frozen=True)
class TrustRoots:
    """The CA certificates a leaf must chain to."""

    certs: tuple[Any, ...]

    @classmethod
    def load(cls) -> TrustRoots | None:
        """The PEM bundle at ``STRANDS_MESH_TLS_CA``, or ``None`` when unset or unreadable.

        An unreadable or empty bundle is logged and treated as no roots: a
        receiver with no roots cannot require signatures under ``auto``, and
        under ``1`` it refuses every signed message, both of which fail closed.
        """
        raw = os.getenv("STRANDS_MESH_TLS_CA", "").strip()
        if not raw:
            return None
        path = Path(raw)
        try:
            if path.is_symlink():
                raise ValueError("is a symlink")
            data = path.read_bytes()
        except (OSError, ValueError) as exc:
            logger.warning("[mesh] STRANDS_MESH_TLS_CA is not a readable trust root (%s); no roots loaded", exc)
            return None
        return cls.from_pem(data)

    @classmethod
    def from_pem(cls, data: bytes) -> TrustRoots | None:
        """Parse a PEM bundle; ``None`` when it holds no certificate."""
        from cryptography import x509

        try:
            certs = tuple(x509.load_pem_x509_certificates(data))
        except ValueError as exc:
            logger.warning("[mesh] trust root bundle holds no parseable certificate (%s)", exc)
            return None
        return cls(certs=certs) if certs else None


def _alg_for_key(key: Any) -> str | None:
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

    if isinstance(key, (rsa.RSAPrivateKey, rsa.RSAPublicKey)):
        return ALG_RSA
    if isinstance(key, (ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey)):
        return ALG_EC
    if isinstance(key, (ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey)):
        return ALG_ED25519
    return None


def _common_name(cert: x509.Certificate) -> str | None:
    """The certificate's single CN when it is shaped like a peer id, else ``None``."""
    from cryptography.x509.oid import NameOID

    try:
        attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    except ValueError:
        return None
    if len(attrs) != 1:
        return None
    value = attrs[0].value
    if not isinstance(value, str) or not _CN_RE.fullmatch(value):
        return None
    return value


def _canonical(fields: dict[str, Any]) -> bytes:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _signed_bytes(body: dict[str, Any], envelope: dict[str, Any], cert_sha256: str) -> bytes:
    covered = {
        "v": envelope["v"],
        "alg": envelope["alg"],
        "t": envelope["t"],
        "nonce": envelope["nonce"],
        "cert_sha256": cert_sha256,
    }
    return _canonical({"body": body, "sig": covered})


def sign(
    identity: WireIdentity, body: dict[str, Any], *, t: float | None = None, nonce: str | None = None
) -> dict[str, Any]:
    """*body* plus a ``sig`` envelope signed by *identity*.

    The signature covers the canonical JSON of the body and of the envelope's
    own version, algorithm, time, nonce and certificate fingerprint, so neither
    the body nor the certificate can be swapped under a signature.

    Raises:
        ValueError: *body* already carries a ``sig`` field, or is not JSON.
    """
    if SIG_FIELD in body:
        raise ValueError("body already carries a signature envelope")
    envelope: dict[str, Any] = {
        "v": ENVELOPE_VERSION,
        "alg": identity.alg,
        "cert": identity.cert_b64,
        "t": time.time() if t is None else float(t),
        "nonce": nonce or uuid.uuid4().hex,
    }
    message = _signed_bytes(body, envelope, identity.cert_sha256)
    raw = _sign_bytes(identity._private_key, identity.alg, message)
    envelope["sig"] = base64.urlsafe_b64encode(raw).decode()
    return {**body, SIG_FIELD: envelope}


def _sign_bytes(key: Any, alg: str, message: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding

    signed: bytes
    if alg == ALG_RSA:
        signed = key.sign(
            message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=_PSS_SALT_LEN), hashes.SHA256()
        )
    elif alg == ALG_EC:
        signed = key.sign(message, ec.ECDSA(hashes.SHA256()))
    else:
        signed = key.sign(message)
    return signed


def _verify_bytes(public_key: Any, alg: str, signature: bytes, message: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding

    try:
        if alg == ALG_RSA:
            public_key.verify(
                signature,
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=_PSS_SALT_LEN),
                hashes.SHA256(),
            )
        elif alg == ALG_EC:
            public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        else:
            public_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def _b64decode(text: str, *, urlsafe: bool) -> bytes | None:
    padded = text + "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(padded) if urlsafe else base64.b64decode(padded, validate=False)
    except ValueError:
        return None


def _envelope_shape_problem(envelope: Any) -> str | None:
    """Why *envelope* is not a well-formed ``sig`` block, or ``None``."""
    if not isinstance(envelope, dict):
        return "message carries no signature envelope"
    if set(envelope) != _SIG_KEYS:
        return "signature envelope has unexpected or missing fields"
    if envelope["v"] != ENVELOPE_VERSION:
        return "signature envelope version not supported"
    alg = envelope["alg"]
    if not isinstance(alg, str) or alg not in ALGORITHMS:
        return "signature algorithm not supported"
    t = envelope["t"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        return "signature time is not a finite number"
    nonce = envelope["nonce"]
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        return "signature nonce is malformed"
    cert = envelope["cert"]
    # 4/3 of the DER cap plus padding: a longer string is refused before decoding.
    if not isinstance(cert, str) or len(cert) > (MAX_CERT_DER_BYTES * 4) // 3 + 4 or not _B64_RE.fullmatch(cert):
        return "signature certificate is oversized or not base64"
    sig = envelope["sig"]
    if not isinstance(sig, str) or len(sig) > 2048 or not _B64URL_RE.fullmatch(sig):
        return "signature value is malformed"
    return None


def verify(
    roots: TrustRoots | None,
    message: dict[str, Any],
    *,
    now: float | None = None,
    freshness_s: float = DEFAULT_FRESHNESS_S,
    forward_skew_s: float = DEFAULT_FORWARD_SKEW_S,
) -> Verified | str:
    """Who signed *message*, or a one-line reason it is not trusted.

    Checks, in order: the envelope's shape, the leaf decodes and is under the
    size cap, the leaf chains to one of *roots* (issuer name match and the
    root's key verifies the leaf), the leaf is inside its validity window at
    *now*, ``t`` is within ``freshness_s`` back and ``forward_skew_s`` ahead of
    *now*, the leaf's key type matches ``alg``, and the signature verifies over
    the canonical bytes. Any failure is a refusal; nothing raises on wire input.

    Args:
        roots: The trust roots; ``None`` refuses every signed message.
        message: The decoded body, ``sig`` included.
        now: Wall-clock seconds; ``time.time()`` when omitted.
        freshness_s: How far back ``t`` may lie.
        forward_skew_s: How far ahead ``t`` may lie.
    """
    envelope = message.get(SIG_FIELD) if isinstance(message, dict) else None
    if not isinstance(envelope, dict):
        return "message carries no signature envelope"
    if (problem := _envelope_shape_problem(envelope)) is not None:
        return problem
    if roots is None or not roots.certs:
        return "no trust roots configured to verify the signing certificate against"
    der = _b64decode(envelope["cert"], urlsafe=False)
    if der is None or not der or len(der) > MAX_CERT_DER_BYTES:
        return "signature certificate is oversized or not base64"
    raw_sig = _b64decode(envelope["sig"], urlsafe=True)
    if raw_sig is None or not raw_sig:
        return "signature value is malformed"

    from cryptography import x509

    try:
        leaf = x509.load_der_x509_certificate(der)
    except ValueError:
        return "signature certificate does not parse"
    cn = _common_name(leaf)
    if cn is None:
        return "signature certificate has no single common name shaped like a peer id"
    if not _chains_to(leaf, roots):
        return "signature certificate is not issued by a configured trust root"
    moment = time.time() if now is None else float(now)
    when = _dt.datetime.fromtimestamp(moment, tz=_dt.UTC)
    if when < leaf.not_valid_before_utc or when > leaf.not_valid_after_utc:
        return "signature certificate is outside its validity window"
    age = moment - float(envelope["t"])
    if age > freshness_s:
        return "signature is stale"
    if age < -forward_skew_s:
        return "signature time is in the future"
    public_key = leaf.public_key()
    if _alg_for_key(public_key) != envelope["alg"]:
        return "signature algorithm does not match the certificate's key"
    cert_sha256 = hashlib.sha256(der).hexdigest()
    body = {k: v for k, v in message.items() if k != SIG_FIELD}
    if not _verify_bytes(public_key, envelope["alg"], raw_sig, _signed_bytes(body, envelope, cert_sha256)):
        return "signature does not verify over the message"
    return Verified(
        cn=cn, cert_sha256=cert_sha256, alg=envelope["alg"], t=float(envelope["t"]), nonce=envelope["nonce"]
    )


def _chains_to(leaf: x509.Certificate, roots: TrustRoots) -> bool:
    """Whether a configured root issued *leaf*: issuer name equal and the root's key verifies it."""
    from cryptography.exceptions import InvalidSignature

    for root in roots.certs:
        if leaf.issuer != root.subject:
            continue
        try:
            leaf.verify_directly_issued_by(root)
        except (InvalidSignature, ValueError, TypeError):
            continue
        return True
    return False


def cn_speaks_for(cn: str, peer_id: str) -> bool:
    """Whether a certificate with common name *cn* may speak as *peer_id*.

    Exactly two shapes: the CN itself, and ``<cn>__<robot>``, the child a
    ``Robot(mesh=True)`` announces alongside itself from one session and one
    certificate. Nothing else, so a certificate for ``lab-op`` cannot announce
    ``lab-op2`` or ``arm-1``.
    """
    if not isinstance(cn, str) or not isinstance(peer_id, str) or not cn or not peer_id:
        return False
    return peer_id == cn or peer_id.startswith(cn + "__")


class ReplayGuard:
    """Single use of each ``(certificate, nonce)`` within a freshness window.

    A signed message is good for one delivery. The guard remembers every pair
    it has seen for *ttl_s* seconds (the verifier refuses anything older than
    that on its own) and holds at most *max_size* entries, evicting the oldest.
    """

    def __init__(self, ttl_s: float = DEFAULT_FRESHNESS_S + DEFAULT_FORWARD_SKEW_S, max_size: int = 4096) -> None:
        self._ttl_s = float(ttl_s)
        self._max_size = int(max_size)
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def seen_before(self, cert_sha256: str, nonce: str, *, now_mono: float | None = None) -> bool:
        """Record the pair; ``True`` when it was already recorded within the window."""
        now = time.monotonic() if now_mono is None else now_mono
        key = (cert_sha256, nonce)
        with self._lock:
            cutoff = now - self._ttl_s
            for stale in [k for k, seen_at in self._seen.items() if seen_at < cutoff]:
                self._seen.pop(stale, None)
            if key in self._seen:
                return True
            if len(self._seen) >= self._max_size:
                oldest = min(self._seen, key=self._seen.__getitem__)
                self._seen.pop(oldest, None)
            self._seen[key] = now
        return False


_required_warned: set[str] = set()
_required_warned_lock = threading.Lock()


def signing_required(roots: TrustRoots | None = None) -> bool:
    """Whether this peer refuses an unsigned or unverifiable identity.

    Reads ``STRANDS_MESH_REQUIRE_SIGNED_IDENTITY``: ``1`` always, ``0`` never,
    ``auto`` (default) exactly when the auth mode is ``mtls`` and a trust root
    is configured (*roots* when given, else :meth:`TrustRoots.load`). A value
    that is none of the three is treated as ``1`` and warned about once: on a
    safety setting a typo must not quietly turn the check off.
    """
    raw = os.getenv(REQUIRE_ENV, "auto").strip().lower()
    if raw == "1":
        return True
    if raw == "0":
        return False
    if raw != "auto":
        with _required_warned_lock:
            first = raw not in _required_warned
            _required_warned.add(raw)
        if first:
            logger.warning(
                "[mesh] %s=%s is not one of 1, 0, auto; treating it as 1 (signatures required)",
                REQUIRE_ENV,
                refusal_str(raw),
            )
        return True
    from strands_robots.mesh._zenoh_config import resolve_auth_mode

    try:
        mode = resolve_auth_mode()
    except ValueError:
        # A mode the session will refuse to start under: nothing to protect yet.
        return False
    if mode != "mtls":
        return False
    return (roots if roots is not None else TrustRoots.load()) is not None
