"""Per-peer cryptographic identity for the mesh.

Background
----------
``STRANDS_MESH_PSK`` proves *fleet membership* — every peer holds the same
symmetric key, so any insider with the PSK can mint a valid HMAC for any
``sender_id``. That makes ``sender_id`` (and everything keyed off it —
per-sender rate buckets, presence claims, audit-log attribution) forgeable
by an authenticated insider.

This module adds a second, per-peer key layered on top of the PSK so that
``sender_id`` cannot be forged by another fleet member. The PSK still
proves "you are in the fleet"; the per-peer key proves "you are
specifically *this* peer".

Design
------
Each peer owns a private 32-byte HMAC key stored at
``~/.strands_robots/mesh/<peer_id>.key`` (mode 0o600). The key is generated
on first use with :func:`os.urandom` if absent.

Identity is bound on the wire by an extra envelope field ``kid`` (= peer
id). The signature now covers ``v|ts|nonce|kid|payload`` and is computed
with the per-peer key when one is configured. The receiver:

1. Reads ``kid`` from the envelope.
2. Looks up that ``kid`` in its **peer directory** (a TOFU map of
   ``kid -> verifier_key``).
3. If the kid is known, recomputes the HMAC under the directory key and
   accepts/rejects accordingly.
4. If the kid is unknown, the envelope is verified under the PSK only
   (legacy / bootstrap path) AND the peer's verifier_key claim from the
   payload is **pinned** for future messages (TOFU). After pinning, any
   subsequent message from the same ``kid`` MUST verify under the pinned
   key — this is what closes presence-spoofing once the directory has
   been populated.

Symmetric-key TOFU model
~~~~~~~~~~~~~~~~~~~~~~~~
Pure HMAC means *both* peers need to share the per-peer key for the
receiver to verify it. Distribution options:

* **TOFU via PSK-signed presence** (default): the peer puts a random
  per-peer key into its presence broadcast, signed under the PSK. The
  first PSK-valid presence carrying a key for a given ``peer_id``
  pins the directory entry. After pinning, the per-peer key is the
  authority — PSK-only signatures from a *different* sender_id are
  rejected as identity spoofs. This closes the threat for any
  attacker who joins after the first peer is pinned; an attacker
  who races the first presence wins TOFU but loses to operator
  audit (the audit log records the pin event).

* **Operator-distributed peer keys** (stronger): operators
  pre-distribute peer keys via secrets manager and load them with
  :func:`load_peer_directory_from_dir`. No TOFU race, no
  PSK-only-spoof window.

* **Per-thing IoT cert/key** (strongest, IoT mode): when running
  under :mod:`strands_robots.mesh.iot`, the cert authority itself
  proves identity at the MQTT-mTLS layer. We additionally anchor
  the per-peer HMAC key file under the cert directory so the same
  filesystem ACLs that protect the IoT private key also protect
  the mesh-layer key.

Out-of-scope
~~~~~~~~~~~~
* **Asymmetric crypto** — Ed25519 would let receivers verify without
  ever holding the signer's secret, but at the cost of taking
  ``cryptography`` as a hard dependency. Once we adopt the IoT
  per-thing cert path (follow-up issue #N), we get asymmetric
  identity for free; until then, this symmetric-with-TOFU scheme
  is the pragmatic minimum that closes vector #9 / #10.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── Constants ───────────────────────────────────────────────────────────

#: Length in bytes of a per-peer HMAC key. 32 bytes = 256 bits, matches the
#: digest size of SHA-256 so an attacker cannot do better than brute-force
#: the digest itself.
PEER_KEY_LEN: int = 32

#: Default location of the per-peer key directory. Each peer stores its
#: secret as ``<peer_id>.key`` here. Override with ``STRANDS_MESH_PEER_KEY_DIR``.
DEFAULT_KEY_DIR: Path = Path.home() / ".strands_robots" / "mesh"

#: Allowed characters in a ``peer_id`` (also used as a key file name).
#: Must be a subset of common filesystem-safe + IoT thing-name characters
#: so the mapping ``peer_id -> file path`` is unambiguous and traversal-free.
_PEER_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

#: Maximum length of a ``peer_id``. Aligned with AWS IoT thing-name limits
#: (128) and short enough that the directory map cannot be DoS'd by giant
#: ids in a TOFU race.
PEER_ID_MAX_LEN: int = 128

#: Maximum number of peers we will ever pin. Bounds memory under a TOFU
#: flood from a malicious peer that rotates ``peer_id`` rapidly. Operators
#: with fleets larger than this should pre-distribute keys with
#: :func:`load_peer_directory_from_dir` (operator-distributed mode), in
#: which case pinning is a no-op.
PEER_DIRECTORY_MAX: int = 10_000


# ─── Exceptions ──────────────────────────────────────────────────────────


class IdentityError(Exception):
    """Base class for identity-layer rejections."""


class UnknownPeerError(IdentityError):
    """Envelope's ``kid`` is not in the directory and TOFU is disabled."""


class IdentityMismatchError(IdentityError):
    """Envelope ``kid`` does not match the bound key for that peer.

    Raised when an attacker holding only the PSK tries to mint a message
    with another peer's ``sender_id`` after that peer's per-peer key has
    been pinned.
    """


class IdentitySpoofError(IdentityError):
    """``payload.sender_id`` does not match envelope ``kid``.

    Raised when an authenticated peer claims to be another peer in the
    payload — closes the audit-attribution forgery surface (vector #10).
    """


# ─── Peer ID validation ──────────────────────────────────────────────────


def is_valid_peer_id(peer_id: Any) -> bool:
    """Return True when *peer_id* is a non-empty, filesystem-safe id.

    Rejects path separators, traversal sequences, NUL bytes, and any byte
    outside the ``[A-Za-z0-9_.-]`` set so that ``peer_id`` cannot be
    hostile when used as a filename or a directory key.
    """
    if not isinstance(peer_id, str) or not peer_id:
        return False
    if len(peer_id) > PEER_ID_MAX_LEN:
        return False
    if not _PEER_ID_RE.fullmatch(peer_id):
        return False
    # Defense in depth: even though the regex blocks ``.`` segments, an id
    # of literally ``.`` or ``..`` is dangerous if anyone bypasses the
    # regex via str subclass shenanigans.
    if peer_id in (".", ".."):
        return False
    return True


# ─── Per-process directory state ─────────────────────────────────────────


class _PeerDirectory:
    """Map ``peer_id -> 32-byte HMAC key`` with an optional bind timestamp.

    The directory is shared process-wide. Pinning is monotonic: once a
    ``peer_id`` is bound to a key, that binding cannot be silently
    overwritten — calling :meth:`pin` again with a different key for an
    already-bound peer raises :class:`IdentityMismatchError`. Operators
    rotating a peer key must call :meth:`drop` first.
    """

    __slots__ = ("_lock", "_keys", "_bound_at", "_tofu_enabled")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, bytes] = {}
        self._bound_at: dict[str, float] = {}
        self._tofu_enabled: bool = True

    # ── pin / lookup ──────────────────────────────────────────────────

    def pin(self, peer_id: str, key: bytes) -> bool:
        """Bind *peer_id* to *key*. Return True iff this is a new pin.

        Re-pinning the same key is a no-op (returns False). Re-pinning a
        DIFFERENT key for an already-bound peer raises
        :class:`IdentityMismatchError` so an attacker cannot silently
        rotate a peer's identity by replaying TOFU.
        """
        if not is_valid_peer_id(peer_id):
            raise IdentityError(f"peer_id={peer_id!r} is not valid")
        if not isinstance(key, (bytes, bytearray)) or len(key) != PEER_KEY_LEN:
            raise IdentityError(
                f"per-peer key must be {PEER_KEY_LEN} bytes, got {type(key).__name__} len={len(key) if isinstance(key, (bytes, bytearray)) else '?'}"
            )
        key_b = bytes(key)
        with self._lock:
            existing = self._keys.get(peer_id)
            if existing is not None:
                if hmac.compare_digest(existing, key_b):
                    return False
                raise IdentityMismatchError(
                    f"peer_id={peer_id!r} already bound to a different key — refusing silent rotation"
                )
            if len(self._keys) >= PEER_DIRECTORY_MAX:
                raise IdentityError(
                    f"peer directory full ({PEER_DIRECTORY_MAX} entries); "
                    "preallocate keys via load_peer_directory_from_dir"
                )
            self._keys[peer_id] = key_b
            self._bound_at[peer_id] = time.time()
            return True

    def lookup(self, peer_id: str) -> bytes | None:
        """Return the bound key for *peer_id* or None."""
        with self._lock:
            return self._keys.get(peer_id)

    def is_bound(self, peer_id: str) -> bool:
        with self._lock:
            return peer_id in self._keys

    def drop(self, peer_id: str) -> bool:
        """Remove *peer_id* from the directory. Return True iff it was present."""
        with self._lock:
            removed = self._keys.pop(peer_id, None)
            self._bound_at.pop(peer_id, None)
            return removed is not None

    def clear(self) -> None:
        """Test-only: drop every binding."""
        with self._lock:
            self._keys.clear()
            self._bound_at.clear()

    def known_peers(self) -> list[str]:
        with self._lock:
            return sorted(self._keys.keys())

    def bound_at(self, peer_id: str) -> float | None:
        with self._lock:
            return self._bound_at.get(peer_id)

    # ── TOFU mode ─────────────────────────────────────────────────────

    @property
    def tofu_enabled(self) -> bool:
        return self._tofu_enabled

    @tofu_enabled.setter
    def tofu_enabled(self, value: bool) -> None:
        self._tofu_enabled = bool(value)


# Module-level singleton. A process-wide directory is correct for the mesh:
# every Mesh peer in the same Python interpreter shares one view of the
# fleet, so pinning observed by one Mesh applies to every other Mesh.
_DIRECTORY = _PeerDirectory()


def get_directory() -> _PeerDirectory:
    """Return the process-wide peer directory."""
    return _DIRECTORY


def reset_directory() -> None:
    """Test-only helper: clear all pinned peers."""
    _DIRECTORY.clear()
    _DIRECTORY.tofu_enabled = True


# ─── Local peer key ──────────────────────────────────────────────────────


def _peer_key_dir() -> Path:
    raw = os.getenv("STRANDS_MESH_PEER_KEY_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_KEY_DIR


def _peer_key_path(peer_id: str) -> Path:
    if not is_valid_peer_id(peer_id):
        raise IdentityError(f"peer_id={peer_id!r} is not valid")
    return _peer_key_dir() / f"{peer_id}.key"


def load_or_create_peer_key(peer_id: str) -> bytes:
    """Return this process's per-peer HMAC key, creating one if needed.

    Persistent across runs: the key is stored at ``<key_dir>/<peer_id>.key``
    with mode 0o600. The directory is created with mode 0o700 if missing.
    Operators who have pre-distributed keys should drop them at the same
    path before first run.

    Raises :class:`IdentityError` on permission failures so callers can
    fall back gracefully (PSK-only mode) instead of crashing the mesh.
    """
    if not is_valid_peer_id(peer_id):
        raise IdentityError(f"peer_id={peer_id!r} is not valid")

    path = _peer_key_path(peer_id)
    try:
        if path.exists():
            data = path.read_bytes()
            if len(data) != PEER_KEY_LEN:
                raise IdentityError(f"per-peer key at {path} is {len(data)} bytes, expected {PEER_KEY_LEN}")
            # Best-effort permission check on POSIX. If the key is world-
            # readable we keep using it (operators may have intentionally
            # mounted it from a secrets-manager-managed file with their
            # own ACLs) but warn loudly. We do NOT auto-chmod foreign
            # files.
            try:
                mode = path.stat().st_mode & 0o777
                if mode & 0o077:
                    logger.warning(
                        "[identity] per-peer key at %s has permissive mode 0o%o; "
                        "expected 0o600. Consider tightening with chmod 600.",
                        path,
                        mode,
                    )
            except OSError:
                pass
            return data

        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = os.urandom(PEER_KEY_LEN)
        # Atomic write with restrictive umask. Use os.open to set mode at
        # creation rather than chmodding an already-readable file.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)
        logger.info("[identity] generated new per-peer key for %s at %s", peer_id, path)
        return key
    except OSError as exc:
        raise IdentityError(f"cannot read/create per-peer key for {peer_id!r} at {path}: {exc}") from exc


def configure_local_peer(peer_id: str) -> bytes | None:
    """Resolve the local peer's signing key.

    Returns the key bytes on success, ``None`` when the local peer should
    fall back to PSK-only signing (per-peer identity disabled).
    Resolution order:

    1. ``STRANDS_MESH_PEER_KEY`` env var (hex-encoded 32 bytes) — useful
       for ephemeral test peers.
    2. ``STRANDS_MESH_PEER_KEY_FILE`` env var — absolute path override.
    3. ``<STRANDS_MESH_PEER_KEY_DIR>/<peer_id>.key`` (default
       ``~/.strands_robots/mesh``).

    Returns ``None`` when ``STRANDS_MESH_PEER_IDENTITY=false`` is set, so
    operators who want to opt out (e.g. while migrating) can do so
    explicitly.
    """
    if os.getenv("STRANDS_MESH_PEER_IDENTITY", "true").strip().lower() == "false":
        return None
    if not is_valid_peer_id(peer_id):
        logger.warning("[identity] peer_id=%r is not valid; per-peer identity disabled", peer_id)
        return None

    raw_hex = os.getenv("STRANDS_MESH_PEER_KEY", "").strip()
    if raw_hex:
        try:
            key = bytes.fromhex(raw_hex)
        except ValueError as exc:
            raise IdentityError(f"STRANDS_MESH_PEER_KEY is not valid hex: {exc}") from exc
        if len(key) != PEER_KEY_LEN:
            raise IdentityError(f"STRANDS_MESH_PEER_KEY must decode to {PEER_KEY_LEN} bytes")
        return key

    file_override = os.getenv("STRANDS_MESH_PEER_KEY_FILE", "").strip()
    if file_override:
        path = Path(file_override)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise IdentityError(f"cannot read STRANDS_MESH_PEER_KEY_FILE={path}: {exc}") from exc
        if len(data) != PEER_KEY_LEN:
            raise IdentityError(f"STRANDS_MESH_PEER_KEY_FILE={path} is {len(data)} bytes, expected {PEER_KEY_LEN}")
        return data

    try:
        return load_or_create_peer_key(peer_id)
    except IdentityError as exc:
        # Filesystem unavailable (read-only container, etc.). Log and let
        # the caller fall back to PSK-only mode rather than crash.
        logger.warning(
            "[identity] could not provision per-peer key for %s; falling back to PSK-only signing: %s",
            peer_id,
            exc,
        )
        return None


# ─── Operator pre-distribution ───────────────────────────────────────────


def load_peer_directory_from_dir(directory: str | os.PathLike[str]) -> int:
    """Pre-populate the directory from a key-file folder.

    Each file ``<directory>/<peer_id>.key`` (32 bytes) is loaded and
    pinned. Returns the number of peers pinned. Stronger than TOFU
    because there is no race window — the operator establishes trust
    before the first wire message.

    Files with an invalid peer_id-ish basename or wrong length are
    skipped with a WARNING. Caller is responsible for filesystem ACLs.
    """
    root = Path(directory)
    if not root.is_dir():
        raise IdentityError(f"peer-directory source {root} is not a directory")
    pinned = 0
    for entry in sorted(root.iterdir()):
        if entry.suffix != ".key" or not entry.is_file():
            continue
        peer_id = entry.stem
        if not is_valid_peer_id(peer_id):
            logger.warning("[identity] skipping %s: peer_id=%r is invalid", entry, peer_id)
            continue
        try:
            data = entry.read_bytes()
        except OSError as exc:
            logger.warning("[identity] skipping %s: %s", entry, exc)
            continue
        if len(data) != PEER_KEY_LEN:
            logger.warning(
                "[identity] skipping %s: key is %d bytes, expected %d",
                entry,
                len(data),
                PEER_KEY_LEN,
            )
            continue
        try:
            if _DIRECTORY.pin(peer_id, data):
                pinned += 1
        except IdentityMismatchError:
            logger.warning(
                "[identity] skipping %s: peer_id=%s already bound to a different key",
                entry,
                peer_id,
            )
    return pinned


# ─── Envelope signing / verification helpers ────────────────────────────


def _identity_canonical_bytes(envelope: dict[str, Any]) -> bytes:
    """Encode the identity-signed subset of *envelope* deterministically.

    The identity signature covers ``v|ts|nonce|kid|payload``. ``sig`` is
    excluded (it IS the signature). Keys are sorted and separators tight
    so semantically-equal envelopes always serialize identically.
    """
    body = {k: envelope[k] for k in ("v", "ts", "nonce", "kid", "payload") if k in envelope}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_identity_sig(envelope: dict[str, Any], key: bytes) -> str:
    """Return ``HMAC-SHA256(key, canonical(envelope))`` as hex."""
    return hmac.new(key, _identity_canonical_bytes(envelope), hashlib.sha256).hexdigest()


def verify_identity_sig(envelope: dict[str, Any], key: bytes, sig: str) -> bool:
    """Constant-time compare ``sig`` against the recomputed HMAC."""
    if not isinstance(sig, str):
        return False
    expected = compute_identity_sig(envelope, key)
    return hmac.compare_digest(expected, sig)


# ─── Public surface ──────────────────────────────────────────────────────

__all__ = [
    "DEFAULT_KEY_DIR",
    "IdentityError",
    "IdentityMismatchError",
    "IdentitySpoofError",
    "PEER_DIRECTORY_MAX",
    "PEER_ID_MAX_LEN",
    "PEER_KEY_LEN",
    "UnknownPeerError",
    "compute_identity_sig",
    "configure_local_peer",
    "get_directory",
    "is_valid_peer_id",
    "load_or_create_peer_key",
    "load_peer_directory_from_dir",
    "reset_directory",
    "verify_identity_sig",
]
