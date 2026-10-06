"""Signed resume assertions: who may clear an emergency-stop lockout.

Clearing a lockout is an operator decision, so the authority to do it is an
Ed25519 key pair rather than a shared string. The operator keeps the private
half in a passphrase-protected key file; every peer is configured with the
public half only (``STRANDS_MESH_RESUME_PUBLIC_KEY``). A peer can check a
resume but cannot mint one, so reading a peer's environment, its disk or the
command topic never yields the power to clear another robot.

An assertion names the lockout it clears and the robots it clears::

    {"v": 1, "epoch": "<lockout id>", "targets": ["arm-1", "arm-2"],
     "t": 1760000000.0, "nonce": "<hex>", "sig": "<base64url Ed25519>"}

``epoch`` is the id of the e-stop that engaged the lockout (every peer that
honoured one fleet e-stop holds the same id), so a captured assertion cannot
clear a later lockout, and ``targets`` names the peers it is for, so it cannot
clear any other robot. Freshness of ``t`` and single use of ``nonce`` are
checked by the receiving :class:`~strands_robots.mesh.core.Mesh`.

Create a key pair once, on the operator's machine::

    python -m strands_robots.mesh.resume_authority keygen ~/.strands_robots/resume_key.pem
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # cryptography is imported where a key is used, so importing this module stays cheap
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

logger = logging.getLogger(__name__)

#: Environment variable holding the public verification key(s): base64 of the
#: 32-byte raw Ed25519 public key, comma-separated when more than one operator
#: key is trusted (a rotation in progress, two consoles).
PUBLIC_KEY_ENV = "STRANDS_MESH_RESUME_PUBLIC_KEY"

#: Environment variable naming the operator's signing key file, read by the
#: dashboard host that signs a fleet resume.
SIGNING_KEY_FILE_ENV = "STRANDS_MESH_RESUME_SIGNING_KEY_FILE"

#: Wire version of the assertion; a peer refuses any other.
ASSERTION_VERSION = 1

#: Most peers one assertion may name.
MAX_TARGETS = 256

#: Shortest passphrase a signing key is written under.
PASSPHRASE_MIN_LEN = 16

#: Fewest of the four character classes (lower, upper, digit, other) a
#: passphrase must draw from, and fewest distinct characters it must hold, so
#: a long run of one letter is not mistaken for a strong secret.
PASSPHRASE_MIN_CLASSES = 3
PASSPHRASE_MIN_DISTINCT = 10

_EPOCH_RE = re.compile(r"^[0-9a-f]{32}\Z")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}\Z")
_PEER_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}\Z")
_SIG_RE = re.compile(r"^[A-Za-z0-9_\-]{86}(==)?\Z")


def new_epoch() -> str:
    """A fresh lockout id: 32 lowercase hex characters."""
    return uuid.uuid4().hex


def is_epoch(value: object) -> bool:
    """Whether *value* is shaped like a lockout id :func:`new_epoch` mints."""
    return isinstance(value, str) and _EPOCH_RE.fullmatch(value) is not None


def passphrase_problem(passphrase: str) -> str | None:
    """Why *passphrase* is too weak to protect a signing key, or ``None``.

    Length alone admits ``"aaaaaaaaaaaaaaaa"``; the check also asks for three
    character classes and ten distinct characters.
    """
    if len(passphrase) < PASSPHRASE_MIN_LEN:
        return f"passphrase is {len(passphrase)} characters; use at least {PASSPHRASE_MIN_LEN}"
    classes = sum(
        (
            any(c.islower() for c in passphrase),
            any(c.isupper() for c in passphrase),
            any(c.isdigit() for c in passphrase),
            any(not c.isalnum() for c in passphrase),
        )
    )
    if classes < PASSPHRASE_MIN_CLASSES:
        return f"passphrase uses {classes} of lower/upper/digit/symbol; use at least {PASSPHRASE_MIN_CLASSES}"
    if len(set(passphrase)) < PASSPHRASE_MIN_DISTINCT:
        return (
            f"passphrase repeats too much ({len(set(passphrase))} distinct characters, need {PASSPHRASE_MIN_DISTINCT})"
        )
    return None


def generate_signing_key(path: str | os.PathLike[str], passphrase: str) -> str:
    """Write a new passphrase-protected signing key to *path* (mode 0600).

    Args:
        path: Where the PEM file is written. An existing file is never replaced.
        passphrase: Encrypts the key at rest; must pass :func:`passphrase_problem`.

    Returns:
        The public key text to set as ``STRANDS_MESH_RESUME_PUBLIC_KEY`` on every peer.

    Raises:
        ValueError: The passphrase is too weak.
        FileExistsError: *path* already exists.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    problem = passphrase_problem(passphrase)
    if problem is not None:
        raise ValueError(problem)
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(passphrase.encode()),
    )
    fd = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    return public_key_text(key.public_key())


def load_signing_key(path: str | os.PathLike[str], passphrase: str) -> Ed25519PrivateKey:
    """Open the operator's signing key.

    Raises:
        ValueError: Wrong passphrase, or the file is not an Ed25519 key.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=passphrase.encode())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not open the resume signing key at {path}: wrong passphrase or not a key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an Ed25519 key")
    return key


def public_key_text(key: Ed25519PublicKey) -> str:
    """The text form of *key* that ``STRANDS_MESH_RESUME_PUBLIC_KEY`` holds."""
    from cryptography.hazmat.primitives import serialization

    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.urlsafe_b64encode(raw).decode()


def verify_keys() -> list[Ed25519PublicKey]:
    """The trusted public keys from ``STRANDS_MESH_RESUME_PUBLIC_KEY``.

    Empty when unset. A malformed entry is skipped with a WARNING, never
    treated as "accept anything": no key means no resume is admitted.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    keys: list[Ed25519PublicKey] = []
    for item in os.getenv(PUBLIC_KEY_ENV, "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            raw = base64.urlsafe_b64decode(item + "=" * (-len(item) % 4))
            keys.append(Ed25519PublicKey.from_public_bytes(raw))
        except (ValueError, TypeError):
            logger.warning("[safety] %s entry %r is not a base64 Ed25519 public key; ignored", PUBLIC_KEY_ENV, item)
    return keys


def signing_key_path() -> str | None:
    """The operator's signing key file from ``STRANDS_MESH_RESUME_SIGNING_KEY_FILE``, or ``None``.

    Only the host that signs a resume (the operator's dashboard) sets it; a
    robot never needs it.
    """
    return os.getenv(SIGNING_KEY_FILE_ENV, "").strip() or None


def _signed_bytes(fields: dict[str, Any]) -> bytes:
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()


def sign_assertion(
    key: Ed25519PrivateKey,
    *,
    epoch: str,
    targets: list[str],
    t: float | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Sign an assertion clearing lockout *epoch* on each peer in *targets*.

    Raises:
        ValueError: *epoch* or a target is malformed, or *targets* is empty.
    """
    if not is_epoch(epoch):
        raise ValueError(f"epoch must be a lockout id (32 hex characters), got {epoch!r}")
    names = sorted(set(targets))
    if not names or len(names) > MAX_TARGETS or not all(_PEER_RE.fullmatch(n) for n in names):
        raise ValueError(f"targets must be 1-{MAX_TARGETS} peer ids, got {targets!r}")
    fields: dict[str, Any] = {
        "v": ASSERTION_VERSION,
        "epoch": epoch,
        "targets": names,
        "t": time.time() if t is None else float(t),
        "nonce": nonce or uuid.uuid4().hex,
    }
    sig = base64.urlsafe_b64encode(key.sign(_signed_bytes(fields))).decode()
    return {**fields, "sig": sig}


def check_assertion(assertion: object, *, keys: list[Ed25519PublicKey], peer_id: str, epoch: str | None) -> str | None:
    """Why *assertion* may not clear *peer_id*'s lockout *epoch*, or ``None``.

    Checks the shape, the signature against *keys*, that *peer_id* is a target
    and that the assertion names *epoch*. Fail closed: no key, no epoch, an
    absent target or any malformed field is a refusal.
    """
    if not keys:
        return f"no resume verification key configured ({PUBLIC_KEY_ENV})"
    if epoch is None:
        return "no lockout epoch to resume"
    if not isinstance(assertion, dict):
        return "resume carries no signed assertion"
    fields = {k: assertion.get(k) for k in ("v", "epoch", "targets", "t", "nonce")}
    sig = assertion.get("sig")
    if set(assertion) != {*fields, "sig"}:
        return "assertion has unexpected or missing fields"
    if fields["v"] != ASSERTION_VERSION:
        return "assertion version not supported"
    targets = fields["targets"]
    t = fields["t"]
    if (
        not is_epoch(fields["epoch"])
        or not isinstance(fields["nonce"], str)
        or not _NONCE_RE.fullmatch(fields["nonce"])
        or not isinstance(targets, list)
        or not 0 < len(targets) <= MAX_TARGETS
        or not all(isinstance(n, str) and _PEER_RE.fullmatch(n) for n in targets)
        or isinstance(t, bool)
        or not isinstance(t, (int, float))
        or not math.isfinite(t)
        or not isinstance(sig, str)
        or not _SIG_RE.fullmatch(sig)
    ):
        return "assertion is malformed"
    from cryptography.exceptions import InvalidSignature

    raw_sig = base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))
    message = _signed_bytes(fields)
    for key in keys:
        try:
            key.verify(raw_sig, message)
            break
        except InvalidSignature:
            continue
    else:
        return "assertion signature does not verify"
    if peer_id not in targets:
        return "assertion does not name this peer"
    if fields["epoch"] != epoch:
        return "assertion is for a different lockout"
    return None


def _main(argv: list[str]) -> int:
    import getpass

    if len(argv) != 2 or argv[0] != "keygen":
        print("usage: python -m strands_robots.mesh.resume_authority keygen <key-file>", file=sys.stderr)
        return 2
    passphrase = getpass.getpass("passphrase for the new signing key: ")
    if getpass.getpass("again: ") != passphrase:
        print("passphrases differ", file=sys.stderr)
        return 1
    try:
        public = generate_signing_key(argv[1], passphrase)
    except (ValueError, FileExistsError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"signing key written to {argv[1]}; keep it on the operator's machine only.")
    print(f"set on every peer:  {PUBLIC_KEY_ENV}={public}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
