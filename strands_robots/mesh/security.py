"""Mesh authentication, authorization, and rate-limiting primitives.

This module provides the security boundary that sits between the wire and
:class:`strands_robots.mesh.core.Mesh`. It owns three concerns:

1. **Message authentication** (:func:`sign_envelope` / :func:`verify_envelope`):
   pre-shared-key HMAC-SHA256 over a canonical JSON encoding, with replay
   protection via per-message nonces and a freshness window. Any party that
   does not hold ``STRANDS_MESH_PSK`` cannot forge or replay a command.

2. **Command validation** (:func:`validate_command`): an action allowlist
   plus per-action schema and bounds, including a ``policy_host`` allowlist
   that prevents a remote caller from steering VLA policy fetches at an
   attacker-controlled inference server.

3. **Per-sender rate limiting** (:func:`consume_peer_token`,
   :class:`TokenBucket`): a process-wide token bucket keyed by sender id.
   The mesh consumes one token per inbound command before spawning the
   exec thread, so a flood from one peer cannot exhaust resources.

Operating modes
---------------
The module supports two modes, selected by env vars:

* **Permissive** (default; ``STRANDS_MESH_PSK`` unset): unsigned legacy
  payloads pass through ``verify_envelope`` so existing zero-config
  Zenoh-LAN setups keep working. A one-time WARNING is logged.
* **Strict** (``STRANDS_MESH_REQUIRE_AUTH=true`` and/or PSK configured):
  unsigned envelopes are rejected with :class:`AuthenticationError`.

Trust model
-----------
The PSK is the symmetric secret distributed at provisioning time to every
peer that should be allowed to issue commands. It is intended to be loaded
from secrets manager, IAM-protected SSM Parameter Store, or equivalent —
never committed to source.

Configuration env vars
----------------------
* ``STRANDS_MESH_PSK`` — pre-shared key. Enables HMAC signing.
* ``STRANDS_MESH_REQUIRE_AUTH`` — ``"true"`` to reject unsigned envelopes
  even when the PSK is unset (useful for tests / staging gates).
* ``STRANDS_MESH_REPLAY_WINDOW`` — past-tolerance for envelope ``ts``,
  in seconds. Default 60, capped at :data:`_MAX_REPLAY_WINDOW_S`.
* ``STRANDS_MESH_POLICY_HOST_ALLOW`` — comma-separated host/CIDR list to
  extend :data:`_DEFAULT_POLICY_HOSTS`.
* ``STRANDS_MESH_PEER_RATE`` — ``"<count>/<seconds>"`` per-sender command
  rate (default ``"20/60"``). ``count`` clamped to
  :data:`_MAX_PEER_RATE_BURST`.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ─── Module configuration ────────────────────────────────────────────────

#: Maximum duration (seconds) accepted for ``execute`` / ``start`` commands.
MAX_DURATION_S: float = 3600.0

#: Maximum RPC timeout (seconds) accepted from peers.
MAX_TIMEOUT_S: float = 300.0

#: Maximum length (characters) of a natural-language ``instruction`` payload.
MAX_INSTRUCTION_LEN: int = 2000

# Maximum forward clock skew accepted on signed envelopes. NTP drift on a
# healthy fleet is sub-second; 5s is a generous safety margin while still
# being well below the replay window so a future-stamped message cannot
# survive long enough for its nonce to age out of the cache.
_MAX_FORWARD_SKEW_S: float = 5.0

# Hard upper cap on STRANDS_MESH_REPLAY_WINDOW. A value larger than this is
# almost certainly an operator typo or a misguided attempt to disable replay
# protection altogether.
_MAX_REPLAY_WINDOW_S: float = 600.0  # 10 minutes

# Hard upper cap on the burst component of STRANDS_MESH_PEER_RATE. Prevents
# misconfiguration like ``999999/0.1`` which would effectively disable the
# per-sender rate limiter.
_MAX_PEER_RATE_BURST: int = 1000

#: Action vocabulary accepted by :func:`validate_command`. Mirrors the
#: dispatch table in :meth:`Mesh._dispatch`. Keep these two sets in sync
#: when adding a new action.
ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "status",
        "stop",
        "features",
        "state",
        "execute",
        "start",
        "step",
        "reset",
        "teleop_status",
        "teleop_receive",
        "teleop_stop",
        # ``resume`` clears the emergency-stop lockout. It is the only
        # action other than ``status`` permitted while the lockout is engaged.
        "resume",
    }
)

# Default allowlist for VLA policy server targets. Loopback only — operators
# extend explicitly via STRANDS_MESH_POLICY_HOST_ALLOW.
_DEFAULT_POLICY_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

#: Replay-protection nonce cache size. Once exceeded, expired entries are
#: pruned and the oldest 20 % are dropped.
_NONCE_CACHE_MAX: int = 10_000


# ─── Exception hierarchy ─────────────────────────────────────────────────


class SecurityError(Exception):
    """Base class for all mesh security rejections."""


class AuthenticationError(SecurityError):
    """Envelope signature is missing, malformed, or invalid."""


class AuthorizationError(SecurityError):
    """Sender is authenticated but not allowed to perform the action."""


class ValidationError(SecurityError):
    """Command payload failed schema or bounds checks."""


class RateLimitError(SecurityError):
    """Sender exceeded a configured rate limit."""


class LockoutError(SecurityError):
    """Command rejected because the local mesh is in emergency-stop lockout.

    Raised from :meth:`Mesh._dispatch` when an action other than ``status``
    or ``resume`` arrives while ``_estop_lockout`` is engaged. The wire
    response is intentionally generic — the exception type carries the real
    semantics so the dispatch wrapper can audit the rejection symmetrically
    with :class:`ValidationError`.
    """


# ─── PSK / configuration helpers ─────────────────────────────────────────

# One-shot flag so the "PSK not set" warning only fires once per process,
# regardless of how many envelopes are signed. Module-private state.
_PSK_WARNED: bool = False


def _warn_psk_unset_once() -> None:
    """Emit the one-time "STRANDS_MESH_PSK not set" warning.

    Subsequent calls are no-ops. Centralised here so :func:`sign_envelope`
    stays focused on envelope construction and so static analysers see a
    self-contained read + write of the module-level flag.
    """
    global _PSK_WARNED
    if _PSK_WARNED:
        return
    _PSK_WARNED = True
    logger.warning(
        "[security] STRANDS_MESH_PSK not set — mesh messages are "
        "unsigned. Set STRANDS_MESH_PSK to enable HMAC authentication "
        "(strict mode: STRANDS_MESH_REQUIRE_AUTH=true)."
    )


def _get_psk() -> bytes | None:
    psk = os.getenv("STRANDS_MESH_PSK")
    return psk.encode("utf-8") if psk else None


def psk_configured() -> bool:
    """Return True when ``STRANDS_MESH_PSK`` is present (signing enabled)."""
    return _get_psk() is not None


def auth_required() -> bool:
    """Return True when ``STRANDS_MESH_REQUIRE_AUTH`` is enabled.

    In this mode the verifier rejects un-enveloped (legacy) messages and
    refuses to fall back to permissive behaviour even when no PSK is set.
    """
    return os.getenv("STRANDS_MESH_REQUIRE_AUTH", "false").strip().lower() == "true"


def _replay_window_s() -> float:
    """Resolve the replay-window past-tolerance in seconds.

    Reads ``STRANDS_MESH_REPLAY_WINDOW``, clamps to
    ``[1, _MAX_REPLAY_WINDOW_S]``, and falls back to 60 on parse error.
    """
    try:
        value = float(os.getenv("STRANDS_MESH_REPLAY_WINDOW", "60"))
    except ValueError:
        return 60.0
    return max(1.0, min(value, _MAX_REPLAY_WINDOW_S))


# ─── Replay-protection nonce cache ───────────────────────────────────────

_NONCE_LOCK = threading.Lock()
# Cache key is a (scope, nonce) tuple so multiple verifiers running in the
# same process — for example several Mesh peers under tests or in a single
# Simulation host — each maintain an independent replay window. Without the
# scope, every broadcast envelope verified by the second peer would be
# rejected as a replay even though it was the FIRST arrival at THAT peer.
_NONCE_CACHE: dict[tuple[str, str], float] = {}

# Sentinel scope used when the caller does not supply one (back-compat for
# tests and any process where there is provably one verifier).
_DEFAULT_NONCE_SCOPE = ""


def clear_replay_cache(scope: str | None = None) -> None:
    """Drop cached nonces.

    With no argument or ``scope=None``, drop every cached nonce. With an
    explicit scope, drop only that scope's entries. Test-only helper.
    """
    with _NONCE_LOCK:
        if scope is None:
            _NONCE_CACHE.clear()
        else:
            for key in [k for k in _NONCE_CACHE if k[0] == scope]:
                _NONCE_CACHE.pop(key, None)


def _record_nonce(nonce: str, now: float, scope: str = _DEFAULT_NONCE_SCOPE) -> bool:
    """Record *nonce* as seen at *now* in the given *scope*.

    Returns True if the (scope, nonce) tuple was novel (and is now
    recorded), False if it was already in the cache within the replay
    window — i.e. a replay.
    """
    window = _replay_window_s()
    with _NONCE_LOCK:
        # Lazy GC: prune expired entries when the cache is over its cap.
        if len(_NONCE_CACHE) > _NONCE_CACHE_MAX:
            cutoff = now - window
            for stale in [k for k, t in _NONCE_CACHE.items() if t < cutoff]:
                _NONCE_CACHE.pop(stale, None)
            # If pruning didn't help (cache full of fresh entries — likely
            # a flood), drop the oldest 20 %.
            if len(_NONCE_CACHE) > _NONCE_CACHE_MAX:
                ordered = sorted(_NONCE_CACHE.items(), key=lambda kv: kv[1])
                drop = max(1, len(ordered) // 5)
                for key, _ in ordered[:drop]:
                    _NONCE_CACHE.pop(key, None)

        cache_key = (scope, nonce)
        if cache_key in _NONCE_CACHE:
            return False
        _NONCE_CACHE[cache_key] = now
        return True


# ─── Canonical encoding + HMAC primitives ────────────────────────────────


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Encode *payload* to deterministic bytes for HMAC computation.

    Sorted keys and compact separators ensure that semantically-equal dicts
    always produce identical bytes — without this, an attacker could swap
    key order to keep the dict semantically equivalent but break the
    receiver's ability to recompute the signature.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hmac_hex(key: bytes, body: bytes) -> str:
    return hmac.new(key, body, hashlib.sha256).hexdigest()


# ─── Envelope sign / verify ──────────────────────────────────────────────


def sign_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap *payload* in an authenticated envelope.

    The envelope shape is::

        {
            "v": 1,
            "ts": <unix seconds>,
            "nonce": <uuid4 hex, 32 chars>,
            "payload": <original payload>,
            "sig": <sha256 hex hmac, omitted in permissive mode>,
        }

    When no PSK is configured the envelope is still emitted (so the wire
    format stays stable across modes) but the ``sig`` field is absent.
    Verifiers running in strict mode reject such envelopes; in permissive
    mode they pass through after replay-window and nonce checks.
    """
    if not isinstance(payload, dict):
        raise TypeError("sign_envelope requires a dict payload")

    envelope: dict[str, Any] = {
        "v": 1,
        "ts": time.time(),
        "nonce": uuid.uuid4().hex,
        "payload": payload,
    }

    psk = _get_psk()
    if psk is None:
        _warn_psk_unset_once()
        return envelope

    body = _canonical_bytes({k: envelope[k] for k in ("v", "ts", "nonce", "payload")})
    envelope["sig"] = _hmac_hex(psk, body)
    return envelope


def verify_envelope(envelope: dict[str, Any], scope: str = _DEFAULT_NONCE_SCOPE) -> dict[str, Any]:
    """Validate signature, freshness, and replay window. Return the payload.

    *scope* identifies the verifier within the process. When several Mesh
    peers run in one Python process they pass their ``peer_id`` so each
    maintains an independent nonce cache; otherwise the first peer to
    receive a broadcast would block every other peer from accepting it.

    The verifier enforces, in order:

    1. Envelope shape (versioned dict with ``ts`` and ``nonce``).
    2. Freshness: ``ts`` must be within ``[now - replay_window, now +
       _MAX_FORWARD_SKEW_S]``. Asymmetric tolerance prevents a peer from
       borrowing forward time to defer a replay until its nonce has aged
       out of the cache.
    3. HMAC-SHA256 signature against ``STRANDS_MESH_PSK`` when configured.
       Constant-time compared via :func:`hmac.compare_digest`.
    4. Replay protection: the nonce is recorded; a repeat is rejected.

    Raises :class:`AuthenticationError` on any failure.

    Permissive mode: when no PSK is configured AND
    ``STRANDS_MESH_REQUIRE_AUTH`` is false, envelopes without ``sig`` are
    accepted but still subject to freshness and replay checks. Bare legacy
    dicts (no ``v``/``payload`` keys) pass through unchanged for back-compat.
    """
    if not isinstance(envelope, dict):
        raise AuthenticationError("envelope must be a dict")

    # Legacy shape: a bare payload dict with no envelope wrapper. Strict
    # mode rejects this; permissive mode passes it through unchanged so
    # existing un-upgraded peers still interop.
    if "v" not in envelope or "payload" not in envelope:
        if auth_required():
            raise AuthenticationError("missing envelope (strict mode requires signed messages)")
        return envelope

    if envelope.get("v") != 1:
        raise AuthenticationError(f"unsupported envelope version: {envelope.get('v')!r}")

    ts = envelope.get("ts")
    if not isinstance(ts, (int, float)):
        raise AuthenticationError("envelope ts missing or wrong type")

    # Nonce minimum 16 chars (64 bits). sign_envelope emits 32 chars from
    # uuid4; the minimum guards against attackers shipping short / reused
    # values that could cause cache collisions.
    nonce = envelope.get("nonce")
    if not isinstance(nonce, str) or len(nonce) < 16:
        raise AuthenticationError("envelope nonce missing or malformed")

    now = time.time()
    window = _replay_window_s()
    forward_skew = min(window, _MAX_FORWARD_SKEW_S)
    if ts > now + forward_skew:
        raise AuthenticationError(
            f"envelope ts {ts:.3f} too far in future (forward_skew_s={forward_skew}, now={now:.3f})"
        )
    if (now - ts) > window:
        raise AuthenticationError(f"envelope ts {ts:.3f} too old (window_s={window}, now={now:.3f})")

    psk = _get_psk()
    sig = envelope.get("sig")

    if psk is None:
        # Permissive: no signature to check, but still enforce replay.
        if auth_required():
            raise AuthenticationError("PSK not configured but auth required")
        if not _record_nonce(nonce, now, scope=scope):
            raise AuthenticationError(f"replay detected for nonce {nonce}")
        return envelope.get("payload") or {}

    if not isinstance(sig, str):
        raise AuthenticationError("envelope sig missing")

    body = _canonical_bytes({k: envelope[k] for k in ("v", "ts", "nonce", "payload")})
    expected = _hmac_hex(psk, body)
    if not hmac.compare_digest(sig, expected):
        raise AuthenticationError("HMAC signature mismatch")

    if not _record_nonce(nonce, now, scope=scope):
        raise AuthenticationError(f"replay detected for nonce {nonce}")

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise AuthenticationError("envelope payload not a dict")
    return payload


# ─── Policy-host allowlist ───────────────────────────────────────────────


def _policy_host_allowlist() -> list[str]:
    raw = os.getenv("STRANDS_MESH_POLICY_HOST_ALLOW", "")
    extra = [host.strip() for host in raw.split(",") if host.strip()]
    return list(_DEFAULT_POLICY_HOSTS) + extra


def is_safe_policy_host(host: str) -> bool:
    """Return True when *host* is permitted as a VLA policy server target.

    The default allowlist is loopback only (``localhost``, ``127.0.0.1``,
    ``::1``). Operators extend it via ``STRANDS_MESH_POLICY_HOST_ALLOW``,
    a comma-separated list of hostnames or CIDR ranges (``"vla.internal,
    10.0.0.0/24"``).

    Hostnames are matched literally (case-insensitive); IP literals are
    additionally matched against any CIDR entries in the operator list.
    """
    if not isinstance(host, str) or not host:
        return False
    host_lc = host.strip().lower()
    allowlist = _policy_host_allowlist()

    # Exact host / IP literal match.
    for entry in allowlist:
        if host_lc == entry.strip().lower():
            return True

    # CIDR match: only meaningful when *host* is itself an IP literal.
    try:
        ip = ipaddress.ip_address(host_lc)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            net = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if ip in net:
            return True
    return False


# ─── Command schema and bounds ───────────────────────────────────────────


def _coerce_float(name: str, value: Any, *, lo: float, hi: float, default: float | None) -> float:
    if value is None:
        if default is None:
            raise ValidationError(f"{name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    coerced = float(value)
    if coerced < lo or coerced > hi:
        raise ValidationError(f"{name}={coerced} out of bounds [{lo}, {hi}]")
    return coerced


def _coerce_int(name: str, value: Any, *, lo: int, hi: int, default: int | None) -> int:
    if value is None:
        if default is None:
            raise ValidationError(f"{name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    coerced = int(value)
    if coerced < lo or coerced > hi:
        raise ValidationError(f"{name}={coerced} out of bounds [{lo}, {hi}]")
    return coerced


def validate_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Validate a mesh command and return a sanitized copy.

    Performed checks:

    * ``action`` must be a string and a member of :data:`ALLOWED_ACTIONS`.
    * ``execute`` and ``start`` actions require:
        - ``instruction``: non-empty str up to :data:`MAX_INSTRUCTION_LEN`.
        - ``policy_host``: in the allowlist (defaults to ``"localhost"``).
        - ``duration``: ``[0, MAX_DURATION_S]``, defaults to 30.
        - ``policy_port`` (optional): integer in ``[1, 65535]``.
    * ``step``: ``steps`` integer in ``[1, 10_000]``, defaults to 1.
    * ``teleop_receive``: ``source_peer_id`` non-empty str.

    Raises :class:`ValidationError` on any rule violation.
    """
    if not isinstance(cmd, dict):
        raise ValidationError("command must be a dict")

    action = cmd.get("action", "status")
    if not isinstance(action, str):
        raise ValidationError("action must be a string")
    if action not in ALLOWED_ACTIONS:
        raise ValidationError(f"unknown action: {action!r} (allowed: {sorted(ALLOWED_ACTIONS)})")

    out = dict(cmd)
    out["action"] = action  # propagate the resolved default

    if action in ("execute", "start"):
        instruction = cmd.get("instruction", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValidationError("execute/start requires non-empty `instruction`")
        if len(instruction) > MAX_INSTRUCTION_LEN:
            raise ValidationError(f"instruction exceeds {MAX_INSTRUCTION_LEN} chars (got {len(instruction)})")

        policy_host = cmd.get("policy_host", "localhost")
        if not is_safe_policy_host(str(policy_host)):
            raise ValidationError(
                f"policy_host={policy_host!r} not in allowlist. Set STRANDS_MESH_POLICY_HOST_ALLOW to extend."
            )
        out["policy_host"] = policy_host

        out["duration"] = _coerce_float(
            "duration",
            cmd.get("duration", 30.0),
            lo=0.0,
            hi=MAX_DURATION_S,
            default=30.0,
        )

        if "policy_port" in cmd and cmd["policy_port"] is not None:
            out["policy_port"] = _coerce_int("policy_port", cmd["policy_port"], lo=1, hi=65535, default=None)

    elif action == "step":
        out["steps"] = _coerce_int("steps", cmd.get("steps", 1), lo=1, hi=10_000, default=1)

    elif action == "teleop_receive":
        source = cmd.get("source_peer_id", "")
        if not isinstance(source, str) or not source:
            raise ValidationError("teleop_receive requires non-empty source_peer_id")

    return out


# ─── Per-sender token-bucket rate limiter ────────────────────────────────


class TokenBucket:
    """Thread-safe token bucket.

    Tokens regenerate continuously at *rate_per_s* up to *capacity*.
    :meth:`consume` returns ``True`` on success and ``False`` when the
    bucket is starved.
    """

    __slots__ = ("capacity", "rate", "tokens", "last", "_lock")

    def __init__(self, capacity: float, rate_per_s: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(rate_per_s)
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, n: float = 1.0) -> bool:
        """Try to consume *n* tokens. Return True on success, False if starved."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


_PEER_RATE_LIMITS: dict[str, TokenBucket] = {}
_PEER_RATE_LOCK = threading.Lock()


def _peer_rate_config() -> tuple[int, float]:
    """Resolve the per-sender rate config as ``(burst, window_seconds)``.

    Reads ``STRANDS_MESH_PEER_RATE`` formatted ``"<count>/<seconds>"``,
    clamps the burst to :data:`_MAX_PEER_RATE_BURST`, and falls back to
    ``(20, 60)`` on parse error.
    """
    raw = os.getenv("STRANDS_MESH_PEER_RATE", "20/60")
    try:
        cnt, win = raw.split("/", 1)
        burst = max(1, min(int(cnt), _MAX_PEER_RATE_BURST))
        window = max(0.1, float(win))
    except Exception:
        return 20, 60.0
    return burst, window


def consume_peer_token(sender_id: str) -> bool:
    """Consume one token from *sender_id*'s bucket. ``False`` means drop.

    Buckets are lazily allocated on first sight of a sender. The registry
    self-prunes when it grows past 1000 buckets (any peer untouched for
    more than 10 windows is evicted).

    The actual ``consume()`` call happens inside the registry lock so the
    GC pass cannot pop the bucket between lookup and use.
    """
    if not sender_id:
        sender_id = "<anonymous>"
    burst, window = _peer_rate_config()
    rate_per_s = burst / window

    with _PEER_RATE_LOCK:
        bucket = _PEER_RATE_LIMITS.get(sender_id)
        if bucket is None:
            bucket = TokenBucket(capacity=burst, rate_per_s=rate_per_s)
            _PEER_RATE_LIMITS[sender_id] = bucket

        if len(_PEER_RATE_LIMITS) > 1000:
            cutoff = time.monotonic() - window * 10
            for stale in [key for key, value in _PEER_RATE_LIMITS.items() if value.last < cutoff]:
                if stale != sender_id:
                    _PEER_RATE_LIMITS.pop(stale, None)

        return bucket.consume(1.0)


def reset_peer_rate_limits() -> None:
    """Clear every peer's rate-limit bucket. Test-only helper."""
    with _PEER_RATE_LOCK:
        _PEER_RATE_LIMITS.clear()


__all__ = [
    "ALLOWED_ACTIONS",
    "AuthenticationError",
    "AuthorizationError",
    "LockoutError",
    "MAX_DURATION_S",
    "MAX_TIMEOUT_S",
    "RateLimitError",
    "SecurityError",
    "TokenBucket",
    "ValidationError",
    "auth_required",
    "clear_replay_cache",
    "consume_peer_token",
    "is_safe_policy_host",
    "psk_configured",
    "reset_peer_rate_limits",
    "sign_envelope",
    "validate_command",
    "verify_envelope",
]
