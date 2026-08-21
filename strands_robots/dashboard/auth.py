"""WebAuthn (passkey) authentication for the dashboard.

Why this exists: the dashboard commands real hardware (SO-101 arms). Before
exposing it beyond localhost (robots.cagatay.my), every /api and /ws route
must require a session that only a registered passkey can mint.

The model: a passkey is a public/private keypair. The private key never
leaves the user's secure enclave (Touch ID / Face ID / phone). The dashboard
stores only the public key and verifies signed challenges, then issues a
short-lived HS256 JWT the middleware checks on every request.

Flow:
  1. FIRST RUN: no credentials -> /api/auth/status reports setup_required.
     The first enrollment seals the dashboard (optionally gated by a
     bootstrap token against the open-enrollment window).
  2. LOGIN: challenge -> passkey signature -> JWT session token.
  3. GUARD: TokenAuthMiddleware in server.py accepts the JWT (or the static
     security.auth_token) on /api/* and /ws/*.

Storage: one JSON file (STRANDS_DASH_AUTH_STORE, default
~/.strands_dashboard/auth.json, chmod 600). The store is re-read whenever
the file's mtime changes, so credential edits and re-enrollment need no
restart.

Env knobs (all optional):
  STRANDS_DASH_AUTH_ENABLED   OVERRIDE only. Unset (the default): auth is ON
                              exactly when a passkey exists in the store.
                              "true"/"false" force it either way. When auth is
                              off, require_session still PASSES only loopback
                              clients: disabled means local-only, never open.
  STRANDS_DASH_AUTH_STORE     store path (default ~/.strands_dashboard/auth.json)
  STRANDS_DASH_AUTH_RP_ID     force the relying-party id (e.g. robots.cagatay.my)
  STRANDS_DASH_AUTH_RP_NAME   display name (default "strands robots dashboard")
  STRANDS_DASH_AUTH_ORIGIN    force the expected WebAuthn origin
  STRANDS_DASH_AUTH_TOKEN_TTL session lifetime seconds (default 86400)
  STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN  one-time secret required to enroll the
                              FIRST passkey
"""

from __future__ import annotations

import logging
import ipaddress
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import jwt  # PyJWT
from fastapi import HTTPException

from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

_ENV = "STRANDS_DASH_AUTH_"


def auth_enabled() -> bool:
    """Whether passkey auth guards the API.

    The STORE is the source of truth: the moment a passkey is enrolled, auth
    is ON. Anything else is a trap - an operator who enrolls a credential and
    still needs to remember an env flag has a dashboard that LOOKS guarded
    (credentials listed in /api/auth/status) while every request rides the
    open posture.

    STRANDS_DASH_AUTH_ENABLED, when SET, is an explicit override in either
    direction (force-on before the first enrollment, or force-off for local
    debugging). Unset means: follow the store.
    """
    raw = os.getenv(_ENV + "ENABLED", "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes", "on")
    return has_credentials()


def _store_path() -> Path:
    default = Path.home() / ".strands_dashboard" / "auth.json"
    return Path(os.getenv(_ENV + "STORE", str(default))).expanduser().resolve()


def _rp_name() -> str:
    return os.getenv(_ENV + "RP_NAME", "strands robots dashboard")


def _token_ttl() -> int:
    try:
        return int(os.getenv(_ENV + "TOKEN_TTL", "86400"))
    except ValueError:
        return 86400


def _bootstrap_token() -> str:
    return os.getenv(_ENV + "BOOTSTRAP_TOKEN", "").strip()


def _forced_rp_id() -> str:
    return os.getenv(_ENV + "RP_ID", "").strip()


def _forced_origin() -> str:
    return os.getenv(_ENV + "ORIGIN", "").strip()


# --- store: one JSON file, thread-safe, hot-reloaded on mtime change -------

_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_cache_key: Optional[tuple] = None


def _default_store() -> Dict[str, Any]:
    return {
        "jwt_secret": secrets.token_urlsafe(48),
        "credentials": [],  # {id, public_key, sign_count, name, created}
        "created": time.time(),
    }


# Set when a store on disk could not be parsed: the backup path plus why. Read via
# store_corruption(); begin_registration consults it, because the credential-less window it
# opens must not be usable by a stranger who merely benefited from a disk error.
_corrupt: Optional[Dict[str, str]] = None


def store_corruption() -> Optional[Dict[str, str]]:
    """The unreadable store this process rescued, if any: {'backup': path, 'reason': str}."""
    return dict(_corrupt) if _corrupt else None


def _preserve_corrupt(path: Path, exc: Exception) -> None:
    """Move an unparseable store aside instead of clobbering it, and remember that we did.

    The bytes may still hold the operator's credential id and public key, so they are kept
    verbatim under a timestamped name. Failing to move it must not stop the dashboard from
    coming up, but it MUST still be recorded - the flag is what re-seals enrollment.
    """
    global _corrupt
    backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    try:
        os.replace(path, backup)
        where = str(backup)
    except OSError:
        where = ""
    _corrupt = {"backup": where, "reason": f"{type(exc).__name__}: {exc}"}
    logging.getLogger(__name__).warning(
        "dashboard auth store at %s is unreadable (%s); kept as %s. Enrollment is limited to "
        "this machine until a passkey exists again.",
        path, _corrupt["reason"], where or "<could not move it>",
    )


def _load() -> Dict[str, Any]:
    """Read the store, re-reading the file whenever it changes on disk."""
    global _cache, _cache_key
    path = _store_path()
    with _lock:
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        if key is not None and key == _cache_key:
            return _cache
        if key is not None:
            try:
                _cache = json.loads(path.read_text())
                _cache_key = key
                return _cache
            except (OSError, ValueError) as exc:
                # A half-written or unreadable store used to fall straight through to a fresh
                # default one -- which OVERWROTE the file. Two consequences, neither obvious from
                # here: auth_enabled() IS has_credentials(), so zero credentials silently
                # unseals every /api and /ws route (through the tunnel, that is the public
                # internet); and the operator's only passkey record was destroyed by the same
                # line, so even fixing the JSON by hand could not bring it back. One truncated
                # write - a crash mid-save, a full disk - was enough.
                _preserve_corrupt(path, exc)
        store = _default_store()
        _save_locked(store)
        return store


def _save_locked(store: Dict[str, Any]) -> None:
    global _cache, _cache_key
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _cache = store
    try:
        stat = path.stat()
        _cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        _cache_key = None


def _save(store: Dict[str, Any]) -> None:
    with _lock:
        _save_locked(store)


def _jwt_secret() -> str:
    return _load()["jwt_secret"]


def has_credentials() -> bool:
    return len(_load().get("credentials", [])) > 0


def list_credentials() -> List[Dict[str, Any]]:
    return [
        {"id": c["id"], "name": c.get("name", "passkey"), "created": c.get("created")}
        for c in _load().get("credentials", [])
    ]


def delete_credential(cred_id: str) -> Dict[str, Any]:
    """Revoke a passkey. Refuses to remove the LAST one (would re-open the
    dashboard to anyone via the setup flow)."""
    store = _load()
    creds = store.get("credentials", [])
    if not any(c["id"] == cred_id for c in creds):
        raise HTTPException(404, "credential not found")
    if len(creds) <= 1:
        raise HTTPException(409, "cannot remove the last passkey - enroll another first")
    store["credentials"] = [c for c in creds if c["id"] != cred_id]
    _save(store)
    return {"ok": True, "removed": cred_id, "remaining": len(store["credentials"])}


# --- relying-party id / origin derivation -----------------------------------

def _host_only(host: str) -> str:
    return host.split(":")[0]


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def rpid_is_usable(host_only: str) -> bool:
    """WebAuthn rpId must be a registrable domain or 'localhost'; a raw IP is
    rejected by browsers before the ceremony starts."""
    if host_only == "localhost":
        return True
    if not host_only or _is_ip(host_only):
        return False
    return True


def _headers(request_or_ws: Any) -> Any:
    return request_or_ws.headers


#: Hostnames that are always acceptable as a relying-party id: a browser on this
#: machine is the operator, and local dev must never depend on remote config.
_LOOPBACK_RP_IDS = frozenset({"localhost", "127.0.0.1", "::1"})


def known_rp_ids(store: Optional[dict] = None) -> set:
    """Every rp_id this deployment has PROVEN it uses.

    A credential is cryptographically bound to the rp_id it was created under, so
    once one is recorded here it is the only value a login can ever verify
    against. Credentials enrolled before this was recorded carry no rp_id: they
    contribute nothing (see rp_id_verdict's "legacy" case) and are healed on their
    first successful authentication, which proves the binding rather than guessing
    it.
    """
    s = store if store is not None else _load()
    return {c["rp_id"] for c in s.get("credentials", []) if c.get("rp_id")}


def rp_id_verdict(host_rp_id: str, forced: str = "", known: Optional[set] = None) -> tuple:
    """Decide the rp_id for a ceremony: ``(rp_id, reason)``, or ``(None, reason)``.

    ``_derive_rp_id`` used to return the client's ``Host`` header verbatim, and
    ``finish_registration`` then verified against that same claimed value -- a
    self-consistent check that enforced nothing. Anyone who could reach the port
    could run a ceremony bound to a domain THEY control; combined with anonymous
    first-enrollment that burned the owner's enrollment slot with a credential
    unusable from the real hostname.

    The store is the authority, in this order:

    * ``STRANDS_DASH_AUTH_RP_ID`` wins outright -- explicit operator config.
    * loopback is always allowed, so a browser on this machine and local dev keep
      working no matter what is recorded.
    * a host matching an rp_id already recorded on an enrolled credential is
      allowed: it is the value that can actually verify.
    * otherwise, if the store records NO rp_id at all (a fresh install, or a store
      written before rp_ids were recorded), fall back to the host. First
      enrollment has to be able to bind SOMETHING, and it is gated by the
      bootstrap token; the value it binds is then recorded, which closes this door
      behind it.
    * once any rp_id is known, a different host is REFUSED. It could only produce
      a credential nobody can use, or a login that cannot verify.
    """
    # Loopback outranks even the pin, and that ordering is deliberate: a browser
    # at http://localhost:8090 CANNOT use 'robots.cagatay.my' as an rp_id -- the
    # spec requires the rp_id to be a registrable suffix of the page's origin, so
    # honouring the pin here would make the browser refuse the ceremony before it
    # starts, and a passkey bound to the domain could not be used from localhost
    # anyway. Pinning describes the remote hostname; it must not confiscate the
    # local door.
    if host_rp_id in _LOOPBACK_RP_IDS:
        return (host_rp_id, "loopback")
    if forced:
        return (forced, "forced by STRANDS_DASH_AUTH_RP_ID")
    known = known_rp_ids() if known is None else known
    if host_rp_id in known:
        return (host_rp_id, "matches an enrolled credential")
    if not known:
        return (host_rp_id, "legacy: no rp_id recorded yet, binding on first use")
    return (None, f"host {host_rp_id!r} is not one of the enrolled {sorted(known)}")


def _derive_rp_id(request_or_ws: Any) -> str:
    host = _host_only(_headers(request_or_ws).get("host", "localhost"))
    rp_id, reason = rp_id_verdict(host, _forced_rp_id())
    if rp_id is None:
        logger.warning("refused WebAuthn ceremony: %s", reason)
        raise HTTPException(400, {
            "error": "this host cannot be used for a passkey ceremony",
            "detail": reason,
            "hint": "reach the dashboard on its enrolled hostname, or set "
                    "STRANDS_DASH_AUTH_RP_ID if it legitimately changed",
        })
    return rp_id


def _derive_origin(request_or_ws: Any) -> str:
    forced = _forced_origin()
    if forced:
        return forced
    headers = _headers(request_or_ws)
    origin = headers.get("origin")
    if origin:
        return origin.rstrip("/")
    host = headers.get("host", "localhost:8090")
    scheme = "https" if headers.get("x-forwarded-proto") == "https" else "http"
    return f"{scheme}://{host}"


def _rpid_error(rp_id: str) -> HTTPException:
    return HTTPException(
        400,
        f"WebAuthn cannot use '{rp_id}' as the relying-party id (needs a "
        "hostname or domain, not a raw IP). Open the dashboard via a hostname "
        "or set STRANDS_DASH_AUTH_RP_ID.",
    )


# --- challenge cache (short-lived, in-memory) --------------------------------

logger = logging.getLogger(__name__)

_challenges: Dict[str, Dict[str, Any]] = {}
_chal_lock = threading.Lock()
_CHAL_TTL = 300.0


#: Caps on the challenge table. Both are per-process and generous: a challenge
#: measures ~0.5KB, so 512 of them is ~256KB. The point is not memory -- the
#: measured growth was slow -- it is that an unauthenticated caller could hold an
#: unbounded number of entries on a public path.
_CHAL_MAX = int(os.getenv("STRANDS_DASH_AUTH_CHAL_MAX", "512"))
#: The property that actually matters: no single client may fill the table and
#: push out the operator's pending login. A cap alone does not give this -- with
#: only a global limit, a flood at ~470 req/s evicts a legitimate challenge within
#: a second of it being issued.
_CHAL_MAX_PER_IP = int(os.getenv("STRANDS_DASH_AUTH_CHAL_MAX_PER_IP", "16"))


def _evict_oldest(where: Dict[str, Dict[str, Any]], keep: int, ip: Optional[str] = None) -> int:
    """Drop the oldest entries (optionally only one ip's) until ``keep`` remain."""
    pool = [(v["t"], k) for k, v in where.items() if ip is None or v.get("ip") == ip]
    dropped = 0
    for _t, k in sorted(pool)[: max(0, len(pool) - keep)]:
        where.pop(k, None)
        dropped += 1
    return dropped


def _stash_challenge(
    kind: str, challenge: bytes, extra: Optional[dict] = None, ip: Optional[str] = None,
) -> str:
    cid = secrets.token_urlsafe(16)
    now = time.time()
    with _chal_lock:
        for k in [k for k, v in _challenges.items() if now - v["t"] > _CHAL_TTL]:
            _challenges.pop(k, None)
        # Evict the flooder's OWN oldest entries first, so one noisy client
        # cannot cost anybody else their in-flight ceremony.
        if ip:
            evicted = _evict_oldest(_challenges, _CHAL_MAX_PER_IP - 1, ip=ip)
            if evicted:
                logger.warning("challenge cap: dropped %d stale challenge(s) from %s", evicted, ip)
        if len(_challenges) >= _CHAL_MAX:
            _evict_oldest(_challenges, _CHAL_MAX - 1)
            logger.warning("challenge table full (%d); evicted oldest", _CHAL_MAX)
        _challenges[cid] = {
            "kind": kind, "challenge": challenge, "t": now, "extra": extra or {}, "ip": ip,
        }
    return cid


def _client_ip(request_or_ws: Any) -> Optional[str]:
    """Best-effort client identity for the per-ip cap only -- NEVER for trust.

    A forwarded header is attacker-settable, so it is used solely to spread the
    cap across claimed identities; fb5f2a0a is the rule for anything that grants
    access, and it stands: a forwarded request is never loopback.
    """
    try:
        h = _headers(request_or_ws)
        fwd = h.get("cf-connecting-ip") or h.get("x-forwarded-for") or h.get("x-real-ip")
        if fwd:
            return fwd.split(",")[0].strip()[:64] or None
        client = getattr(request_or_ws, "client", None)
        return getattr(client, "host", None)
    except Exception:
        return None


def _pop_challenge(cid: str, kind: str) -> Dict[str, Any]:
    with _chal_lock:
        rec = _challenges.pop(cid, None)
    if not rec or rec["kind"] != kind:
        raise HTTPException(400, "invalid or expired challenge")
    if time.time() - rec["t"] > _CHAL_TTL:
        raise HTTPException(400, "challenge expired")
    return rec


# --- JWT sessions ------------------------------------------------------------

def issue_token(
    subject: str, name: str = "", iat0: int | None = None, exp: int | None = None
) -> str:
    """A session token. `iat0` is the ORIGINAL sign-in, carried unchanged through every
    renewal so the absolute cap in renewal_verdict() cannot be reset by re-issuing."""
    now = int(time.time())
    payload = {
        "sub": subject,
        "name": name,
        "iat": now,
        "iat0": int(iat0) if iat0 else now,
        "exp": int(exp) if exp else now + _token_ttl(),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _session_max_age() -> int:
    """Absolute lifetime of a session, however often it is renewed (default 30 days)."""
    try:
        return int(os.getenv(_ENV + "SESSION_MAX_AGE", "2592000"))
    except ValueError:
        return 2592000


def renewal_verdict(
    claims: Mapping[str, Any] | None,
    now: float,
    ttl: int | None = None,
    max_age: int | None = None,
) -> Dict[str, Any]:
    """Should this session be handed a fresh token? (U21 — pure, so the clock is testable.)

    The dashboard's JWT lives 24h and had no renewal path at all, so a phone that
    signed in on Monday was locked out on Tuesday with a passkey ceremony as the
    only way back. Measured consequence: a 44h websocket reconnect storm,
    18,968 refusals, from the one device cagatay uses to check the lab remotely
    (Q109). Expiry is not an edge case here — it is a DAILY event.

    SLIDING, not longer: a valid token past its half-life earns a new one, so an
    active session never dies mid-use, while the window a STOLEN token stays
    useful in remains one TTL. A blunt 30-day `exp` would have widened exactly
    that window instead.

    The refusals are the security of it, and each is a way sessions go wrong:
    * an EXPIRED token is never renewed. "Almost valid" is how a session becomes
      unrevokable — revocation works by waiting a TTL out.
    * renewal cannot outlive `iat0` + max_age (default 30d), carried as its own
      claim so it survives every re-issue. Without that cap a session renewed by
      a background poller is immortal, and the authenticator is never asked again.
    * a token BEFORE its half-life is left alone: re-issuing on every request
      would rewrite the client's credential dozens of times a minute and make
      the absolute cap the only real limit.
    * no claims, or claims with no `exp`, earn nothing. A renewal decision needs
      evidence.

    Returns ``{"renew": bool, "reason": str, "exp": int|None, "iat0": int|None}``
    — `reason` is written to be shown to a person, because a session that
    silently refuses to renew is the bug this is fixing.
    """
    ttl = _token_ttl() if ttl is None else ttl
    max_age = _session_max_age() if max_age is None else max_age
    if not isinstance(claims, Mapping):
        return {"renew": False, "reason": "no session claims to renew", "exp": None, "iat0": None}
    try:
        exp = float(claims["exp"])
    except (KeyError, TypeError, ValueError):
        return {"renew": False, "reason": "session has no expiry to extend", "exp": None, "iat0": None}
    if exp <= now:
        return {"renew": False, "reason": "session already expired - sign in again", "exp": None, "iat0": None}
    # The original sign-in: `iat0` once a session has been renewed, `iat` the first time,
    # and `exp - ttl` for a token issued before this claim existed (a session already in
    # a phone's storage must not be treated as brand new, which would restart its cap).
    try:
        iat0 = float(claims.get("iat0") or claims.get("iat") or (exp - ttl))
    except (TypeError, ValueError):
        iat0 = exp - ttl
    hard_deadline = iat0 + max_age
    if now >= hard_deadline:
        return {
            "renew": False,
            "reason": "this session has reached its maximum age - sign in with your passkey again",
            "exp": None,
            "iat0": int(iat0),
        }
    if now < exp - ttl / 2:
        return {"renew": False, "reason": "session still fresh", "exp": int(exp), "iat0": int(iat0)}
    # Never past the cap, and never SHORTER than what the client already holds: a renewal
    # that shaved time off would be a downgrade the client cannot refuse.
    new_exp = int(min(now + ttl, hard_deadline))
    if new_exp <= exp:
        return {"renew": False, "reason": "renewal would not extend this session", "exp": int(exp), "iat0": int(iat0)}
    return {"renew": True, "reason": "past half-life, extended", "exp": new_exp, "iat0": int(iat0)}


def verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "session expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid session")


def renew_if_due(token: str, now: float | None = None) -> Optional[str]:
    """A fresh token if this one is past its half-life, else None (U21).

    None means "keep what you have" for every reason: still fresh, already expired,
    past the absolute cap, unreadable. The caller never has to distinguish, because
    the only action available to it is to pass a new token along or not — the
    REASONS matter to auth/status, which can afford to explain them.
    """
    if not token:
        return None
    try:
        claims = verify_token(token)
    except HTTPException:
        return None  # an expired or forged token is a login problem, not a renewal one
    verdict = renewal_verdict(claims, time.time() if now is None else now)
    if not verdict.get("renew"):
        return None
    return issue_token(
        str(claims.get("sub") or ""),
        str(claims.get("name") or ""),
        iat0=verdict.get("iat0"),
        exp=verdict.get("exp"),
    )


def session_is_valid(token: str) -> bool:
    """Non-raising check for the ASGI middleware."""
    if not token:
        return False
    try:
        verify_token(token)
        return True
    except HTTPException:
        return False


def client_is_loopback(client_host: Optional[str]) -> bool:
    """True when the connecting client is this machine. Used so that
    auth-disabled means LOCAL-ONLY rather than open to the network."""
    if not client_host:
        return False
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return client_host == "localhost"


# --- WebAuthn ceremonies ------------------------------------------------------

def begin_registration(request: Any, label: str = "passkey", bootstrap: str = "") -> Dict[str, Any]:
    """Start a passkey enrollment. The FIRST enrollment seals the dashboard;
    later ones require a valid session (enforced by the route)."""
    store = _load()
    first_time = len(store.get("credentials", [])) == 0
    required = _bootstrap_token()
    if first_time and required:
        if not secrets.compare_digest(bootstrap or "", required):
            raise HTTPException(403, "bootstrap token required for first enrollment")

    # A first enrollment that exists only because the store was unreadable is a DIFFERENT event
    # from a genuinely new dashboard: nobody chose it, and a stranger must not be able to seize
    # the dashboard on the strength of a disk error. The person at the machine still can, and a
    # bootstrap token still works from anywhere - this narrows the accident, it does not add a
    # dead end.
    damage = store_corruption()
    if first_time and damage and not required:
        if not client_is_loopback(_client_ip(request)):
            raise HTTPException(
                403,
                "the credential store was unreadable and has been kept as "
                f"{damage['backup'] or 'a backup'} ({damage['reason']}). Enrolling a new passkey "
                "is limited to the machine itself until one exists again - open the dashboard on "
                "that machine, or set STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN and pass it.",
            )

    rp_id = _derive_rp_id(request)
    if not rpid_is_usable(rp_id):
        raise _rpid_error(rp_id)

    user_id = store.get("user_id")
    if not user_id:
        user_id = bytes_to_base64url(secrets.token_bytes(16))
        store["user_id"] = user_id
        _save(store)

    exclude = [
        PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"]))
        for c in store.get("credentials", [])
    ]
    opts = generate_registration_options(
        rp_id=rp_id,
        rp_name=_rp_name(),
        user_id=base64url_to_bytes(user_id),
        user_name="dashboard-admin",
        user_display_name="Dashboard Admin",
        exclude_credentials=exclude or None,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
    )
    cid = _stash_challenge("reg", opts.challenge, {"label": label, "rp_id": rp_id}, ip=_client_ip(request))
    return {"challenge_id": cid, "options": json.loads(options_to_json(opts))}


def finish_registration(request: Any, challenge_id: str, credential: dict) -> Dict[str, Any]:
    rec = _pop_challenge(challenge_id, "reg")
    verification = verify_registration_response(
        credential=credential,
        expected_challenge=rec["challenge"],
        expected_rp_id=rec["extra"]["rp_id"],
        expected_origin=_derive_origin(request),
    )
    store = _load()
    cred_id = bytes_to_base64url(verification.credential_id)
    if any(c["id"] == cred_id for c in store.get("credentials", [])):
        raise HTTPException(409, "credential already registered")
    store.setdefault("credentials", []).append({
        "id": cred_id,
        "public_key": bytes_to_base64url(verification.credential_public_key),
        "sign_count": verification.sign_count,
        "name": rec["extra"].get("label", "passkey"),
        "created": time.time(),
        # The binding, recorded: from here on the Host header cannot introduce a
        # different rp_id (see rp_id_verdict).
        "rp_id": rec["extra"]["rp_id"],
    })
    _save(store)
    token = issue_token(cred_id, name=rec["extra"].get("label", "passkey"))
    return {"ok": True, "token": token, "credential_id": cred_id}


def begin_authentication(request: Any) -> Dict[str, Any]:
    store = _load()
    if not store.get("credentials"):
        raise HTTPException(400, "no credentials enrolled - setup required")
    rp_id = _derive_rp_id(request)
    if not rpid_is_usable(rp_id):
        raise _rpid_error(rp_id)
    allow = [
        PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"]))
        for c in store["credentials"]
    ]
    opts = generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    cid = _stash_challenge("auth", opts.challenge, {"rp_id": rp_id}, ip=_client_ip(request))
    return {"challenge_id": cid, "options": json.loads(options_to_json(opts))}


def finish_authentication(request: Any, challenge_id: str, credential: dict) -> Dict[str, Any]:
    rec = _pop_challenge(challenge_id, "auth")
    store = _load()
    cred_id = credential.get("id") or credential.get("rawId")
    match = next((c for c in store.get("credentials", []) if c["id"] == cred_id), None)
    if not match:
        raise HTTPException(404, "unknown credential")
    verification = verify_authentication_response(
        credential=credential,
        expected_challenge=rec["challenge"],
        expected_rp_id=rec["extra"]["rp_id"],
        expected_origin=_derive_origin(request),
        credential_public_key=base64url_to_bytes(match["public_key"]),
        credential_current_sign_count=match.get("sign_count", 0),
        require_user_verification=False,
    )
    match["sign_count"] = verification.new_sign_count
    # Self-heal the binding for credentials enrolled before rp_ids were recorded:
    # this authentication VERIFIED against rec["extra"]["rp_id"], which is proof,
    # not a guess. From here on rp_id_verdict refuses any other Host.
    if not match.get("rp_id") and rec["extra"].get("rp_id"):
        match["rp_id"] = rec["extra"]["rp_id"]
        logger.info("recorded rp_id %r for credential %s", match["rp_id"], match.get("name"))
    _save(store)
    token = issue_token(cred_id, name=match.get("name", "passkey"))
    return {"ok": True, "token": token, "credential_id": cred_id}


def status(request: Any = None) -> Dict[str, Any]:
    store = _load()
    out: Dict[str, Any] = {
        "enabled": auth_enabled(),
        "setup_required": len(store.get("credentials", [])) == 0,
        "credentials": list_credentials(),
        "bootstrap_required": bool(_bootstrap_token()) and len(store.get("credentials", [])) == 0,
    }
    if request is not None:
        try:
            host = _host_only(request.headers.get("host", ""))
            origin = _derive_origin(request)
            forced = _forced_rp_id()
            out["rp_id"] = forced or host
            out["secure_context"] = origin.startswith("https://") or host == "localhost"
            out["rpid_usable"] = True if forced else rpid_is_usable(host)
            if not out["secure_context"]:
                out["warning"] = ("This origin is not a secure context. WebAuthn needs "
                                  "HTTPS or http://localhost.")
            elif not out["rpid_usable"]:
                out["warning"] = (f"'{host}' cannot be a WebAuthn rpId - use a hostname "
                                  "or set STRANDS_DASH_AUTH_RP_ID.")
        except Exception:
            pass
    return out
