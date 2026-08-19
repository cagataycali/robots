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

import ipaddress
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _bool_env(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


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
            except (OSError, ValueError):
                pass
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


def _derive_rp_id(request_or_ws: Any) -> str:
    forced = _forced_rp_id()
    if forced:
        return forced
    return _host_only(_headers(request_or_ws).get("host", "localhost"))


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

_challenges: Dict[str, Dict[str, Any]] = {}
_chal_lock = threading.Lock()
_CHAL_TTL = 300.0


def _stash_challenge(kind: str, challenge: bytes, extra: Optional[dict] = None) -> str:
    cid = secrets.token_urlsafe(16)
    now = time.time()
    with _chal_lock:
        for k in [k for k, v in _challenges.items() if now - v["t"] > _CHAL_TTL]:
            _challenges.pop(k, None)
        _challenges[cid] = {"kind": kind, "challenge": challenge, "t": now, "extra": extra or {}}
    return cid


def _pop_challenge(cid: str, kind: str) -> Dict[str, Any]:
    with _chal_lock:
        rec = _challenges.pop(cid, None)
    if not rec or rec["kind"] != kind:
        raise HTTPException(400, "invalid or expired challenge")
    if time.time() - rec["t"] > _CHAL_TTL:
        raise HTTPException(400, "challenge expired")
    return rec


# --- JWT sessions ------------------------------------------------------------

def issue_token(subject: str, name: str = "") -> str:
    now = int(time.time())
    payload = {"sub": subject, "name": name, "iat": now, "exp": now + _token_ttl()}
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "session expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid session")


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
    cid = _stash_challenge("reg", opts.challenge, {"label": label, "rp_id": rp_id})
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
    cid = _stash_challenge("auth", opts.challenge, {"rp_id": rp_id})
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
