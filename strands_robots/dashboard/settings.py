"""Persistent dashboard settings - the store behind ``/api/config``.

Two stores, deliberately separate:

* **this file** (``settings.json``) holds operator *preferences*: agent model /
  prompt / sampling, voice provider, mesh endpoints, runtime toggles. Plain
  JSON, safe to read, safe to commit to a bug report.
* :mod:`strands_robots.dashboard.config_api` handles the ``.env`` file, which
  holds *credentials*. Those are masked on read and chmod 0600 on write.

Every setting resolves in the same order::

    settings.json  ->  environment variable  ->  built-in default

so an operator who exports ``DASHBOARD_SYSTEM_PROMPT`` still gets what they
expect, and anything set in the UI wins over the env for the next agent build.
Mesh settings additionally get pushed *into* ``os.environ`` before the first
``get_session()`` call (:func:`apply_mesh_env`) because that is the only knob
upstream reads - see ``mesh/session.py::_build_config``.
"""

from __future__ import annotations

import copy
import json
import math
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path(
    os.getenv(
        "DASHBOARD_SETTINGS_FILE",
        os.path.join(Path.home(), ".strands_robots", "dashboard", "settings.json"),
    )
).expanduser()

#: Section -> key -> (env var fallback, built-in default). ``None`` for the env
#: var means "no env fallback".
_SCHEMA: dict[str, dict[str, tuple[str | None, Any]]] = {
    "agent": {
        "model_id": ("DASHBOARD_MODEL_ID", None),
        "system_prompt": ("DASHBOARD_SYSTEM_PROMPT", None),  # None -> DEFAULT_SYSTEM_PROMPT
        "temperature": (None, None),
        "max_tokens": (None, None),
    },
    "voice": {
        "provider": ("VOICE_PROVIDER", "openai"),
        "voice_name": ("VOICE_NAME", None),
    },
    "mesh": {
        # Mesh endpoints. Empty list/None means "leave the env alone".
        "connect": ("ZENOH_CONNECT", []),
        "listen": ("ZENOH_LISTEN", []),
        "port": ("STRANDS_MESH_PORT", None),
        "backend": ("STRANDS_MESH_BACKEND", None),
        "camera_hz": ("STRANDS_MESH_CAMERA_HZ", None),
        "policy_type_allow": ("STRANDS_MESH_POLICY_TYPE_ALLOW", []),
    },
    "runtime": {
        "trust_remote_code": ("STRANDS_TRUST_REMOTE_CODE", False),
    },
    "security": {
        # When set, every /api and /ws request must present this token
        # (Authorization: Bearer <t>, X-Dashboard-Token, or ?token=).
        "auth_token": ("DASHBOARD_AUTH_TOKEN", None),
        # Comma-separated origins allowed to make BROWSER cross-origin calls.
        # Default: none - same-origin only. An API that moves motors must not
        # answer a drive-by fetch from whatever tab the operator has open;
        # LAN-dev against a separate frontend port opts in explicitly, e.g.
        # DASHBOARD_CORS_ORIGINS=http://localhost:4319.
        "cors_origins": ("DASHBOARD_CORS_ORIGINS", []),
    },
}

_LIST_KEYS = {
    ("mesh", "connect"),
    ("mesh", "listen"),
    ("mesh", "policy_type_allow"),
    ("security", "cors_origins"),
}

_lock = threading.RLock()
_cache: dict[str, dict[str, Any]] | None = None


# ----------------------------------------------------------------------
# Coercion
# ----------------------------------------------------------------------

def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(p).strip() for p in value if str(p).strip()]
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


#: Public alias - other dashboard modules parse comma-separated endpoint
#: strings with the same rules the settings store uses.
def as_list(value: Any) -> list[str]:
    return _as_list(value)


class CoercionError(ValueError):
    """A settings value that must be REPORTED, not silently defaulted.

    Raised only on the strict path (UI/API writes). The lenient path (env
    vars / built-in defaults at import time) keeps the old degrade-to-None
    behavior, because a typo'd env var must not kill startup.
    """


def _finite_float(key: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise CoercionError(f"{key}: {value!r} is not a number")
    if not math.isfinite(out):
        # json.dumps would emit bare NaN/Infinity - not JSON (RFC 8259); one
        # such write bricks the config screen for every browser forever.
        raise CoercionError(f"{key}: {value!r} is not a finite number")
    return out


def _coerce(section: str, key: str, value: Any, strict: bool = False) -> Any:
    try:
        return _coerce_strict(section, key, value)
    except CoercionError:
        if strict:
            raise
        # Lenient degrade (env/CLI/file paths) must still degrade to the key's
        # own SHAPE: a list key that fell back to a scalar poisons every
        # comma-split consumer, which is worse than the empty default.
        if (section, key) in _LIST_KEYS:
            return []
        return None if key in ("temperature", "camera_hz", "max_tokens", "port") else value


#: What "true" and "false" may be spelled like. Anything else is a typo the
#: strict path reports rather than resolving to False (Q15: _as_bool("banana")
#: silently disabled a setting the operator believed they turned on).
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off", "")


def _coerce_strict(section: str, key: str, value: Any) -> Any:
    if (section, key) in _LIST_KEYS:
        # Q15: cors_origins=5 became [] - silently REPLACING a security posture
        # with a different one. A list key takes a list or a comma-separated
        # string; any other type is the client's bug, reported as such.
        if value is not None and not isinstance(value, (str, list, tuple)):
            raise CoercionError(
                f"{key}: expected a list or comma-separated string, got {type(value).__name__}"
            )
        return _as_list(value)
    if key in ("trust_remote_code",):
        if not isinstance(value, bool):
            spelled = str(value).strip().lower()
            if spelled not in _TRUTHY and spelled not in _FALSY:
                raise CoercionError(
                    f"{key}: {value!r} is not a boolean (use true/false)"
                )
        return _as_bool(value)
    if key == "temperature":
        if value in (None, ""):
            return None
        out = _finite_float(key, value)
        if not 0.0 <= out <= 2.0:
            raise CoercionError(f"temperature: {out} is outside 0..2")
        return out
    if key == "camera_hz":
        if value in (None, ""):
            return None
        out = _finite_float(key, value)
        if not 0.0 < out <= 240.0:
            # a publisher sleeps 1/hz between frames: 0 divides by zero,
            # negative sleeps never, huge busy-loops the camera thread.
            raise CoercionError(f"camera_hz: {out} is outside (0, 240]")
        return out
    if key in ("max_tokens", "port"):
        if value in (None, ""):
            return None
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise CoercionError(f"{key}: {value!r} is not an integer")
        if key == "port" and not 1 <= out <= 65535:
            raise CoercionError(f"port: {out} is outside 1..65535")
        if key == "max_tokens" and out < 1:
            raise CoercionError(f"max_tokens: {out} must be at least 1")
        return out
    if value is None:
        return None
    # The string fallback (model_id, auth_token, backend, ...) must not repr a
    # structure into a value: str({'a': 1}) as auth_token locks the operator
    # out of the UI that set it (Q15). Numbers are fine - ids are sometimes
    # typed unquoted - but containers are always a client bug.
    if isinstance(value, (dict, list, tuple, set)):
        raise CoercionError(
            f"{key}: expected a string, got {type(value).__name__}"
        )
    return str(value)


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------

def _defaults() -> dict[str, dict[str, Any]]:
    """Built-in defaults resolved through the environment."""
    out: dict[str, dict[str, Any]] = {}
    for section, keys in _SCHEMA.items():
        out[section] = {}
        for key, (env_name, default) in keys.items():
            raw = os.getenv(env_name) if env_name else None
            out[section][key] = _coerce(section, key, raw if raw not in (None, "") else default)
    return out


def _read_file() -> dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            # parse_constant fires only for NaN/Infinity, which are not JSON;
            # a file poisoned by an old write must count as corrupt (browsers
            # already refuse it), so it heals to defaults instead of being
            # handed back to JSON.parse forever.
            data = json.loads(
                SETTINGS_FILE.read_text(),
                parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite {c}")),
            )
            if isinstance(data, dict):
                return data
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not kill startup
        logger.warning("could not read %s: %s (using defaults)", SETTINGS_FILE, exc)
    return {}


def load(refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Full settings tree: file values layered over env/defaults."""
    global _cache
    with _lock:
        if _cache is not None and not refresh:
            return copy.deepcopy(_cache)
        merged = _defaults()
        stored = _read_file()
        for section, values in stored.items():
            if section not in merged or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key in _SCHEMA[section]:
                    merged[section][key] = _coerce(section, key, value)
        _cache = merged
        return copy.deepcopy(merged)


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    tree = load()
    if section not in tree:
        return default
    if key is None:
        return tree[section]
    value = tree[section].get(key)
    return default if value in (None, "", []) else value


def update(patch: dict[str, Any]) -> list[str]:
    """Merge *patch* into the settings file. Returns the changed dotted keys.

    Lenient (bad values degrade to None) - kept for the CLI path where flags
    were already validated by argparse. UI/API writes go through
    :func:`update_strict` so a bad value is an ERROR the user sees, not a
    silent default (Q14/Q15).
    """
    changed, _ = _update(patch, strict=False)
    return changed


def unknown_keys(patch: dict[str, Any]) -> list[str]:
    """Dotted names in ``patch`` that this schema does not know.

    ``_update`` ``continue``s past an unknown section and an unknown key, so such
    a name is neither stored NOR reported: it lands in no ``changed`` list and
    raises no coercion error. The settings drawer builds its status line out of
    those two lists, so a whole patch of names the backend does not recognise -
    a frontend field renamed on one side only, a typo in an env-ish key, a
    section that moved - produced the same reassuring "nothing changed" as
    re-saving values that were already correct.

    The two cases are not the same. This one names them so they can be said out
    loud, without changing what gets stored: reporting is not enforcement, and a
    caller sending an extra field must never be refused because of it.
    """
    out: list[str] = []
    for section, values in (patch or {}).items():
        if section not in _SCHEMA:
            out.append(f"{section}.*" if isinstance(values, dict) else str(section))
            continue
        if not isinstance(values, dict):
            continue
        for key in values:
            if key not in _SCHEMA[section]:
                out.append(f"{section}.{key}")
    return sorted(out)


def update_strict(patch: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Like :func:`update`, but invalid VALUES are reported, never stored.

    Returns ``(changed, errors)`` where each error names the dotted key and
    the reason. Valid keys in the same patch still apply. An unrecognised
    section or key is a different thing - a name, not a value - and is skipped
    here without an error; :func:`unknown_keys` is what names those.
    """
    return _update(patch, strict=True)


def _update(patch: dict[str, Any], strict: bool) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    errors: list[str] = []
    with _lock:
        current = load()
        stored = _read_file()
        for section, values in (patch or {}).items():
            if section not in _SCHEMA or not isinstance(values, dict):
                continue
            for key, raw in values.items():
                if key not in _SCHEMA[section]:
                    continue
                try:
                    value = _coerce(section, key, raw, strict=strict)
                except CoercionError as exc:
                    errors.append(f"{section}.{exc}")
                    continue
                if value == current[section].get(key):
                    continue
                stored.setdefault(section, {})[key] = value
                changed.append(f"{section}.{key}")
        if changed:
            _write_file(stored)
            global _cache
            _cache = None
            load(refresh=True)
    return changed, errors


def _write_file(data: dict[str, Any]) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(SETTINGS_FILE)
    # settings.json holds no secrets, but it does hold the auth token - so
    # keep it owner-only like the .env file.
    try:
        os.chmod(SETTINGS_FILE, 0o600)
    except OSError:
        pass


# ----------------------------------------------------------------------
# Mesh env application
# ----------------------------------------------------------------------

#: Settings key -> env var read by ``mesh/session.py`` / the transport factory.
MESH_ENV = {
    "connect": "ZENOH_CONNECT",
    "listen": "ZENOH_LISTEN",
    "port": "STRANDS_MESH_PORT",
    "backend": "STRANDS_MESH_BACKEND",
    "camera_hz": "STRANDS_MESH_CAMERA_HZ",
    "policy_type_allow": "STRANDS_MESH_POLICY_TYPE_ALLOW",
}


def apply_mesh_env() -> dict[str, str]:
    """Push mesh settings into ``os.environ``.

    Upstream reads these at ``get_session()`` time only (``_build_config``
    inserts ``connect/endpoints`` / ``listen/endpoints`` from ZENOH_CONNECT /
    ZENOH_LISTEN, and ``STRANDS_MESH_PORT`` is read at session-open), so this
    must run *before* ``MeshBridge.start()`` - and again before any re-open.
    """
    mesh = load()["mesh"]
    applied: dict[str, str] = {}
    for key, env_name in MESH_ENV.items():
        value = mesh.get(key)
        if isinstance(value, list):
            value = ",".join(value)
        if value in (None, ""):
            continue
        os.environ[env_name] = str(value)
        applied[env_name] = str(value)
    if applied:
        logger.info("mesh env from settings: %s", applied)
    also = {
        "runtime.trust_remote_code": ("STRANDS_TRUST_REMOTE_CODE", load()["runtime"]["trust_remote_code"]),
    }
    for _, (env_name, value) in also.items():
        if value:
            os.environ[env_name] = "1"
            applied[env_name] = "1"
    return applied
