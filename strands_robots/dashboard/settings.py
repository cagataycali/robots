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


def _coerce(section: str, key: str, value: Any) -> Any:
    if (section, key) in _LIST_KEYS:
        return _as_list(value)
    if key in ("trust_remote_code",):
        return _as_bool(value)
    if key in ("temperature", "camera_hz"):
        try:
            return None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            return None
    if key in ("max_tokens", "port"):
        try:
            return None if value in (None, "") else int(value)
        except (TypeError, ValueError):
            return None
    if value is None:
        return None
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
            data = json.loads(SETTINGS_FILE.read_text())
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

    Only keys declared in :data:`_SCHEMA` are accepted - a typo'd key from the
    UI is dropped rather than silently persisted forever.
    """
    changed: list[str] = []
    with _lock:
        current = load()
        stored = _read_file()
        for section, values in (patch or {}).items():
            if section not in _SCHEMA or not isinstance(values, dict):
                continue
            for key, raw in values.items():
                if key not in _SCHEMA[section]:
                    continue
                value = _coerce(section, key, raw)
                if value == current[section].get(key):
                    continue
                stored.setdefault(section, {})[key] = value
                changed.append(f"{section}.{key}")
        if changed:
            _write_file(stored)
            global _cache
            _cache = None
            load(refresh=True)
    return changed


def _write_file(data: dict[str, Any]) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
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
