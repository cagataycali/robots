"""Configuration surface for the dashboard - the ``/api/config`` payload.

Handles the parts of configuration that :mod:`strands_robots.dashboard.settings`
deliberately does not: the ``.env`` file holding credentials, and assembling /
applying the combined config document the UI edits.

Secret hygiene (the rules exist because each one has a failure mode):

* keys matching :data:`SECRET_RX` are **masked** on read, never sent in full;
* a value that still *looks* masked is **skipped** on write - otherwise the UI
  helpfully overwrites a live API key with its own bullet characters, which is
  unrecoverable;
* the env file is written with an order-preserving upsert (comments and
  unrelated keys survive) and chmod 0600;
* applied values also land in ``os.environ`` so providers pick them up without
  a restart.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Mapping

from strands_robots.dashboard import settings

logger = logging.getLogger(__name__)

ENV_FILE = Path(os.getenv("DASHBOARD_ENV_FILE", ".env")).expanduser()

SECRET_RX = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|BEARER|API_?KEY)", re.I)

#: Characters our masks are made of. A submitted value containing one of these
#: is assumed to be an untouched mask and is not written back.
_MASK_MARKERS = ("•", "…")

#: Models offered as chips in the UI. Free text is always allowed too - this is
#: a convenience list, not an allowlist.
KNOWN_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

VOICE_PROVIDERS = ["openai", "nova_sonic", "gemini"]

#: Env vars worth surfacing even when absent from the .env file, so an operator
#: can discover what the dashboard actually reads.
INTERESTING_ENV = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "AWS_REGION",
    "AWS_PROFILE",
    "VOICE_MODEL",
    "STRANDS_MESH_LOCAL_DEV",
    "STRANDS_MESH_MULTICAST",
    "STRANDS_ROBOTS_VIDEO_ROOT",
    "STRANDS_ROBOTS_NO_DYLD_SHIM",
]


def is_secret(key: str) -> bool:
    return bool(SECRET_RX.search(key))


#: .env is read by every process the dashboard spawns, so an unrestricted
#: upsert is configuration -> code execution (PATH=/tmp/evil hijacks python/
#: ffmpeg for every child). Only variables the dashboard actually owns are
#: writable from the UI (Q13).
ALLOWED_ENV_PREFIXES: tuple[str, ...] = ("STRANDS_", "DASHBOARD_", "VOICE_", "HF_")
ALLOWED_ENV_KEYS = frozenset(INTERESTING_ENV)
ENV_VALUE_MAX_LEN = 4096


def env_key_allowed(key: str) -> bool:
    return key in ALLOWED_ENV_KEYS or key.startswith(ALLOWED_ENV_PREFIXES)


def env_entry_error(key: str, value: str) -> str | None:
    """Why this key/value pair must not reach the env file, or None if fine."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
        return f"invalid env key {key!r}"
    if not env_key_allowed(key):
        return (
            f"env key {key!r} is not dashboard-managed - allowed: "
            f"{', '.join(ALLOWED_ENV_PREFIXES)}* and {', '.join(sorted(ALLOWED_ENV_KEYS))}"
        )
    if any(ord(ch) < 0x20 for ch in value):
        # a newline in a VALUE writes a second variable on its own line,
        # defeating any key allow-list - so control chars are refused outright.
        return f"env value for {key} contains control characters"
    if len(value) > ENV_VALUE_MAX_LEN:
        return f"env value for {key} exceeds {ENV_VALUE_MAX_LEN} characters"
    return None


def mask(value: str) -> str:
    """``sk-abc...xyz`` -> ``sk-••••••yz``. Short values are fully hidden."""
    if not value:
        return ""
    if len(value) <= 6:
        return "•" * 6
    return f"{value[:3]}{'•' * 6}{value[-2:]}"


def looks_masked(value: Any) -> bool:
    return isinstance(value, str) and any(m in value for m in _MASK_MARKERS)


# ----------------------------------------------------------------------
# .env read / write
# ----------------------------------------------------------------------

def read_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        if not ENV_FILE.exists():
            return out
        for line in ENV_FILE.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                out[key] = value
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", ENV_FILE, exc)
    return out


def upsert_env_file(updates: dict[str, str]) -> list[str]:
    """Order-preserving upsert into the env file. Returns the keys written.

    Belt-and-braces: refuses disallowed keys and control characters itself,
    so no future caller can reintroduce Q13 by skipping apply()'s checks.
    """
    if not updates:
        return []
    for key, value in updates.items():
        problem = env_entry_error(str(key), str(value))
        if problem:
            raise ValueError(problem)
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()
    remaining = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.partition("=")[0].strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    for key, value in remaining.items():
        lines.append(f"{key}={value}")
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        pass
    return list(updates)


def bootstrap_env(
    from_file: Mapping[str, str], environ: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """(values to export, keys the process environment already decides).

    Q50: the Env tab wrote .env and exported into ``os.environ`` in the same breath, so a
    saved key worked... until the next start. NOTHING in this codebase ever read .env back:
    no load_dotenv, no `set -a; source .env` in restart_dashboard.sh. After a restart the tab
    still listed HF_TOKEN as set (it reads the FILE) while the process had never heard of it,
    so Hub downloads 401'd and voice providers reported a missing credential the settings
    screen swore was configured.

    A key already in the process environment WINS and is reported as shadowed instead: the
    operator who typed `HF_TOKEN=... ./restart_dashboard.sh` is making a deliberate statement
    about this run, and a file written weeks ago must not overrule it. An identical value is
    not a conflict and is not reported.
    """
    to_set: dict[str, str] = {}
    shadowed: list[str] = []
    for key, value in from_file.items():
        live = environ.get(key)
        if live is None:
            to_set[key] = value
        elif live != value:
            shadowed.append(key)
    return to_set, sorted(shadowed)


def load_env_file() -> tuple[list[str], list[str]]:
    """Apply .env to this process (once, at startup). Returns (exported, shadowed)."""
    to_set, shadowed = bootstrap_env(read_env_file(), os.environ)
    for key, value in to_set.items():
        os.environ[key] = value
    return sorted(to_set), shadowed


def env_view() -> list[dict[str, Any]]:
    """Masked env listing for the UI: .env contents + interesting live vars."""
    from_file = read_env_file()
    keys = list(from_file)
    for key in INTERESTING_ENV:
        if key not in keys:
            keys.append(key)
    rows: list[dict[str, Any]] = []
    for key in keys:
        live = os.environ.get(key, "")
        in_file = key in from_file
        # Q50: show what the PROCESS uses, not what the file says. Those differ only when
        # something in the launch environment overrides the file, and in that case the file's
        # value is the one nothing is acting on - displaying it made the screen a plausible
        # liar precisely when an operator was debugging a credential.
        raw = live if live else from_file.get(key, "")
        shadowed = bool(in_file and live and live != from_file.get(key))
        secret = is_secret(key)
        rows.append({
            "key": key,
            "value": mask(raw) if secret and raw else raw,
            "secret": secret,
            "set": bool(raw),
            "in_file": in_file,
            "shadowed": shadowed,
        })
    return rows


# ----------------------------------------------------------------------
# Combined document
# ----------------------------------------------------------------------

#: Keys ``mesh.security.validate_command()`` admits for execute/start, minus
#: the ones the dashboard sets itself (action/instruction/policy_provider/
#: duration). The validator builds its output from this allowlist and drops
#: every other key *silently*, so a form field outside this set would look
#: accepted and never reach the policy. Keep in sync with
#: ``validate_command``'s execute/start branch.
WIRE_CMD_KEYS: tuple[str, ...] = (
    "policy_host",
    "policy_port",
    "policy_type",
    "server_address",
    "model_path",
    "pretrained_name_or_path",
    "robot_name",
    "target_pose",
    "target_joints",
    "world_update",
    "control_frequency",
    "action_horizon",
    "fast_mode",
    "n_steps",
)

#: How each wire key should be rendered / parsed by the run form.
WIRE_KEY_TYPES: dict[str, str] = {
    "policy_host": "string",
    "policy_port": "int",
    "policy_type": "string",
    "server_address": "string",
    "model_path": "string",
    "pretrained_name_or_path": "string",
    "robot_name": "string",
    "target_pose": "json",
    "target_joints": "json",
    "world_update": "json",
    "control_frequency": "float",
    "action_horizon": "int",
    "fast_mode": "bool",
    "n_steps": "int",
}

#: Registry key -> the wire key that actually carries it. The registry names a
#: provider's constructor kwargs; the wire schema names its own fields, and the
#: two only partly overlap.
_WIRE_ALIASES = {
    "port": "policy_port",
    "host": "policy_host",
    "checkpoint": "model_path",
    "policy_path": "model_path",
    "repo_id": "pretrained_name_or_path",
    "server_address": "server_address",
}


def _policy_catalog() -> list[dict[str, Any]]:
    """Full provider objects from ``registry/policies.json``.

    The registry IS the run-form schema: ``requires`` are the mandatory
    inputs, ``config_keys`` the advanced set, ``defaults`` the prefill. Sending
    only ``{instruction, policy_provider}`` for a provider that requires a port
    or a checkpoint is a guaranteed failed run, so the UI builds
    its form from this instead of a hardcoded list.
    """
    try:
        from strands_robots.registry.policies import get_policy_provider, list_policy_providers
    except Exception as exc:  # noqa: BLE001
        logger.warning("policy registry unavailable: %s", exc)
        return []

    try:
        from strands_robots.mesh.security import is_safe_policy_provider
    except Exception:  # noqa: BLE001
        def is_safe_policy_provider(_name: str) -> bool:  # type: ignore[misc]
            return True

    out: list[dict[str, Any]] = []
    for name in list_policy_providers():
        spec = get_policy_provider(name) or {}
        requires = list(spec.get("requires") or [])
        config_keys = list(spec.get("config_keys") or [])
        # Split the provider's inputs by what the wire will actually carry, so
        # the form can render the deliverable fields and *say* that the rest
        # only work when the policy is built locally.
        wire_fields: list[dict[str, Any]] = []
        unsettable: list[str] = []
        for key in dict.fromkeys(requires + config_keys):
            wire_key = _WIRE_ALIASES.get(key, key)
            if wire_key in WIRE_CMD_KEYS:
                wire_fields.append({
                    "key": key,
                    "wire_key": wire_key,
                    "type": WIRE_KEY_TYPES.get(wire_key, "string"),
                    "required": key in requires,
                    "default": (spec.get("defaults") or {}).get(key),
                })
            else:
                unsettable.append(key)
        out.append({
            "name": name,
            "description": spec.get("description", ""),
            "requires": requires,
            "config_keys": config_keys,
            "defaults": dict(spec.get("defaults") or {}),
            "shorthands": list(spec.get("shorthands") or []),
            "url_patterns": list(spec.get("url_patterns") or []),
            "extra": spec.get("extra"),
            "trainable": bool(spec.get("trainer")),
            "wire_fields": wire_fields,
            "unsettable_over_mesh": unsettable,
            # False -> the mesh security gate rejects it; the card shows a lock
            # and points at STRANDS_MESH_POLICY_TYPE_ALLOW rather than letting
            # the operator discover it as a wire rejection.
            "wire_safe": bool(is_safe_policy_provider(name)),
            # Hardware peers cannot build checkpoint policies over the wire
            # (they only accept {port, host, data_config}).
            "server_based": bool({"port", "policy_port", "server_address", "host"}
                                & set(spec.get("requires") or [])),
        })
    return out


def snapshot(*, bridge: Any = None, agent_status: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``GET /api/config`` document."""
    tree = settings.load(refresh=True)
    agent = dict(tree["agent"])
    from strands_robots.dashboard.agent_bridge import DEFAULT_SYSTEM_PROMPT

    prompt = agent.get("system_prompt")
    return {
        "agent": {
            "model_id": agent.get("model_id"),
            "known_models": KNOWN_MODELS,
            "system_prompt": prompt or DEFAULT_SYSTEM_PROMPT,
            "is_default_prompt": not prompt,
            "temperature": agent.get("temperature"),
            "max_tokens": agent.get("max_tokens"),
            **(agent_status or {}),
        },
        "voice": {
            "provider": tree["voice"].get("provider") or "openai",
            "voice_name": tree["voice"].get("voice_name"),
            "providers": VOICE_PROVIDERS,
        },
        "mesh": bridge.mesh_info() if bridge is not None else {},
        "runtime": dict(tree["runtime"]),
        "security": {
            # Never echo the token back - only whether one is configured.
            "auth_enabled": bool(tree["security"].get("auth_token")),
            "cors_origins": tree["security"].get("cors_origins") or ["*"],
        },
        "policies": _policy_catalog(),
        "env": env_view(),
        "env_file": str(ENV_FILE),
        "settings_file": str(settings.SETTINGS_FILE),
    }


#: Settings keys that only take effect on a new mesh session. Everything else
#: is hot-applied, and the response says which is which per field rather than
#: making the operator guess.
_RESTART_KEYS = {"mesh.connect", "mesh.listen", "mesh.port", "mesh.backend", "mesh.camera_hz"}

#: Body fields of ``POST /api/config`` that are NOT settings sections: the
#: caller's own vocabulary, so they must never be reported as unknown settings.
_BODY_NON_SECTION_KEYS = frozenset(
    {"env", "reset_prompt", "reset_agent", "clear_history", "restart_mesh", "force"}
)

#: Changing these rebuilds the agent on the next turn.
_AGENT_KEYS = {"agent.model_id", "agent.system_prompt", "agent.temperature", "agent.max_tokens"}


def apply(body: dict[str, Any]) -> dict[str, Any]:
    """Apply a ``POST /api/config`` body.

    Returns ``{applied, restart_required, env_written, skipped_masked,
    agent_reset, errors}``.
    """
    body = body or {}
    errors: list[str] = []
    patch: dict[str, dict[str, Any]] = {}

    for section in ("agent", "voice", "mesh", "runtime", "security"):
        values = body.get(section)
        if isinstance(values, dict):
            patch[section] = dict(values)

    # "Reset to default prompt" is an explicit action, not an empty string -
    # an empty prompt field should not silently wipe a customised prompt.
    if body.get("reset_prompt"):
        patch.setdefault("agent", {})["system_prompt"] = None

    # Never let the UI persist the resolved default prompt as an override:
    # is_default_prompt would then be wrong forever.
    if isinstance(patch.get("agent"), dict):
        from strands_robots.dashboard.agent_bridge import DEFAULT_SYSTEM_PROMPT

        prompt = patch["agent"].get("system_prompt")
        if isinstance(prompt, str) and prompt.strip() == DEFAULT_SYSTEM_PROMPT.strip():
            patch["agent"]["system_prompt"] = None

    # Endpoint schemes: mtls refuses non-TLS endpoints loudly at session open
    # (mesh/session.py::_validate_endpoint_schemes). Catch it at the form.
    mesh_patch = patch.get("mesh") or {}
    if mesh_patch.get("connect") or mesh_patch.get("listen"):
        local_dev = os.getenv("STRANDS_MESH_LOCAL_DEV", "") not in ("", "0", "false")
        if not local_dev:
            bad = [
                ep for key in ("connect", "listen")
                for ep in settings.as_list(mesh_patch.get(key))
                if ep.split("/", 1)[0] in ("tcp", "udp", "ws")
            ]
            if bad:
                errors.append(
                    f"non-TLS endpoints {bad} are rejected when mesh auth is mtls - "
                    "use tls/... or quic/..., or run with --local-dev"
                )
                for key in ("connect", "listen"):
                    mesh_patch.pop(key, None)

    # Names the schema does not know are dropped without an error, so without
    # this the drawer says "nothing changed" for a patch that changed nothing
    # BECAUSE IT WAS NOT UNDERSTOOD. Computed off the BODY, not the patch: an
    # unknown SECTION never makes it into the patch at all. Non-section fields
    # of the body ("env" and the action flags) are the caller's own vocabulary
    # and are not settings names.
    ignored = settings.unknown_keys(
        {
            key: value
            for key, value in body.items()
            if isinstance(value, dict) and key not in _BODY_NON_SECTION_KEYS
        }
    )

    if patch:
        changed, coercion_errors = settings.update_strict(patch)
        errors.extend(coercion_errors)
    else:
        changed = []

    # --- env upsert -------------------------------------------------
    env_written: list[str] = []
    skipped_masked: list[str] = []
    raw_env = body.get("env")
    if isinstance(raw_env, dict):
        updates: dict[str, str] = {}
        for key, value in raw_env.items():
            key = str(key).strip()
            if looks_masked(value):
                skipped_masked.append(key)
                continue
            value_str = "" if value is None else str(value)
            problem = env_entry_error(key, value_str)
            if problem:
                errors.append(problem)
                continue
            updates[key] = value_str
        env_written = upsert_env_file(updates)
        for key, value in updates.items():
            os.environ[key] = value

    # --- hot apply --------------------------------------------------
    agent_reset = False
    if any(k in _AGENT_KEYS for k in changed) or body.get("reset_agent"):
        from strands_robots.dashboard.agent_bridge import reset_agent

        reset_agent(clear_history=bool(body.get("clear_history")))
        agent_reset = True

    restart_required = sorted(k for k in changed if k in _RESTART_KEYS)
    return {
        "ignored": ignored,
        "applied": sorted(k for k in changed if k not in _RESTART_KEYS),
        "restart_required": restart_required,
        "env_written": env_written,
        "skipped_masked": skipped_masked,
        "agent_reset": agent_reset,
        "errors": errors,
    }
