"""Payload validation for the strands-robots mesh.

This module owns the *payload-semantic* security boundary -- the rules
that depend on what is inside a mesh command, not who sent it. Wire-
level authentication (peer identity, replay protection, rate limiting,
fleet membership) is delegated entirely to Zenoh: mTLS at
``transport/link/tls``, role policy at ``access_control``, frequency
caps at ``downsampling``, byte caps at ``low_pass_filter``. See
:mod:`strands_robots.mesh._zenoh_config` and
:mod:`strands_robots.mesh._acl_config` for the transport-side
configuration.

What this module covers:

* :func:`validate_command` -- action allowlist plus per-action bounds
  (instruction length, duration, step count, ...).
* :func:`is_safe_policy_host` -- VLA inference target host / CIDR
  allowlist.
* :func:`is_safe_model_path` -- HuggingFace repo / local model path
  validation with optional ``<org>/<repo>`` prefix gating.
* :func:`is_safe_policy_type` / :func:`is_safe_policy_provider` --
  policy registry allowlist.
* :func:`is_safe_server_address` -- composite host[:port] check.

Everything here defends against an *authenticated* peer that has
already cleared mTLS + ACL but whose payload contents we still need
to bound. Without these checks an authorised operator could steer a
robot at an attacker-controlled inference server, request a 24-hour
``execute`` action, or drive the robot to download an arbitrary
HuggingFace model.

Configuration env vars
----------------------
``STRANDS_MESH_POLICY_HOST_ALLOW``
    Comma-separated host / CIDR list extending the loopback-only
    default ``policy_host`` allowlist.
``STRANDS_MESH_HF_REPO_ALLOW``
    Comma-separated HF org prefixes (or full ``<org>/<repo>`` prefixes)
    accepted in ``pretrained_name_or_path``. Defaults to
    ``nvidia,huggingface,lerobot``.
``STRANDS_MESH_POLICY_TYPE_ALLOW``
    Comma-separated extra policy_type / policy_provider values.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# --- Constants -----------------------------------------------------------

#: Maximum duration (seconds) accepted for ``execute`` / ``start`` commands.
MAX_DURATION_S: float = 3600.0

#: Maximum RPC timeout (seconds) accepted from peers.
MAX_TIMEOUT_S: float = 300.0

#: Maximum length (characters) of a natural-language ``instruction`` payload.
MAX_INSTRUCTION_LEN: int = 2000

#: Maximum length of a HuggingFace repo id / local model path. Real-world
#: HF ids are well under 200 chars; 512 leaves headroom for nested local
#: paths without becoming a DoS vector.
MAX_MODEL_PATH_LEN: int = 512

#: Allowed characters for HF repo ids and local model paths. We reject
#: ``..`` traversal, shell metacharacters, NUL bytes, whitespace, and any
#: byte outside the printable ASCII subset below.
_MODEL_PATH_RE = re.compile(r"^[A-Za-z0-9_./\-]+$")

#: Built-in policy_type allowlist. Mirrors the LeRobot policy registry
#: families plus the generic providers DevDuck ships with. Operators
#: extend via ``STRANDS_MESH_POLICY_TYPE_ALLOW`` (comma-separated).
_DEFAULT_POLICY_TYPES: frozenset[str] = frozenset(
    {
        "mock",
        "groot",
        "lerobot",
        "lerobot_local",
        "act",
        "diffusion",
        "tdmpc",
        "vqbet",
        "pi0",
        "pi0fast",
        "smolvla",
        "sac",
    }
)

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
        # ``resume`` clears the emergency-stop lockout; the only action
        # other than ``status`` permitted while the lockout is engaged.
        "resume",
    }
)

#: Default allowlist for VLA policy server targets (loopback only).
_DEFAULT_POLICY_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


# --- Exception hierarchy -------------------------------------------------


class SecurityError(Exception):
    """Base class for payload-validation rejections."""


class ValidationError(SecurityError):
    """Command payload failed schema or bounds checks."""


class LockoutError(SecurityError):
    """Command rejected because the local mesh is in emergency-stop lockout.

    Raised from :meth:`Mesh._dispatch` when an action other than
    ``status`` or ``resume`` arrives while ``_estop_lockout`` is engaged.
    The wire response is intentionally generic -- the exception type
    carries the real semantics so the dispatch wrapper can audit the
    rejection symmetrically with :class:`ValidationError`.
    """


# --- Policy-host allowlist -----------------------------------------------


def _policy_host_allowlist() -> list[str]:
    """Return the configured policy-host allowlist (defaults + env extras)."""
    raw = os.getenv("STRANDS_MESH_POLICY_HOST_ALLOW", "")
    extra = [host.strip() for host in raw.split(",") if host.strip()]
    return list(_DEFAULT_POLICY_HOSTS) + extra


def is_safe_policy_host(host: str) -> bool:
    """Return True when *host* is permitted as a VLA policy server target.

    The default allowlist is loopback only (``localhost``, ``127.0.0.1``,
    ``::1``). Operators extend it via ``STRANDS_MESH_POLICY_HOST_ALLOW``,
    a comma-separated list of hostnames or CIDR ranges
    (``"vla.internal,10.0.0.0/24"``).

    Hostnames are matched literally (case-insensitive); IP literals are
    additionally matched against any CIDR entries in the operator list.

    .. warning::
       Hostname entries are matched LITERALLY against the caller's input string;
       no DNS resolution is performed at allowlist time. Adding ``vla.internal``
       to the allowlist therefore implicitly trusts whatever resolver the
       inference call uses at runtime. Deployments on a hostile or weak DNS path
       should prefer IP literals or CIDR ranges (``10.0.0.0/24``) over hostnames
       so the trust boundary stays under operator control.
    """
    if not isinstance(host, str) or not host:
        return False
    host_lc = host.strip().lower()
    allowlist = _policy_host_allowlist()

    for entry in allowlist:
        if host_lc == entry.strip().lower():
            return True

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


# --- HuggingFace repo / local model path / policy_type allowlists -------


def _hf_repo_allowlist() -> list[str]:
    """Return operator-extensible HF repo prefix allowlist.

    Defaults to ``["nvidia", "huggingface", "lerobot"]`` covering GR00T
    and LeRobot models. Operators extend via
    ``STRANDS_MESH_HF_REPO_ALLOW`` (comma-separated ``<org>`` or
    ``<org>/<repo>`` prefixes).
    """
    raw = os.getenv("STRANDS_MESH_HF_REPO_ALLOW", "")
    extra = [pfx.strip().strip("/").lower() for pfx in raw.split(",") if pfx.strip()]
    builtin = ["nvidia", "huggingface", "lerobot"]
    return builtin + extra


def is_safe_model_path(path: str, *, hf_only: bool = False) -> bool:
    """Return True when *path* is a permitted HF repo id or local path.

    Checks performed:

    * Type and length: ``str``, non-empty, ``<= MAX_MODEL_PATH_LEN``.
    * Charset: ``[A-Za-z0-9_./-]+`` only (rejects shell metacharacters,
      whitespace, NUL bytes, non-ASCII).
    * No path traversal: rejects any segment equal to ``..``.
    * If *hf_only* is True (recommended for cross-mesh kwargs): the path
      MUST resemble ``<org>/<repo>`` and the org prefix MUST be in
      :func:`_hf_repo_allowlist`. This prevents an authenticated peer
      from steering a robot at an attacker-controlled HF repo.
    """
    if not isinstance(path, str) or not path:
        return False
    if len(path) > MAX_MODEL_PATH_LEN:
        return False
    if not _MODEL_PATH_RE.fullmatch(path):
        return False
    parts = path.replace("\\", "/").split("/")
    if any(seg == ".." for seg in parts):
        return False

    if hf_only:
        if path.startswith("/") or "/" not in path:
            return False
        org = parts[0].lower()
        allow = _hf_repo_allowlist()
        for entry in allow:
            entry_low = entry.lower()
            if "/" in entry_low:
                if path.lower().startswith(entry_low + "/") or path.lower() == entry_low:
                    return True
            elif org == entry_low:
                return True
        return False

    return True


def _policy_type_allowlist() -> frozenset[str]:
    """Return the configured policy_type allowlist (defaults + env extras)."""
    raw = os.getenv("STRANDS_MESH_POLICY_TYPE_ALLOW", "")
    extra = {pt.strip().lower() for pt in raw.split(",") if pt.strip()}
    return frozenset(_DEFAULT_POLICY_TYPES | extra)


def is_safe_policy_type(policy_type: str) -> bool:
    """Return True iff *policy_type* is in the allowlist."""
    if not isinstance(policy_type, str) or not policy_type:
        return False
    return policy_type.strip().lower() in _policy_type_allowlist()


def is_safe_policy_provider(provider: str) -> bool:
    """Return True iff *provider* is in the allowlist.

    ``policy_provider`` is the registry key the dispatcher passes to
    ``r._execute_task_sync`` / ``r.start_task`` to choose the policy
    class. Without this gate an authenticated peer could steer a robot
    to any registered provider, bypassing the spirit of the other
    allowlists. Shares the allowlist with :func:`is_safe_policy_type`.
    """
    if not isinstance(provider, str) or not provider:
        return False
    return provider.strip().lower() in _policy_type_allowlist()


def is_safe_server_address(addr: str) -> bool:
    """Validate a remote policy ``server_address`` (host[:port] or URL).

    Strips any scheme + port; the host portion is then checked against
    :func:`is_safe_policy_host`. Reuses the operator-controlled
    ``STRANDS_MESH_POLICY_HOST_ALLOW`` rather than introducing a
    parallel one.
    """
    if not isinstance(addr, str) or not addr:
        return False
    if len(addr) > MAX_MODEL_PATH_LEN:
        return False
    s = addr.strip()

    # 1. Strip optional scheme prefix (single ://)
    if "://" in s:
        s = s.split("://", 1)[1]

    # 2. Strip optional path (everything from first /)
    s = s.split("/", 1)[0]

    # 3. Detect bracketed IPv6: [host]:port or [host]
    if s.startswith("["):
        if "]" not in s:
            return False  # Malformed bracketed address
        bracket_end = s.index("]")
        host = s[1:bracket_end]  # Extract host without brackets
        remainder = s[bracket_end + 1 :]  # Everything after ]

        if remainder:
            # Must be :port
            if not remainder.startswith(":"):
                return False  # Malformed
            port_str = remainder[1:]
            if not port_str:
                return False  # Empty port
            # Validate port is digits in [1, 65535]
            if not port_str.isdigit():
                return False
            port = int(port_str)
            if port < 1 or port > 65535:
                return False

        return is_safe_policy_host(host)

    # 4. For unbracketed: count colons
    colon_count = s.count(":")

    if colon_count == 0:
        # No colons: treat whole string as host
        return is_safe_policy_host(s)

    elif colon_count == 1:
        # One colon: treat as host:port
        host, port_str = s.rsplit(":", 1)
        # Validate port is digits in [1, 65535]
        if not port_str.isdigit():
            return False
        port = int(port_str)
        if port < 1 or port > 65535:
            return False
        return is_safe_policy_host(host)

    else:
        # Two or more colons: MUST be an unbracketed IPv6 literal
        # Try to parse as IPv6 address
        try:
            ipaddress.ip_address(s)
            # Valid IPv6, treat whole string as host
            return is_safe_policy_host(s)
        except ValueError:
            # Not a valid IPv6 address
            return False


# --- Command schema and bounds -------------------------------------------


def _coerce_float(name: str, value: Any, *, lo: float, hi: float, default: float | None) -> float:
    """Coerce *value* to a float in ``[lo, hi]`` or raise ValidationError."""
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
    """Coerce *value* to an int in ``[lo, hi]`` or raise ValidationError."""
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
    """Validate a mesh command and return a sanitised copy.

    Performed checks:

    * ``action`` must be a string and a member of :data:`ALLOWED_ACTIONS`.
    * ``execute`` and ``start`` actions require:
        - ``instruction``: non-empty str up to :data:`MAX_INSTRUCTION_LEN`.
        - ``policy_host``: in the allowlist (defaults to ``"localhost"``).
        - ``duration``: ``[0, MAX_DURATION_S]``, defaults to 30.
        - ``policy_port`` (optional): integer in ``[1, 65535]``.
        - ``pretrained_name_or_path`` (optional): HF repo, allowlist-gated.
        - ``model_path`` (optional): HF id or local path, no traversal.
        - ``policy_type`` (optional): in :func:`is_safe_policy_type`.
        - ``policy_provider`` (optional): in :func:`is_safe_policy_provider`,
          defaults to ``"mock"``.
        - ``server_address`` (optional): in :func:`is_safe_server_address`.
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
    out["action"] = action

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

        if "pretrained_name_or_path" in cmd:
            value = cmd["pretrained_name_or_path"]
            if not isinstance(value, str) or not is_safe_model_path(value, hf_only=True):
                raise ValidationError(
                    f"pretrained_name_or_path={value!r} not in allowlist. Set "
                    "STRANDS_MESH_HF_REPO_ALLOW to add an org/repo prefix."
                )
            out["pretrained_name_or_path"] = value

        if "model_path" in cmd:
            value = cmd["model_path"]
            if not isinstance(value, str) or not is_safe_model_path(value, hf_only=False):
                raise ValidationError(
                    f"model_path={value!r} contains disallowed characters or path-traversal segments."
                )
            out["model_path"] = value

        if "policy_type" in cmd:
            value = cmd["policy_type"]
            if not isinstance(value, str) or not is_safe_policy_type(value):
                raise ValidationError(
                    f"policy_type={value!r} not in allowlist. Set STRANDS_MESH_POLICY_TYPE_ALLOW to extend."
                )
            out["policy_type"] = value.strip().lower()

        if "policy_provider" in cmd:
            value = cmd["policy_provider"]
            if not isinstance(value, str) or not is_safe_policy_provider(value):
                raise ValidationError(
                    f"policy_provider={value!r} not in allowlist. "
                    "Set STRANDS_MESH_POLICY_TYPE_ALLOW to extend "
                    "(provider and policy_type share one allowlist)."
                )
            out["policy_provider"] = value.strip().lower()
        else:
            raise ValidationError(
                "policy_provider is required for execute/start actions; "
                "set it explicitly (e.g. 'mock' for the noop policy). "
                "Silent defaults are not honoured on the security boundary."
            )

        if "server_address" in cmd:
            value = cmd["server_address"]
            if not isinstance(value, str) or not is_safe_server_address(value):
                raise ValidationError(
                    f"server_address={value!r} host not in allowlist. Set STRANDS_MESH_POLICY_HOST_ALLOW to extend."
                )
            out["server_address"] = value

    elif action == "step":
        out["steps"] = _coerce_int("steps", cmd.get("steps", 1), lo=1, hi=10_000, default=1)

    elif action == "teleop_receive":
        source = cmd.get("source_peer_id", "")
        if not isinstance(source, str) or not source:
            raise ValidationError("teleop_receive requires non-empty source_peer_id")

    return out


__all__ = [
    "ALLOWED_ACTIONS",
    "LockoutError",
    "MAX_DURATION_S",
    "MAX_INSTRUCTION_LEN",
    "MAX_MODEL_PATH_LEN",
    "MAX_TIMEOUT_S",
    "SecurityError",
    "ValidationError",
    "is_safe_model_path",
    "is_safe_policy_host",
    "is_safe_policy_provider",
    "is_safe_policy_type",
    "is_safe_server_address",
    "validate_command",
]
