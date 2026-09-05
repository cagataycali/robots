"""Caller-authorization helpers for Device Connect robot/sim drivers.

Security hardening: Device Connect RPC handlers run on the device side with no
built-in per-call authorization. State-mutating RPCs (execute / stop / step /
reset) and lifecycle events (emergencyStop) must therefore verify the calling
device against an operator-controlled allowlist before acting on physical (or
simulated) hardware.

Allowlists are sourced from environment variables so deployments opt in without
code changes:

* ``DEVICE_CONNECT_RPC_ALLOW`` - comma-separated device ids permitted to call
  state-mutating RPCs. ``*`` (or unset) means "allow all" but logs a warning so
  the permissive posture is visible. An empty value is treated as unset, and a
  value is empty when it holds no non-blank entry after stripping - so ``""``,
  ``" "`` and ``","`` are all unset.
* ``DEVICE_CONNECT_ESTOP_ALLOW`` - comma-separated device ids permitted to
  trigger emergency-stop handling. Falls back to ``DEVICE_CONNECT_RPC_ALLOW``
  when unset.

Matching supports trailing ``*`` glob prefixes (e.g. ``safety-*``).

Caller-identity semantics (READ THIS before relying on the allowlist):

* The caller id is whatever the messaging layer reported as the RPC's
  ``source_device``. A device-to-device caller (another ``DeviceRuntime``) and
  an agent that sets ``STRANDS_ROBOT_MESH_AGENT_ID`` both carry an id; an
  anonymous client carries **none** (``caller=None``).
* When an allowlist IS set, a missing/None caller cannot be authorized and is
  denied (fail-closed). So setting ``DEVICE_CONNECT_RPC_ALLOW`` will reject
  every anonymous caller - configure an id on the caller side to allow it.
* The id is only as trustworthy as the transport. Under authenticated
  transport (mTLS) it is bound to the sender's certificate. Under insecure
  transport it is **self-asserted** - any peer can claim any id - so the
  allowlist is advisory there, not a cryptographic boundary. A one-time warning
  is logged in that case.
* Which of those two holds is a property of the ``DeviceRuntime`` the driver is
  attached to, not of the environment alone. A runtime resolves its posture from
  its own ``allow_insecure`` argument first and ``DEVICE_CONNECT_ALLOW_INSECURE``
  second, so this module reads the runtime's resolved answer and falls back to
  the variable only when no runtime is attached. Reading the variable
  unconditionally answered the lower-precedence half of the question: a runtime
  brought up with ``allow_insecure=True`` and the variable unset is insecure and
  went unwarned, and one brought up with ``allow_insecure=False`` while the
  variable opted in is authenticated and was warned about anyway.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_RPC_ALLOW_ENV = "DEVICE_CONNECT_RPC_ALLOW"
_ESTOP_ALLOW_ENV = "DEVICE_CONNECT_ESTOP_ALLOW"

_warned_permissive: set[str] = set()
_warned_insecure_acl: set[str] = set()

_INSECURE_ENV = "DEVICE_CONNECT_ALLOW_INSECURE"

#: The string spellings of ``DEVICE_CONNECT_ALLOW_INSECURE`` that opt in. Spelled
#: once for the whole package: this module is stdlib-only, so the two other
#: readers of the variable can import it here, while a module that needs the
#: ``[device-connect]`` extra cannot be imported from either of them.
INSECURE_TRUE = ("true", "1", "yes")


def insecure_env_opts_in(env_value: str | None) -> bool:
    """Whether a raw ``DEVICE_CONNECT_ALLOW_INSECURE`` value opts in.

    The one reader of that variable's string vocabulary. Every surface that
    answers the question - the ``allow_insecure`` resolver, this module's
    fallback and the agent-side connector's warning - asks here, so the three
    cannot come to disagree about what ``"yes"`` means.

    Args:
        env_value: The raw variable value, or ``None`` when it is unset.

    Returns:
        Whether the value is one of :data:`INSECURE_TRUE`, case-insensitively.
        ``None`` and every other spelling are secure.
    """
    return env_value is not None and env_value.lower() in INSECURE_TRUE


def _insecure_transport_active(device: Any = None) -> bool:
    """Whether the transport carrying an RPC is unauthenticated.

    Args:
        device: The ``DeviceRuntime`` the driver is attached to (its
            ``allow_insecure`` is the resolved posture), or ``None`` when the
            driver is not attached to one.

    Returns:
        The runtime's resolved posture when there is a runtime, otherwise what
        ``DEVICE_CONNECT_ALLOW_INSECURE`` says on its own.
    """
    resolved = getattr(device, "allow_insecure", None)
    if isinstance(resolved, bool):
        return resolved
    return insecure_env_opts_in(os.environ.get(_INSECURE_ENV))


def _warn_insecure_acl_once(scope: str) -> None:
    """Warn (once per scope) that an allowlist is being enforced against a
    self-asserted caller id because the transport is insecure."""
    if scope in _warned_insecure_acl:
        return
    _warned_insecure_acl.add(scope)
    logger.warning(
        "Device Connect %s allowlist is enforced against a SELF-ASSERTED caller "
        "identity: this device's transport is insecure (allow_insecure, or %s "
        "when no runtime setting was given), so any peer can claim an allowed "
        "id. Treat the allowlist as advisory here; use authenticated transport "
        "(mTLS) for a cryptographic authorization boundary.",
        scope,
        _INSECURE_ENV,
    )


def _parse_allowlist(raw: str | None) -> list[str] | None:
    """Parse a comma-separated allowlist. Returns None when unset/empty.

    This is the one place that decides whether an allowlist is set. A value is
    empty when it holds no non-blank entry after stripping, so ``""``, ``" "``
    and ``","`` all parse to None. Every emptiness question routes here rather
    than testing the raw string's truthiness, which would call a whitespace- or
    comma-only value "set" and leave it parsing to nothing.
    """
    if raw is None:
        return None
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    return entries or None


def _matches(caller: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat == "*" or fnmatch.fnmatchcase(caller, pat):
            return True
    return False


def _warn_permissive_once(scope: str) -> None:
    if scope not in _warned_permissive:
        _warned_permissive.add(scope)
        logger.warning(
            "Device Connect %s authorization is permissive (no %s allowlist set). "
            "Any device that can reach the network may invoke state-mutating "
            "operations. Set the allowlist to restrict callers.",
            scope,
            _RPC_ALLOW_ENV if scope == "rpc" else _ESTOP_ALLOW_ENV,
        )


def is_authorized_caller(caller: str | None, *, scope: str = "rpc", device: Any = None) -> bool:
    """Return True iff *caller* is authorized for the given *scope*.

    Args:
        caller: The id the messaging layer reported as the RPC's source device,
            or ``None`` for an anonymous caller.
        scope: ``"rpc"`` for state-mutating RPCs (execute/stop/step/reset), or
            ``"estop"`` for emergency-stop event handling.
        device: The ``DeviceRuntime`` this driver is attached to, so the
            self-asserted-identity advisory follows the transport that actually
            carries the call rather than the environment variable alone. Callers
            pass ``self._device``, which ``DeviceDriver.set_device`` fills in on
            every bring-up path. ``None`` falls back to the variable.

    Returns:
        Whether the call may proceed. Authorization itself does not consult
        *device*: an allowlist is enforced under either posture, and *device*
        only decides whether the advisory that the enforcement is advisory
        fires.
    """
    if scope == "estop":
        # Fall back through the parser, not through the raw string's truthiness:
        # a whitespace- or comma-only value (a templated list whose ids never got
        # populated) is an empty allowlist, so it must inherit the RPC allowlist.
        # Testing truthiness here would call it "set", skip the fallback, and
        # then parse it to nothing - opening emergencyStop to every caller,
        # anonymous ones included.
        patterns = _parse_allowlist(os.environ.get(_ESTOP_ALLOW_ENV))
        if patterns is None:
            patterns = _parse_allowlist(os.environ.get(_RPC_ALLOW_ENV))
        env_scope = "estop"
    else:
        patterns = _parse_allowlist(os.environ.get(_RPC_ALLOW_ENV))
        env_scope = "rpc"

    if patterns is None:
        # No allowlist configured - preserve out-of-the-box dev usability but
        # make the permissive posture loud so operators notice.
        _warn_permissive_once(env_scope)
        return True

    # An allowlist is configured. If the transport is insecure the caller id is
    # self-asserted, so the allowlist is advisory - say so once, loudly.
    if _insecure_transport_active(device):
        _warn_insecure_acl_once(env_scope)

    # Allowlist configured: a missing caller identity cannot be authorized.
    if not caller:
        return False
    return _matches(caller, patterns)


def authz_error(caller: str | None, function: str) -> dict[str, str]:
    """Standard structured rejection for an unauthorized RPC call."""
    logger.warning("Rejected unauthorized Device Connect RPC %s from caller=%r", function, caller)
    return {
        "status": "error",
        "reason": f"caller not authorized for {function!r}",
        "caller": caller or "unknown",
    }
