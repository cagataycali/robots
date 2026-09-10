"""Caller-authorization helpers for Device Connect robot/sim drivers.

Security hardening: Device Connect RPC handlers run on the device side with no
built-in per-call authorization. State-mutating RPCs (execute / stop / step /
reset) and lifecycle events (emergencyStop) must therefore verify the calling
device against an operator-controlled allowlist before acting on physical (or
simulated) hardware.

Allowlists are sourced from environment variables so deployments opt in without
code changes:

* ``DEVICE_CONNECT_RPC_ALLOW`` - comma-separated device ids permitted to call
  state-mutating RPCs. **Unset means nobody**: with no allowlist every
  state-mutating RPC is refused, and the refusal is logged once naming the
  variable to set. ``*`` means "allow all" and logs a warning so the permissive
  posture is visible - it is the development spelling, and it has to be typed.
  An empty value is treated as unset, and a value is empty when it holds no
  non-blank entry after stripping - so ``""``, ``" "`` and ``","`` are all
  unset. (Before F-003 an unset allowlist meant "allow all": a device that was
  simply never configured executed every peer's ``execute`` / ``stop`` /
  ``step`` / ``reset``, and the warning was the only sign.)
* ``DEVICE_CONNECT_ESTOP_ALLOW`` - comma-separated device ids permitted to
  trigger emergency-stop handling. Falls back to ``DEVICE_CONNECT_RPC_ALLOW``
  when unset. With neither set, a **named** caller may still stop the robot -
  stopping must never get harder than moving - but an anonymous caller
  (``caller=None``) is refused, so the fallback stays permissive for peers that
  say who they are and the warning still fires.

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
_warned_unconfigured: set[str] = set()

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


def attached_runtime(driver: Any) -> Any:
    """The ``DeviceRuntime`` *driver* is attached to, or ``None`` when it is not.

    ``DeviceDriver._device`` belongs to ``device_connect_edge``, whose
    ``__init__`` initializes it to ``None`` and whose ``set_device`` later
    rebinds it to the runtime - ``drivers/base.py`` lines 146 and 184 on both
    the published 0.2.5 this tree locks and ``arm/device-connect@main``, the ref
    CI redirects to when the integration changes. So an unattached driver
    presents as a ``None`` *value*, which is the case
    :func:`is_authorized_caller` documents the environment fallback for, and
    every driver here reaches that initializer: all three call
    ``super().__init__()``.

    Read through ``getattr`` with a default rather than as ``driver._device``
    so the contract is stated once instead of assumed at each of the 21 call
    sites, and so the two shapes answer alike. A driver that has not reached
    ``super().__init__()`` carries no attribute at all rather than a ``None``
    -- ``ReachyMiniDriver`` refuses an unusable ``api_port`` or ``prefix``
    before that call -- and the direct read raises ``AttributeError`` on that
    shape where this returns ``None``. The distinction is worth the default
    because the ``emergencyStop`` handlers read the posture before deciding a
    stop, and an ``AttributeError`` there is a stop that neither authorizes nor
    refuses.

    Args:
        driver: The ``DeviceDriver`` handling the call, normally ``self``.

    Returns:
        The attached runtime, or ``None`` when no runtime has been set - which
        is what :func:`_insecure_transport_active` falls back to the environment
        variable for.
    """
    return getattr(driver, "_device", None)


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
    """Warn (once per scope) that every named caller is being allowed.

    Fires for an explicit ``*`` on either scope, and for the emergency-stop
    scope when no allowlist is set at all (the one place an unset list still
    allows anyone who carries an id).
    """
    if scope not in _warned_permissive:
        _warned_permissive.add(scope)
        logger.warning(
            "Device Connect %s authorization is permissive (the %s allowlist "
            "allows every named caller). Any device that can reach the network "
            "may invoke these operations. List specific device ids to restrict callers.",
            scope,
            _RPC_ALLOW_ENV if scope == "rpc" else _ESTOP_ALLOW_ENV,
        )


def _warn_unconfigured_once(scope: str) -> None:
    """Log (once per scope) that a call was refused because no allowlist exists."""
    if scope in _warned_unconfigured:
        return
    _warned_unconfigured.add(scope)
    logger.warning(
        "Refused a Device Connect %s call: no %s allowlist is set, and an unset "
        "allowlist authorizes nobody. Set %s to the comma-separated device ids "
        "that may call state-mutating operations, or to '*' to allow every "
        "caller during development.",
        scope,
        _RPC_ALLOW_ENV,
        _RPC_ALLOW_ENV,
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
            pass ``attached_runtime(self)``, which reports the runtime
            ``DeviceDriver.set_device`` set or ``None`` when none was set.
            ``None`` falls back to the variable.

    Returns:
        Whether the call may proceed. Authorization itself does not consult
        *device*: an allowlist is enforced under either posture, and *device*
        only decides whether the advisory that the enforcement is advisory
        fires.

        With no allowlist configured the answer depends on the scope. A
        state-mutating RPC is refused - authorization that nobody configured
        authorizes nobody (F-003, CWE-862) - and the refusal is logged once
        naming ``DEVICE_CONNECT_RPC_ALLOW``. An emergency stop from a named
        caller is still honoured, because a stop must never be harder to
        deliver than the motion it ends, but an anonymous stop is refused: a
        caller that carries no id cannot be told apart from one that forged
        none.
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
        if env_scope != "estop":
            # No allowlist configured: nobody is authorized to move the robot.
            # Development opts in by spelling '*' rather than by forgetting.
            _warn_unconfigured_once(env_scope)
            return False
        # Emergency stop with no allowlist anywhere: a named caller may still
        # stop the robot (stopping must not get harder than moving), and the
        # permissive posture stays loud. An anonymous caller is refused: no id
        # means nothing to hold the stop against.
        _warn_permissive_once(env_scope)
        return bool(caller)

    if "*" in patterns:
        # An explicit '*' is the dev spelling of "allow all" - loud, like before.
        _warn_permissive_once(env_scope)

    # An allowlist is configured. If the transport is insecure the caller id is
    # self-asserted, so the allowlist is advisory - say so once, loudly.
    if _insecure_transport_active(device):
        _warn_insecure_acl_once(env_scope)

    # Allowlist configured: a missing caller identity cannot be authorized, and
    # neither can one that is not a name - the transport reports a device id as
    # a string, so anything else is not an identity the allowlist can speak to.
    if not caller or not isinstance(caller, str):
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
