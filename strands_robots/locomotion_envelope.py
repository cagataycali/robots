"""The locomotion velocity envelope a ``target_velocity`` command may ask for.

One definition, two readers. :mod:`strands_robots.mesh.security` holds the
mesh ``start`` / ``execute`` payload's ``target_velocity`` to it before the
command is dispatched, and :class:`~strands_robots.policies.wbc.policy.WBCPolicy`
holds the kwarg to it again before the value is scaled into the observation the
network is given - so the wire and the sink cannot come to disagree about how
fast a legged robot may be asked to go (F-005, CWE-20). It lives here, at the
top of the package with only the standard library above it, because the two
readers sit in layers that must not import each other: ``mesh`` must not pull
``policies`` (and NumPy) into its import, and a policy must not depend on the
transport that happens to carry its commands.

The numbers are a plausibility envelope, not a per-robot spec. 2 m/s is a brisk
human walk and above what any shipped humanoid locomotion policy here was
trained on (the G1 WBC checkpoints are commanded in tenths of a metre per
second); 2 rad/s is a full turn in ~3 s. A command past either is not a bold
gait, it is an input the policy never saw, so it is refused rather than clamped:
a clamped ``[1e6, 0, 0]`` would still be a silent 2 m/s sprint the caller did
not ask for. Operators with a faster platform raise the bound through the
environment; the resolver re-reads it on every call so a change takes effect
without a restart, the pattern ``STRANDS_MESH_INPUT_VALUE_ABS`` already uses.

Component layout follows the receivers' own convention: index 0 and 1 are
linear (``vx``, ``vy`` in m/s), index 2 is angular (``omega`` in rad/s). The
arity verdict belongs to the receiving policy, not to the wire
(:data:`strands_robots.mesh.security.MAX_TARGET_VELOCITY_COMPONENTS` admits
more than three), so any further component is held to the linear bound.
"""

from __future__ import annotations

import math
import os

#: Bound on ``|vx|`` and ``|vy|`` (and any component past the third), in m/s.
MAX_TARGET_LINEAR_VELOCITY_MPS: float = 2.0

#: Bound on ``|omega|`` (component 2), in rad/s.
MAX_TARGET_ANGULAR_VELOCITY_RPS: float = 2.0

#: Operator override for the linear bound. A positive finite float in m/s;
#: anything else is ignored and the default stands.
LINEAR_ENV_VAR: str = "STRANDS_MAX_TARGET_LINEAR_VELOCITY_MPS"

#: Operator override for the angular bound, in rad/s. Same parsing rule.
ANGULAR_ENV_VAR: str = "STRANDS_MAX_TARGET_ANGULAR_VELOCITY_RPS"


def _positive_float_env(env_var: str, default: float) -> float:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def max_linear_velocity_mps() -> float:
    """The linear bound in force right now (env override or the default)."""
    return _positive_float_env(LINEAR_ENV_VAR, MAX_TARGET_LINEAR_VELOCITY_MPS)


def max_angular_velocity_rps() -> float:
    """The angular bound in force right now (env override or the default)."""
    return _positive_float_env(ANGULAR_ENV_VAR, MAX_TARGET_ANGULAR_VELOCITY_RPS)


def component_bound(index: int) -> tuple[float, str]:
    """``(bound, unit)`` for ``target_velocity[index]`` under the current envelope."""
    if index == 2:
        return max_angular_velocity_rps(), "rad/s"
    return max_linear_velocity_mps(), "m/s"


def target_velocity_component_error(index: int, value: float, context: str) -> str | None:
    """Refusal text when a finite ``value`` exceeds the envelope for its slot, else ``None``.

    Runs after the caller's finiteness check: ``nan`` compares false against
    every bound and would slip past a bare comparison.

    Args:
        index: Position in the ``target_velocity`` list.
        value: The component, already known to be a finite float.
        context: Message prefix naming the surface that received it.

    Returns:
        A message naming the component, the value, the bound, its unit and the
        environment variable that raises it - or ``None`` when in envelope.
    """
    bound, unit = component_bound(index)
    if abs(value) <= bound:
        return None
    env_var = ANGULAR_ENV_VAR if index == 2 else LINEAR_ENV_VAR
    return (
        f"{context}: target_velocity[{index}]={value!r} exceeds the locomotion envelope "
        f"of +/-{bound} {unit}; refusing rather than clamping. Raise {env_var} for a faster platform."
    )
