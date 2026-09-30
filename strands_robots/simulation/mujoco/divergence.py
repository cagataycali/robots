"""Read whether MuJoCo declared the physics unstable during a run of steps.

``mj_step`` checks ``qpos``, ``qvel`` and ``qacc`` every step. When one holds a
NaN, an inf or a value past ``mjMAXVAL``, MuJoCo resets the ENTIRE state - every
joint of every robot and every object back to the model's initial pose, and
``data.time`` back to zero - and says so only as a ``WARNING`` on stderr plus a
bump of ``data.warning[...].number``. A caller that stepped through that saw
``success`` and a clock that ran backwards. These helpers turn the counter bump
into a sentence the step surfaces can return.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# The three checks mj_step runs before it integrates, and the state each names.
_INSTABILITY_WARNINGS = (
    ("mjWARN_BADQPOS", "joint position"),
    ("mjWARN_BADQVEL", "joint velocity"),
    ("mjWARN_BADQACC", "joint acceleration"),
)


def instability_counts(mj: Any, data: Any) -> tuple[int, ...]:
    """Return how many times MuJoCo has declared ``data`` unstable so far, per check.

    Args:
        mj: The ``mujoco`` module.
        data: The ``MjData`` being stepped.

    Returns:
        The ``BADQPOS``, ``BADQVEL`` and ``BADQACC`` warning counters, in that order.
    """
    return tuple(int(data.warning[getattr(mj.mjtWarning, name)].number) for name, _ in _INSTABILITY_WARNINGS)


def _clear_instability_counts(mj: Any, data: Any) -> None:
    """Zero the three instability counters so the next read starts from a clean baseline."""
    for name, _ in _INSTABILITY_WARNINGS:
        data.warning[getattr(mj.mjtWarning, name)].number = 0


def _state_is_finite(data: Any) -> bool:
    return bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())


def divergence_error(mj: Any, model: Any, data: Any, before: tuple[int, ...], verb: str) -> str | None:
    """Describe a divergence since ``before`` was read, or return ``None``.

    Args:
        mj: The ``mujoco`` module.
        model: The ``MjModel`` being stepped.
        data: The ``MjData`` being stepped.
        before: :func:`instability_counts` read before the steps ran.
        verb: The surface reporting it (``"step"``, ``"send_action"``, ...).

    Returns:
        An error sentence naming what diverged, the joint, and how to recover;
        ``None`` when MuJoCo declared nothing and the state is finite.

    Side effect: when a divergence is reported the three counters are zeroed,
    because MuJoCo's autoreset leaves the flagged one latched at exactly 1
    (``mj_resetData`` zeroes every counter, then the check re-increments its
    own). Without that, a second divergence of the same kind after a caller
    ignored the first, or recovered with ``load_state`` (``mj_setState`` never
    touches ``data.warning``), would read as "no change" and step through as a
    success, the very outcome this module exists to refuse. ``reset`` gives the
    same clean baseline through ``mj_resetData``.
    """
    after = instability_counts(mj, data)
    if after == before and _state_is_finite(data):
        return None
    _clear_instability_counts(mj, data)
    what, joint = "state", None
    for (name, label), was, now in zip(_INSTABILITY_WARNINGS, before, after, strict=True):
        stat = data.warning[getattr(mj.mjtWarning, name)]
        if now > was:
            what = label
            dof = int(stat.lastinfo)
            if 0 <= dof < model.nv:
                jnt = int(model.dof_jntid[dof])
                joint = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt) or f"joint {jnt}"
            break
    where = f" at '{joint}'" if joint else ""
    if _state_is_finite(data):
        outcome = (
            "MuJoCo reset every robot and object to the model's initial pose and the clock "
            f"to t={data.time:.4f}s, so the state now is not a continuation of the run"
        )
    else:
        outcome = "the state is no longer finite (NaN/inf)"
    return (
        f"{verb}: the physics diverged - MuJoCo found a NaN, inf or huge {what}{where}, and {outcome}. "
        "Usual causes: a very large force, gain, mass change or timestep. Call reset (or load_state) "
        "and retry with smaller values."
    )
