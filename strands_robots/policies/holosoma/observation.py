"""Pure-NumPy observation builder and gait clock for the Holosoma provider.

Reproduces ``holosoma_inference/policies/base.py`` (``_initialize_history_state``,
``_update_obs_history``) and ``policies/locomotion.py`` (``update_phase_time``,
``_get_obs_sin_phase`` / ``_get_obs_cos_phase``) with no torch or onnxruntime so
the layout is unit-testable on any machine.

The one fact that decides the layout: upstream sorts the term names of a group
alphabetically before concatenating them (``holosoma_inference/policies/base.py:135``
``self.obs_terms_sorted[group] = sorted(term_names)``). The ``loco-g1-29dof``
preset therefore lays ``actor_obs`` out as

    actions(29) base_ang_vel(3) command_ang_vel(1) command_lin_vel(2) cos_phase(2)
    dof_pos(29) dof_vel(29) projected_gravity(3) sin_phase(2)          = 100

and not in the order the preset lists them. lerobot's port
(``robots/unitree_g1/controllers/holosoma_locomotion.py``) hand-writes the same
slices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from strands_robots.policies.holosoma.config import HOLOSOMA_OBS_DIM, HolosomaConfig

#: Term names of ``actor_obs`` in the order the flattened vector carries them.
ACTOR_OBS_TERMS: tuple[str, ...] = (
    "actions",
    "base_ang_vel",
    "command_ang_vel",
    "command_lin_vel",
    "cos_phase",
    "dof_pos",
    "dof_vel",
    "projected_gravity",
    "sin_phase",
)

#: Width of every term (``config_values/observation.py:36-47``).
ACTOR_OBS_DIMS: dict[str, int] = {
    "actions": 29,
    "base_ang_vel": 3,
    "command_ang_vel": 1,
    "command_lin_vel": 2,
    "cos_phase": 2,
    "dof_pos": 29,
    "dof_vel": 29,
    "projected_gravity": 3,
    "sin_phase": 2,
}

assert sum(ACTOR_OBS_DIMS[t] for t in ACTOR_OBS_TERMS) == HOLOSOMA_OBS_DIM
assert tuple(sorted(ACTOR_OBS_TERMS)) == ACTOR_OBS_TERMS


def actor_obs_slices() -> dict[str, slice]:
    """Return ``{term: slice}`` into the flattened 100-wide vector."""
    out: dict[str, slice] = {}
    start = 0
    for term in ACTOR_OBS_TERMS:
        width = ACTOR_OBS_DIMS[term]
        out[term] = slice(start, start + width)
        start += width
    return out


def _require_len(vec: NDArray[np.float64], n: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"build_actor_obs: {name} must have {n} entries, got {arr.shape[0]}")
    return arr


def build_actor_obs(
    config: HolosomaConfig,
    *,
    last_action: NDArray[np.float64],
    base_ang_vel: NDArray[np.float64],
    command_lin_vel: NDArray[np.float64],
    command_ang_vel: float,
    phase: NDArray[np.float64],
    qj: NDArray[np.float64],
    dqj: NDArray[np.float64],
    proj_gravity: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Assemble one ``actor_obs`` row (float32, shape ``(100,)``).

    Args:
        config: Scales and defaults.
        last_action: Previous RAW network output after the clip, before
            ``action_scale`` (upstream ``last_policy_action``), 29 wide.
        base_ang_vel: Body-frame angular velocity, rad/s, 3 wide.
        command_lin_vel: ``[vx, vy]`` m/s in the base frame.
        command_ang_vel: Yaw rate command, rad/s.
        phase: ``[left, right]`` foot phase in radians (see :class:`GaitPhase`).
        qj: Measured joint positions, rad, ``dof_names`` order, 29 wide. The
            builder subtracts ``config.default_angles``.
        dqj: Measured joint velocities, rad/s, 29 wide.
        proj_gravity: Gravity direction ``[0, 0, -1]`` rotated into the base
            frame, 3 wide.

    Raises:
        ValueError: If any block has the wrong width.
    """
    scales = config.obs_scales
    n = config.n_obs_joints
    default = _require_len(np.asarray(config.default_angles), n, "config.default_angles")
    blocks = {
        "actions": _require_len(last_action, config.num_actions, "last_action") * scales.get("actions", 1.0),
        "base_ang_vel": _require_len(base_ang_vel, 3, "base_ang_vel") * scales.get("base_ang_vel", 1.0),
        "command_ang_vel": np.array([float(command_ang_vel)], dtype=np.float64) * scales.get("command_ang_vel", 1.0),
        "command_lin_vel": _require_len(command_lin_vel, 2, "command_lin_vel") * scales.get("command_lin_vel", 1.0),
        "cos_phase": np.cos(_require_len(phase, 2, "phase")) * scales.get("cos_phase", 1.0),
        "dof_pos": (_require_len(qj, n, "qj") - default) * scales.get("dof_pos", 1.0),
        "dof_vel": _require_len(dqj, n, "dqj") * scales.get("dof_vel", 1.0),
        "projected_gravity": _require_len(proj_gravity, 3, "proj_gravity") * scales.get("projected_gravity", 1.0),
        "sin_phase": np.sin(_require_len(phase, 2, "phase")) * scales.get("sin_phase", 1.0),
    }
    return np.concatenate([blocks[t] for t in ACTOR_OBS_TERMS]).astype(np.float32)


class GaitPhase:
    """The two-foot gait clock of ``LocomotionPolicy`` (``locomotion.py:56-67``).

    Left foot starts at ``0``, right at ``pi``. Every control tick the phase
    advances by ``config.phase_dt`` and wraps to ``(-pi, pi]``. When the
    commanded velocity is below ``0.01`` in both norm and yaw rate the robot
    is standing: both feet are pinned to ``pi``. The first moving tick after
    standing restarts the clock at ``[0, pi]`` instead of advancing.

    Upstream advances the phase BEFORE building the tick's observation
    (``holosoma_inference/policies/base.py:860-863``: ``update_phase_time()`` then ``policy_action()``);
    :meth:`step` is that call, so read :attr:`phase` after it.
    """

    STAND_EPS = 0.01

    def __init__(self, config: HolosomaConfig) -> None:
        self._dt = config.phase_dt
        self.phase = np.array([0.0, np.pi], dtype=np.float64)
        self.is_standing = False

    def reset(self) -> None:
        """Back to the start-of-walk clock, not standing (upstream ``_handle_start_policy``)."""
        self.phase = np.array([0.0, np.pi], dtype=np.float64)
        self.is_standing = False

    def step(self, command_lin_vel: NDArray[np.float64], command_ang_vel: float) -> NDArray[np.float64]:
        """Advance one control tick for the given command and return the phase."""
        advanced = self.phase + self._dt
        self.phase = np.fmod(advanced + np.pi, 2.0 * np.pi) - np.pi
        lin = float(np.linalg.norm(np.asarray(command_lin_vel, dtype=np.float64)))
        ang = abs(float(command_ang_vel))
        if lin < self.STAND_EPS and ang < self.STAND_EPS:
            self.phase = np.array([np.pi, np.pi], dtype=np.float64)
            self.is_standing = True
        elif self.is_standing:
            self.phase = np.array([0.0, np.pi], dtype=np.float64)
            self.is_standing = False
        return self.phase


__all__ = ["ACTOR_OBS_DIMS", "ACTOR_OBS_TERMS", "GaitPhase", "actor_obs_slices", "build_actor_obs"]
