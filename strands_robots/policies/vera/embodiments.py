"""VERA embodiment registry — declares the supported targets + their action layouts.

The VERA server advertises *most* of this at connect time (via
``VeraServerConfig``: ``view_keys``, ``view_widths``, ``action_dim``,
``action_horizon``, ``control_dt``, ``action_space``, ``embodiment``,
``gripper_dim_index``). We keep an embodiment table on the client side for:

1. **Default action layouts** — names per action dim so we can return per-step
   actuator dicts (the strands-robots :class:`Policy` contract) without
   forcing the user to handcraft an ``action_mapping`` for the common case.
2. **Built-in robot mappings** — e.g. VERA's ``mimicgen`` embodiment uses the
   Panda layout (joint_0..joint_6 + gripper); we ship a default mapping onto
   MuJoCo Panda actuator names (mirrors ``cosmos3``'s
   ``ROBOT_ACTION_MAPPINGS``).
3. **Sensible defaults** — port, render-resolution hint, expected view keys —
   so a one-liner ``create_policy("vera", embodiment="pusht")`` Just Works
   against a vera server launched with matching defaults.

The actual contract is whatever the server declares on connect; this table is
client convenience only. The :class:`VeraPolicy` re-reads ``view_keys`` /
``view_widths`` / ``action_horizon`` / ``action_dim`` from
``client.get_server_metadata()`` and uses them when present, falling back to
the embodiment defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VeraEmbodiment:
    """Per-embodiment client-side defaults for the VERA policy."""

    name: str
    aliases: tuple[str, ...] = ()
    # Per-view names the server expects (width-concat order). The server
    # actually drives this on connect; these are fallbacks for testing /
    # introspection.
    view_keys: tuple[str, ...] = ()
    # Per-view widths (px). Server-driven on connect; sane default below.
    view_widths: tuple[int, ...] = ()
    # Default action layout — names per action dim. Used to unpack the
    # ``(H, D)`` action chunk into per-timestep actuator dicts.
    action_layout: tuple[str, ...] = ()
    # Which action dim is the gripper (``-1`` = no gripper). The server
    # actually advertises this in ``gripper_dim_index``; this is a fallback.
    gripper_dim_index: int = -1
    # Default server port (matches the vera README quickstart examples).
    default_port: int = 8000
    # Optional default ``action_space`` advertised by the server
    # (informational only — the server is the source of truth).
    default_action_space: str = "joint_position"
    # Default render size (per view). Used by the example notebooks /
    # ``examples/vera_sim_rollout.py`` when sizing cameras.
    default_render_size: int = 252
    # Free-form extras (e.g. recommended ``--text`` prompt).
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in embodiments (Wave 1 ships PushT + MimicGen; Wave 2 adds DROID/Allegro)
# ---------------------------------------------------------------------------
PUSHT = VeraEmbodiment(
    name="pusht",
    aliases=("planar_push", "push_t"),
    view_keys=("image",),
    view_widths=(252,),
    # PushT is a 2-DoF planar pusher: (dx, dy) velocity commands. No gripper.
    action_layout=("dx", "dy"),
    gripper_dim_index=-1,
    default_port=8820,
    default_action_space="cartesian_velocity",
    default_render_size=252,
    extras={
        "description": "Planar T-block pushing (gym-pusht) — DFoT planner, fastest to load",
        "needs_prompt": False,
    },
)

MIMICGEN = VeraEmbodiment(
    name="mimicgen",
    aliases=("panda", "panda_stack", "mimicgen_stack"),
    view_keys=("front", "side", "agentview"),
    view_widths=(128, 128, 128),
    # MimicGen serves a Panda 7-joint + 1 gripper: matches the cosmos3 DROID
    # ``joint_pos`` layout (joint_0..joint_6 + gripper). The server advertises
    # ``action_dim`` and ``gripper_dim_index`` on connect (= 8 / 7 here).
    action_layout=(
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ),
    gripper_dim_index=7,
    default_port=8800,
    default_action_space="joint_position",
    default_render_size=128,
    extras={
        "description": "MimicGen 2-block stacking (Panda) — WAN 1.3B planner",
        "default_prompt": "A robot arm stacks one block on top of another block",
    },
)

DROID = VeraEmbodiment(
    name="droid",
    aliases=("franka", "fr3", "panda_real"),
    view_keys=("ext1", "ext2", "wrist"),
    view_widths=(128, 128, 128),
    # DROID FR3: 7 joints + 1 gripper (cartesian-velocity served by default;
    # the IDM also exposes joint_position). Wave-2 release: code is in-tree
    # but checkpoints land later — adjust ``action_dim`` if your server uses
    # a different head.
    action_layout=(
        "dx",
        "dy",
        "dz",
        "drx",
        "dry",
        "drz",
        "gripper",
    ),
    gripper_dim_index=6,
    default_port=8000,
    default_action_space="cartesian_velocity",
    default_render_size=128,
    extras={
        "description": "DROID FR3 (Franka Research 3) — real-world cartesian-velocity policy",
        "default_prompt": "pick up the object",
    },
)

ALLEGRO = VeraEmbodiment(
    name="allegro",
    aliases=("allegro_hand", "dexterous", "in_hand"),
    view_keys=("ext1", "ext2"),
    view_widths=(128, 128),
    # 16-DoF Allegro hand: per-joint position commands. No "gripper" — the
    # whole hand IS the manipulator.
    action_layout=tuple(f"joint_{i}" for i in range(16)),
    gripper_dim_index=-1,
    default_port=8001,
    default_action_space="joint_position",
    default_render_size=128,
    extras={
        "description": "Allegro 16-DoF hand — dexterous in-hand cube reorientation",
        "default_prompt": "rotate the cube clockwise around the vertical axis",
    },
)


_EMBODIMENTS: dict[str, VeraEmbodiment] = {
    e.name: e for e in (PUSHT, MIMICGEN, DROID, ALLEGRO)
}
_ALIASES: dict[str, str] = {
    alias: e.name for e in _EMBODIMENTS.values() for alias in e.aliases
}


def get_embodiment(name: str) -> VeraEmbodiment:
    """Look up an embodiment by name or alias.

    Raises:
        ValueError: with an actionable list of available embodiments.
    """
    if name in _EMBODIMENTS:
        return _EMBODIMENTS[name]
    if name in _ALIASES:
        return _EMBODIMENTS[_ALIASES[name]]
    raise ValueError(
        f"Unknown VERA embodiment {name!r}. Available: "
        f"{sorted(_EMBODIMENTS)} (aliases: {sorted(_ALIASES)})"
    )


def list_embodiments() -> list[str]:
    """Return the canonical embodiment names."""
    return sorted(_EMBODIMENTS)


# ---------------------------------------------------------------------------
# Built-in robot action mappings — apply when the user passes ``robot="panda"``
# so per-step dicts use real MuJoCo actuator names without manual mapping.
# Mirrors cosmos3's ``ROBOT_ACTION_MAPPINGS``.
# ---------------------------------------------------------------------------
ROBOT_ACTION_MAPPINGS: dict[str, dict[str, str]] = {
    # MuJoCo Panda model (the strands-robots ``franka_panda`` asset).
    "panda": {
        "joint_0": "joint1",
        "joint_1": "joint2",
        "joint_2": "joint3",
        "joint_3": "joint4",
        "joint_4": "joint5",
        "joint_5": "joint6",
        "joint_6": "joint7",
        "gripper": "finger_joint1",
    },
    "franka": {
        "joint_0": "joint1",
        "joint_1": "joint2",
        "joint_2": "joint3",
        "joint_3": "joint4",
        "joint_4": "joint5",
        "joint_5": "joint6",
        "joint_6": "joint7",
        "gripper": "finger_joint1",
    },
    # PushT planar pusher (gym-pusht): the strands-robots simulation owns the
    # 2D actuator names; this maps VERA's (dx, dy) onto them.
    "pusht": {"dx": "vx", "dy": "vy"},
}


def get_robot_action_mapping(robot: str) -> dict[str, str] | None:
    """Return a built-in DROID/MimicGen-layout → robot-actuator name map, or None."""
    return ROBOT_ACTION_MAPPINGS.get(robot)


def list_robot_action_mappings() -> list[str]:
    """Available robot names for the ``robot=`` constructor sugar."""
    return sorted(ROBOT_ACTION_MAPPINGS)
