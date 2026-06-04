"""Cosmos 3 embodiment specs — data-driven action/observation layouts.

Each embodiment maps a Cosmos 3 ``domain_name`` (the world-model conditioning
domain) to its raw action dimensionality, default action-chunk size, and the
named layout of the action vector columns so the policy can emit per-actuator
dicts instead of opaque float rows.

Dimensions verified against ``cosmos_framework.data.vfm.action.domain_utils``
(``EMBODIMENT_TO_RAW_ACTION_DIM``) and the released RoboLab DROID policy server
defaults (``action_policy_server_robolab.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cosmos3Embodiment:
    """Static description of a Cosmos 3 action embodiment.

    Attributes:
        name: Short embodiment key (``"droid"``).
        domain_name: Cosmos 3 conditioning domain (``"droid_lerobot"``).
        raw_action_dim: Raw model action dim for the domain (DROID = 10).
        action_chunk_size: Default number of predicted action steps.
        fps: Conditioning FPS the policy was trained at.
        camera_keys: Server observation image keys (OpenPI ``/`` namespace).
        action_layouts: ``{action_space: [column_name, ...]}`` naming each
            output action column so :class:`Cosmos3Policy` can build
            per-actuator step dicts. The released DROID policy serves
            ``joint_pos`` (8D = 7 joints + gripper) and ``midtrain``
            (10D = 3 pos + 4 quat + ... + gripper).
        default_action_space: The action space the server serves by default.
    """

    name: str
    domain_name: str
    raw_action_dim: int
    action_chunk_size: int
    fps: int
    camera_keys: list[str] = field(default_factory=list)
    action_layouts: dict[str, list[str]] = field(default_factory=dict)
    default_action_space: str = "joint_pos"


# Canonical 7-DOF Franka joint names (DROID/RoboMIND-Franka), matching the
# ordered joint convention used by the released Cosmos3-Nano-Policy-DROID.
_FRANKA_JOINTS = [f"joint_{i}" for i in range(7)]

# DROID joint_pos action = [7 joint deltas/targets, 1 gripper].
_DROID_JOINT_POS = _FRANKA_JOINTS + ["gripper"]
# DROID midtrain action = [3 EE position, 4 quaternion (xyzw), gripper].
_DROID_MIDTRAIN = ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw", "gripper"]


EMBODIMENTS: dict[str, Cosmos3Embodiment] = {
    "droid": Cosmos3Embodiment(
        name="droid",
        domain_name="droid_lerobot",
        raw_action_dim=10,
        action_chunk_size=32,
        fps=15,
        camera_keys=[
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
        ],
        action_layouts={
            "joint_pos": _DROID_JOINT_POS,
            "midtrain": _DROID_MIDTRAIN,
        },
        default_action_space="joint_pos",
    ),
    "umi": Cosmos3Embodiment(
        name="umi",
        domain_name="umi",
        raw_action_dim=10,
        action_chunk_size=16,
        fps=20,
        camera_keys=["observation/image"],
        action_layouts={
            # EE 9D pose delta (3D translation + 6D rotation) + 1D grasp.
            "midtrain": [
                "tx", "ty", "tz",
                "r0", "r1", "r2", "r3", "r4", "r5",
                "grasp",
            ],
        },
        default_action_space="midtrain",
    ),
    "av": Cosmos3Embodiment(
        name="av",
        domain_name="av",
        raw_action_dim=9,
        action_chunk_size=60,
        fps=10,
        camera_keys=["observation/image"],
        action_layouts={
            # Ego pose 9D (3D translation + 6D rotation), no gripper.
            "midtrain": ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5"],
        },
        default_action_space="midtrain",
    ),
    "bridge": Cosmos3Embodiment(
        name="bridge",
        domain_name="bridge_orig_lerobot",
        raw_action_dim=10,
        action_chunk_size=16,
        fps=5,
        camera_keys=["observation/image"],
        action_layouts={
            "midtrain": ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5", "grasp"],
        },
        default_action_space="midtrain",
    ),
}

# Aliases → canonical embodiment key.
_EMBODIMENT_ALIASES = {
    "droid_lerobot": "droid",
    "robomind-franka": "droid",
    "franka": "droid",
    "bridge_orig_lerobot": "bridge",
    "autonomous_vehicle": "av",
}


def get_embodiment(name: str) -> Cosmos3Embodiment:
    """Resolve an embodiment by name or alias.

    Args:
        name: Embodiment key or alias (``"droid"``, ``"droid_lerobot"``, ...).

    Returns:
        The matching :class:`Cosmos3Embodiment`.

    Raises:
        KeyError: If the embodiment is unknown.
    """
    key = name.lower().strip()
    key = _EMBODIMENT_ALIASES.get(key, key)
    if key not in EMBODIMENTS:
        raise KeyError(
            f"Unknown Cosmos 3 embodiment {name!r}. "
            f"Available: {sorted(EMBODIMENTS)} (+ aliases {sorted(_EMBODIMENT_ALIASES)})"
        )
    return EMBODIMENTS[key]


def list_embodiments() -> list[str]:
    """List canonical embodiment names."""
    return sorted(EMBODIMENTS)
