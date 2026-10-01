"""Feature layouts for whole-body teleoperation and recording.

A layout is the fixed table of state and action column names a recorder writes
and a checkpoint expects. The same 29 Unitree G1 joints are spelled three ways
in the wild, and a dataset that mixes them is silently unusable:

* our MuJoCo actuator names (``left_hip_pitch_joint``), the order
  ``robot_action_keys("g1")`` returns: left leg, right leg, waist, left arm,
  right arm;
* lerobot's ``G1_29_JointIndex`` names (``kLeftHipPitch.q``), the same order,
  which the ``nepyope/can_clean_final`` dataset uses for ``observation.state``;
* lerobot's SONIC controller token keys (``motion_token.{i}.pos``), while that
  dataset spells its action ``motion_token_{i}``.

The two orders are the same (the hardware order); the ZMQ wire NVIDIA's
teleop streamer speaks uses IsaacLab's breadth-first order instead, and
:data:`ISAACLAB_TO_HARDWARE` is that permutation, so a source consuming the
wire converts once at its edge and nothing inside the package sees two orders.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Unitree G1 29 joints in hardware order: the order our MuJoCo model's
#: actuators, lerobot's ``G1_29_JointIndex`` and the blog dataset all share.
G1_HARDWARE_JOINTS: tuple[str, ...] = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

#: lerobot ``G1_29_JointIndex`` spellings, same order as :data:`G1_HARDWARE_JOINTS`.
G1_LEROBOT_JOINTS: tuple[str, ...] = tuple(
    "k" + "".join(part.capitalize() for part in name.split("_")) + ".q" for name in G1_HARDWARE_JOINTS
)

#: Our MuJoCo actuator names, same order.
G1_SIM_JOINTS: tuple[str, ...] = tuple(f"{name}_joint" for name in G1_HARDWARE_JOINTS)

#: Breadth-first (IsaacLab, the ZMQ wire) order expressed as hardware indices:
#: entry ``i`` is the hardware index of breadth-first joint ``i``. This is the
#: argsort of lerobot's ``g1_utils.ISAACLAB_TO_MUJOCO``; the two projects name
#: the orders the other way round, so the table is defined by structure here
#: (wrists are the breadth-first tail 23..28, which the wire spec states).
ISAACLAB_TO_HARDWARE: tuple[int, ...] = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)  # fmt: skip

#: Size of a SONIC motion token.
TOKEN_DIM = 64

GRIPPERS: tuple[str, ...] = ("left_gripper", "right_gripper")


@dataclass(frozen=True)
class Layout:
    """The columns one recording writes.

    Attributes:
        name: The layout's registry name.
        state_names: ``observation.state`` column names, in order.
        action_names: ``action`` column names, in order.
        joint_names: The 29 body joints as the follower spells them, in hardware
            order; the names ``send_action`` receives.
        is_token: Whether ``action`` carries a 64-D token instead of joints.
    """

    name: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    is_token: bool

    def joint_action(self, joints: dict[str, float], grippers: dict[str, float]) -> dict[str, float]:
        """Return the action dict for joint targets plus gripper fractions."""
        out = {name: float(joints[name]) for name in self.joint_names if name in joints}
        out.update(self._gripper_columns(grippers))
        return out

    def _gripper_columns(self, grippers: dict[str, float]) -> dict[str, float]:
        """Gripper values for the gripper columns this layout has; a layout without them drops the values."""
        return {name: float(grippers[name]) for name in GRIPPERS if name in grippers and name in self.action_names}

    def token_action(self, token: list[float], grippers: dict[str, float]) -> dict[str, float]:
        """Return the action dict for one 64-D token plus gripper fractions."""
        if len(token) != TOKEN_DIM:
            raise ValueError(f"layout {self.name!r} wants a {TOKEN_DIM}-D token, got {len(token)}")
        names = self.action_names[:TOKEN_DIM]
        out = {name: float(value) for name, value in zip(names, token, strict=True)}
        out.update(self._gripper_columns(grippers))
        return out


_LAYOUTS: dict[str, Layout] = {
    "g1_joint_29": Layout(
        name="g1_joint_29",
        state_names=G1_SIM_JOINTS,
        action_names=G1_SIM_JOINTS,
        joint_names=G1_SIM_JOINTS,
        is_token=False,
    ),
    "blog_31_66": Layout(
        name="blog_31_66",
        state_names=G1_LEROBOT_JOINTS + GRIPPERS,
        action_names=tuple(f"motion_token_{i}" for i in range(TOKEN_DIM)) + GRIPPERS,
        joint_names=G1_SIM_JOINTS,
        is_token=True,
    ),
    "blog_31_31": Layout(
        name="blog_31_31",
        state_names=G1_LEROBOT_JOINTS + GRIPPERS,
        action_names=G1_LEROBOT_JOINTS + GRIPPERS,
        joint_names=G1_SIM_JOINTS,
        is_token=False,
    ),
    "lerobot_token_64": Layout(
        name="lerobot_token_64",
        state_names=tuple(f"motion_token_state.{i}.pos" for i in range(TOKEN_DIM)),
        action_names=tuple(f"motion_token.{i}.pos" for i in range(TOKEN_DIM)),
        joint_names=G1_SIM_JOINTS,
        is_token=True,
    ),
}


def teleop_layout(name: str) -> Layout:
    """Return the layout registered under ``name``.

    Raises:
        ValueError: ``name`` is not a layout; the message lists the choices.
    """
    try:
        return _LAYOUTS[name]
    except KeyError:
        raise ValueError(f"Unknown teleop layout {name!r}. Choose from: {sorted(_LAYOUTS)}") from None


def list_layouts() -> tuple[str, ...]:
    """Return the registered layout names, sorted."""
    return tuple(sorted(_LAYOUTS))
