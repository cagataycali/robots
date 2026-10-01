"""Whole-body teleoperation: a pose source, a retarget stage, an optional encoder.

The three stages are small Protocols so each can be swapped without touching the
others: a headset, a body tracker, an exoskeleton pair or a scripted mock as the
:class:`PoseSource`; a joint map, an IK solver or an SMPL fit as the
:class:`Retarget`; a SONIC encoder or nothing as the :class:`Encoder`.
:class:`WholeBodyTeleoperator` composes them and duck-types to a lerobot
``Teleoperator`` (``connect``, ``disconnect``, ``is_connected``, ``get_action``,
``action_features``), which is exactly the surface
:meth:`~strands_robots.teleop_mixin.TeleopMixin.attach_teleop` drives, so a
follower's existing 50 Hz ``teleoperate()`` loop runs it unchanged.

Units inside the package are SI: radians, metres, seconds; gripper commands are
open fractions in ``[0, 1]``. A source that reports anything else converts at its
own edge. Rates: a source owns its thread and answers :meth:`PoseSource.latest`
without blocking; the follower's loop samples it.

This module ships the pieces the design can prove without hardware or model
weights: the frame type, a scripted mock source, a joint-map retarget and the
composing teleoperator. The ZMQ, headset, SMPL and SONIC stages are described in
``docs/project/design-wholebody-teleop.md`` and land in their own PRs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from strands_robots.teleop.layouts import GRIPPERS, Layout, teleop_layout


@dataclass(frozen=True)
class PoseFrame:
    """One operator sample in the robot's base frame.

    Attributes:
        t_mono: ``time.monotonic()`` at capture, for staleness and resampling.
        frame_index: Monotonic counter from the source.
        joints: Joint targets a joint-space source already produced, radians,
            keyed by the source's own names (a leader arm, an exoskeleton, the
            mock). ``None`` for pose-only sources.
        grippers: Open fractions in ``[0, 1]`` keyed ``left_gripper`` /
            ``right_gripper``; absent keys leave the gripper alone.
        poses: Tracked points keyed ``head``, ``left_wrist``, ``right_wrist``,
            ``left_ankle``, ``right_ankle``: ``(x, y, z, qw, qx, qy, qz)`` each,
            metres and a unit quaternion. Empty for joint-space sources.
        mode: ``"joints"``, ``"three_point"``, ``"pose"`` or ``"planner"``: which
            of the streamer's modes produced the frame.
        episode_toggle: True on the one frame where the operator pressed the
            episode button; the recorder bridge saves on that edge.
    """

    t_mono: float
    frame_index: int
    joints: Mapping[str, float] | None = None
    grippers: Mapping[str, float] = field(default_factory=dict)
    poses: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    mode: str = "joints"
    episode_toggle: bool = False


class PoseSource(Protocol):
    """An operator input that yields :class:`PoseFrame` samples."""

    name: str

    def connect(self) -> None:
        """Open the device or stream."""

    def disconnect(self) -> None:
        """Close it; idempotent."""

    @property
    def is_connected(self) -> bool:
        """Whether :meth:`latest` can answer."""

    def latest(self) -> PoseFrame | None:
        """The newest frame, or ``None`` when nothing fresh is available. Never blocks."""


@dataclass
class RetargetOut:
    """What a retarget stage produces for one frame.

    Attributes:
        joints: Follower joint targets, radians, keyed by follower joint name.
        grippers: Open fractions keyed ``left_gripper`` / ``right_gripper``.
        reference: Whatever the encoder stage needs beyond joints (SMPL joints,
            three-point poses); empty when no encoder runs.
    """

    joints: dict[str, float]
    grippers: dict[str, float] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)


class Retarget(Protocol):
    """Turns a :class:`PoseFrame` into follower joint targets."""

    def __call__(self, frame: PoseFrame) -> RetargetOut:
        """Retarget one frame."""


class Encoder(Protocol):
    """Turns a retargeted frame into a latent action (a SONIC token)."""

    dim: int

    def reset(self) -> None:
        """Clear any reference history at an episode boundary."""

    def encode(self, out: RetargetOut) -> list[float]:
        """Return ``dim`` floats for this frame."""


class JointMapRetarget:
    """Rename and scale source joints onto follower joints: ``target = scale * source + offset``.

    The identity map for a source that already speaks the follower's names, and
    the whole retarget for a leader arm or an exoskeleton whose joints have a
    fixed affine relation to the follower's. A source key absent from the table
    is dropped, never forwarded under its own name, so a leader's extra axes do
    not reach the follower.
    """

    def __init__(self, table: Mapping[str, str | tuple[str, float, float]]):
        """Build the map.

        Args:
            table: ``source_name -> target_name`` or ``source_name -> (target_name,
                scale, offset)``.

        Raises:
            ValueError: Two source names map onto one target name.
        """
        self._rows: dict[str, tuple[str, float, float]] = {}
        for source, spec in table.items():
            target, scale, offset = (spec, 1.0, 0.0) if isinstance(spec, str) else spec
            self._rows[source] = (target, float(scale), float(offset))
        targets = [row[0] for row in self._rows.values()]
        duplicates = sorted({name for name in targets if targets.count(name) > 1})
        if duplicates:
            raise ValueError(f"JointMapRetarget: several source joints map onto {duplicates}")

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Follower joint names this map can produce."""
        return tuple(row[0] for row in self._rows.values())

    def __call__(self, frame: PoseFrame) -> RetargetOut:
        """Apply the map to ``frame.joints``; grippers pass through unchanged."""
        joints: dict[str, float] = {}
        for source, value in (frame.joints or {}).items():
            row = self._rows.get(source)
            if row is not None:
                target, scale, offset = row
                joints[target] = scale * float(value) + offset
        return RetargetOut(joints=joints, grippers={k: float(v) for k, v in frame.grippers.items() if k in GRIPPERS})


class MockPoseSource:
    """A scripted operator: sinusoidal arm joints and a slow gripper sweep.

    Deterministic in time so a test can predict what the follower was told.
    Produces joint-space frames keyed by the names given, so paired with an
    identity :class:`JointMapRetarget` it drives any follower whose joints it
    names; the default names are the sim G1's arm actuators.
    """

    name = "mock"

    def __init__(
        self,
        joint_names: tuple[str, ...] = (
            "left_shoulder_pitch_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_elbow_joint",
        ),
        amplitude_rad: float = 0.4,
        period_s: float = 4.0,
        episode_every_s: float | None = None,
    ):
        """Configure the script.

        Args:
            joint_names: Joints the script animates; each gets a phase offset.
            amplitude_rad: Peak excursion of every animated joint.
            period_s: Period of the arcs in seconds.
            episode_every_s: When set, ``episode_toggle`` is raised once each
                interval, for exercising a recorder bridge.
        """
        if not math.isfinite(amplitude_rad) or amplitude_rad <= 0:
            raise ValueError(f"amplitude_rad must be a positive finite number, got {amplitude_rad!r}")
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError(f"period_s must be a positive finite number, got {period_s!r}")
        self._joint_names = tuple(joint_names)
        self._amplitude = float(amplitude_rad)
        self._period = float(period_s)
        self._episode_every = episode_every_s
        self._t0: float | None = None
        self._frames = 0
        self._last_episode_slot = 0

    @property
    def is_connected(self) -> bool:
        """True between :meth:`connect` and :meth:`disconnect`."""
        return self._t0 is not None

    def connect(self) -> None:
        """Start the clock."""
        self._t0 = time.monotonic()
        self._frames = 0
        self._last_episode_slot = 0

    def disconnect(self) -> None:
        """Stop the clock; :meth:`latest` answers ``None`` afterwards."""
        self._t0 = None

    def scripted_joints(self, t: float) -> dict[str, float]:
        """The joint targets the script commands at elapsed time ``t`` (seconds)."""
        phase = 2.0 * math.pi * t / self._period
        return {
            name: self._amplitude * math.sin(phase + index * math.pi / 2.0)
            for index, name in enumerate(self._joint_names)
        }

    def latest(self) -> PoseFrame | None:
        """One frame at the current clock, or ``None`` when disconnected."""
        if self._t0 is None:
            return None
        now = time.monotonic()
        t = now - self._t0
        self._frames += 1
        toggle = False
        if self._episode_every is not None:
            slot = int(t // self._episode_every)
            toggle = slot != self._last_episode_slot
            self._last_episode_slot = slot
        sweep = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / (2.0 * self._period))
        return PoseFrame(
            t_mono=now,
            frame_index=self._frames,
            joints=self.scripted_joints(t),
            grippers={"left_gripper": sweep, "right_gripper": 1.0 - sweep},
            mode="joints",
            episode_toggle=toggle,
        )


class WholeBodyTeleoperator:
    """Source, retarget and optional encoder behind one lerobot-shaped teleoperator.

    ``get_action()`` returns the layout's action dict: joint targets when no
    encoder is set, the 64-D token plus grippers when one is. Either way the
    follower-facing joint targets of the last frame are kept in
    :attr:`last_joint_targets`, so a simulated follower can be driven by them
    while the token is what gets recorded.
    """

    def __init__(
        self,
        source: PoseSource,
        retarget: Retarget,
        encoder: Encoder | None = None,
        layout: str | Layout = "g1_joint_29",
        stale_after_s: float = 0.2,
    ):
        """Compose the stages.

        Args:
            source: Where frames come from.
            retarget: Frame to follower joints.
            encoder: Optional latent encoder; when set the layout must be a
                token layout, and vice versa.
            layout: Name or :class:`~strands_robots.teleop.layouts.Layout`.
            stale_after_s: A frame older than this is not acted on; the
                follower holds and nothing is sent.

        Raises:
            ValueError: Encoder and layout disagree, or the encoder's ``dim`` is
                not the layout's token width.
        """
        self.source = source
        self.retarget = retarget
        self.encoder = encoder
        self.layout = teleop_layout(layout) if isinstance(layout, str) else layout
        if self.layout.is_token != (encoder is not None):
            want = "an encoder" if self.layout.is_token else "no encoder"
            raise ValueError(f"layout {self.layout.name!r} wants {want}; got {'one' if encoder else 'none'}")
        if encoder is not None and int(encoder.dim) != len(self.layout.action_names) - len(
            [n for n in self.layout.action_names if n in GRIPPERS]
        ):
            raise ValueError(f"encoder dim {encoder.dim} does not match layout {self.layout.name!r}")
        if not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError(f"stale_after_s must be a positive finite number, got {stale_after_s!r}")
        self._stale_after = float(stale_after_s)
        self.last_joint_targets: dict[str, float] = {}
        self.last_frame: PoseFrame | None = None
        self.frames_used = 0
        self.frames_stale = 0

    @property
    def name(self) -> str:
        """``wholebody:<source>`` for logs and teleop status lines."""
        return f"wholebody:{self.source.name}"

    @property
    def action_features(self) -> dict[str, type]:
        """The layout's action columns, all floats; what a recorder declares."""
        return dict.fromkeys(self.layout.action_names, float)

    @property
    def is_connected(self) -> bool:
        """The source's connection state."""
        return bool(self.source.is_connected)

    def connect(self, calibrate: bool = True) -> None:
        """Connect the source and reset the encoder. ``calibrate`` is accepted for lerobot parity."""
        del calibrate
        self.source.connect()
        if self.encoder is not None:
            self.encoder.reset()

    def disconnect(self) -> None:
        """Disconnect the source."""
        self.source.disconnect()

    def get_action(self) -> dict[str, float]:
        """One tick: sample, retarget, encode, lay out. Empty when nothing fresh arrived."""
        frame = self.source.latest()
        if frame is None or (time.monotonic() - frame.t_mono) > self._stale_after:
            self.frames_stale += 1
            return {}
        out = self.retarget(frame)
        self.last_frame = frame
        self.last_joint_targets = dict(out.joints)
        self.frames_used += 1
        if self.encoder is None:
            return self.layout.joint_action(out.joints, out.grippers)
        return self.layout.token_action(self.encoder.encode(out), out.grippers)


__all__ = [
    "Encoder",
    "JointMapRetarget",
    "MockPoseSource",
    "PoseFrame",
    "PoseSource",
    "Retarget",
    "RetargetOut",
    "WholeBodyTeleoperator",
]
