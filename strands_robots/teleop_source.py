"""Turn a robot into its own teleop source.

Why this module exists: ``start_teleop_publish`` takes a lerobot
``Teleoperator`` object, and the mesh has no verb that can create one -- so a
dashboard (or any remote caller) could tell a follower to SUBSCRIBE to a leader
stream, but nothing on the mesh could ever make that stream exist. The follower
then waited out its subscribe budget and answered with a shrug. The teleop chain
was half-built: receive without publish.

Constructing a second ``Teleoperator`` inside the process that already drives the
leader arm is not the answer either. It would open a SECOND serial connection to
the same port and collide with every existing reader on that bus (the state
probe, the sensors probe, the camera publisher) -- the exact "Port is in use!"
failure that :mod:`strands_robots.bus_access` exists to prevent.

An arm that is already spawned as a ``Robot`` is a perfectly good leader as it
is: its measured joint positions ARE the action stream, they come off the wire
through the shared bus lock, and ``get_observation`` already names them the way
``send_action`` expects (``shoulder_pan.pos``). So the leader publishes ITSELF.

This is read-only on the leader: nothing here writes to a bus or moves anything.
The follower moves only when someone separately points it at this stream with
``teleop_receive``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from strands_robots.bus_access import read_joints
from strands_robots.utils import partial_construction_repr

logger = logging.getLogger(__name__)

__all__ = ["RobotAsTeleoperator", "positions_from_observation"]


def positions_from_observation(obs: Any) -> dict[str, float]:
    """The joint-position half of an observation, as a teleop action dict.

    An observation also carries camera frames (numpy arrays) and, depending on
    the driver, velocities and effort. Only ``*.pos`` travels: that is what a
    follower's ``send_action`` consumes, and shipping a 4K frame at 50Hz down the
    input topic would be an accident, not a feature.

    A joint whose value is not a finite number is DROPPED rather than published
    as ``NaN``: a follower that receives NaN either refuses the frame or drives
    somewhere undefined, and both are worse than a frame with one joint missing.

    Args:
        obs: Whatever the driver's ``get_observation()`` returned.

    Returns:
        ``{joint_key: float}`` for every finite ``*.pos`` entry.
    """
    if not hasattr(obs, "items"):
        return {}
    out: dict[str, float] = {}
    for key, value in obs.items():
        if not isinstance(key, str) or not key.endswith(".pos"):
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[key] = f
    return out


class RobotAsTeleoperator:
    """Adapter making a robot host publishable as a teleop source.

    Satisfies the only contract :class:`~strands_robots.mesh.input.InputPublisher`
    requires -- ``get_action() -> dict`` -- by reading the host's own joints.

    Args:
        host: The robot host (hardware ``Robot`` or sim). Its inner lerobot
            driver is used when present, so this reader shares the SAME bus lock
            as the state probe and the camera publisher.
        robot_name: Sim only: which robot in the world is the leader.
    """

    def __init__(self, host: Any, robot_name: str | None = None) -> None:
        self.host = host
        self.robot_name = robot_name
        #: Frames whose read raised, and frames that carried no joint at all.
        #: Kept apart: a wire that is failing and an arm that reports nothing
        #: need different fixes, and "0 frames published" says neither.
        self.read_errors = 0
        self.empty_reads = 0

    def __repr__(self) -> str:
        # A publisher is constructed on the refusal path as often as the happy
        # one - start_teleop_publish_self probes the host and raises when it has
        # no joints - so this repr is read from a traceback holding a half-built
        # instance. Reading self.host there would raise a SECOND AttributeError
        # that names an attribute unrelated to the refusal being investigated.
        try:
            return f"RobotAsTeleoperator(host={type(self.host).__name__}, robot_name={self.robot_name!r})"
        except AttributeError:
            return partial_construction_repr(self)

    # The device whose bus lock protects this read. Hardware hosts wrap a lerobot
    # driver in .robot; a sim host is its own device.
    def _device(self) -> Any:
        inner = getattr(self.host, "robot", None)
        if inner is not None and hasattr(inner, "get_observation"):
            return inner
        return self.host

    def get_action(self) -> dict[str, float]:
        """One frame of the leader's measured positions.

        Never raises: the publish loop counts an exception as an error frame and
        keeps going, and a leader that goes quiet for one frame must not tear
        down a live teleop session.
        """
        device = self._device()
        if not hasattr(device, "get_observation"):
            self.read_errors += 1
            return {}
        try:
            # Joints only. A leader arm with a broken camera must still be able to
            # drive a follower: get_observation() would have discarded the joint
            # positions it had already read the moment a frame grab raised.
            obs = read_joints(device)
        except Exception as exc:  # noqa: BLE001 - one bad frame is not a session
            self.read_errors += 1
            logger.debug("[teleop] leader read failed: %r", exc)
            return {}
        action = positions_from_observation(obs)
        if not action:
            self.empty_reads += 1
        return action

    @property
    def stats(self) -> dict[str, Any]:
        """Counters for the status verb, so a silent stream is diagnosable."""
        return {
            "source": "self",
            "robot_name": self.robot_name,
            "read_errors": self.read_errors,
            "empty_reads": self.empty_reads,
        }
