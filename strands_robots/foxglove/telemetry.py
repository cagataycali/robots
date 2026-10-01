"""Fan one telemetry publish out to several bridges (a ROS 2 bridge and a Foxglove bridge at once).

Both engines hold ONE ``_ros_bridge`` slot and call three methods on it. When
an operator asks for ``ros2_bridge=True`` and ``foxglove=True`` together, the
slot holds a :class:`TelemetryFanout` and each member sees every call. A member
that raises does not silence the others; the engine's own per-step guard logs
the failure.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TelemetryFanout:
    """The three-method telemetry interface, forwarded to every member in order.

    Args:
        members: Bridges exposing ``publish_joint_states``, ``publish_image``
            and ``shutdown``. At least two; one bridge needs no fanout.
    """

    def __init__(self, members: list[Any]) -> None:
        if len(members) < 2:
            raise ValueError("TelemetryFanout needs at least two bridges; hand a single bridge to the engine directly.")
        self.members = list(members)

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """Forward one joint-state publish to every member."""
        for member in self.members:
            member.publish_joint_states(robot, names, positions)

    def publish_image(self, robot: str, key: str, frame: Any) -> None:
        """Forward one camera frame to every member."""
        for member in self.members:
            member.publish_image(robot, key, frame)

    def wants_images(self) -> bool:
        """True when any member wants a frame now (a member without the hint always does)."""
        return any(getattr(member, "wants_images", lambda: True)() for member in self.members)

    def shutdown(self) -> None:
        """Shut every member down; a member that fails does not stop the rest."""
        failures: list[BaseException] = []
        for member in self.members:
            try:
                member.shutdown()
            except Exception as exc:  # noqa: BLE001 - every member must get its shutdown
                failures.append(exc)
                logger.warning("telemetry bridge %r did not shut down cleanly: %s", member, exc)
        if failures:
            raise failures[0]
