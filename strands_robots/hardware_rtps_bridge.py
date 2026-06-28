"""Pure-RTPS (cyclonedds) hardware bridge - a real robot on ROS 2, no rclpy.

This is the rclpy-free sibling of
:class:`strands_robots.hardware_ros_bridge.HardwareRosBridge`. Both derive from
:class:`strands_robots.ros_telemetry.RosTelemetryBase` - the single source of
truth for the ROS 2 topic names and the inbound ``joint_command`` contract - so
they are byte-compatible on the wire by construction. ``HardwareRtpsBridge``
exposes the exact same ROS 2 topics for a physical
:class:`strands_robots.hardware_robot.Robot` - and to real ROS 2 nodes - but
speaks DDS/RTPS directly through the pip-installable ``cyclonedds`` binding
instead of a sourced ROS 2 distro:

* **publish** (outbound) - ``/<robot>/joint_states``
  (``sensor_msgs/msg/JointState``) and, per camera,
  ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``).
* **subscribe** (inbound) - ``/<robot>/joint_command``
  (``sensor_msgs/msg/JointState``) forwarded into
  ``robot.send_action({motor.pos: float})`` over a background poll thread, so an
  external ROS 2 stack can drive the physical arm. Full duplex, same contract as
  the rclpy bridge (shared via :class:`~strands_robots.ros_telemetry.RosTelemetryBase`).

Why this exists alongside ``HardwareRosBridge``: ``rclpy`` needs a *sourced ROS 2
distro* (apt / RoboStack / docker), which is heavy and version-pinned (Humble vs
Jazzy vs Rolling). ``cyclonedds`` is a single self-contained pip wheel that
speaks the RTPS wire protocol every ROS 2 distro shares, so this bridge runs on
a bare dev laptop or a minimal robot image with ``pip install
'strands-robots[ros2]'`` and nothing else. The trade-off is type coverage: RTPS
publishing needs a *local* IDL definition, so only the messages in
:mod:`strands_robots.rtps.idl` work (now ``geometry_msgs`` + the ``sensor_msgs``
``JointState``/``Image`` chain this bridge needs). The rclpy bridge keeps full
``sensor_msgs`` fidelity for anything outside the bundle.

Selection is the hardware ``Robot``'s job (``ros2_transport="rclpy"|"rtps"``);
this module only implements the RTPS path. Both bridges present an identical
``publish_joint_states`` / ``publish_image`` / ``shutdown`` surface so the
``Robot`` telemetry path is transport-agnostic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from strands_robots.ros_telemetry import RosTelemetryBase
from strands_robots.utils import require_optional

if TYPE_CHECKING:
    import numpy as np

    from strands_robots.hardware_robot import Robot

logger = logging.getLogger(__name__)

# ROS 2 type strings this bridge publishes/subscribes. All must be present in
# the RTPS IDL bundle (strands_robots.rtps.idl.REGISTRY).
_JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
_IMAGE_TYPE = "sensor_msgs/msg/Image"


class HardwareRtpsBridge(RosTelemetryBase):
    """Full-duplex hardware ROS 2 bridge over pure RTPS (cyclonedds, no rclpy).

    The rclpy-free sibling of
    :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge`. Both derive
    from :class:`~strands_robots.ros_telemetry.RosTelemetryBase`, so they share
    the topic names and the ``joint_command`` -> ``send_action`` contract and are
    wire-compatible by construction; they differ only in transport (cyclonedds
    RTPS vs rclpy) and in type coverage (bounded by the local IDL bundle).

    Args:
        robot: The hardware ``Robot`` to drive on inbound commands. When
            ``None``, no command surface is created (telemetry-only), mirroring
            the rclpy bridge's pure-publisher mode.
        domain_id: ROS 2 / DDS domain id to publish/subscribe on.
        enable_commands: When True (default) and a ``robot`` is bound, subscribe
            to ``/<robot>/joint_command`` and drive the arm.
        command_robot_name: Topic namespace for the command topic; defaults to
            the bound robot's name (the namespace we publish ``joint_states``
            under).
        poll_period: Seconds between inbound command reads on the poll thread.

    Raises:
        ImportError: If ``cyclonedds`` (the ``[ros2]`` extra) is not installed.
    """

    def __init__(
        self,
        robot: Robot | None = None,
        *,
        domain_id: int = 0,
        enable_commands: bool = True,
        command_robot_name: str | None = None,
        poll_period: float = 0.02,
    ) -> None:
        # cyclonedds is the only dependency - no rclpy, no sourced ROS 2 distro.
        require_optional(
            "cyclonedds",
            extra="ros2",
            purpose="the pure-RTPS hardware bridge (Robot ros2_transport='rtps')",
        )
        from cyclonedds.domain import DomainParticipant

        from strands_robots.rtps.idl import get_type
        from strands_robots.rtps.mangling import dds_topic_name

        self._get_type = get_type
        self._dds_topic_name = dds_topic_name

        self._robot = robot
        self._domain_id = int(domain_id)
        self._participant: Any = DomainParticipant(self._domain_id)

        self._robot_name = self._safe(self._resolve_robot_name(robot) if robot is not None else "robot")
        self._joint_writer: Any = None
        self._image_writers: dict[str, Any] = {}

        # Cache the resolved IDL classes once (KeyError here = the bundle is
        # missing a type, a packaging bug, surfaced at construction not mid-run).
        self._JointState = get_type(_JOINT_STATE_TYPE)
        self._Image = get_type(_IMAGE_TYPE)

        self._enable_commands = bool(enable_commands) and robot is not None
        self._poll_period = float(poll_period)
        self._command_reader: Any = None
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

        if self._enable_commands:
            name = command_robot_name or self._resolve_robot_name(robot)
            self._command_robot_name = self._safe(name)
            self._command_reader = self._make_reader(self.joint_command_topic(name), self._JointState)
            self._start_poll()

    # -- helpers ----------------------------------------------------------

    def _make_writer(self, ros_topic: str, idl_cls: Any) -> Any:
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        topic = Topic(self._participant, self._dds_topic_name(ros_topic), idl_cls)
        return DataWriter(self._participant, topic)

    def _make_reader(self, ros_topic: str, idl_cls: Any) -> Any:
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        topic = Topic(self._participant, self._dds_topic_name(ros_topic), idl_cls)
        return DataReader(self._participant, topic)

    # -- publish (outbound) ----------------------------------------------

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """Publish one ``JointState`` for ``robot`` on ``/<robot>/joint_states``.

        Signature matches ``RosTelemetryBridge.publish_joint_states`` so the
        hardware ``Robot`` telemetry path is transport-agnostic.
        """
        if self._joint_writer is None:
            self._joint_writer = self._make_writer(self.joint_states_topic(robot), self._JointState)
        msg = self._JointState(
            header=self._header(self._safe(robot)),
            name=list(names),
            position=[float(p) for p in positions],
            velocity=[],
            effort=[],
        )
        self._joint_writer.write(msg)

    def publish_image(self, robot: str, camera: str, image: np.ndarray) -> None:
        """Publish one RGB ``Image`` on ``/<robot>/<camera>/image_raw``."""
        if image.ndim != 3 or image.shape[2] != 3:
            return
        key = f"{robot}/{camera}"
        writer = self._image_writers.get(key)
        if writer is None:
            writer = self._make_writer(self.image_topic(robot, camera), self._Image)
            self._image_writers[key] = writer
        height, width = int(image.shape[0]), int(image.shape[1])
        msg = self._Image(
            header=self._header(f"{self._safe(robot)}/{self._safe(camera)}"),
            height=height,
            width=width,
            encoding="rgb8",
            is_bigendian=0,
            step=width * 3,
            data=image.astype("uint8", copy=False).tobytes(),
        )
        writer.write(msg)

    def _header(self, frame_id: str) -> Any:
        """Build a std_msgs/Header with a wall-clock stamp (sec/nanosec)."""
        Header = self._get_type("std_msgs/msg/Header")
        Time = self._get_type("builtin_interfaces/msg/Time")
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        return Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)

    # -- subscribe (inbound) ---------------------------------------------

    def _on_command(self, msg: Any) -> None:
        """Forward an inbound ``joint_command`` JointState to ``send_action``.

        Delegates to :meth:`RosTelemetryBase._drive_from_command` (shared with
        the rclpy bridge): zip ``name``/``position`` into a flat
        ``{motor.pos: float}`` action; reject mismatched messages rather than
        partially applying; surface (never raise) ``send_action`` errors.
        ``skip_empty=True`` because cyclonedds ``take()`` may surface a wholly
        empty sample (DDS dispose / keep-alive) that is not a real actuation
        request, so it is dropped quietly rather than warned on.
        """
        self._drive_from_command(self._robot, msg, skip_empty=True)

    def _poll_loop(self) -> None:
        """Poll the command reader and dispatch new samples to ``_on_command``.

        cyclonedds has no rclpy-style executor; we ``take()`` available samples
        each tick. ``take`` (not ``read``) so each command is delivered once.
        """
        while not self._stop.is_set():
            try:
                for sample in self._command_reader.take(N=10):
                    self._on_command(sample)
            except Exception:
                logger.debug("HardwareRtpsBridge: command poll raised", exc_info=True)
            self._stop.wait(self._poll_period)

    def _start_poll(self) -> None:
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name=f"{self._command_robot_name}_rtps_cmd",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(
            "HardwareRtpsBridge: driving %r from /%s/joint_command (cyclonedds, no rclpy)",
            self._command_robot_name,
            self._command_robot_name,
        )

    # -- lifecycle --------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the poll thread and drop DDS entities. Idempotent."""
        self._stop.set()
        thread = self._poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._poll_thread = None
        # Dropping references lets cyclonedds reclaim the readers/writers and
        # the participant; there is no explicit close() in the python binding.
        self._command_reader = None
        self._joint_writer = None
        self._image_writers = {}
        self._participant = None

    def __repr__(self) -> str:
        return (
            f"HardwareRtpsBridge(robot={self._robot_name!r}, domain_id={self._domain_id}, "
            f"enable_commands={self._enable_commands})"
        )
