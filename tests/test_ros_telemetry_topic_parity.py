"""The rclpy and RTPS hardware bridges are one symmetric pair on the ROS 2 graph.

A real arm exposed over rclpy (``HardwareRosBridge``) and the same arm exposed
over pure RTPS/cyclonedds (``HardwareRtpsBridge``) must advertise byte-identical
topics so an external ROS 2 node cannot tell the transports apart. These tests
pin the *structural* guarantee behind that promise: both transports - and the
sim bridge - derive from :class:`RosTelemetryBase`, the single source of truth
for topic names and the inbound ``joint_command`` contract, so the two
codepaths cannot drift as the contract evolves.

These tests need neither rclpy nor cyclonedds: they exercise the
transport-agnostic base directly.
"""

from __future__ import annotations

import pytest

from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
from strands_robots.ros_telemetry import RosTelemetryBase, RosTelemetryBridge
from strands_robots.simulation.ros_bridge import SimRosBridge


def test_every_telemetry_bridge_derives_from_the_shared_base() -> None:
    # The structural guarantee: one base owns the wire contract for both
    # transports (and the sim bridge), so they cannot diverge.
    for cls in (RosTelemetryBridge, HardwareRosBridge, SimRosBridge, HardwareRtpsBridge):
        assert issubclass(cls, RosTelemetryBase), f"{cls.__name__} must derive from RosTelemetryBase"


@pytest.mark.parametrize(
    "robot,camera",
    [
        ("so101", "wrist"),
        ("arm 1", "cam/2"),  # characters needing sanitization
        ("--lead--", "front cam"),
        ("", ""),  # degenerate -> both fall back identically
    ],
)
def test_topic_names_identical_across_transports(robot: str, camera: str) -> None:
    # rclpy and RTPS resolve the same wire names for every input, because both
    # inherit the same classmethods from the base.
    assert RosTelemetryBridge.joint_states_topic(robot) == HardwareRtpsBridge.joint_states_topic(robot)
    assert RosTelemetryBridge.image_topic(robot, camera) == HardwareRtpsBridge.image_topic(robot, camera)
    assert RosTelemetryBridge.joint_command_topic(robot) == HardwareRtpsBridge.joint_command_topic(robot)


def test_topic_name_format_is_the_documented_contract() -> None:
    assert RosTelemetryBase.joint_states_topic("so101") == "/so101/joint_states"
    assert RosTelemetryBase.image_topic("so101", "wrist") == "/so101/wrist/image_raw"
    assert RosTelemetryBase.joint_command_topic("so101") == "/so101/joint_command"
    # Unsafe segments are sanitized to alnum/underscore and never leak.
    assert RosTelemetryBase.joint_states_topic("a/b 1") == "/a_b_1/joint_states"


class _Msg:
    def __init__(self, name: list[str], position: list[float]) -> None:
        self.name = name
        self.position = position


def test_command_action_shared_contract() -> None:
    base = RosTelemetryBase()
    assert base._command_action(_Msg(["a", "b"], [0.1, 0.2])) == {"a": 0.1, "b": 0.2}
    # Length mismatch is rejected (returns None) rather than partially applied.
    assert base._command_action(_Msg(["a", "b"], [0.1])) is None
    # An empty message is rejected; with skip_empty it is dropped silently.
    assert base._command_action(_Msg([], [])) is None
    assert base._command_action(_Msg([], []), skip_empty=True) is None


def test_drive_from_command_dispatches_and_surfaces_errors() -> None:
    base = RosTelemetryBase()

    sent: list[dict[str, float]] = []

    class _OkRobot:
        def send_action(self, action: dict[str, float]) -> dict[str, str]:
            sent.append(action)
            return {"status": "success"}

    base._drive_from_command(_OkRobot(), _Msg(["a"], [0.5]))
    assert sent == [{"a": 0.5}]

    # A raising send_action is surfaced (logged), never propagated.
    class _BoomRobot:
        def send_action(self, action: dict[str, float]) -> dict[str, str]:
            raise RuntimeError("bus fault")

    base._drive_from_command(_BoomRobot(), _Msg(["a"], [0.5]))  # must not raise
