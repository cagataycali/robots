"""Behavior tests for the hardware ROS 2 telemetry bridge.

``rclpy`` is a system-provided, non-PyPI dependency, so these tests inject a
fake ``rclpy`` + ``sensor_msgs.msg`` into ``sys.modules`` to exercise the
publisher wiring with NO ROS 2 installed - the same approach the simulation
bridge tests use. They assert that:

* :class:`HardwareRosBridge` is a thin, symmetric sibling of
  :class:`SimRosBridge`: both subclass
  :class:`~strands_robots.ros_telemetry.RosTelemetryBridge` and produce
  identical topics/messages, differing only in their default node name.
* The hardware :class:`~strands_robots.hardware_robot.Robot` publishes its live
  observation (joint scalars -> ``joint_states``, ``(H, W, 3)`` arrays ->
  per-camera ``image_raw``) when ``ros2_bridge=True``, and is a no-op when the
  bridge is disabled.
* ``publish_ros_observation`` reads the robot once and publishes on demand, and
  reports a clean error (never raises) when the bridge is off.
* Enabling the bridge with no ``rclpy`` raises a clear :class:`ImportError`.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import strands_robots.utils as utils_mod
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.ros_telemetry import RosTelemetryBridge
from strands_robots.simulation.ros_bridge import SimRosBridge


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.messages.append(msg)


class _FakeClock:
    def now(self) -> Any:
        return self

    def to_msg(self) -> str:
        return "stamp"


class _FakeSubscription:
    def __init__(self, topic: str, callback: Any) -> None:
        self.topic = topic
        self.callback = callback


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.publishers: list[_FakePublisher] = []
        self.subscriptions: list[_FakeSubscription] = []
        self.destroyed = False

    def get_clock(self) -> _FakeClock:
        return _FakeClock()

    def create_publisher(self, _msg_type: Any, topic: str, _depth: int) -> _FakePublisher:
        pub = _FakePublisher(topic)
        self.publishers.append(pub)
        return pub

    def create_subscription(self, _msg_type: Any, topic: str, callback: Any, _depth: int) -> _FakeSubscription:
        sub = _FakeSubscription(topic, callback)
        self.subscriptions.append(sub)
        return sub

    def destroy_node(self) -> None:
        self.destroyed = True


class _Header:
    def __init__(self) -> None:
        self.stamp: Any = None
        self.frame_id: str = ""


class _JointState:
    def __init__(self) -> None:
        self.header = _Header()
        self.name: list[str] = []
        self.position: list[float] = []


class _Image:
    def __init__(self) -> None:
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""


@pytest.fixture
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake rclpy + sensor_msgs.msg and clear require_optional's cache."""
    state: dict[str, Any] = {"inited": False, "shutdown": False, "nodes": [], "spun": 0}

    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: state["inited"]  # type: ignore[attr-defined]

    def _init() -> None:
        state["inited"] = True

    def _shutdown() -> None:
        state["shutdown"] = True
        state["inited"] = False

    def _create_node(name: str) -> _FakeNode:
        node = _FakeNode(name)
        state["nodes"].append(node)
        return node

    def _spin_once(_node: Any, timeout_sec: float = 0.0) -> None:
        state["spun"] += 1

    rclpy.init = _init  # type: ignore[attr-defined]
    rclpy.shutdown = _shutdown  # type: ignore[attr-defined]
    rclpy.create_node = _create_node  # type: ignore[attr-defined]
    rclpy.spin_once = _spin_once  # type: ignore[attr-defined]

    sensor_pkg = ModuleType("sensor_msgs")
    sensor_msg = ModuleType("sensor_msgs.msg")
    sensor_msg.JointState = _JointState  # type: ignore[attr-defined]
    sensor_msg.Image = _Image  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_pkg)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msg)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    return state


class _FakeLeRobot:
    """Minimal lerobot Robot: serves an observation and records actions.

    Supports the inbound (drive) path too so the full-duplex end-to-end test can
    push a ``joint_command`` through ``HwRobot.send_action`` into the device.
    """

    def __init__(self, observation: dict[str, Any]) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self._obs = observation
        self.is_connected = True
        self.sent_actions: list[dict[str, Any]] = []

    def get_observation(self) -> dict[str, Any]:
        return self._obs

    def connect(self, _calibrate: bool = False) -> None:
        self.is_connected = True

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


# Every Robot built by ``_make_robot`` is tracked here so an autouse teardown can
# stop its command spin thread even when a test does not shut the robot down
# itself. The fake ``rclpy.spin_once`` returns immediately, so a command bridge
# left running busy-spins a daemon thread for the rest of the session; a few
# such leaks starve slower, CPU-bound tests on a loaded runner. Unconditional
# teardown keeps the suite hermetic regardless of any single test's omission.
_BUILT_ROBOTS: list[HwRobot] = []


def _shutdown_built_robots() -> None:
    """Tear down every robot built via ``_make_robot`` (stops command threads)."""
    while _BUILT_ROBOTS:
        _BUILT_ROBOTS.pop()._shutdown_ros_bridge()


def _live_command_threads() -> int:
    """Count live hardware-bridge command spin threads (named ``*_ros_cmd``)."""
    return sum(1 for t in threading.enumerate() if t.is_alive() and t.name.endswith("_ros_cmd"))


@pytest.fixture(autouse=True)
def _reap_bridge_threads() -> Any:
    """Reap any command spin thread a test forgot to shut down."""
    yield
    _shutdown_built_robots()


def _make_robot(observation: dict[str, Any], *, ros2_bridge: bool = False, ros2_domain: int = 0) -> HwRobot:
    """Build a Robot via __new__ and wire only what the bridge path touches."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.data_config = None
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = _FakeLeRobot(observation)
    hw._init_ros_bridge(ros2_bridge=ros2_bridge, ros2_domain=ros2_domain)
    _BUILT_ROBOTS.append(hw)
    return hw


# --- bridge symmetry --------------------------------------------------------


def test_bridges_are_symmetric_subclasses() -> None:
    assert issubclass(SimRosBridge, RosTelemetryBridge)
    assert issubclass(HardwareRosBridge, RosTelemetryBridge)
    assert SimRosBridge.default_node_name == "strands_sim"
    assert HardwareRosBridge.default_node_name == "strands_hardware"


def test_hardware_bridge_default_node_name(fake_ros: dict[str, Any]) -> None:
    HardwareRosBridge()
    assert fake_ros["nodes"][0].name == "strands_hardware"


def test_hardware_bridge_publishes_identically_to_sim(fake_ros: dict[str, Any]) -> None:
    """Same robot/joints -> byte-identical JointState topic + fields on both."""
    hw = HardwareRosBridge()
    sim = SimRosBridge()
    hw.publish_joint_states("so101", ["shoulder_pan", "elbow"], [0.1, 0.2])
    sim.publish_joint_states("so101", ["shoulder_pan", "elbow"], [0.1, 0.2])

    (hw_pub,) = fake_ros["nodes"][0].publishers
    (sim_pub,) = fake_ros["nodes"][1].publishers
    assert hw_pub.topic == sim_pub.topic == "/so101/joint_states"
    (hw_msg,) = hw_pub.messages
    (sim_msg,) = sim_pub.messages
    assert hw_msg.name == sim_msg.name == ["shoulder_pan", "elbow"]
    assert hw_msg.position == sim_msg.position == [0.1, 0.2]


class TestJointStateArraysAreRefusedWhenTheyDisagree:
    """A ``JointState`` whose two arrays name different joints never reaches the wire.

    ``name`` and ``position`` are paired by index, so a caller that supplies a
    different number of each has no pose to publish: a consumer's
    ``zip(name, position)`` reports every joint after the gap under its
    neighbour's name and the tail unreported. Dropped whole, with the reason
    logged, for the reason :meth:`_command_action` refuses a malformed inbound
    command whole rather than applying part of it.
    """

    @pytest.mark.parametrize(
        ("names", "positions"),
        [
            (["hip", "knee", "ankle"], [0.1, 0.2]),  # a value column short by one
            (["hip", "knee"], [0.1, 0.2, 0.3]),  # a name column short by one
            (["hip"], []),  # every value missing
        ],
    )
    def test_a_mismatched_pair_publishes_nothing(
        self, fake_ros: dict[str, Any], caplog: pytest.LogCaptureFixture, names: list[str], positions: list[float]
    ) -> None:
        bridge = HardwareRosBridge()
        with caplog.at_level(logging.WARNING):
            bridge.publish_joint_states("so101", names, positions)

        assert [m for pub in fake_ros["nodes"][0].publishers for m in pub.messages] == []
        # The reason names both counts, so the caller can see which column is short.
        assert f"{len(names)} joint name(s) and {len(positions)} position(s)" in caplog.text
        assert "no partial publication" in caplog.text

    def test_a_matching_pair_still_publishes(self, fake_ros: dict[str, Any]) -> None:
        # Control: the guard refuses no more than the mismatch.
        bridge = HardwareRosBridge()
        bridge.publish_joint_states("so101", ["hip", "knee"], [0.1, 0.2])

        (pub,) = fake_ros["nodes"][0].publishers
        (msg,) = pub.messages
        assert msg.name == ["hip", "knee"]
        assert msg.position == [0.1, 0.2]

    def test_two_empty_arrays_are_a_valid_empty_state(self, fake_ros: dict[str, Any]) -> None:
        # Control: JointState allows empty arrays; equal lengths agree.
        bridge = HardwareRosBridge()
        bridge.publish_joint_states("so101", [], [])

        (pub,) = fake_ros["nodes"][0].publishers
        (msg,) = pub.messages
        assert msg.name == [] and msg.position == []


def test_hardware_bridge_sets_domain_env(fake_ros: dict[str, Any]) -> None:
    import os

    HardwareRosBridge(domain_id=11)
    assert os.environ["ROS_DOMAIN_ID"] == "11"


# --- Robot wiring -----------------------------------------------------------


def test_robot_publishes_joint_states_and_images(fake_ros: dict[str, Any]) -> None:
    obs = {
        "shoulder_pan.pos": 0.5,
        "elbow.pos": -0.2,
        "front": np.zeros((4, 5, 3), dtype=np.uint8),
        "enabled": True,  # bool must NOT be treated as a joint scalar
    }
    hw = _make_robot(obs, ros2_bridge=True, ros2_domain=3)
    hw._publish_ros_telemetry(hw.robot.get_observation())

    node = fake_ros["nodes"][0]
    joint_pub = next(p for p in node.publishers if p.topic == "/fake_arm/joint_states")
    image_pub = next(p for p in node.publishers if p.topic == "/fake_arm/front/image_raw")
    (joint_msg,) = joint_pub.messages
    # Sorted joint keys; bool 'enabled' excluded.
    assert joint_msg.name == ["elbow.pos", "shoulder_pan.pos"]
    assert joint_msg.position == [-0.2, 0.5]
    (image_msg,) = image_pub.messages
    assert (image_msg.height, image_msg.width, image_msg.encoding) == (4, 5, "rgb8")


def test_robot_skip_images_publishes_joints_only(fake_ros: dict[str, Any]) -> None:
    obs = {"j0.pos": 0.0, "cam": np.zeros((2, 2, 3), dtype=np.uint8)}
    hw = _make_robot(obs, ros2_bridge=True)
    hw._publish_ros_telemetry(hw.robot.get_observation(), skip_images=True)
    topics = {p.topic for p in fake_ros["nodes"][0].publishers}
    assert topics == {"/fake_arm/joint_states"}


def test_robot_telemetry_noop_when_disabled() -> None:
    """Bridge off: publishing is a silent no-op and creates no rclpy node."""
    hw = _make_robot({"j0.pos": 1.0}, ros2_bridge=False)
    assert hw._ros_bridge is None
    hw._publish_ros_telemetry(hw.robot.get_observation())  # must not raise


def test_robot_telemetry_noop_without_bridge_attr() -> None:
    """A Robot built without _init_ros_bridge still no-ops (getattr guard)."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "bare"
    hw.robot = _FakeLeRobot({"j0.pos": 1.0})
    hw._publish_ros_telemetry({"j0.pos": 1.0})  # no _ros_bridge attribute


def test_publish_ros_observation_on_demand(fake_ros: dict[str, Any]) -> None:
    hw = _make_robot({"j0.pos": 0.7}, ros2_bridge=True, ros2_domain=2)
    result = hw.publish_ros_observation()
    assert result["status"] == "success"
    joint_pub = next(p for p in fake_ros["nodes"][0].publishers if p.topic == "/fake_arm/joint_states")
    assert joint_pub.messages[-1].position == [0.7]


def test_publish_ros_observation_errors_when_disabled() -> None:
    hw = _make_robot({"j0.pos": 0.7}, ros2_bridge=False)
    result = hw.publish_ros_observation()
    assert result["status"] == "error"
    assert "ros2_bridge=True" in result["content"][0]["text"]


def test_shutdown_destroys_node_and_is_idempotent(fake_ros: dict[str, Any]) -> None:
    hw = _make_robot({"j0.pos": 0.0}, ros2_bridge=True)
    node = fake_ros["nodes"][0]
    hw._shutdown_ros_bridge()
    assert node.destroyed is True
    assert hw._ros_bridge is None
    hw._shutdown_ros_bridge()  # idempotent


def test_enabling_bridge_without_rclpy_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """No fake_ros fixture: require_optional cannot find rclpy -> ImportError."""
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    monkeypatch.setitem(sys.modules, "rclpy", None)
    with pytest.raises(ImportError):
        _make_robot({"j0.pos": 0.0}, ros2_bridge=True)


def _raise_lerobot_missing(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Stand-in for _initialize_robot in an environment without the [lerobot] extra."""
    raise ImportError("No module named 'lerobot'")


def test_rclpy_importerror_beats_lerobot_when_both_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ros2_bridge=True must report the rclpy [ros2] hint, not a lerobot error.

    Regression for the constructor ordering bug: _initialize_robot (which imports
    lerobot) used to run BEFORE the ROS 2 dependency check, so a fresh install
    without the [lerobot] extra surfaced "No module named 'lerobot'" and masked
    the documented "pip install 'strands-robots[ros2]'" hint the operator who set
    ros2_bridge=True actually needs. The precondition check must win.
    """
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.setattr(HwRobot, "_initialize_robot", _raise_lerobot_missing)
    with pytest.raises(ImportError) as exc:
        HwRobot(tool_name="arm", robot="so101", ros2_bridge=True)
    message = str(exc.value)
    assert "rclpy" in message
    assert "strands-robots[ros2]" in message
    assert "lerobot" not in message


def test_cyclonedds_importerror_beats_lerobot_for_rtps_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """ros2_transport='rtps' must report the cyclonedds [ros2] hint, not lerobot."""
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    monkeypatch.setitem(sys.modules, "cyclonedds", None)
    monkeypatch.setattr(HwRobot, "_initialize_robot", _raise_lerobot_missing)
    with pytest.raises(ImportError) as exc:
        HwRobot(tool_name="arm", robot="so101", ros2_bridge=True, ros2_transport="rtps")
    message = str(exc.value)
    assert "cyclonedds" in message
    assert "strands-robots[ros2]" in message
    assert "lerobot" not in message


def test_invalid_transport_rejected_before_lerobot_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid ros2_transport raises ValueError before _initialize_robot runs."""
    monkeypatch.setattr(HwRobot, "_initialize_robot", _raise_lerobot_missing)
    with pytest.raises(ValueError, match="ros2_transport"):
        HwRobot(tool_name="arm", robot="so101", ros2_bridge=True, ros2_transport="zenoh")


# --- full duplex: inbound /<robot>/joint_command -> send_action -------------


class _FakeDrivableRobot:
    """Hardware-Robot stand-in that records send_action calls (inbound path)."""

    def __init__(self, *, name: str = "so101", reject: bool = False) -> None:
        self.tool_name_str = "test_arm"
        self.robot = _FakeLeRobot({})
        self.robot.name = name
        self.sent_actions: list[dict[str, Any]] = []
        self._reject = reject

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.sent_actions.append(action)
        if self._reject:
            return {"status": "error", "content": [{"text": "rejected"}]}
        return {"status": "success", "content": [{"text": "ok"}]}


def test_bound_bridge_subscribes_to_joint_command(fake_ros: dict[str, Any]) -> None:
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot)  # type: ignore[arg-type]
    sub = next(s for s in fake_ros["nodes"][0].subscriptions if s.topic == "/so101/joint_command")
    assert sub is not None
    bridge.shutdown()


def test_joint_command_drives_send_action(fake_ros: dict[str, Any]) -> None:
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot)  # type: ignore[arg-type]
    sub = next(s for s in fake_ros["nodes"][0].subscriptions if s.topic == "/so101/joint_command")

    cmd = _JointState()
    cmd.name = ["shoulder_pan.pos", "elbow.pos"]
    cmd.position = [0.5, -0.5]
    sub.callback(cmd)

    assert robot.sent_actions == [{"shoulder_pan.pos": 0.5, "elbow.pos": -0.5}]
    bridge.shutdown()


def test_joint_limits_reject_out_of_range_command_whole(fake_ros: dict[str, Any]) -> None:
    # joint_limits threaded into the rclpy bridge are enforced by the shared
    # base: any out-of-range joint rejects the WHOLE command (no partial apply).
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot, joint_limits={"shoulder_pan.pos": (-1.0, 1.0)})  # type: ignore[arg-type]
    sub = next(s for s in fake_ros["nodes"][0].subscriptions if s.topic == "/so101/joint_command")

    over = _JointState()
    over.name = ["shoulder_pan.pos", "elbow.pos"]
    over.position = [5.0, 0.0]  # shoulder_pan out of [-1, 1]
    sub.callback(over)
    assert robot.sent_actions == []

    ok = _JointState()
    ok.name = ["shoulder_pan.pos", "elbow.pos"]
    ok.position = [0.5, 9.0]  # elbow has no declared bound -> unconstrained
    sub.callback(ok)
    assert robot.sent_actions == [{"shoulder_pan.pos": 0.5, "elbow.pos": 9.0}]
    bridge.shutdown()


def test_invalid_joint_limits_raise_at_construction(fake_ros: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="min .* > max"):
        HardwareRosBridge(_FakeDrivableRobot(), joint_limits={"j0.pos": (1.0, -1.0)})  # type: ignore[arg-type]


def test_joint_command_length_mismatch_is_ignored(fake_ros: dict[str, Any]) -> None:
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot)  # type: ignore[arg-type]
    sub = next(s for s in fake_ros["nodes"][0].subscriptions if s.topic == "/so101/joint_command")

    bad = _JointState()
    bad.name = ["j0.pos", "j1.pos"]
    bad.position = [0.5]  # mismatched length
    sub.callback(bad)

    assert robot.sent_actions == []  # never partially applied
    bridge.shutdown()


def test_read_only_bridge_creates_no_subscription(fake_ros: dict[str, Any]) -> None:
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot, enable_commands=False)  # type: ignore[arg-type]
    assert fake_ros["nodes"][0].subscriptions == []
    bridge.shutdown()


def test_pure_publisher_bridge_has_no_command_surface(fake_ros: dict[str, Any]) -> None:
    """No robot bound -> telemetry-only, symmetric with the sim sibling."""
    bridge = HardwareRosBridge()  # robot=None
    assert fake_ros["nodes"][0].subscriptions == []
    bridge.shutdown()


def test_command_subscription_torn_down_on_shutdown(fake_ros: dict[str, Any]) -> None:
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot)  # type: ignore[arg-type]
    node = fake_ros["nodes"][0]
    bridge.shutdown()
    assert node.destroyed is True
    assert bridge._spin_thread is None
    assert bridge._command_sub is None
    bridge.shutdown()  # idempotent


def test_robot_with_commands_drives_arm_end_to_end(fake_ros: dict[str, Any]) -> None:
    """Full duplex through the Robot: inbound joint_command reaches send_action."""
    hw = _make_robot({"j0.pos": 0.0, "j1.pos": 0.0}, ros2_bridge=True, ros2_domain=4)
    node = fake_ros["nodes"][0]
    # Robot publishes joint_states AND subscribes to joint_command (duplex).
    sub = next(s for s in node.subscriptions if s.topic == "/fake_arm/joint_command")

    cmd = _JointState()
    cmd.name = ["j0.pos", "j1.pos"]
    cmd.position = [0.25, 0.75]
    sub.callback(cmd)

    # The action reached the underlying lerobot device via Robot.send_action.
    assert hw.robot.sent_actions == [{"j0.pos": 0.25, "j1.pos": 0.75}]
    hw._shutdown_ros_bridge()


def test_make_robot_command_bridges_do_not_leak_spin_threads(fake_ros: dict[str, Any]) -> None:
    """A bridge robot left un-shut-down must not leak its command spin thread.

    ``ros2_commands`` defaults to True, so ``Robot(ros2_bridge=True)`` starts a
    daemon thread that services inbound ``joint_command`` callbacks. Under the
    fake ``rclpy`` whose ``spin_once`` returns immediately, that thread busy-spins
    until its stop event is set. If a test builds such a robot and never shuts it
    down, the thread would run for the rest of the session and starve slower,
    CPU-bound tests on a loaded runner. The autouse teardown must reap every
    robot built via ``_make_robot`` so one test's omission cannot leak threads.
    """
    before = _live_command_threads()
    for _ in range(3):
        # commands on by default -> each starts a command spin thread, and none
        # of these are shut down explicitly here.
        _make_robot({"j0.pos": 0.0}, ros2_bridge=True)
    assert _live_command_threads() == before + 3

    # The teardown the autouse fixture runs must stop every tracked thread.
    _shutdown_built_robots()
    deadline = time.monotonic() + 2.0
    while _live_command_threads() > before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _live_command_threads() == before


# --- Robot() threading of joint_limits / dds_security_config ----------------


def test_joint_limits_threaded_into_rclpy_bridge(fake_ros: dict[str, Any]) -> None:
    hw = _make_robot({"j0.pos": 0.0})
    hw._init_ros_bridge(ros2_bridge=True, ros2_transport="rclpy", joint_limits={"j0.pos": (-2.0, 2.0)})
    assert hw._ros_bridge._joint_limits == {"j0.pos": (-2.0, 2.0)}


def test_dds_security_config_rejected_for_rclpy_transport(fake_ros: dict[str, Any]) -> None:
    hw = _make_robot({"j0.pos": 0.0})
    with pytest.raises(ValueError, match="only supported with ros2_transport='rtps'"):
        hw._init_ros_bridge(
            ros2_bridge=True,
            ros2_transport="rclpy",
            dds_security_config={"identity_ca": "file:ca.pem"},
        )


def test_inert_safety_config_without_bridge_is_rejected(fake_ros: dict[str, Any]) -> None:
    hw = _make_robot({"j0.pos": 0.0})
    with pytest.raises(ValueError, match="require ros2_bridge=True"):
        hw._init_ros_bridge(ros2_bridge=False, joint_limits={"j0.pos": (-1.0, 1.0)})


def test_init_ros_bridge_rejects_unknown_transport(fake_ros: dict[str, Any]) -> None:
    """``_init_ros_bridge`` guards the transport itself, not just ``__init__``.

    ``__init__`` validates the transport up front via ``_check_ros2_bridge_deps``,
    but ``_init_ros_bridge`` is a public plain method the test doubles (and any
    future re-init path) call directly. It must reject an unknown transport with
    the documented message before it tries to import a bridge backend, rather
    than fail later with an opaque ImportError.
    """
    hw = _make_robot({"j0.pos": 0.0})
    with pytest.raises(ValueError, match="must be 'rclpy' or 'rtps'"):
        hw._init_ros_bridge(ros2_bridge=True, ros2_transport="zenoh")


# --- inbound command spin loop ----------------------------------------------
# The command spin thread runs in a background daemon that coverage cannot
# trace, so these tests drive `_spin_loop` synchronously to pin its documented
# contract: it services `spin_once` until the stop event is set, and a single
# transient `spin_once` failure must not kill the loop.


def test_command_spin_loop_services_callbacks_until_stopped(fake_ros: dict[str, Any]) -> None:
    # Telemetry-only bridge starts no thread, so _spin_loop can be driven in
    # the (traced) test thread with no race against a live daemon.
    bridge = HardwareRosBridge(enable_commands=False)
    bridge._spin_period = 0.001
    calls: list[tuple[Any, float]] = []

    def spin_once(node: Any, timeout_sec: float = 0.0) -> None:
        calls.append((node, timeout_sec))
        if len(calls) >= 3:
            bridge._stop.set()

    bridge._rclpy.spin_once = spin_once  # type: ignore[attr-defined]
    bridge._stop.clear()
    bridge._spin_loop()

    # Serviced exactly until stop was signalled, always against our node and at
    # the configured spin period.
    assert len(calls) == 3
    assert calls[0][0] is bridge._node
    assert calls[0][1] == pytest.approx(0.001)
    bridge.shutdown()


def test_command_spin_loop_survives_a_transient_spin_once_failure(fake_ros: dict[str, Any]) -> None:
    bridge = HardwareRosBridge(enable_commands=False)
    bridge._spin_period = 0.001
    attempts = {"n": 0}

    def spin_once(node: Any, timeout_sec: float = 0.0) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("executor hiccup")
        # The retry proves the loop kept running past the exception.
        bridge._stop.set()

    bridge._rclpy.spin_once = spin_once  # type: ignore[attr-defined]
    bridge._stop.clear()
    bridge._spin_loop()  # must not propagate the RuntimeError

    assert attempts["n"] == 2  # retried after swallowing the failure
    bridge.shutdown()


def test_start_spin_is_idempotent(fake_ros: dict[str, Any]) -> None:
    robot = _FakeDrivableRobot()
    bridge = HardwareRosBridge(robot)  # type: ignore[arg-type]  # enable_commands default -> thread started
    first = bridge._spin_thread
    assert first is not None and first.is_alive()

    # A second start while the thread is alive is a no-op (no duplicate thread).
    bridge._start_spin()
    assert bridge._spin_thread is first
    assert _live_command_threads() == 1
    bridge.shutdown()


# --- a refused construction costs the process nothing ------------------------

#: Every constructor argument :class:`HardwareRosBridge` documents a
#: ``ValueError`` for, paired with a value from outside its domain. That the
#: refusal happens is already pinned elsewhere; what these cells pin is where it
#: happens - the base constructor writes the process-wide ``ROS_DOMAIN_ID``,
#: initializes the rclpy context when nothing else has, and creates the node, so
#: a guard placed after it charges a rejected caller for state no one can
#: release: ``__init__`` raised, so there is no bridge to call ``shutdown`` on.
_REFUSED_CONSTRUCTIONS: list[tuple[str, dict[str, Any]]] = [
    ("domain_id", {"domain_id": 233}),
    ("qos_depth", {"qos_depth": 0}),
    ("spin_period", {"spin_period": 0.0}),
    ("enable_commands", {"enable_commands": "false"}),
    ("joint_limits/order", {"joint_limits": {"j0.pos": (1.0, -1.0)}}),
    ("joint_limits/non-finite", {"joint_limits": {"j0.pos": (1.0, float("nan"))}}),
    ("command_robot_name/truthy-non-str", {"command_robot_name": 7}),
    ("command_robot_name/falsy-non-str", {"command_robot_name": 0}),
]


@pytest.mark.parametrize(
    "kwargs",
    [kwargs for _, kwargs in _REFUSED_CONSTRUCTIONS],
    ids=[param for param, _ in _REFUSED_CONSTRUCTIONS],
)
def test_a_refused_bridge_leaves_the_process_as_it_found_it(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    """No refusal writes ROS_DOMAIN_ID, starts rclpy, or creates a node.

    Each cell asks for a bridge on domain 42 with one argument outside its
    domain, from a process whose shell pointed at domain 7. A refusal that lands
    after the base constructor leaves 42 behind for every later participant to
    inherit, the rclpy context up, and a live node in the graph.
    """
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    with pytest.raises(ValueError):
        HardwareRosBridge(_FakeDrivableRobot(), **{"domain_id": 42, **kwargs})  # type: ignore[arg-type]

    assert os.environ["ROS_DOMAIN_ID"] == "7"
    assert fake_ros["inited"] is False
    assert fake_ros["nodes"] == []


def test_a_refused_bridge_does_not_strand_the_rclpy_context(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrected retry can still release the context, because nothing took it.

    ``_owns_context`` is recorded from ``rclpy.ok()``, so whichever bridge starts
    the context is the only one that will shut it down. When a refused
    construction starts it, that bridge is never returned: the retry sees the
    context already up, declines ownership, and its ``shutdown`` releases the
    node it made but not the context or the node the refusal left behind.
    """
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    with pytest.raises(ValueError, match="min .* > max"):
        HardwareRosBridge(_FakeDrivableRobot(), joint_limits={"j0.pos": (1.0, -1.0)})  # type: ignore[arg-type]

    bridge = HardwareRosBridge(_FakeDrivableRobot(), joint_limits={"j0.pos": (-1.0, 1.0)})  # type: ignore[arg-type]
    assert bridge._owns_context is True
    bridge.shutdown()

    assert fake_ros["shutdown"] is True
    assert fake_ros["inited"] is False
    assert [node.name for node in fake_ros["nodes"] if not node.destroyed] == []


# --- the command namespace is the one caller-supplied name in a topic --------

#: ``command_robot_name`` values that cannot name a topic segment, paired with
#: how each one failed before it was graded. A truthy value reached the
#: sanitiser's ``re.sub`` and raised ``TypeError`` naming no parameter; a falsy
#: one was filtered by the ``command_robot_name or <derived>`` default and never
#: raised at all, so the bridge read commands under a namespace the caller had
#: not asked for.
_UNNAMEABLE_COMMAND_NAMESPACES: list[Any] = [7, ["left_arm"], b"left_arm", 0, [], False]


@pytest.mark.parametrize("name", _UNNAMEABLE_COMMAND_NAMESPACES, ids=repr)
def test_a_command_namespace_that_cannot_name_a_topic_is_refused(fake_ros: dict[str, Any], name: Any) -> None:
    """The refusal names the parameter and the input that restores the default.

    The old failures named neither: ``TypeError: expected string or bytes-like
    object, got 'int'`` came out of the sanitiser, and the falsy values produced
    no message because the default-selecting ``or`` swallowed them.
    """
    with pytest.raises(ValueError) as excinfo:
        HardwareRosBridge(_FakeDrivableRobot(), command_robot_name=name)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "'command_robot_name'" in message
    assert type(name).__name__ in message
    assert "Pass None" in message


@pytest.mark.parametrize(
    ("name", "expected_topic"),
    [
        (None, "/so101/joint_command"),
        ("", "/so101/joint_command"),
        ("left_arm", "/left_arm/joint_command"),
    ],
    ids=["derive/None", "derive/empty", "override"],
)
def test_an_accepted_command_namespace_selects_the_topic_it_names(
    fake_ros: dict[str, Any], name: Any, expected_topic: str
) -> None:
    """Both documented ways of asking for the default still get it.

    ``None`` and ``""`` select the bound robot's own name - the namespace this
    bridge publishes ``joint_states`` under - and a string overrides it. Pinned
    alongside the refusals so the guard cannot be satisfied by refusing more than
    it should.
    """
    bridge = HardwareRosBridge(_FakeDrivableRobot(), command_robot_name=name)  # type: ignore[arg-type]
    assert [sub.topic for sub in fake_ros["nodes"][0].subscriptions] == [expected_topic]
    bridge.shutdown()
