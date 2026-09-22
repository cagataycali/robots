"""The Yahboom M3 Pro native driver: the ROS 2 wire, the unit maps, the refusals.

Everything here runs with no robot, no rosbridge and no ROS install. One double
stands in for the transport - :func:`strands_robots.rosbridge.rosbridge_action`
- and it is faithful in the one respect the tests depend on: it **records**
every publish, with the topic, the interface type and the fields, rather than
discarding it. A double that dropped the message could not tell "the driver
converted radians to degrees" from "the driver sent nothing", and the trailing
zero Twist after a held ``move`` is exactly one recorded publish - the property
that stops the base.

The operator gate is real: ``/cmd_vel`` is on the shared blocklist, so the
tests that drive the base pre-approve it through ``STRANDS_ROS2_COMMAND_ALLOW``
the way a headless caller does, and one test proves that without the
pre-approval the twist never reaches the recorder.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

from strands_robots._command_gate import COMMAND_ALLOW_ENV
from strands_robots.drivers import get_native_driver_class, list_driver_coverage, resolve_driver
from strands_robots.drivers.base import (
    HardwareDriver,
    declared_verbs,
    drifted_driver_parameters,
    missing_driver_members,
)
from strands_robots.drivers.yahboom_m3pro import (
    ARM_CHANNELS,
    ARM_TOPIC,
    ARM_TYPE,
    BASE_CHANNELS,
    CMD_VEL_TOPIC,
    DEFAULT_ROSBRIDGE,
    GRIPPER_CLOSED_DEG,
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_DEG,
    GRIPPER_OPEN_RAD,
    HOME_DEG,
    IMU_TOPIC,
    MAX_ANGULAR_RPS,
    MAX_LINEAR_MPS,
    MAX_MOVE_DURATION_S,
    ODOM_TOPIC,
    PUBLISH_RATE_HZ,
    REQUIRED_TOPICS,
    SERVO_RANGES_DEG,
    TWIST_TYPE,
    YahboomM3ProDriver,
    arm_deg_to_rad,
    arm_rad_to_deg,
    endpoint_error,
    gripper_deg_to_rad,
    gripper_rad_to_deg,
    parse_echo,
    servo_deg_error,
    split_endpoint,
    twist_axis_error,
)

# --------------------------------------------------------------------------- #
# The transport double.                                                       #
# --------------------------------------------------------------------------- #

_GRAPH = (
    "/arm6_joints [arm_msgs/msg/ArmJoints]\n"
    "/cmd_vel [geometry_msgs/msg/Twist]\n"
    "/imu/data_raw [sensor_msgs/msg/Imu]\n"
    "/odom_raw [nav_msgs/msg/Odometry]\n"
    "/parameter_events [rcl_interfaces/msg/ParameterEvent]\n"
    "/rosout [rcl_interfaces/msg/Log]\n"
    "/scan0 [sensor_msgs/msg/LaserScan]\n"
    "/scan1 [sensor_msgs/msg/LaserScan]"
)

_ODOM = {"pose": {"pose": {"position": {"x": 1.25, "y": -0.5, "z": 0.0}}}, "twist": {"twist": {"linear": {"x": 0.1}}}}
_IMU = {"linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.8}}


class _FakeTransport:
    """Records every publish; answers status / list_topics / echo from a table."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.publishes: list[dict[str, Any]] = []
        self.gate_asked: list[tuple[str, str]] = []
        self.connected = True
        self.available = True
        self.graph = _GRAPH
        self.samples: dict[str, list[dict[str, Any]]] = {ODOM_TOPIC: [_ODOM], IMU_TOPIC: [_IMU]}
        self.publish_error: str | None = None

    def __call__(self, action: str, *, host: str, port: int, gate: Any, **options: Any) -> dict[str, Any]:
        self.calls.append({"action": action, "host": host, "port": port, **options})
        if action == "status":
            if not self.available:
                return {"status": "success", "content": [{"text": "backend: none - roslibpy is not importable"}]}
            if not self.connected:
                return {"status": "success", "content": [{"text": "backend: roslibpy; not connected - dial failed"}]}
            return {"status": "success", "content": [{"text": f"backend: roslibpy; connected to ws://{host}:{port}"}]}
        if action == "list_topics":
            return {"status": "success", "content": [{"text": self.graph}]}
        if action == "echo":
            samples = self.samples.get(options["topic"], [])
            body = json.dumps(samples, indent=2)
            note = "" if samples else "\n(no messages within 5s - topic may be silent or the type mismatched)"
            return {
                "status": "success",
                "content": [{"text": f"echo {options['topic']} ({options['type']}):\n{body}{note}"}],
            }
        if action == "publish":
            refusal = gate("publish", options["topic"])
            self.gate_asked.append(("publish", options["topic"]))
            if refusal is not None:
                return {"status": "error", "content": [{"text": f"use_rosbridge: {refusal}"}]}
            if self.publish_error is not None:
                return {"status": "error", "content": [{"text": f"use_rosbridge: {self.publish_error}"}]}
            self.publishes.append(dict(options))
            return {
                "status": "success",
                "content": [{"text": f"published {options['count']} message(s) to {options['topic']}"}],
            }
        raise AssertionError(f"unexpected transport action {action!r}")

    def twists(self) -> list[dict[str, Any]]:
        return [p for p in self.publishes if p["topic"] == CMD_VEL_TOPIC]

    def arm_messages(self) -> list[dict[str, Any]]:
        return [p for p in self.publishes if p["topic"] == ARM_TOPIC]


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _FakeTransport:
    fake = _FakeTransport()
    import strands_robots.rosbridge as rosbridge_mod

    monkeypatch.setattr(rosbridge_mod, "rosbridge_action", fake)
    return fake


@pytest.fixture
def allow_cmd_vel(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a headless caller sets: the pre-approval for the blocklisted drive surface."""
    monkeypatch.setenv(COMMAND_ALLOW_ENV, CMD_VEL_TOPIC)
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)


@pytest.fixture
def driver(transport: _FakeTransport) -> YahboomM3ProDriver:
    built = YahboomM3ProDriver(port="m3pro.local:9090")
    assert built.connect_eagerly() is None
    return built


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _invoke(driver: YahboomM3ProDriver, **request: Any) -> dict[str, Any]:
    tool_use: Any = {"toolUseId": "t1", "name": driver.tool_name, "input": request}
    results = [event async for event in driver.stream(tool_use, {})]
    assert len(results) == 1
    return results[0]


# --------------------------------------------------------------------------- #
# The seam.                                                                   #
# --------------------------------------------------------------------------- #


class TestTheSeamCanBuildIt:
    def test_the_driver_satisfies_the_whole_surface(self) -> None:
        assert missing_driver_members(YahboomM3ProDriver) == ()
        assert drifted_driver_parameters(YahboomM3ProDriver) == ()
        assert isinstance(YahboomM3ProDriver(), HardwareDriver)

    def test_the_shipped_registration_names_this_class(self) -> None:
        assert get_native_driver_class("yahboom_m3pro") is YahboomM3ProDriver
        assert get_native_driver_class("m3pro") is YahboomM3ProDriver

    def test_the_registry_entry_resolves_to_the_native_driver_by_default(self) -> None:
        """``Robot("yahboom_m3pro", mode="real")`` needs no ``driver=`` keyword."""
        assert list_driver_coverage()["yahboom_m3pro"] == ("strands",)
        assert resolve_driver("yahboom_m3pro") == "strands"

    def test_the_factory_extras_are_tolerated(self) -> None:
        built = YahboomM3ProDriver(tool_name="bench", cameras=None, data_config=None, port=None, robot_ip="ignored")
        assert built.tool_name == "bench"
        assert built.tool_type == "robot"
        assert built.endpoint == f"ws://{DEFAULT_ROSBRIDGE.replace(':', ':')}"


class TestConstructionRefusesTheWrongShape:
    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"port": "/dev/ttyUSB0"}, "filesystem path"),
            ({"port": "ws://10.0.0.9:9090"}, "URL"),
            ({"port": "10.0.0.9:abc"}, "whole number"),
            ({"port": "10.0.0.9:70000"}, "port"),
            ({"port": "  "}, "rosbridge endpoint"),
            ({"transport": "dds"}, "transport must be one of"),
            ({"timeout_s": 0}, "timeout_s"),
            ({"joint_signs": (1.0, 1.0)}, "joint_signs"),
            ({"joint_signs": (1.0, 1.0, 1.0, 1.0, 2.0)}, "joint_signs"),
            ({"move_time_ms": 0}, "move_time_ms"),
            ({"move_time_ms": 40_000}, "ceiling"),
        ],
    )
    def test_a_wrong_argument_is_refused_by_name(self, kwargs: dict[str, Any], needle: str) -> None:
        with pytest.raises(ValueError, match=needle):
            YahboomM3ProDriver(**kwargs)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("m3pro.local", ("m3pro.local", 9090)),
            ("10.0.0.9:9091", ("10.0.0.9", 9091)),
            ("localhost:9090", ("localhost", 9090)),
        ],
    )
    def test_the_endpoint_splits_and_defaults_the_port(self, value: str, expected: tuple[str, int]) -> None:
        assert endpoint_error(value, "port", "t") is None
        assert split_endpoint(value) == expected

    def test_the_ros2_transport_names_no_endpoint(self) -> None:
        assert YahboomM3ProDriver(transport="ros2").endpoint == "rclpy (in-process)"


# --------------------------------------------------------------------------- #
# The unit maps - sim radians <-> servo degrees.                              #
# --------------------------------------------------------------------------- #


class TestTheSimToServoMapsAreTheDescriptionsInverse:
    """``q = (deg - 90) * pi / 180`` is how the MJCF's ``home`` keyframe was written."""

    @pytest.mark.parametrize(
        ("degrees", "radians"), [(90, 0.0), (0, -math.pi / 2), (180, math.pi / 2), (120, math.pi / 6)]
    )
    def test_arm_joints_one_to_four_land_on_the_urdf_range(self, degrees: int, radians: float) -> None:
        assert arm_rad_to_deg(1, radians) == degrees
        assert arm_deg_to_rad(1, degrees) == pytest.approx(radians)

    def test_arm_five_maps_its_270_degree_travel_onto_the_urdf_range(self) -> None:
        """The URDF's ``-pi/2 .. pi`` on ``arm5`` is exactly the wrist servo's 0 .. 270."""
        assert arm_rad_to_deg(5, -math.pi / 2) == 0
        assert arm_rad_to_deg(5, math.pi) == 270
        assert servo_deg_error(5, arm_rad_to_deg(5, math.pi), "j5", "t") is None

    def test_a_reversed_servo_flips_about_centre(self) -> None:
        assert arm_rad_to_deg(2, math.pi / 4, sign=-1.0) == 45
        assert arm_deg_to_rad(2, 45, sign=-1.0) == pytest.approx(math.pi / 4)

    def test_the_home_keyframe_round_trips(self) -> None:
        """The vendor grasp-init pose the description encodes: 90/120/0/0/90."""
        radians = [arm_deg_to_rad(i + 1, deg) for i, deg in enumerate(HOME_DEG[:5])]
        assert [arm_rad_to_deg(i + 1, q) for i, q in enumerate(radians)] == list(HOME_DEG[:5])

    def test_the_gripper_endpoints_coincide(self) -> None:
        assert gripper_rad_to_deg(GRIPPER_CLOSED_RAD) == GRIPPER_CLOSED_DEG
        assert gripper_rad_to_deg(GRIPPER_OPEN_RAD) == GRIPPER_OPEN_DEG
        assert gripper_deg_to_rad(GRIPPER_CLOSED_DEG) == pytest.approx(GRIPPER_CLOSED_RAD)
        assert gripper_deg_to_rad(GRIPPER_OPEN_DEG) == pytest.approx(GRIPPER_OPEN_RAD)
        assert gripper_rad_to_deg(GRIPPER_CLOSED_RAD / 2) == 105

    @pytest.mark.parametrize("servo_id", sorted(SERVO_RANGES_DEG))
    def test_the_servo_travel_is_refused_past_either_end(self, servo_id: int) -> None:
        low, high = SERVO_RANGES_DEG[servo_id]
        assert servo_deg_error(servo_id, low, "j", "t") is None
        assert servo_deg_error(servo_id, high, "j", "t") is None
        assert "travel" in (servo_deg_error(servo_id, low - 1, "j", "t") or "")
        assert "travel" in (servo_deg_error(servo_id, high + 1, "j", "t") or "")
        assert servo_deg_error(servo_id, float("nan"), "j", "t") is not None

    @pytest.mark.parametrize(
        ("param", "value", "ok"),
        [
            ("linear.x", MAX_LINEAR_MPS, True),
            ("linear.y", -MAX_LINEAR_MPS, True),
            ("angular.z", MAX_ANGULAR_RPS, True),
            ("linear.x", MAX_LINEAR_MPS + 0.01, False),
            ("angular.z", -MAX_ANGULAR_RPS - 0.01, False),
            ("linear.x", "0.5", False),
            ("linear.x", True, False),
        ],
    )
    def test_the_twist_envelope_is_the_vendor_teleop_ceiling(self, param: str, value: Any, ok: bool) -> None:
        assert (twist_axis_error(value, param, "t") is None) is ok


# --------------------------------------------------------------------------- #
# Connect is proven, not assumed.                                             #
# --------------------------------------------------------------------------- #


class TestConnectIsProvenNotAssumed:
    def test_success_records_the_graph(self, transport: _FakeTransport) -> None:
        built = YahboomM3ProDriver(port="10.0.0.9:9090")
        assert built.connect_eagerly() is None
        assert built.is_connected
        assert [c["action"] for c in transport.calls] == ["status", "list_topics"]
        assert transport.calls[0]["host"] == "10.0.0.9" and transport.calls[0]["port"] == 9090
        assert built.connect_eagerly() is None and len(transport.calls) == 2, "idempotent"

    def test_a_missing_transport_package_is_a_named_reason(self, transport: _FakeTransport) -> None:
        transport.available = False
        reason = YahboomM3ProDriver().connect_eagerly()
        assert reason is not None and "roslibpy" in reason

    def test_an_unreachable_bridge_is_a_named_reason(self, transport: _FakeTransport) -> None:
        transport.connected = False
        reason = YahboomM3ProDriver(port="10.0.0.9").connect_eagerly()
        assert reason is not None and "ws://10.0.0.9:9090" in reason

    def test_a_graph_without_the_board_names_the_micro_ros_remedy(self, transport: _FakeTransport) -> None:
        """Two topics on the graph is the agent handshake failure, not a dead robot."""
        transport.graph = "/parameter_events [rcl_interfaces/msg/ParameterEvent]\n/rosout [rcl_interfaces/msg/Log]"
        built = YahboomM3ProDriver()
        reason = built.connect_eagerly()
        assert reason is not None
        for topic in REQUIRED_TOPICS:
            assert topic in reason
        assert "micro_ros_agent" in reason and "restart" in reason
        assert not built.is_connected

    def test_the_driver_stays_usable_off_hardware(self, transport: _FakeTransport, allow_cmd_vel: None) -> None:
        transport.connected = False
        built = YahboomM3ProDriver()
        built.connect_eagerly()
        assert built.send_action({"arm1.pos": 0.0})["status"] == "error"
        assert built.move(0.1, duration_s=1.0)["status"] == "error"
        assert built.stop_task()["status"] == "error"
        assert built.read_state() == {"odom": None, "imu": None}
        assert built.get_observation() == {}
        assert transport.publishes == []


# --------------------------------------------------------------------------- #
# The arm wire.                                                               #
# --------------------------------------------------------------------------- #


class TestTheArmWire:
    def test_send_action_converts_the_models_radians_to_one_arm_joints_message(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        result = driver.send_action({"arm1.pos": 0.0, "arm2.pos": math.pi / 6, "gripper.pos": GRIPPER_CLOSED_RAD})
        assert result["status"] == "success", result
        [message] = transport.arm_messages()
        assert message["type"] == ARM_TYPE
        assert message["count"] == 1
        assert message["fields"] == {
            "joint1": 90,
            "joint2": 120,
            "joint3": HOME_DEG[2],
            "joint4": HOME_DEG[3],
            "joint5": HOME_DEG[4],
            "joint6": GRIPPER_CLOSED_DEG,
            "time": 1500,
        }
        assert result["content"][0]["json"]["arm_deg"] == [90, 120, 0, 0, 90, 30]

    def test_an_omitted_joint_keeps_its_last_commanded_angle(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        driver.send_action({"arm3.pos": math.pi / 4})
        driver.send_action({"arm1.pos": -math.pi / 4})
        assert transport.arm_messages()[-1]["fields"]["joint3"] == 135
        assert transport.arm_messages()[-1]["fields"]["joint1"] == 45

    def test_joint_signs_reach_the_wire(self, transport: _FakeTransport) -> None:
        built = YahboomM3ProDriver(joint_signs=(-1.0, 1.0, 1.0, 1.0, 1.0))
        built.connect_eagerly()
        built.send_action({"arm1.pos": math.pi / 4})
        assert transport.arm_messages()[-1]["fields"]["joint1"] == 45
        pose = built.last_arm_command()
        assert pose is not None and pose["arm1.pos"] == pytest.approx(math.pi / 4)

    def test_a_radian_past_the_servo_travel_is_refused_not_clamped(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        result = driver.send_action({"arm1.pos": math.pi})
        assert result["status"] == "error"
        assert "travel" in result["content"][0]["text"] and "converted" in result["content"][0]["text"]
        assert transport.arm_messages() == []

    def test_the_last_command_is_reported_in_sim_units_but_not_as_an_observation(
        self, driver: YahboomM3ProDriver
    ) -> None:
        assert driver.last_arm_command() is None
        driver.send_action({"arm5.pos": math.pi, "gripper.pos": GRIPPER_OPEN_RAD})
        pose = driver.last_arm_command()
        assert pose is not None
        assert pose["arm5.pos"] == pytest.approx(math.pi)
        assert pose["gripper.pos"] == pytest.approx(GRIPPER_OPEN_RAD)
        assert set(pose) == set(ARM_CHANNELS)
        assert driver.get_observation() == {}, "a command is not a reading"

    def test_set_arm_degrees_speaks_the_vendor_units(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        result = driver.set_arm_degrees([90, 90, 90, 90, 90, 180], time_ms=2000)
        assert result["status"] == "success"
        assert transport.arm_messages()[-1]["fields"] == {
            "joint1": 90,
            "joint2": 90,
            "joint3": 90,
            "joint4": 90,
            "joint5": 90,
            "joint6": 180,
            "time": 2000,
        }

    @pytest.mark.parametrize(
        ("joints", "time_ms", "needle"),
        [
            ([90, 90, 90], None, "six servo angles"),
            ([90, 90, 90, 90, 90, 10], None, "gripper"),
            ([90, 90, 90, 90, 300, 90], None, "servo 5"),
            ([90, 90, 90, 90, 90, 90], 0, "time_ms"),
            ([90, 90, 90, 90, 90, 90], 60_000, "ceiling"),
            ([90, 90, 90, 90, 90, "90"], None, "joint6"),
        ],
    )
    def test_a_bad_arm_request_is_refused_before_the_wire(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, joints: Any, time_ms: Any, needle: str
    ) -> None:
        result = driver.set_arm_degrees(joints, time_ms=time_ms)
        assert result["status"] == "error"
        assert needle in result["content"][0]["text"]
        assert transport.arm_messages() == []

    def test_the_gripper_verb_moves_only_the_sixth_servo(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        driver.set_arm_degrees([10, 20, 30, 40, 50, 180])
        assert driver.set_gripper(False)["status"] == "success"
        assert transport.arm_messages()[-1]["fields"] == {
            "joint1": 10,
            "joint2": 20,
            "joint3": 30,
            "joint4": 40,
            "joint5": 50,
            "joint6": GRIPPER_CLOSED_DEG,
            "time": 1500,
        }
        assert driver.set_gripper("closed")["status"] == "error", "read as a boolean, not for truthiness"

    def test_home_is_the_descriptions_keyframe_with_the_gripper_open(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        assert driver.home()["status"] == "success"
        fields = transport.arm_messages()[-1]["fields"]
        assert [fields[f"joint{i}"] for i in range(1, 7)] == list(HOME_DEG)
        assert fields["joint6"] == GRIPPER_OPEN_DEG

    def test_the_arm_topic_is_not_gated(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No pre-approval set, no agent: the arm still moves - it is not on the blocklist."""
        monkeypatch.delenv(COMMAND_ALLOW_ENV, raising=False)
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        assert driver.home()["status"] == "success"
        assert len(transport.arm_messages()) == 1


# --------------------------------------------------------------------------- #
# The base wire, and its gate.                                                #
# --------------------------------------------------------------------------- #


class TestTheBaseWire:
    def test_move_streams_the_twist_above_the_watchdog_then_stops(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        result = driver.move(0.2, -0.1, 0.5, duration_s=1.5)
        assert result["status"] == "success", result
        held, stop = transport.twists()
        assert held["type"] == TWIST_TYPE
        assert held["count"] == 15 and held["rate"] == PUBLISH_RATE_HZ
        assert held["fields"] == {"linear": {"x": 0.2, "y": -0.1, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.5}}
        assert stop["count"] == 1
        assert stop["fields"] == {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}
        assert result["content"][0]["json"] == {
            "commanded": {"linear.x": 0.2, "linear.y": -0.1, "angular.z": 0.5},
            "held_s": 1.5,
            "frames": 15,
            "stopped": True,
        }

    def test_a_move_without_a_duration_is_a_twitch_and_is_refused(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        result = driver.move(0.2)
        assert result["status"] == "error" and "duration_s is required" in result["content"][0]["text"]
        assert transport.twists() == []

    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"linear_x": MAX_LINEAR_MPS * 2, "duration_s": 1.0}, "envelope"),
            ({"angular_z": -MAX_ANGULAR_RPS * 2, "duration_s": 1.0}, "envelope"),
            ({"linear_x": 0.1, "duration_s": MAX_MOVE_DURATION_S + 1}, "at most"),
            ({"linear_x": 0.1, "duration_s": -1}, "duration_s"),
        ],
    )
    def test_a_bad_move_is_refused_before_the_wire(
        self,
        driver: YahboomM3ProDriver,
        transport: _FakeTransport,
        allow_cmd_vel: None,
        kwargs: dict[str, Any],
        needle: str,
    ) -> None:
        result = driver.move(**kwargs)
        assert result["status"] == "error" and needle in result["content"][0]["text"]
        assert transport.twists() == []

    def test_send_action_sends_one_twist_frame(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        result = driver.send_action({"linear.y": 0.3})
        assert result["status"] == "success"
        [frame] = transport.twists()
        assert frame["count"] == 1
        assert frame["fields"]["linear"] == {"x": 0.0, "y": 0.3, "z": 0.0}
        assert result["content"][0]["json"]["twist"] == {"linear.x": 0.0, "linear.y": 0.3, "angular.z": 0.0}

    def test_an_arm_and_a_base_command_in_one_action_send_arm_first(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        result = driver.send_action({"gripper.pos": GRIPPER_OPEN_RAD, "linear.x": 0.1})
        assert result["status"] == "success"
        assert [p["topic"] for p in transport.publishes] == [ARM_TOPIC, CMD_VEL_TOPIC]

    def test_an_unknown_channel_is_refused_with_the_valid_set(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        result = driver.send_action({"wheel.vel": 1.0})
        text = result["content"][0]["text"]
        assert result["status"] == "error" and "wheel.vel" in text
        for channel in (*ARM_CHANNELS, *BASE_CHANNELS):
            assert channel in text
        assert transport.publishes == []

    def test_a_base_command_without_approval_never_reaches_the_wire(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared gate is real: no allowlist, no bypass, no agent -> refused, nothing sent."""
        monkeypatch.delenv(COMMAND_ALLOW_ENV, raising=False)
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        result = driver.move(0.1, duration_s=0.5)
        assert result["status"] == "error"
        assert "safety-critical" in result["content"][0]["text"]
        assert COMMAND_ALLOW_ENV in result["content"][0]["text"], "the refusal names the pre-approval"
        assert transport.twists() == []
        assert transport.gate_asked == [("publish", CMD_VEL_TOPIC)]

    def test_stop_task_is_a_zero_twist_and_a_failed_one_is_not_reported_as_stopped(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        assert driver.stop_task()["status"] == "success"
        assert transport.twists()[-1]["fields"]["linear"] == {"x": 0.0, "y": 0.0, "z": 0.0}
        transport.publish_error = "bridge went away"
        stopped = driver.stop_task()
        assert stopped["status"] == "error" and "bridge went away" in stopped["content"][0]["text"]
        _run(driver.stop())  # never raises, whatever the halt outcome

    def test_a_move_whose_trailing_stop_fails_is_an_error(
        self,
        driver: YahboomM3ProDriver,
        transport: _FakeTransport,
        allow_cmd_vel: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = transport.__call__

        def _fail_the_stop(action: str, **options: Any) -> dict[str, Any]:
            if action == "publish" and options.get("count") == 1 and options["topic"] == CMD_VEL_TOPIC:
                return {"status": "error", "content": [{"text": "use_rosbridge: dropped"}]}
            return original(action, **options)

        import strands_robots.rosbridge as rosbridge_mod

        monkeypatch.setattr(rosbridge_mod, "rosbridge_action", _fail_the_stop)
        result = driver.move(0.1, duration_s=0.5)
        assert result["status"] == "error"
        assert result["content"][1]["json"]["stopped"] is False
        assert "watchdog" in result["content"][0]["text"]

    def test_cleanup_sends_a_parting_zero_twist_and_is_idempotent(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        driver.cleanup()
        assert len(transport.twists()) == 1
        assert not driver.is_connected
        driver.cleanup()
        assert len(transport.twists()) == 1


# --------------------------------------------------------------------------- #
# The read path.                                                              #
# --------------------------------------------------------------------------- #


class TestTheReadPath:
    def test_read_state_echoes_odometry_and_imu(self, driver: YahboomM3ProDriver, transport: _FakeTransport) -> None:
        state = driver.read_state()
        assert state["odom"] == _ODOM and state["imu"] == _IMU
        echoed = [(c["topic"], c["type"]) for c in transport.calls if c["action"] == "echo"]
        assert echoed == [(ODOM_TOPIC, "nav_msgs/Odometry"), (IMU_TOPIC, "sensor_msgs/Imu")]

    def test_a_silent_topic_falls_back_to_the_cache(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        driver.read_state()
        transport.samples = {}
        assert driver.read_state()["odom"] == _ODOM

    def test_parse_echo_reads_the_transports_text_block(self) -> None:
        envelope = {
            "status": "success",
            "content": [{"text": f"echo /x (a/B):\n{json.dumps([{'k': 1}, 'noise'])}\n(note)"}],
        }
        assert parse_echo(envelope) == [{"k": 1}]
        assert parse_echo({"status": "error", "content": [{"text": "[]"}]}) == []
        assert parse_echo({"status": "success", "content": [{"text": "echo /x (a/B):\nnot json"}]}) == []

    def test_status_reports_the_graph_and_the_last_commands(
        self, driver: YahboomM3ProDriver, allow_cmd_vel: None
    ) -> None:
        driver.home()
        driver.send_action({"angular.z": 0.5})
        status = _run(driver.get_status())["content"][0]["json"]
        assert status["connected"] is True and status["transport"] == "rosbridge"
        assert status["endpoint"] == "ws://m3pro.local:9090"
        assert CMD_VEL_TOPIC in status["topics"] and ARM_TOPIC in status["topics"]
        assert status["last_arm_deg"] == list(HOME_DEG)
        assert status["last_twist"] == {"linear.x": 0.0, "linear.y": 0.0, "angular.z": 0.5}

    def test_status_carries_the_shared_driver_triple(self, driver: YahboomM3ProDriver) -> None:
        """Every native driver's status reports tool_name/connected/battery_pct.

        The triple is the fleet-wide contract a caller reads without knowing the
        driver; the M3 Pro's board publishes no battery-percent topic this
        driver has verified, so it reports the field as ``None`` rather than
        omitting it or inventing a reading (the reason ``get_observation`` is
        empty too).
        """
        status = _run(driver.get_status())["content"][0]["json"]
        for key in ("tool_name", "connected", "battery_pct"):
            assert key in status
        assert status["battery_pct"] is None

    def test_policy_paths_refuse_with_a_route(self, driver: YahboomM3ProDriver) -> None:
        assert "send_action" in driver.start_task("wave")["content"][0]["text"]
        assert "send_action" in driver.run_policy(object())["content"][0]["text"]  # type: ignore[arg-type]
        assert driver.get_task_status()["content"][0]["json"]["running"] is False


# --------------------------------------------------------------------------- #
# The agent surface.                                                          #
# --------------------------------------------------------------------------- #


class TestTheAgentSurface:
    def test_every_declared_verb_dispatches(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport, allow_cmd_vel: None
    ) -> None:
        verbs = declared_verbs(driver.tool_spec)
        assert verbs == ["status", "sensors", "arm", "gripper", "home", "move", "stop"]
        assert _run(_invoke(driver, action="status"))["status"] == "success"
        sensors = _run(_invoke(driver, action="sensors"))
        assert sensors["status"] == "success" and "odom x 1.25" in sensors["content"][0]["text"]
        assert (
            _run(_invoke(driver, action="arm", joints=[90, 90, 90, 90, 90, 180], time_ms=1000))["status"] == "success"
        )
        assert _run(_invoke(driver, action="gripper", open=False))["status"] == "success"
        assert _run(_invoke(driver, action="home"))["status"] == "success"
        assert _run(_invoke(driver, action="move", linear_x=0.1, duration_s=0.2))["status"] == "success"
        assert _run(_invoke(driver, action="stop"))["status"] == "success"
        assert len(transport.arm_messages()) == 3
        assert len(transport.twists()) == 3  # the held move, its trailing stop, the stop verb

    def test_an_undeclared_verb_is_refused_not_dispatched(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        result = _run(_invoke(driver, action="dance"))
        assert result["status"] == "error" and "dance" in result["content"][0]["text"]
        assert transport.publishes == []

    def test_the_default_verb_is_the_read(self, driver: YahboomM3ProDriver) -> None:
        assert _run(_invoke(driver))["content"][0]["json"]["connected"] is True

    def test_sensors_refuses_a_silent_robot_rather_than_reporting_nothing(
        self, driver: YahboomM3ProDriver, transport: _FakeTransport
    ) -> None:
        transport.samples = {}
        result = _run(_invoke(driver, action="sensors"))
        assert result["status"] == "error" and "micro-ROS" in result["content"][0]["text"]

    def test_the_operator_context_is_released_after_the_call(
        self, driver: YahboomM3ProDriver, allow_cmd_vel: None
    ) -> None:
        _run(_invoke(driver, action="stop"))
        assert driver._operator_context is None
