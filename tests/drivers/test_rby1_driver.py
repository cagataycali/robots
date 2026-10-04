"""The RB-Y1 native driver against a robot double shaped like ``rby1_sdk``.

The double stands in for the whole ``rby1_sdk`` module: a ``Robot_A`` that
records every call, a command stream that records every setpoint, the command
builders (which on the real SDK are opaque C++ objects, so the double's builders
keep the vectors they are given), and a dynamics model whose limit vectors follow
the state's joint order and append the joints the state left out - the shape
the real ``rby1_sdk.dynamics.Robot`` returns. One cell grades every method the
double answers against the real SDK so it cannot drift into an API the vendor
does not have.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class, list_driver_coverage
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.rby1 import BODY_JOINTS, JOINT_NAMES, WHEEL_JOINTS, RBY1Driver

#: The 24 joints in the order ``Model_A().robot_joint_names`` reports them.
MODEL_JOINTS = [*WHEEL_JOINTS, *JOINT_NAMES]
HOME = {name: 0.05 * index for index, name in enumerate(MODEL_JOINTS)}
#: At the default 50 Hz a 1 rad/s limit is a 0.02 rad step.
SPEED, LOWER, UPPER = 1.0, -2.0, 2.0


def _enum(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name)


class Builder:
    """Every command builder: records each ``set_*`` call, returns itself."""

    def __init__(self, kind: str) -> None:
        self.kind, self.fields = kind, dict[str, Any]()

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("set_"):
            raise AttributeError(name)

        def setter(value: Any) -> Builder:
            self.fields[name] = value
            return self

        return setter


def _positions(command: Builder) -> list[float]:
    """The body + head setpoint carried by one ``RobotCommandBuilder``."""
    component = command.fields["set_command"].fields
    body = component["set_body_command"].fields["set_command"].fields["set_position"]
    head = component["set_head_command"].fields["set_command"].fields["set_position"]
    return [float(value) for value in [*body, *head]]


class FakeStream:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot, self.done = robot, False

    def send_command(self, builder: Builder, timeout_ms: int = 1000) -> types.SimpleNamespace:
        self.robot.calls.append(("send_command", (_positions(builder),)))
        if self.robot.finish_code == "Ok":
            for name, value in zip(JOINT_NAMES, _positions(builder), strict=True):
                self.robot.position[name] = value
            return types.SimpleNamespace(status=_enum("Running"), finish_code=_enum("Unknown"))
        return types.SimpleNamespace(status=_enum("Finished"), finish_code=_enum(self.robot.finish_code))

    def is_done(self) -> bool:
        return self.done

    def cancel(self) -> None:
        self.robot.calls.append(("stream.cancel", ()))
        self.done = True


class FakeDynamics:
    def make_state(self, link_names: list[str], joint_names: list[str]) -> list[str]:
        return [*joint_names, *(name for name in MODEL_JOINTS if name not in joint_names)]

    def _limits(self, state: list[str], joint: float, wheel: float) -> list[float]:
        return [wheel if name in WHEEL_JOINTS else joint for name in state]

    def get_limit_q_lower(self, state: list[str]) -> list[float]:
        return self._limits(state, LOWER, float("-inf"))

    def get_limit_q_upper(self, state: list[str]) -> list[float]:
        return self._limits(state, UPPER, float("inf"))

    def get_limit_qdot_upper(self, state: list[str]) -> list[float]:
        return self._limits(state, SPEED, 15.7)


class FakeRobot:
    """Records calls; answers like ``rby1_sdk.create_robot_a(address)``."""

    def __init__(self, address: str) -> None:
        self.address = address
        self.linked = False
        self.manager = "Idle"
        self.emo = ["Released", "Released"]
        self.finish_code = "Ok"
        self.position = dict(HOME)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def connect(self, max_retries: int = 5, timeout_ms: int = 1000) -> bool:
        self.linked = self.address != "10.0.0.9:50051"
        return self.linked

    def is_connected(self) -> bool:
        return self.linked

    def power_on(self, dev_name: str) -> bool:
        self.calls.append(("power_on", (dev_name,)))
        return True

    def servo_on(self, dev_name: str) -> bool:
        self.calls.append(("servo_on", (dev_name,)))
        return True

    def enable_control_manager(self, unlimited_mode_enabled: bool = False) -> bool:
        self.calls.append(("enable_control_manager", ()))
        self.manager = "Enabled"
        return True

    def disable_control_manager(self) -> bool:
        self.calls.append(("disable_control_manager", ()))
        return True

    def get_control_manager_state(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(state=_enum(self.manager))

    def get_state(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            position=[self.position[name] for name in MODEL_JOINTS],
            velocity=[0.0] * len(MODEL_JOINTS),
            torque=[0.5] * len(MODEL_JOINTS),
            emo_states=[types.SimpleNamespace(state=_enum(state)) for state in self.emo],
            battery_state=types.SimpleNamespace(level_percent=87.0),
        )

    @staticmethod
    def model() -> types.SimpleNamespace:
        return types.SimpleNamespace(robot_joint_names=list(MODEL_JOINTS))

    def get_dynamics(self, urdf_model: str = "") -> FakeDynamics:
        return FakeDynamics()

    def create_command_stream(self, priority: int = 1) -> FakeStream:
        self.calls.append(("create_command_stream", (priority,)))
        return FakeStream(self)

    def cancel_control(self) -> bool:
        self.calls.append(("cancel_control", ()))
        return True

    def disconnect(self) -> None:
        self.calls.append(("disconnect", ()))
        self.linked = False


BUILDERS = (
    "RobotCommandBuilder",
    "ComponentBasedCommandBuilder",
    "BodyCommandBuilder",
    "HeadCommandBuilder",
    "JointPositionCommandBuilder",
    "CommandHeaderBuilder",
)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeRobot]:
    """Install the double as ``rby1_sdk`` and hand back the robots it built."""
    built: dict[str, FakeRobot] = {}

    def create_robot_a(address: str) -> FakeRobot:
        built["robot"] = FakeRobot(address)
        return built["robot"]

    module = types.ModuleType("rby1_sdk")
    module.create_robot_a = create_robot_a  # type: ignore[attr-defined]
    for name in BUILDERS:
        setattr(module, name, lambda name=name: Builder(name))
    monkeypatch.setitem(sys.modules, "rby1_sdk", module)
    return built


def _connected(sdk: dict[str, FakeRobot], **kwargs: Any) -> tuple[RBY1Driver, FakeRobot]:
    driver = RBY1Driver("rby1", port="192.168.30.1:50051", **kwargs)
    assert driver.connect_eagerly() is None
    return driver, sdk["robot"]


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0].get("text") or envelope["content"][0].get("json"))


def _commanded_home() -> list[float]:
    return [HOME[name] for name in JOINT_NAMES]


def test_the_rby1_is_built_by_the_native_driver() -> None:
    assert list_driver_coverage()["rby1"] == ("strands",)
    assert get_native_driver_class("rby1") is RBY1Driver
    assert missing_driver_members(RBY1Driver) == ()


def test_the_double_speaks_the_real_sdk() -> None:
    """Every call the double answers exists on the vendor's classes with the keywords used."""
    real = pytest.importorskip("rby1_sdk")
    graded = (
        (FakeRobot, real.Robot_A),
        (FakeStream, real.Robot_A_CommandStreamHandler),
        (FakeDynamics, real.dynamics.Robot_24),
    )
    for fake, vendor in graded:
        for name, member in vars(fake).items():
            if name.startswith("_") or not callable(member) and not isinstance(member, staticmethod):
                continue
            assert hasattr(vendor, name), f"{vendor.__name__}.{name}"
    # The driver's joint table is the vendor's body-then-head order.
    model = real.Model_A()
    names = list(model.robot_joint_names)
    assert [names[index] for index in [*model.body_idx, *model.head_idx]] == list(JOINT_NAMES)
    assert [names[index] for index in model.mobility_idx] == list(WHEEL_JOINTS)
    for name in BUILDERS:
        assert hasattr(real, name), name
    # The command the driver builds is one the real builders accept: body N=20, head N=2.
    driver = RBY1Driver("rby1")
    driver._sdk = real
    assert isinstance(driver._command([0.0] * len(JOINT_NAMES)), real.RobotCommandBuilder)


def test_connect_brings_the_robot_up_in_the_vendor_order(sdk: dict[str, FakeRobot]) -> None:
    driver, robot = _connected(sdk)
    assert robot.address == "192.168.30.1:50051"
    assert robot.calls == [("power_on", (".*",)), ("servo_on", (".*",)), ("enable_control_manager", ())]
    assert driver.get_observation() == HOME


@pytest.mark.parametrize(
    ("port", "setup", "expected"),
    [
        (None, {}, "no robot address"),
        ("10.0.0.9:50051", {}, "did not answer"),
        ("192.168.30.1:50051", {"emo": ["Released", "Pressed"]}, "emergency stop pressed (EMO [1])"),
        ("192.168.30.1:50051", {"manager": "MajorFault"}, "control manager in MajorFault"),
    ],
)
def test_connect_refuses_and_never_powers_a_faulted_robot(
    sdk: dict[str, FakeRobot], monkeypatch: pytest.MonkeyPatch, port: str | None, setup: dict[str, Any], expected: str
) -> None:
    original = FakeRobot.__init__

    def faulted(self: FakeRobot, address: str) -> None:
        original(self, address)
        for key, value in setup.items():
            setattr(self, key, value)

    monkeypatch.setattr(FakeRobot, "__init__", faulted)
    driver = RBY1Driver("rby1", port=port)
    reason = driver.connect_eagerly()
    assert reason is not None and expected in reason, reason
    assert not driver.is_connected
    assert all(call[0] != "power_on" for call in sdk.get("robot", FakeRobot("x")).calls)


def test_connect_without_the_sdk_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rby1_sdk", None)
    reason = RBY1Driver("rby1", port="10.0.0.2:50051").connect_eagerly()
    assert reason is not None and "strands-robots[rby1]" in reason


def test_send_action_streams_body_and_head_holding_omitted_joints(sdk: dict[str, FakeRobot]) -> None:
    driver, robot = _connected(sdk)
    target = {"right_arm_3": HOME["right_arm_3"] + 0.01, "head_1": HOME["head_1"] - 0.01}
    envelope = driver.send_action(target)
    assert envelope["status"] == "success", envelope
    expected = _commanded_home()
    expected[JOINT_NAMES.index("right_arm_3")] += 0.01
    expected[JOINT_NAMES.index("head_1")] -= 0.01
    assert robot.calls[-2:] == [("create_command_stream", (1,)), ("send_command", (expected,))]
    assert len(BODY_JOINTS) == 20


@pytest.mark.parametrize(
    ("action", "arrange", "expected"),
    [
        ({"right_wheel": 1.0}, None, "name no commanded RB-Y1 joint"),
        ({"gripper_finger_r1": 0.01}, None, "name no commanded RB-Y1 joint"),
        ({"torso_1": float("nan")}, None, "torso_1"),
        ({"torso_0": 2.5}, None, "outside the robot's range [-2.0000, 2.0000]"),
        # 1 rad/s at 50 Hz is 0.02 rad a period; 0.1 rad is a jump.
        ({"torso_0": HOME["torso_0"] + 0.1}, None, "velocity limit allows"),
        ({}, None, "nothing to command"),
        ({"torso_0": HOME["torso_0"]}, lambda robot: robot.emo.__setitem__(0, "Pressed"), "emergency stop"),
        ({"torso_0": HOME["torso_0"]}, lambda robot: setattr(robot, "manager", "MinorFault"), "MinorFault"),
        ({"torso_0": HOME["torso_0"]}, lambda robot: setattr(robot, "finish_code", "Preempted"), "Preempted"),
    ],
)
def test_send_action_refuses_what_the_robot_would_not_track(
    sdk: dict[str, FakeRobot], action: dict[str, Any], arrange: Any, expected: str
) -> None:
    driver, robot = _connected(sdk)
    if arrange is not None:
        arrange(robot)
    envelope = driver.send_action(action)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    if expected != "Preempted":
        assert all(call[0] != "send_command" for call in robot.calls)


def test_the_step_gate_is_measured_from_the_last_commanded_setpoint(sdk: dict[str, FakeRobot]) -> None:
    """A lagging robot does not turn a stream of small steps into refusals."""
    driver, robot = _connected(sdk)
    for step in range(1, 6):
        robot.position = dict(HOME)  # the robot has not moved at all
        assert driver.send_action({"left_arm_0": HOME["left_arm_0"] + 0.015 * step})["status"] == "success"
    assert sum(call[0] == "create_command_stream" for call in robot.calls) == 1


def test_state_names_all_24_measured_joints(sdk: dict[str, FakeRobot]) -> None:
    driver, _ = _connected(sdk)
    payload = driver.state()["content"][0]["json"]
    assert payload["joints"] == HOME
    assert set(payload["joint_efforts"]) == set(MODEL_JOINTS)
    status = asyncio.run(driver.get_status())["content"][0]["json"]
    assert (status["control_manager"], status["battery_pct"]) == ("Enabled", 87.0)


def test_run_policy_streams_through_send_action_and_stop_task_cancels(sdk: dict[str, FakeRobot]) -> None:
    driver, robot = _connected(sdk)

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        return {"head_0": observation["head_0"] + 0.01}

    assert driver.run_policy(policy, n_steps=5)["status"] == "success"
    deadline = time.monotonic() + 5.0
    while driver.get_task_status()["content"][0]["json"].get("running") and time.monotonic() < deadline:
        time.sleep(0.01)
    status = driver.get_task_status()["content"][0]["json"]
    assert status["steps"] == 5, status
    assert robot.position["head_0"] == pytest.approx(HOME["head_0"] + 0.05)
    stopped = driver.stop_task()
    assert stopped["status"] == "success", stopped
    assert robot.calls[-2:] == [("stream.cancel", ()), ("cancel_control", ())]
    # The next write opens a fresh stream anchored on the measured pose.
    assert driver.send_action({"head_0": robot.position["head_0"] + 0.01})["status"] == "success"
    assert robot.calls[-2][0] == "create_command_stream"


def test_the_agent_stop_verb_halts_and_cleanup_releases(sdk: dict[str, FakeRobot]) -> None:
    driver, robot = _connected(sdk)

    async def invoke() -> list[Any]:
        use = {"toolUseId": "t1", "name": "rby1", "input": {"action": "stop"}}
        return [chunk async for chunk in driver.stream(use, {})]  # type: ignore[arg-type]

    (result,) = asyncio.run(invoke())
    assert result["status"] == "success" and result["toolUseId"] == "t1"
    driver.cleanup()
    assert robot.calls[-3:] == [("cancel_control", ()), ("disable_control_manager", ()), ("disconnect", ())]
    assert not driver.is_connected
