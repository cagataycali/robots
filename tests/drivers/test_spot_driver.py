"""The Spot native driver against a robot double shaped like ``bosdyn-client``.

The double stands in for the SDK pieces the driver touches: a ``Robot`` that
authenticates, time-syncs, reports E-stop, power and arm, and hands out a
robot-state, a robot-command and a lease client; a ``LeaseKeepAlive``; and a
``RobotCommandBuilder`` whose commands are recorded as tuples. One cell grades
every call against the real SDK - the methods and keywords, the commands the
real builder makes from the driver's arguments, and the robot-state message
the driver parses.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import types
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class, list_driver_coverage, spot
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.spot import ARM_JOINTS, JOINT_NAMES, SpotDriver, sdk_joint_name

#: Spot with its arm stowed (radians); the claw closed.
HOME = {name: 0.0 for name in JOINT_NAMES} | {"fl_hy": 0.8, "fl_kn": -1.6, "arm_sh1": -3.1, "arm_el0": 3.1}


class ClaimedError(Exception):
    """``bosdyn.client.lease.ResourceAlreadyClaimedError``."""


class SdkError(Exception):
    """``bosdyn.client.exceptions.Error``, the SDK's base error."""


def _robot_state(joints: dict[str, float], battery: float = 87.0) -> Any:
    measured = [
        types.SimpleNamespace(
            name=sdk_joint_name(name),
            position=types.SimpleNamespace(value=value),
            velocity=types.SimpleNamespace(value=0.0),
        )
        for name, value in joints.items()
    ]
    return types.SimpleNamespace(
        kinematic_state=types.SimpleNamespace(joint_states=measured),
        battery_states=[types.SimpleNamespace(charge_percentage=types.SimpleNamespace(value=battery))],
    )


class FakeStateClient:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot

    def get_robot_state(self) -> Any:
        return self.robot.state


class FakeCommandClient:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot

    def robot_command(self, command: Any, end_time_secs: float | None = None) -> int:
        self.robot.calls.append(("robot_command", command, end_time_secs))
        return 1


class FakeTimeSync:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot

    def wait_for_sync(self) -> None:
        self.robot.calls.append(("wait_for_sync",))

    def stop(self) -> None:
        self.robot.calls.append(("time_sync_stop",))


class FakeRobot:
    """Answers like ``bosdyn.client.robot.Robot``; records every call that changes the robot."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.estopped, self.powered, self.arm, self.lease_taken = False, True, True, False
        self.state = _robot_state(HOME)
        self.time_sync = FakeTimeSync(self)
        self.clients = {"robot-state": FakeStateClient(self), "robot-command": FakeCommandClient(self), "lease": self}

    def authenticate(self, username: str, password: str) -> None:
        self.calls.append(("authenticate", username, password))

    def ensure_client(self, service_name: str) -> Any:
        return self.clients[service_name]

    def is_estopped(self) -> bool:
        return self.estopped

    def is_powered_on(self) -> bool:
        return self.powered

    def power_on(self, timeout_sec: float = 20) -> None:
        self.calls.append(("power_on",))
        self.powered = True

    def power_off(self, cut_immediately: bool = False, timeout_sec: float = 20) -> None:
        self.calls.append(("power_off", cut_immediately))
        self.powered = False

    def has_arm(self) -> bool:
        return self.arm


class FakeKeepAlive:
    """``LeaseKeepAlive``: acquiring raises when another client holds the body."""

    def __init__(self, lease_client: FakeRobot, must_acquire: bool = False, return_at_exit: bool = False) -> None:
        if lease_client.lease_taken:
            raise ClaimedError("body lease is held by tablet")
        self.robot = lease_client
        self.robot.calls.append(("lease_acquire", must_acquire, return_at_exit))

    def is_alive(self) -> bool:
        return True

    def shutdown(self) -> None:
        self.robot.calls.append(("lease_return",))


class FakeBuilder:
    """``RobotCommandBuilder``: each command is a tuple naming what was built."""

    @staticmethod
    def arm_joint_command(*angles: float) -> tuple[Any, ...]:
        return ("arm", angles)

    @staticmethod
    def claw_gripper_open_angle_command(gripper_q: float, build_on_command: Any = None) -> tuple[Any, ...]:
        return ("claw", gripper_q, build_on_command)

    @staticmethod
    def synchro_velocity_command(v_x: float, v_y: float, v_rot: float) -> tuple[Any, ...]:
        return ("twist", v_x, v_y, v_rot)

    @staticmethod
    def stop_command() -> tuple[Any, ...]:
        return ("stop",)


def _sdk(robot: FakeRobot) -> Any:
    def blocking_stand(command_client: Any, timeout_sec: float = 10) -> None:
        robot.calls.append(("blocking_stand",))

    service = types.SimpleNamespace
    return types.SimpleNamespace(
        client=service(create_standard_sdk=lambda name: service(create_robot=lambda host: robot)),
        command=service(
            RobotCommandBuilder=FakeBuilder,
            RobotCommandClient=service(default_service_name="robot-command"),
            blocking_stand=blocking_stand,
        ),
        lease=service(
            LeaseClient=service(default_service_name="lease"),
            LeaseKeepAlive=FakeKeepAlive,
            ResourceAlreadyClaimedError=ClaimedError,
        ),
        state=service(RobotStateClient=service(default_service_name="robot-state")),
        exceptions=service(Error=SdkError),
    )


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> FakeRobot:
    """Install the double as the resolved SDK and hand back the robot it connects to."""
    built = FakeRobot()
    monkeypatch.setattr(spot, "_resolve_sdk", lambda: _sdk(built))
    monkeypatch.setenv("BOSDYN_CLIENT_USERNAME", "user")
    monkeypatch.setenv("BOSDYN_CLIENT_PASSWORD", "secret")
    return built


def _connected(robot: FakeRobot) -> SpotDriver:
    driver = SpotDriver("spot", port="192.168.80.3")
    assert driver.connect_eagerly() is None
    del robot.calls[:]
    return driver


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0].get("text") or envelope["content"][0].get("json"))


def _commands(robot: FakeRobot) -> list[Any]:
    return [call[1] for call in robot.calls if call[0] == "robot_command"]


def test_spot_is_built_by_the_native_driver() -> None:
    assert list_driver_coverage()["spot"] == ("strands",)
    assert get_native_driver_class("spot") is SpotDriver
    assert missing_driver_members(SpotDriver) == ()


def test_the_double_speaks_the_real_sdk() -> None:
    """The calls, keywords, commands and robot-state message the driver uses exist in bosdyn-client."""
    pytest.importorskip("bosdyn.client")
    from bosdyn.api import robot_state_pb2
    from bosdyn.client import lease, robot, robot_command, robot_state

    graded = (
        (FakeRobot, robot.Robot),
        (FakeStateClient, robot_state.RobotStateClient),
        (FakeCommandClient, robot_command.RobotCommandClient),
        (FakeKeepAlive, lease.LeaseKeepAlive),
        (FakeBuilder, robot_command.RobotCommandBuilder),
    )
    for fake, vendor in graded:
        for name in vars(fake):
            if name.startswith("_") or not callable(getattr(fake, name)):
                continue
            params = inspect.signature(getattr(fake, name)).parameters.values()
            keywords = {p.name: None for p in params if p.name != "self" and p.kind is not p.VAR_POSITIONAL}
            inspect.signature(getattr(vendor, name)).bind_partial(**keywords)
    inspect.signature(robot_command.blocking_stand).bind_partial(None, timeout_sec=1)

    real = FakeRobot()
    real.state = robot_state_pb2.RobotState()
    for name, value in HOME.items():
        joint = real.state.kinematic_state.joint_states.add(name=sdk_joint_name(name))
        joint.position.value = value
    driver = SpotDriver("spot", port="192.168.80.3")
    sdk = _sdk(real)
    sdk.command = types.SimpleNamespace(
        **{**vars(sdk.command), "RobotCommandBuilder": robot_command.RobotCommandBuilder}
    )
    driver._sdk, driver._robot, driver._has_arm = sdk, real, True
    driver._state_client, driver._command_client = FakeStateClient(real), FakeCommandClient(real)
    assert driver.get_observation() == HOME
    assert driver.send_action({"arm_sh0": 0.5, "arm_f1x": -1.0})["status"] == "success"
    (command,) = _commands(real)
    point = command.synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0].position
    assert (point.sh0.value, point.sh1.value, point.el0.value) == pytest.approx((0.5, -3.1, 3.1))
    claw = command.synchronized_command.gripper_command.claw_gripper_command.trajectory.points[0]
    assert claw.point == pytest.approx(-1.0)


def test_connect_takes_the_lease_and_reads_every_joint(robot: FakeRobot) -> None:
    driver = SpotDriver("spot", port="192.168.80.3")
    assert driver.connect_eagerly() is None
    assert robot.calls == [("authenticate", "user", "secret"), ("wait_for_sync",), ("lease_acquire", True, True)]
    assert driver.get_observation() == HOME
    status = asyncio.run(driver.get_status())["content"][0]["json"]
    assert (status["estopped"], status["powered_on"], status["lease_held"], status["battery_pct"]) == (
        False,
        True,
        True,
        87.0,
    )


@pytest.mark.parametrize(
    ("port", "env", "arrange", "expected"),
    [
        (None, True, None, "no robot address"),
        ("192.168.80.3", False, None, "BOSDYN_CLIENT_USERNAME"),
        ("192.168.80.3", True, lambda robot: setattr(robot, "lease_taken", True), "holds 192.168.80.3's body lease"),
        ("192.168.80.3", True, lambda robot: setattr(robot, "authenticate", _raise), "SdkError: bad login"),
    ],
)
def test_connect_refuses_by_name(
    robot: FakeRobot, monkeypatch: pytest.MonkeyPatch, port: str | None, env: bool, arrange: Any, expected: str
) -> None:
    if not env:
        monkeypatch.delenv("BOSDYN_CLIENT_PASSWORD")
    if arrange is not None:
        arrange(robot)
    driver = SpotDriver("spot", port=port)
    reason = driver.connect_eagerly()
    assert reason is not None and expected in reason, reason
    assert not driver.is_connected
    assert ("lease_acquire", True, True) not in robot.calls


def _raise(*args: Any) -> None:
    raise SdkError("bad login")


def test_the_sdk_missing_is_a_reason_naming_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(spot.importlib, "import_module", missing)
    monkeypatch.setenv("BOSDYN_CLIENT_USERNAME", "user")
    monkeypatch.setenv("BOSDYN_CLIENT_PASSWORD", "secret")
    reason = SpotDriver("spot", port="192.168.80.3").connect_eagerly()
    assert reason is not None and "strands-robots[spot]" in reason


def test_send_action_moves_the_arm_and_claw_in_one_command(robot: FakeRobot) -> None:
    driver = _connected(robot)
    envelope = driver.send_action({"arm_el1": 0.4, "arm_f1x": -1.2})
    assert envelope["status"] == "success", envelope
    arm = ("arm", tuple(0.4 if name == "arm_el1" else HOME[name] for name in ARM_JOINTS))
    assert _commands(robot) == [("claw", -1.2, arm)]
    assert driver.send_action({"arm_f1x": 0.0})["status"] == "success"
    assert _commands(robot)[-1] == ("claw", 0.0, None)


@pytest.mark.parametrize(
    ("action", "arrange", "expected"),
    [
        ({"fl_kn": -1.0}, None, "locomotion controller owns"),
        ({"arm_el0": -0.1}, None, "outside the arm's limits [0.0000, 3.1416]"),
        ({"arm_f1x": 0.2}, None, "outside the arm's limits [-1.5708, 0.0000]"),
        ({"arm_sh0": float("nan")}, None, "arm_sh0"),
        ({"gripper": 0.0}, None, "name no Spot joint"),
        ({}, None, "nothing to command"),
        ({"arm_sh0": 0.1}, lambda robot: setattr(robot, "estopped", True), "E-stopped"),
        ({"arm_sh0": 0.1}, lambda robot: setattr(robot, "powered", False), "motors are off"),
    ],
)
def test_send_action_refuses_before_it_writes(
    robot: FakeRobot, action: dict[str, Any], arrange: Any, expected: str
) -> None:
    driver = _connected(robot)
    if arrange is not None:
        arrange(robot)
    envelope = driver.send_action(action)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    assert _commands(robot) == []


def test_an_armless_spot_refuses_arm_actions(robot: FakeRobot) -> None:
    robot.arm = False
    driver = _connected(robot)
    envelope = driver.send_action({"arm_sh0": 0.1})
    assert envelope["status"] == "error" and "no arm" in _text(envelope)


def test_set_twist_expires_on_its_own(robot: FakeRobot) -> None:
    driver = _connected(robot)
    before = time.time()
    assert driver.set_twist(vx=0.5, vy=-0.2, wz=0.3)["status"] == "success"
    ((_, command, end_time),) = robot.calls
    assert command == ("twist", 0.5, -0.2, 0.3)
    assert before + spot.VELOCITY_COMMAND_DURATION <= end_time <= time.time() + spot.VELOCITY_COMMAND_DURATION
    robot.powered = False
    assert "motors are off" in _text(driver.set_twist(vx=0.1))
    assert "vx" in _text(driver.set_twist(vx=float("inf")))
    assert len(robot.calls) == 1


def test_stand_powers_on_then_stands_unless_estopped(robot: FakeRobot) -> None:
    robot.powered = False
    driver = _connected(robot)
    assert driver.stand()["status"] == "success"
    assert robot.calls == [("power_on",), ("blocking_stand",)]
    robot.estopped = True
    assert "E-stopped" in _text(driver.stand())
    assert len(robot.calls) == 2


def test_run_policy_writes_through_send_action_and_stop_task_halts(robot: FakeRobot) -> None:
    driver = _connected(robot)

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        return {"arm_wr1": observation["arm_wr1"] + 0.01}

    assert driver.run_policy(policy, n_steps=3)["status"] == "success"
    deadline = time.monotonic() + 5.0
    while driver.get_task_status()["content"][0]["json"].get("running") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver.get_task_status()["content"][0]["json"]["steps"] == 3
    assert len(_commands(robot)) == 3
    assert driver.stop_task()["status"] == "success"
    assert _commands(robot)[-1] == ("stop",)


def test_the_agent_stop_verb_halts_and_cleanup_sits_powers_off_and_returns_the_lease(robot: FakeRobot) -> None:
    driver = _connected(robot)

    async def invoke() -> list[Any]:
        use = {"toolUseId": "t1", "name": "spot", "input": {"action": "stop"}}
        return [chunk async for chunk in driver.stream(use, {})]  # type: ignore[arg-type]

    (result,) = asyncio.run(invoke())
    assert result["status"] == "success" and result["toolUseId"] == "t1"
    assert _commands(robot) == [("stop",)]
    del robot.calls[:]
    driver.cleanup()
    assert robot.calls == [
        ("robot_command", ("stop",), None),
        ("power_off", False),
        ("lease_return",),
        ("time_sync_stop",),
    ]
    assert not driver.is_connected
    driver.cleanup()
    assert len(robot.calls) == 4
