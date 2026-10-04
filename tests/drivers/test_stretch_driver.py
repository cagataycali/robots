"""The Stretch native driver against a robot double shaped like ``stretch_body``.

The double stands in for ``stretch_body.robot``: a ``Robot`` whose lift and arm
are prismatic joints, whose head and end-of-arm are Dynamixel chains keyed by
motor name, whose base takes ``set_velocity`` and whose pimu reports the
runstop - each recording the calls it gets. One cell grades every method the
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

from strands_robots.drivers import get_native_driver_class, list_driver_coverage, stretch
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.stretch import StretchDriver, wheel_speeds

#: A Stretch 3 without a dex wrist: wrist_pitch and wrist_roll are absent.
HOME = {"lift": 0.6, "arm": 0.1, "wrist_yaw": 0.0, "head_pan": 0.0, "head_tilt": -0.5, "stretch_gripper": 0.0}
LIMITS = {"lift": (0.0, 1.1), "arm": (0.0, 0.52), "wrist_yaw": (-1.75, 4.0), "head_pan": (-3.9, 1.5)}
LIMITS |= {"head_tilt": (-1.53, 0.79), "stretch_gripper": (-1.0, 1.0)}
DRIVEN = ("lift", "arm", "wrist_yaw", "head_pan", "head_tilt")
#: Stretch 3 base parameters (robot_params_SE3): 0.3 m/s per wheel.
SEPARATION, WHEEL_MAX = 0.3153, 0.3


class Joint:
    """A prismatic joint (``Lift``/``Arm``) or one Dynamixel motor: status, limits, ``move_to``."""

    def __init__(self, robot: FakeRobot, name: str) -> None:
        self.robot, self.name = robot, name
        self.status = {"pos": HOME[name], "vel": 0.0}

    def get_soft_motion_limits(self) -> list[float]:
        return list(LIMITS[self.name])

    def move_to(self, x_m: float) -> None:
        self.robot.calls.append(("move_to", (self.name, x_m)))


class Chain:
    """``Head`` / ``EndOfArm``: motors by name, ``move_to(joint, x_r)``."""

    def __init__(self, robot: FakeRobot, names: tuple[str, ...]) -> None:
        self.motors = {name: Joint(robot, name) for name in names}

    def move_to(self, joint: str, x_r: float) -> None:
        self.motors[joint].move_to(x_r)


class Base:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.status = {"x": 1.0, "y": 0.5, "theta": 0.25, "x_vel": 0.0, "theta_vel": 0.0}
        self.params = {"wheel_separation_m": SEPARATION, "motion": {"max": {"vel_m": WHEEL_MAX}}}
        self.left_wheel = types.SimpleNamespace(gains={"enable_vel_watchdog": 1})
        self.right_wheel = types.SimpleNamespace(gains={"enable_vel_watchdog": 1})

    def set_velocity(self, v_m: float, w_r: float) -> None:
        self.robot.calls.append(("set_velocity", (v_m, w_r)))


class FakeRobot:
    """Records calls; answers like ``stretch_body.robot.Robot()``."""

    started = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.homed = True
        self.pimu = types.SimpleNamespace(status={"runstop_event": False, "voltage": 12.4})
        self.base = Base(self)
        self.lift, self.arm = Joint(self, "lift"), Joint(self, "arm")
        self.head = Chain(self, ("head_pan", "head_tilt"))
        self.end_of_arm = Chain(self, ("wrist_yaw", "stretch_gripper"))

    def startup(self) -> bool:
        self.calls.append(("startup", ()))
        return self.started

    def stop(self) -> None:
        self.calls.append(("stop", ()))

    def is_homed(self) -> bool:
        return self.homed

    def push_command(self) -> None:
        self.calls.append(("push_command", ()))


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> FakeRobot:
    """Install the double as ``stretch_body.robot`` and hand back the robot it builds."""
    built = FakeRobot()
    module = types.ModuleType("stretch_body.robot")
    module.Robot = lambda: built  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stretch_body.robot", module)
    return built


def _connected(robot: FakeRobot) -> StretchDriver:
    driver = StretchDriver("stretch3")
    assert driver.connect_eagerly() is None
    return driver


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0].get("text") or envelope["content"][0].get("json"))


def test_both_stretch_generations_are_built_by_the_native_driver() -> None:
    coverage = list_driver_coverage()
    assert (coverage["stretch"], coverage["stretch3"]) == (("strands",), ("strands",))
    assert get_native_driver_class("stretch3") is StretchDriver
    assert missing_driver_members(StretchDriver) == ()


def test_the_double_speaks_the_real_sdk() -> None:
    """Every call the double answers exists on the vendor's classes."""
    sdk = pytest.importorskip("stretch_body.robot")
    from stretch_body import base, dynamixel_hello_XL430, end_of_arm, head, prismatic_joint

    graded = (
        (FakeRobot, sdk.Robot),
        (Base, base.Base),
        (Joint, prismatic_joint.PrismaticJoint),
        (Joint, dynamixel_hello_XL430.DynamixelHelloXL430),
        (Chain, head.Head),
        (Chain, end_of_arm.EndOfArm),
    )
    for fake, vendor in graded:
        for name, member in vars(fake).items():
            if not name.startswith("_") and callable(member):
                assert hasattr(vendor, name), f"{vendor.__name__}.{name}"


def test_connect_starts_the_robot_and_reads_the_joints_it_has(robot: FakeRobot) -> None:
    driver = _connected(robot)
    assert robot.calls == [("startup", ())]
    assert driver.get_observation() == {name: HOME[name] for name in DRIVEN}
    payload = driver.state()["content"][0]["json"]
    assert payload["base"]["theta"] == 0.25


def _import_raising(error: BaseException) -> Any:
    def import_module(name: str) -> Any:
        raise error

    return import_module


@pytest.mark.parametrize(
    ("port", "started", "import_error", "expected"),
    [
        ("/dev/ttyUSB0", True, None, "names nothing"),
        (None, False, None, "another process may hold the robot"),
        (None, True, ImportError("No module named 'stretch_body'"), "pip install hello-robot-stretch-body"),
        # stretch_body reads the fleet calibration at import time.
        (None, True, KeyError("HELLO_FLEET_PATH"), "no robot configuration (KeyError"),
        (None, True, SystemExit(1), "no robot configuration (SystemExit"),
    ],
)
def test_connect_refuses_by_name(
    robot: FakeRobot,
    monkeypatch: pytest.MonkeyPatch,
    port: str | None,
    started: bool,
    import_error: BaseException | None,
    expected: str,
) -> None:
    robot.started = started
    if import_error is not None:
        monkeypatch.setattr(stretch.importlib, "import_module", _import_raising(import_error))
    driver = StretchDriver("stretch3", port=port)
    reason = driver.connect_eagerly()
    assert reason is not None and expected in reason, reason
    assert not driver.is_connected
    assert robot.calls == ([("startup", ()), ("stop", ())] if not started else [])


def test_send_action_queues_every_goal_then_pushes_once(robot: FakeRobot) -> None:
    driver = _connected(robot)
    envelope = driver.send_action({"lift": 0.8, "head_pan": -0.2, "wrist_yaw": 1.0})
    assert envelope["status"] == "success", envelope
    assert robot.calls[1:] == [
        ("move_to", ("lift", 0.8)),
        ("move_to", ("head_pan", -0.2)),
        ("move_to", ("wrist_yaw", 1.0)),
        ("push_command", ()),
    ]


@pytest.mark.parametrize(
    ("action", "arrange", "expected"),
    [
        ({"lift": 1.2}, None, "outside its soft limits [0.0000, 1.1000]"),
        ({"arm": float("nan")}, None, "arm"),
        ({"wrist_pitch": 0.1}, None, "name no commanded joint"),
        ({"stretch_gripper": 0.1}, None, "gripper is not driven"),
        ({}, None, "nothing to command"),
        ({"lift": 0.7}, lambda robot: setattr(robot, "homed", False), "not homed"),
        ({"lift": 0.7}, lambda robot: robot.pimu.status.update(runstop_event=True), "runstop is latched"),
    ],
)
def test_send_action_refuses_what_the_sdk_would_silently_change(
    robot: FakeRobot, action: dict[str, Any], arrange: Any, expected: str
) -> None:
    driver = _connected(robot)
    if arrange is not None:
        arrange(robot)
    envelope = driver.send_action(action)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    assert robot.calls == [("startup", ())]


def test_set_twist_drives_the_base_within_the_wheel_limit(robot: FakeRobot) -> None:
    driver = _connected(robot)
    envelope = driver.set_twist(vx=0.1, wz=0.5)
    assert envelope["status"] == "success", envelope
    assert robot.calls[1:] == [("set_velocity", (0.1, 0.5)), ("push_command", ())]
    left, right = envelope["content"][0]["json"]["wheel_speeds"]
    assert (left, right) == pytest.approx(wheel_speeds(0.1, 0.5, SEPARATION))


@pytest.mark.parametrize(
    ("twist", "arrange", "expected"),
    [
        ({"vy": 0.1}, None, "cannot strafe"),
        # 0.25 m/s forward plus 0.5 rad/s puts the outer wheel at 0.329 m/s.
        ({"vx": 0.25, "wz": 0.5}, None, "exceed the base's 0.300 m/s"),
        ({"vx": float("inf")}, None, "vx"),
        ({"vx": 0.1}, lambda robot: robot.base.right_wheel.gains.update(enable_vel_watchdog=0), "watchdog"),
        ({"vx": 0.1}, lambda robot: robot.pimu.status.update(runstop_event=True), "runstop is latched"),
    ],
)
def test_set_twist_refuses_what_the_base_would_clamp_or_run_away_with(
    robot: FakeRobot, twist: dict[str, float], arrange: Any, expected: str
) -> None:
    driver = _connected(robot)
    if arrange is not None:
        arrange(robot)
    envelope = driver.set_twist(**twist)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    assert robot.calls == [("startup", ())]


def test_run_policy_writes_through_send_action_and_stop_task_holds(robot: FakeRobot) -> None:
    driver = _connected(robot)

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        return {"head_pan": observation["head_pan"] + 0.01}

    assert driver.run_policy(policy, n_steps=3)["status"] == "success"
    deadline = time.monotonic() + 5.0
    while driver.get_task_status()["content"][0]["json"].get("running") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver.get_task_status()["content"][0]["json"]["steps"] == 3
    assert sum(call == ("push_command", ()) for call in robot.calls) == 3
    del robot.calls[1:]
    stopped = driver.stop_task()
    assert stopped["status"] == "success", stopped
    # The halt zeroes the twist and holds every joint at its measured position.
    assert robot.calls[1] == ("set_velocity", (0.0, 0.0))
    assert ("move_to", ("lift", HOME["lift"])) in robot.calls and robot.calls[-1] == ("push_command", ())


def test_the_agent_stop_verb_halts_and_cleanup_releases(robot: FakeRobot) -> None:
    driver = _connected(robot)

    async def invoke() -> list[Any]:
        use = {"toolUseId": "t1", "name": "stretch3", "input": {"action": "stop"}}
        return [chunk async for chunk in driver.stream(use, {})]  # type: ignore[arg-type]

    (result,) = asyncio.run(invoke())
    assert result["status"] == "success" and result["toolUseId"] == "t1"
    status = asyncio.run(driver.get_status())["content"][0]["json"]
    assert (status["runstop"], status["homed"], status["voltage"]) == (False, True, 12.4)
    driver.cleanup()
    assert robot.calls[-1] == ("stop", ()) and not driver.is_connected
