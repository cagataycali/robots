"""The xArm 7 native driver against a controller double shaped like ``XArmAPI``.

The double records every call it is handed and answers the SDK's own return
shapes (``code`` or ``(code, value)``), and one cell grades it against the real
``xarm.wrapper.XArmAPI`` so the double cannot drift into an API the vendor does
not have.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
import types
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class, list_driver_coverage
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.xarm import JOINT_NAMES, XArmDriver

HOME = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]


class FakeXArm:
    """Records calls; answers like the SDK's ``XArmAPI(port, is_radian=True)``.

    ``error_code`` and ``state`` start at the SDK's pre-report cache values
    (``0`` and ``4``); the error word the controller really holds is only
    visible through ``get_err_warn_code``, as on a freshly connected arm.
    """

    def __init__(self, port: str, is_radian: bool = False) -> None:
        self.port, self.is_radian = port, is_radian
        self.connected = True
        self.error_code = 0
        self.warn_code = 0
        self.state = 4
        self.held_error = 0
        self.mode = 0
        self.joint_speed_limit = [0.0001, 3.0]
        self.angles = list(HOME)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.servo_code = 0

    def get_err_warn_code(self, show: bool = False, lang: str = "en") -> tuple[int, list[int]]:
        self.error_code = self.held_error
        return 0, [self.held_error, self.warn_code]

    def motion_enable(self, enable: bool = True) -> int:
        self.calls.append(("motion_enable", (enable,)))
        return 0

    def set_mode(self, mode: int = 0) -> int:
        self.calls.append(("set_mode", (mode,)))
        self.mode = mode
        return 0

    def set_state(self, state: int = 0) -> int:
        self.calls.append(("set_state", (state,)))
        self.state = 4 if state == 4 else 1
        return 0

    def get_servo_angle(self, is_radian: bool | None = None) -> tuple[int, list[float]]:
        return 0, list(self.angles)

    def get_joint_states(self, is_radian: bool | None = None) -> tuple[int, list[list[float]]]:
        return 0, [list(self.angles), [0.0] * 7, [0.5] * 7]

    def get_position(self, is_radian: bool | None = None) -> tuple[int, list[float]]:
        return 0, [207.0, 0.0, 112.0, 3.14, 0.0, 0.0]

    def set_servo_angle_j(self, angles: list[float], is_radian: bool | None = None) -> int:
        self.calls.append(("set_servo_angle_j", (list(angles),)))
        if self.servo_code == 0:
            self.angles = list(angles)
        return self.servo_code

    def disconnect(self) -> None:
        self.calls.append(("disconnect", ()))
        self.connected = False


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeXArm]:
    """Install the double as ``xarm.wrapper`` and hand back the arms it built."""
    built: dict[str, FakeXArm] = {}

    def factory(port: str, is_radian: bool = False) -> FakeXArm:
        built["arm"] = FakeXArm(port, is_radian)
        return built["arm"]

    module = types.ModuleType("xarm.wrapper")
    module.XArmAPI = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarm.wrapper", module)
    return built


def _connected(sdk: dict[str, FakeXArm], **kwargs: Any) -> tuple[XArmDriver, FakeXArm]:
    driver = XArmDriver("xarm7", port="192.168.1.185", **kwargs)
    assert driver.connect_eagerly() is None
    return driver, sdk["arm"]


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0].get("text") or envelope["content"][0].get("json"))


def test_the_xarm7_is_built_by_the_native_driver() -> None:
    assert list_driver_coverage()["xarm7"] == ("strands",)
    assert get_native_driver_class("xarm7") is XArmDriver
    assert missing_driver_members(XArmDriver) == ()


def test_the_double_speaks_the_real_sdk() -> None:
    """Every call the double answers exists on the vendor's XArmAPI with the keyword used."""
    real = pytest.importorskip("xarm.wrapper").XArmAPI
    for name, member in vars(FakeXArm).items():
        if name.startswith("_") or not callable(member):
            continue
        params = set(inspect.signature(getattr(real, name)).parameters)
        assert set(inspect.signature(member).parameters) - {"self"} <= params | {"kwargs"}, name
    for prop in ("connected", "error_code", "warn_code", "state", "mode", "joint_speed_limit"):
        assert isinstance(getattr(real, prop), property), prop


def test_connect_enters_servo_mode_in_the_vendor_order(sdk: dict[str, FakeXArm]) -> None:
    driver, arm = _connected(sdk)
    assert arm.port == "192.168.1.185" and arm.is_radian is True
    assert arm.calls == [("motion_enable", (True,)), ("set_mode", (1,)), ("set_state", (0,))]
    assert driver.get_observation() == dict(zip(JOINT_NAMES, HOME, strict=True))


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ({"port": None}, "no controller address"),
        ({"held_error": 22}, "holds error code 22"),
        ({"connected": False}, "did not answer"),
        ({"raises": Exception("connect socket failed")}, "did not answer: connect socket failed"),
    ],
)
def test_connect_refuses_and_never_energises_a_faulted_arm(
    sdk: dict[str, FakeXArm], monkeypatch: pytest.MonkeyPatch, setup: dict[str, Any], expected: str
) -> None:
    original = FakeXArm.__init__

    def faulted(self: FakeXArm, port: str, is_radian: bool = False) -> None:
        if "raises" in setup:
            raise setup["raises"]
        original(self, port, is_radian)
        for key, value in setup.items():
            if key != "port":
                setattr(self, key, value)

    monkeypatch.setattr(FakeXArm, "__init__", faulted)
    driver = XArmDriver("xarm7", port=setup.get("port", "192.168.1.185"))
    reason = driver.connect_eagerly()
    assert reason is not None and expected in reason
    assert not driver.is_connected
    assert all(call[0] != "motion_enable" for call in (sdk["arm"].calls if "arm" in sdk else []))


def test_connect_without_the_sdk_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "xarm.wrapper", None)
    reason = XArmDriver("xarm7", port="10.0.0.2").connect_eagerly()
    assert reason is not None and "strands-robots[xarm]" in reason


def test_send_action_writes_the_whole_arm_holding_omitted_joints(sdk: dict[str, FakeXArm]) -> None:
    driver, arm = _connected(sdk)
    envelope = driver.send_action({"joint1": 0.01, "joint4": 0.92})
    assert envelope["status"] == "success", envelope
    expected = list(HOME)
    expected[0], expected[3] = 0.01, 0.92
    assert arm.calls[-1] == ("set_servo_angle_j", (expected,))


@pytest.mark.parametrize(
    ("action", "arrange", "expected"),
    [
        ({"gripper": 100.0}, None, "name no xArm 7 joint"),
        ({"joint1": float("nan")}, None, "joint1"),
        # 3.0 rad/s at 100 Hz is 0.03 rad a period; 0.5 rad is a jump.
        ({"joint1": 0.5}, None, "joint speed limit"),
        ({"joint1": 0.01}, lambda arm: setattr(arm, "error_code", 31), "holds error code 31"),
        ({"joint1": 0.01}, lambda arm: setattr(arm, "servo_code", 9), "with code 9"),
        ({}, None, "nothing to command"),
    ],
)
def test_send_action_refuses_what_the_controller_would_not_track(
    sdk: dict[str, FakeXArm], action: dict[str, Any], arrange: Any, expected: str
) -> None:
    driver, arm = _connected(sdk)
    if arrange is not None:
        arrange(arm)
    envelope = driver.send_action(action)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    if arrange is None:
        assert all(call[0] != "set_servo_angle_j" for call in arm.calls)


def test_the_step_gate_is_measured_from_the_last_commanded_setpoint(sdk: dict[str, FakeXArm]) -> None:
    """A lagging arm does not turn a stream of small steps into refusals."""
    driver, arm = _connected(sdk)
    arm.servo_code = 0
    for step in range(1, 6):
        arm.angles = list(HOME)  # the arm has not moved at all
        assert driver.send_action({"joint1": 0.02 * step})["status"] == "success"


def test_state_names_positions_velocities_and_efforts(sdk: dict[str, FakeXArm]) -> None:
    driver, _ = _connected(sdk)
    payload = driver.state()["content"][0]["json"]
    assert payload["joints"] == dict(zip(JOINT_NAMES, HOME, strict=True))
    assert set(payload["joint_efforts"]) == set(JOINT_NAMES)
    assert payload["tcp_pose"][:3] == [207.0, 0.0, 112.0]


def test_run_policy_streams_through_send_action_and_stop_task_rearms(sdk: dict[str, FakeXArm]) -> None:
    driver, arm = _connected(sdk)

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        return {"joint1": observation["joint1"] + 0.01}

    assert driver.run_policy(policy, n_steps=5)["status"] == "success"
    deadline = time.monotonic() + 5.0
    while driver.get_task_status()["content"][0]["json"].get("running") and time.monotonic() < deadline:
        time.sleep(0.01)
    status = driver.get_task_status()["content"][0]["json"]
    assert status["steps"] == 5, status
    assert arm.angles[0] == pytest.approx(0.05)
    stopped = driver.stop_task()
    assert stopped["status"] == "success", stopped
    assert arm.calls[-3:] == [("set_state", (4,)), ("set_mode", (1,)), ("set_state", (0,))]


def test_the_agent_stop_verb_halts_and_cleanup_releases(sdk: dict[str, FakeXArm]) -> None:
    driver, arm = _connected(sdk)

    async def invoke() -> list[Any]:
        use = {"toolUseId": "t1", "name": "xarm7", "input": {"action": "stop"}}
        return [chunk async for chunk in driver.stream(use, {})]  # type: ignore[arg-type]

    (result,) = asyncio.run(invoke())
    assert result["status"] == "success" and result["toolUseId"] == "t1"
    driver.cleanup()
    assert arm.calls[-2:] == [("set_state", (4,)), ("disconnect", ())]
    assert not driver.is_connected
