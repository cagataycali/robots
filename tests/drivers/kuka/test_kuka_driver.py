"""The KUKA iiwa native driver against a controller double shaped like ``pyfri``.

The double answers the ``pyfri`` calls the session process makes and walks the
FRI callbacks the way ``ClientApplication::step`` does (``onStateChange`` on a
change, then ``monitor``/``waitForCommand``/``command`` by session state). One
cell grades its names and enum values against the real binding when it is
installed, and one runs the real session subprocess against an on-disk double
whose ``step`` never returns, the way ``pyfri`` blocks on a quiet controller.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import textwrap
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class, kuka, kuka_session, list_driver_coverage
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.kuka import JOINT_LIMITS, JOINT_NAMES, MAX_JOINT_SPEED, KukaDriver

SAMPLE_TIME = 0.002
HOME = [0.1, 0.5, -0.2, -1.2, 0.3, 0.9, -0.4]
#: ``ESessionState`` etc. values the double speaks, as the driver's tables define them.
STATE = {name: index for index, name in enumerate(kuka.SESSION_STATES)}


class Controller:
    """One Sunrise controller: what every message the double decodes carries."""

    def __init__(self) -> None:
        self.session = STATE["COMMANDING_ACTIVE"]
        self.safety, self.drive, self.quality, self.mode = 0, 2, 3, kuka.COMMAND_MODES.index("POSITION")
        self.measured = list(HOME)
        self.ipo = list(HOME)
        self.torque = [0.5 * i for i in range(7)]
        self.commanded: list[list[float]] = []
        self.bind_ok = True


CONTROLLER = Controller()


class _State:
    def getMeasuredJointPosition(self) -> list[float]:
        return list(CONTROLLER.measured)

    def getIpoJointPosition(self) -> list[float]:
        return list(CONTROLLER.ipo)

    def getMeasuredTorque(self) -> list[float]:
        return list(CONTROLLER.torque)

    def getExternalTorque(self) -> list[float]:
        return [0.0] * 7

    def getSessionState(self) -> int:
        return CONTROLLER.session

    def getSafetyState(self) -> int:
        return CONTROLLER.safety

    def getDriveState(self) -> int:
        return CONTROLLER.drive

    def getConnectionQuality(self) -> int:
        return CONTROLLER.quality

    def getClientCommandMode(self) -> int:
        return CONTROLLER.mode

    def getSampleTime(self) -> float:
        return SAMPLE_TIME


class _Command:
    def setJointPosition(self, values: list[float]) -> None:
        CONTROLLER.commanded.append(list(values))
        CONTROLLER.measured = list(values)  # the arm tracks the command exactly


class FakeLBRClient:
    def __init__(self) -> None:
        self._state, self._command = _State(), _Command()

    def robotState(self) -> _State:
        return self._state

    def robotCommand(self) -> _Command:
        return self._command


class FakeClientApplication:
    def __init__(self, client: FakeLBRClient) -> None:
        self.client, self.last = client, STATE["IDLE"]

    def connect(self, port: int, remoteHost: str | None = None) -> bool:
        return CONTROLLER.bind_ok

    def step(self) -> bool:
        time.sleep(SAMPLE_TIME)
        current = CONTROLLER.session
        if current != self.last:
            self.client.onStateChange(self.last, current)  # type: ignore[attr-defined]
            self.last = current
        callback = {1: "monitor", 2: "monitor", 3: "waitForCommand", 4: "command"}.get(current)
        if callback is not None:
            getattr(self.client, callback)()
        return True

    def disconnect(self) -> None:
        pass


class _ThreadSession:
    """``_launch``'s handle for a session run on a thread instead of a process."""

    def __init__(self, sdk: types.ModuleType, host: str | None, fri_port: int) -> None:
        to_child, inbox = os.pipe()[::-1]
        outbox, self.from_child = os.pipe()[::-1]
        self.to_child, self.code = to_child, None

        def run() -> None:
            try:
                self.code = kuka_session.run_session(sdk, inbox, outbox, host, fri_port)
            finally:
                os.close(inbox)
                os.close(outbox)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def poll(self) -> int | None:
        return None if self.thread.is_alive() else self.code

    def wait(self, timeout: float | None = None) -> int | None:
        self.thread.join(timeout)
        return self.code

    def terminate(self) -> None:
        raise AssertionError("the double's session always honours the stop frame")


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> Controller:
    global CONTROLLER
    CONTROLLER = Controller()
    sdk = types.ModuleType("pyfri")
    sdk.LBRClient, sdk.ClientApplication = FakeLBRClient, FakeClientApplication  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyfri", sdk)

    def launch(host: str | None, fri_port: int) -> tuple[_ThreadSession, int, int]:
        session = _ThreadSession(sdk, host, fri_port)
        return session, session.to_child, session.from_child

    monkeypatch.setattr(kuka, "_launch", launch)
    return CONTROLLER


def _connected(**kwargs: Any) -> KukaDriver:
    driver = KukaDriver(control_frequency=50.0, connect_timeout=2.0, **kwargs)
    assert driver.connect_eagerly() is None
    return driver


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0].get("text") or envelope["content"][0].get("json"))


def _settle(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_the_iiwa_is_built_by_the_native_driver() -> None:
    assert list_driver_coverage()["kuka_iiwa"] == ("strands",)
    assert get_native_driver_class("kuka_iiwa") is KukaDriver
    assert missing_driver_members(KukaDriver) == ()


def test_the_double_speaks_the_real_binding() -> None:
    """Every call the double answers exists in ``pyfri``, and the enum tables match its values."""
    fri = pytest.importorskip("pyfri")
    for fake, real in (
        (_State, fri.LBRState),
        (_Command, fri.LBRCommand),
        (FakeClientApplication, fri.ClientApplication),
    ):
        for name in vars(fake):
            if not name.startswith("_"):
                assert hasattr(real, name), (real.__name__, name)
    for table, enum in (
        (kuka.SESSION_STATES, fri.ESessionState),
        (kuka.SAFETY_STATES, fri.ESafetyState),
        (kuka.DRIVE_STATES, fri.EDriveState),
        (kuka.CONNECTION_QUALITIES, fri.EConnectionQuality),
        (kuka.COMMAND_MODES, fri.EClientCommandMode),
    ):
        assert {name: int(value) for name, value in enum.__members__.items()} == {n: i for i, n in enumerate(table)}


def test_the_limits_are_the_iiwa14_datasheet_rows() -> None:
    degrees = [round(math.degrees(v)) for v in JOINT_LIMITS], [round(math.degrees(v)) for v in MAX_JOINT_SPEED]
    assert degrees == ([170, 120, 170, 120, 170, 120, 175], [85, 85, 100, 75, 130, 135, 135])
    assert JOINT_NAMES == tuple(f"joint{i}" for i in range(1, 8))  # the MuJoCo asset's names


@pytest.mark.parametrize(
    ("arrange", "expected"),
    [
        (lambda c, m: _without_pyfri(m), "not on PyPI"),
        (lambda c, m: setattr(c, "bind_ok", False), "could not bind UDP port 30200"),
        (lambda c, m: setattr(c, "session", STATE["IDLE"]), "no FRI monitoring message"),
        # pyfri built on pybind11 2.11 under numpy 2 copies joint 1 into all seven slots.
        (lambda c, m: c.__dict__.update(measured=[0.3] * 7, torque=[1.5] * 7), "pybind11"),
    ],
)
def test_connect_refuses_and_commands_nothing(
    controller: Controller, monkeypatch: pytest.MonkeyPatch, arrange: Any, expected: str
) -> None:
    arrange(controller, monkeypatch)
    driver = KukaDriver(connect_timeout=0.3)
    reason = driver.connect_eagerly()
    assert reason is not None and expected in reason, reason
    assert not driver.is_connected and controller.commanded == []
    assert driver.send_action({"joint1": 0.1})["status"] == "error"


def _without_pyfri(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_module(name: str, package: str | None = None) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.delitem(sys.modules, "pyfri")
    monkeypatch.setattr(kuka.importlib, "import_module", no_module)


@pytest.mark.parametrize(
    ("action", "arrange", "expected"),
    [
        (
            {"joint1": 0.1},
            lambda c: setattr(c, "session", STATE["MONITORING_READY"]),
            "MONITORING_READY, not COMMANDING_ACTIVE",
        ),
        ({"joint1": 0.1}, lambda c: setattr(c, "mode", kuka.COMMAND_MODES.index("TORQUE")), "TORQUE, not POSITION"),
        ({"joint1": 0.1}, lambda c: setattr(c, "safety", 2), "SAFETY_STOP_LEVEL_1"),
        ({"joint1": 0.1}, lambda c: setattr(c, "drive", 0), "drive state is OFF"),
        ({"joint1": 0.1}, lambda c: setattr(c, "quality", 1), "quality is FAIR"),
        ({"gripper": 0.5}, None, "name no iiwa joint"),
        ({"joint3": float("nan")}, None, "joint3"),
        ({"joint2": 2.2}, None, "outside +/-2.0944"),
        # 75 deg/s at 50 Hz is 0.0262 rad a period on A4.
        ({"joint4": -1.2 + 0.05}, None, "rad/s allows"),
        ({}, None, "nothing to command"),
    ],
)
def test_send_action_refuses_what_the_controller_would_not_track(
    controller: Controller, action: dict[str, Any], arrange: Any, expected: str
) -> None:
    driver = _connected()
    if arrange is not None:
        arrange(controller)
        _settle(lambda: driver.state()["status"] == "error" or _changed(driver, controller))
    envelope = driver.send_action(action)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    time.sleep(0.02)
    assert all(position == HOME for position in controller.commanded)
    driver.cleanup()


def _changed(driver: KukaDriver, controller: Controller) -> bool:
    payload = driver.state()["content"][0]["json"]
    return (
        payload["session_state"] != "COMMANDING_ACTIVE"
        or payload["client_command_mode"] != "POSITION"
        or payload["safety_state"] != "NORMAL_OPERATION"
        or payload["drive_state"] != "ACTIVE"
        or payload["connection_quality"] != "EXCELLENT"
    )


def test_a_target_is_reached_no_faster_than_the_joint_speed_per_cycle(controller: Controller) -> None:
    driver = _connected()
    target = HOME[3] + 0.02
    assert driver.send_action({"joint4": target})["status"] == "success"
    _settle(lambda: controller.commanded and abs(controller.commanded[-1][3] - target) < 1e-12)
    steps = [abs(b[3] - a[3]) for a, b in zip(controller.commanded, controller.commanded[1:], strict=False)]
    assert max(steps) == pytest.approx(MAX_JOINT_SPEED[3] * SAMPLE_TIME)
    assert driver.state()["content"][0]["json"]["joints"]["joint4"] == pytest.approx(target)
    driver.cleanup()


def test_stop_holds_the_last_commanded_position_not_the_interpolated_one(controller: Controller) -> None:
    driver = _connected()
    controller.ipo = [0.0] * 7  # the Java motion's pose; going there would be a jump
    assert driver.send_action({"joint1": HOME[0] + 0.02})["status"] == "success"
    _settle(lambda: abs(controller.commanded[-1][0] - (HOME[0] + 0.02)) < 1e-12)
    assert driver.stop_task()["status"] == "success"
    held = len(controller.commanded)
    _settle(lambda: len(controller.commanded) > held + 20)
    assert all(position == controller.commanded[held - 1] for position in controller.commanded[held:])
    driver.cleanup()


def test_run_policy_streams_through_send_action_and_holds_when_it_ends(controller: Controller) -> None:
    driver = _connected()

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        return {"joint6": observation["joint6"] + MAX_JOINT_SPEED[5] / 100.0}

    assert driver.run_policy(policy, n_steps=5)["status"] == "success"
    _settle(lambda: not driver.get_task_status()["content"][0]["json"].get("running"))
    assert driver.get_task_status()["content"][0]["json"]["steps"] == 5
    count = len(controller.commanded)
    _settle(lambda: len(controller.commanded) > count + 10)
    assert controller.commanded[-1] == controller.commanded[-10]
    assert controller.commanded[-1][5] > HOME[5]
    driver.cleanup()


def test_the_agent_verbs_report_the_fri_state_and_stop(controller: Controller) -> None:
    driver = _connected(port="192.170.10.2")

    async def invoke(action: str) -> dict[str, Any]:
        use = {"toolUseId": "t1", "name": "kuka_iiwa", "input": {"action": action}}
        (result,) = [chunk async for chunk in driver.stream(use, {})]  # type: ignore[arg-type]
        return result

    status = asyncio.run(invoke("status"))["content"][0]["json"]
    assert (status["host"], status["session_state"], status["drive_state"]) == (
        "192.170.10.2",
        "COMMANDING_ACTIVE",
        "ACTIVE",
    )
    assert asyncio.run(invoke("stop"))["status"] == "success"
    assert asyncio.run(invoke("state"))["content"][0]["json"]["joint_efforts"]["joint7"] == 3.0
    driver.cleanup()
    assert not driver.is_connected


_SILENT_PYFRI = """
import time
class LBRClient:
    def robotState(self):
        return self
    def robotCommand(self):
        return self
    def setJointPosition(self, values):
        pass
    def getMeasuredJointPosition(self):
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    getIpoJointPosition = getMeasuredJointPosition
    def getMeasuredTorque(self):
        return [0.0] * 7
    getExternalTorque = getMeasuredTorque
    def getSessionState(self):
        return 2
    def getSafetyState(self):
        return 0
    getClientCommandMode = getSafetyState
    def getDriveState(self):
        return 2
    getConnectionQuality = getDriveState
    def getSampleTime(self):
        return 0.005
class ClientApplication:
    def __init__(self, client):
        self.client, self.steps = client, 0
    def connect(self, port, host=None):
        return True
    def step(self):
        self.steps += 1
        if self.steps > 1:
            time.sleep(3600)  # the controller went quiet: recvfrom never returns
        self.client.onStateChange(0, 2)
        self.client.monitor()
        return True
    def disconnect(self):
        pass
"""


def test_cleanup_ends_a_session_blocked_on_a_quiet_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real session subprocess, whose ``step`` never returns: cleanup still ends it."""
    (tmp_path / "pyfri.py").write_text(textwrap.dedent(_SILENT_PYFRI))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "pyfri", raising=False)
    driver = KukaDriver(connect_timeout=10.0)
    assert driver.connect_eagerly() is None
    assert driver.get_observation()["joint7"] == pytest.approx(0.7)
    session = driver._session
    started = time.monotonic()
    driver.cleanup()
    assert time.monotonic() - started < 5.0
    assert session.poll() is not None and not driver.is_connected
