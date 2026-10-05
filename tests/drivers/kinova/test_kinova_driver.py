"""The Kinova Gen3 native driver against a base double shaped like ``kortex_api``.

The double installs the seven ``kortex_api`` modules the driver imports and
answers with the SDK's own shapes (``RefreshFeedback().actuators[i].position``
in degrees on ``0..360``, ``GetArmState().active_state`` as the ``ArmState``
enum). One cell grades it against the real vendor wheel, so the double cannot
drift into an API Kinova does not ship.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import sys
import time
import types
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class, list_driver_coverage
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.kinova import JOINT_NAMES, MAX_JOINT_SPEED, KinovaDriver

#: ``Base_pb2`` enum values, as the vendor wheel defines them.
ARM_STATES = {"ARMSTATE_IN_FAULT": 4, "ARMSTATE_MAINTENANCE": 5, "ARMSTATE_SERVOING_READY": 7}
SINGLE_LEVEL_SERVOING = 2

#: Retract-ish pose in wire degrees; joint_1 sits just short of 180 so a step crosses pi.
HOME_DEG = [179.5, 15.0, 180.0, 230.0, 0.0, 55.0, 90.0]


class FakeBase:
    """One Gen3 base: the state every client of the double reads and writes."""

    def __init__(self) -> None:
        self.positions = list(HOME_DEG)
        self.actuator_count = 7
        self.arm_state = ARM_STATES["ARMSTATE_SERVOING_READY"]
        self.session_error: Exception | None = None
        self.transport_error: OSError | None = None
        self.calls: list[tuple[str, Any]] = []


class FakeTransport:
    def connect(self, ip: str, port: int) -> None:
        if BASE.transport_error is not None:
            raise BASE.transport_error
        BASE.calls.append(("connect", (ip, port)))

    def disconnect(self) -> None:
        BASE.calls.append(("disconnect", ()))


class FakeRouterClient:
    def __init__(self, transport: FakeTransport, errorCallback: Any = None) -> None:
        self.transport = transport


class FakeSessionManager:
    def __init__(self, router: FakeRouterClient) -> None:
        self.router = router

    def CreateSession(self, createSessionInfo: Any) -> None:
        if BASE.session_error is not None:
            raise BASE.session_error
        BASE.calls.append(("CreateSession", createSessionInfo.username))

    def CloseSession(self) -> None:
        BASE.calls.append(("CloseSession", ()))


class FakeBaseClient:
    def __init__(self, router: FakeRouterClient) -> None:
        self.router = router

    def GetActuatorCount(self) -> SimpleNamespace:
        return SimpleNamespace(count=BASE.actuator_count)

    def GetArmState(self) -> SimpleNamespace:
        return SimpleNamespace(active_state=BASE.arm_state)

    def SetServoingMode(self, servoingmodeinformation: Any) -> None:
        BASE.calls.append(("SetServoingMode", servoingmodeinformation.servoing_mode))

    def SendJointSpeedsCommand(self, jointspeeds: Any) -> None:
        BASE.calls.append(("SendJointSpeedsCommand", [(s.joint_identifier, s.value) for s in jointspeeds.joint_speeds]))

    def Stop(self) -> None:
        BASE.calls.append(("Stop", ()))


class FakeBaseCyclicClient:
    def __init__(self, router: FakeRouterClient) -> None:
        self.router = router

    def RefreshFeedback(self) -> SimpleNamespace:
        actuators = [SimpleNamespace(position=p, velocity=0.0, torque=1.5) for p in BASE.positions]
        pose = dict.fromkeys(("x", "y", "z", "theta_x", "theta_y", "theta_z"), 0.0) | {"x": 0.45}
        base = SimpleNamespace(active_state=BASE.arm_state, **{f"tool_pose_{k}": v for k, v in pose.items()})
        return SimpleNamespace(base=base, actuators=actuators)


class FakeKException(Exception):
    pass


BASE = FakeBase()


@pytest.fixture
def base(monkeypatch: pytest.MonkeyPatch) -> FakeBase:
    """Install the double as the ``kortex_api`` modules and hand back the base."""
    global BASE
    BASE = FakeBase()
    names = {v: k for k, v in ARM_STATES.items()}
    pb2 = types.SimpleNamespace(
        ServoingModeInformation=lambda **kw: SimpleNamespace(**kw),
        JointSpeeds=lambda **kw: SimpleNamespace(**kw),
        JointSpeed=lambda **kw: SimpleNamespace(**kw),
        ArmState=SimpleNamespace(Name=lambda value: names.get(value, f"ARMSTATE_{value}")),
        SINGLE_LEVEL_SERVOING=SINGLE_LEVEL_SERVOING,
        **ARM_STATES,
    )
    modules: dict[str, dict[str, Any]] = {
        "kortex_api.TCPTransport": {"TCPTransport": FakeTransport},
        "kortex_api.RouterClient": {"RouterClient": FakeRouterClient},
        "kortex_api.SessionManager": {"SessionManager": FakeSessionManager},
        "kortex_api.autogen.client_stubs.BaseClientRpc": {"BaseClient": FakeBaseClient},
        "kortex_api.autogen.client_stubs.BaseCyclicClientRpc": {"BaseCyclicClient": FakeBaseCyclicClient},
        "kortex_api.autogen.messages.Base_pb2": vars(pb2),
        "kortex_api.autogen.messages.Session_pb2": {"CreateSessionInfo": lambda **kw: SimpleNamespace(**kw)},
        "kortex_api.Exceptions.KException": {"KException": FakeKException},
    }
    for name, members in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    return BASE


def _connected(**kwargs: Any) -> KinovaDriver:
    driver = KinovaDriver("kinova_gen3", port="192.168.1.10", **kwargs)
    assert driver.connect_eagerly() is None
    return driver


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0].get("text") or envelope["content"][0].get("json"))


def _speeds(base: FakeBase) -> list[tuple[str, Any]]:
    return [call for call in base.calls if call[0] == "SendJointSpeedsCommand"]


def test_the_gen3_is_built_by_the_native_driver() -> None:
    assert list_driver_coverage()["kinova_gen3"] == ("strands",)
    assert get_native_driver_class("kinova_gen3") is KinovaDriver
    assert missing_driver_members(KinovaDriver) == ()


def test_the_double_speaks_the_real_sdk() -> None:
    """Every call and enum the double answers exists in the vendor wheel with the keyword used."""
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") != "python":
        pytest.skip("kortex_api's protobuf modules import only under the pure-Python runtime")
    rpc = pytest.importorskip("kortex_api.autogen.client_stubs.BaseClientRpc")
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport

    for fake, real in (
        (FakeBaseClient, rpc.BaseClient),
        (FakeBaseCyclicClient, BaseCyclicClient),
        (FakeSessionManager, SessionManager),
        (FakeTransport, TCPTransport),
    ):
        for name, member in vars(fake).items():
            if name.startswith("_") or not callable(member):
                continue
            params = set(inspect.signature(getattr(real, name)).parameters)
            assert set(inspect.signature(member).parameters) - {"self"} <= params, (fake.__name__, name)
    assert {name: Base_pb2.ArmState.Value(name) for name in ARM_STATES} == ARM_STATES
    assert Base_pb2.SINGLE_LEVEL_SERVOING == SINGLE_LEVEL_SERVOING
    assert {"position", "velocity", "torque"} <= {f.name for f in BaseCyclic_pb2.ActuatorFeedback.DESCRIPTOR.fields}
    assert len(Base_pb2.JointSpeeds(joint_speeds=[Base_pb2.JointSpeed(joint_identifier=0, value=1.0)]).joint_speeds)


def test_connect_opens_a_session_and_enters_single_level_servoing(base: FakeBase) -> None:
    driver = _connected(username="operator")
    assert base.calls[:3] == [
        ("connect", ("192.168.1.10", 10000)),
        ("CreateSession", "operator"),
        ("SetServoingMode", SINGLE_LEVEL_SERVOING),
    ]
    observation = driver.get_observation()
    assert observation["joint_1"] == pytest.approx(math.radians(179.5))
    assert observation["joint_4"] == pytest.approx(math.radians(230.0 - 360.0))  # 0..360 on the wire, (-pi, pi] here
    driver.cleanup()


@pytest.mark.parametrize(
    ("arrange", "expected"),
    [
        (lambda b: setattr(b, "arm_state", ARM_STATES["ARMSTATE_IN_FAULT"]), "is in fault"),
        (lambda b: setattr(b, "arm_state", ARM_STATES["ARMSTATE_MAINTENANCE"]), "ARMSTATE_MAINTENANCE"),
        (lambda b: setattr(b, "actuator_count", 6), "reports 6 actuators"),
        (lambda b: setattr(b, "session_error", FakeKException("bad credentials")), "refused the session"),
        (lambda b: setattr(b, "transport_error", ConnectionRefusedError(111, "refused")), "did not answer"),
        (None, "no base address"),
    ],
)
def test_connect_refuses_and_never_selects_servoing_on_a_base_that_cannot_move(
    base: FakeBase, arrange: Any, expected: str
) -> None:
    if arrange is not None:
        arrange(base)
    driver = KinovaDriver("kinova_gen3", port=None if arrange is None else "192.168.1.10")
    reason = driver.connect_eagerly()
    assert reason is not None and expected in reason, reason
    assert not driver.is_connected
    assert all(call[0] != "SetServoingMode" for call in base.calls)
    assert driver.send_action({"joint_1": 0.0})["status"] == "error"


@pytest.mark.parametrize(
    ("runtime_error", "expected"),
    [
        (ImportError("No module named 'kortex_api'"), "--no-deps"),
        (TypeError("Descriptors cannot be created directly.\nmore"), "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"),
    ],
)
def test_connect_without_a_usable_sdk_names_the_fix(
    monkeypatch: pytest.MonkeyPatch, runtime_error: Exception, expected: str
) -> None:
    import importlib

    def failing(name: str, package: str | None = None) -> Any:
        raise runtime_error

    monkeypatch.setattr(importlib, "import_module", failing)
    reason = KinovaDriver("kinova_gen3", port="192.168.1.10").connect_eagerly()
    assert reason is not None and expected in reason, reason


def test_send_action_writes_the_speed_that_reaches_the_target_in_one_period(base: FakeBase) -> None:
    driver = _connected(control_frequency=40.0)
    here = driver.get_observation()
    target = here["joint_2"] + 0.01
    envelope = driver.send_action({"joint_2": target})
    assert envelope["status"] == "success", envelope
    ((_, speeds),) = _speeds(base)
    assert [joint for joint, _ in speeds] == list(range(7))
    assert speeds[1][1] == pytest.approx(math.degrees(0.01 * 40.0))
    assert all(value == pytest.approx(0.0, abs=1e-9) for index, value in speeds if index != 1)
    driver.cleanup()


def test_a_step_across_pi_is_measured_the_short_way_round(base: FakeBase) -> None:
    """joint_1 at 179.5 deg asked for -179.5 deg is a 1 deg step forward, not a 359 deg one back."""
    driver = _connected(control_frequency=40.0)
    envelope = driver.send_action({"joint_1": math.radians(-179.5)})
    assert envelope["status"] == "success", envelope
    assert _speeds(base)[0][1][0][1] == pytest.approx(1.0 * 40.0)
    driver.cleanup()


@pytest.mark.parametrize(
    ("action", "arrange", "expected"),
    [
        ({"gripper": 0.5}, None, "name no Gen3 joint"),
        ({"joint_3": float("nan")}, None, "joint_3"),
        # 0.8727 rad/s at 40 Hz is 0.0218 rad a period; 0.1 rad is a jump.
        ({"joint_3": math.pi + 0.1}, None, "rad/s allows"),
        ({"joint_3": math.pi}, lambda b: setattr(b, "arm_state", ARM_STATES["ARMSTATE_IN_FAULT"]), "is in fault"),
        ({}, None, "nothing to command"),
    ],
)
def test_send_action_refuses_what_the_base_would_not_track(
    base: FakeBase, action: dict[str, Any], arrange: Any, expected: str
) -> None:
    driver = _connected()
    if arrange is not None:
        arrange(base)
    envelope = driver.send_action(action)
    assert envelope["status"] == "error" and expected in _text(envelope), envelope
    assert _speeds(base) == []
    driver.cleanup()


def test_a_quiet_stream_is_stopped_because_kortex_speeds_never_expire(base: FakeBase) -> None:
    driver = _connected(control_frequency=100.0)
    assert driver.send_action({"joint_5": 0.005})["status"] == "success"
    deadline = time.monotonic() + 2.0
    while ("Stop", ()) not in base.calls and time.monotonic() < deadline:
        time.sleep(0.005)
    assert ("Stop", ()) in base.calls
    assert base.calls.index(("Stop", ())) > base.calls.index(_speeds(base)[0])
    driver.cleanup()


def test_state_names_positions_velocities_torques_and_the_tool_pose(base: FakeBase) -> None:
    driver = _connected()
    payload = driver.state()["content"][0]["json"]
    assert list(payload["joints"]) == list(JOINT_NAMES)
    assert payload["joint_efforts"]["joint_7"] == 1.5
    assert payload["tool_pose"][0] == 0.45 and payload["arm_state"] == "ARMSTATE_SERVOING_READY"
    driver.cleanup()


def test_run_policy_streams_through_send_action_and_stops_the_arm_when_it_ends(base: FakeBase) -> None:
    driver = _connected(control_frequency=100.0)

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        return {"joint_6": observation["joint_6"] + MAX_JOINT_SPEED / 200.0}

    assert driver.run_policy(policy, n_steps=5)["status"] == "success"
    deadline = time.monotonic() + 5.0
    while driver.get_task_status()["content"][0]["json"].get("running") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver.get_task_status()["content"][0]["json"]["steps"] == 5
    assert len(_speeds(base)) == 5 and base.calls[-1] == ("Stop", ())
    driver.cleanup()


def test_the_agent_stop_verb_halts_and_cleanup_closes_the_session(base: FakeBase) -> None:
    driver = _connected()

    async def invoke() -> list[Any]:
        use = {"toolUseId": "t1", "name": "kinova_gen3", "input": {"action": "stop"}}
        return [chunk async for chunk in driver.stream(use, {})]  # type: ignore[arg-type]

    (result,) = asyncio.run(invoke())
    assert result["status"] == "success" and result["toolUseId"] == "t1"
    assert base.calls[-1] == ("Stop", ())
    driver.cleanup()
    assert base.calls[-2:] == [("CloseSession", ()), ("disconnect", ())]
    assert not driver.is_connected
