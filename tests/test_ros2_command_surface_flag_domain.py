"""A read-only ROS 2 bridge stays read-only, whatever spelling asked for it.

``Robot(ros2_bridge=True)`` can expose an inbound ``/<robot>/joint_command``
topic that drives a physical arm, and ``docs/security.md`` names the flag that
closes it as the ungated posture: "Telemetry-only is ungated. ``ros2_commands=
False`` is publish-only (no inbound surface) and needs no security config."

That promise is only as good as how the flag is read. Every non-empty string is
truthy, so ``"false"``, ``"no"``, ``"off"`` and ``"0"`` - the spellings a
YAML/env deployment config yields for a disabled feature - each select the
*permissive* posture when the flag is read by truthiness. ``None``, ``0``,
``""`` and ``[]`` take the other branch without ever being a declared spelling
of it. Nothing raises and nothing logs, so on the rclpy transport a read-only
request became a live arm-driving subscription indistinguishable from a full
duplex one, and ``ros2_bridge="false"`` built a bridge - with commands enabled by
default - for a caller who asked for none.

The DDS Security gate does not stand in for the check, and on the pure-RTPS
transport it inherits the same inversion: ``_require_secure_command_surface``
branches on this flag, so a truthy non-boolean makes it refuse a *read-only*
request with a message about "an enabled command bridge" and advise the
``STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE`` opt-out - the one remedy that
turns the refusal into a silent open of the surface the caller asked to close.

So the three entry points that accept this posture - ``Robot._init_ros_bridge``
(reached from ``Robot(...)``, the documented route) and the two publicly
exported bridge classes - check it on the shared ``boolean_flag_error`` domain.
The tests below parametrize over that domain rather than a copied spelling list,
so a spelling it grows is covered without an edit here, and each guard is
asserted to land ahead of its transport dependency so the refusal reports
identically with and without the ``[ros2]`` extra.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import strands_robots.utils as utils_mod
from strands_robots.hardware_robot import Robot, RobotTaskState
from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.utils import boolean_flag_error

#: Spellings of "off" that are every one of them truthy, so reading the flag by
#: truthiness selects the surface they ask to close.
TRUTHY_OFF_SPELLINGS = ("false", "False", "no", "off", "0")

#: Falsy values that are not a declared spelling of either posture, so reading
#: the flag by truthiness takes a branch the caller never named.
UNDECLARED_FALSY: tuple[Any, ...] = (None, 0, "", [])

#: The inbound topic the RTPS bridge subscribes to for ``test_arm``.
COMMAND_TOPIC = "rt/test_arm/joint_command"


class _FakeTopic:
    def __init__(self, _participant: Any, name: str, _idl: Any) -> None:
        self.name = name


class _FakeEndpoint:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def take(self, N: int = 1) -> list[Any]:  # noqa: N803 - cyclonedds spells it N
        return []

    def write(self, _sample: Any) -> None:
        return None


class _Idl:
    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


@pytest.fixture
def fake_dds(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """A cyclonedds stand-in that records the topics a bridge subscribes to.

    Injected into ``sys.modules`` rather than requiring the real wheel, because
    the ``[ros2]`` extra is not part of ``[all]``: the inbound surface these
    tests measure has to be observable wherever the suite runs.
    """
    state: dict[str, list[str]] = {"readers": [], "writers": []}

    def reader(_participant: Any, topic: _FakeTopic) -> _FakeEndpoint:
        state["readers"].append(topic.name)
        return _FakeEndpoint(topic.name)

    def writer(_participant: Any, topic: _FakeTopic) -> _FakeEndpoint:
        state["writers"].append(topic.name)
        return _FakeEndpoint(topic.name)

    cyclonedds = ModuleType("cyclonedds")
    modules = {
        "cyclonedds": cyclonedds,
        "cyclonedds.domain": ModuleType("cyclonedds.domain"),
        "cyclonedds.pub": ModuleType("cyclonedds.pub"),
        "cyclonedds.sub": ModuleType("cyclonedds.sub"),
        "cyclonedds.topic": ModuleType("cyclonedds.topic"),
    }
    modules["cyclonedds.domain"].DomainParticipant = lambda domain_id=0, qos=None: SimpleNamespace(  # type: ignore[attr-defined]
        domain_id=domain_id, qos=qos
    )
    modules["cyclonedds.pub"].DataWriter = writer  # type: ignore[attr-defined]
    modules["cyclonedds.sub"].DataReader = reader  # type: ignore[attr-defined]
    modules["cyclonedds.topic"].Topic = _FakeTopic  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {"cyclonedds": cyclonedds}, raising=False)

    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda _t: _Idl, raising=True)
    monkeypatch.setattr(idl_mod, "have_cyclonedds", lambda: True, raising=True)

    # These tests are about the flag, not about DDS Security. Run them under the
    # explicit operator opt-out, which is the posture that makes the defect
    # observable rather than masked: with neither a security config nor the
    # opt-out the gate refuses, so the surface a truthy value opens would never
    # be reachable to measure. It is also an ordinary lab posture, and the docs
    # tell a telemetry-only caller they need no security config at all.
    monkeypatch.setenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", "1")
    return state


def _robot() -> Robot:
    """A ``Robot`` skeleton carrying only what ``_init_ros_bridge`` reads.

    ``__new__`` is deliberate: ``_init_ros_bridge`` is documented as a plain
    method precisely so a lightweight double need not thread ``__init__``
    through, and building one here keeps the test off the hardware path.
    """
    robot: Any = Robot.__new__(Robot)
    robot.tool_name_str = "test_arm"
    robot.robot = SimpleNamespace(name="test_arm")
    # What the finalizer reads, so a collected double reports nothing.
    robot._shutdown_event = threading.Event()
    robot._stop_requested = threading.Event()
    robot._task_state = RobotTaskState()
    robot._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flag_domain")
    robot.mesh = None
    return robot


def _init_bridge(robot: Robot, **kwargs: Any) -> Any:
    """Run ``_init_ros_bridge`` on ``robot`` and hand back whatever it built."""
    robot._init_ros_bridge(ros2_transport="rtps", ros2_domain=7, **kwargs)
    bridge = robot._ros_bridge
    if bridge is not None:
        bridge.shutdown()
    return bridge


class TestAReadOnlyRequestDoesNotOpenTheCommandSurface:
    """``ros2_commands`` off must leave no inbound arm-driving subscription."""

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
    def test_a_truthy_spelling_of_off_subscribes_to_no_command_topic(
        self, spelling: Any, fake_dds: dict[str, list[str]]
    ) -> None:
        assert boolean_flag_error(spelling, "ros2_commands", "Robot"), (
            f"premise: {spelling!r} is not a boolean, so the shared domain refuses it"
        )
        try:
            bridge = _init_bridge(_robot(), ros2_bridge=True, ros2_commands=spelling)
        except ValueError as exc:
            assert "ros2_commands" in str(exc), f"refused, but not for the flag: {exc}"
            return
        assert COMMAND_TOPIC not in fake_dds["readers"], (
            f"ros2_commands={spelling!r} asks for a read-only telemetry bridge, yet an inbound "
            f"arm-driving subscription is live on {COMMAND_TOPIC!r} "
            f"(enable_commands={bridge._enable_commands}); any participant on the DDS domain "
            f"can now drive the arm"
        )

    @pytest.mark.parametrize("value", UNDECLARED_FALSY)
    def test_an_undeclared_falsy_value_does_not_silently_pick_a_posture(
        self, value: Any, fake_dds: dict[str, list[str]]
    ) -> None:
        assert boolean_flag_error(value, "ros2_commands", "Robot"), (
            f"premise: {value!r} is not a boolean, so the shared domain refuses it"
        )
        with pytest.raises(ValueError, match="ros2_commands"):
            _init_bridge(_robot(), ros2_bridge=True, ros2_commands=value)


class TestNoBridgeIsBuiltForACallerWhoAskedForNone:
    """``ros2_bridge`` off must build nothing - not a bridge with commands on."""

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
    def test_a_truthy_spelling_of_off_builds_no_bridge(self, spelling: Any, fake_dds: dict[str, list[str]]) -> None:
        assert boolean_flag_error(spelling, "ros2_bridge", "Robot"), (
            f"premise: {spelling!r} is not a boolean, so the shared domain refuses it"
        )
        robot = _robot()
        try:
            bridge = _init_bridge(robot, ros2_bridge=spelling)
        except ValueError as exc:
            assert "ros2_bridge" in str(exc), f"refused, but not for the flag: {exc}"
            return
        assert bridge is None, (
            f"ros2_bridge={spelling!r} asks for no ROS 2 bridge, yet {type(bridge).__name__} was "
            f"built and - because ros2_commands defaults to True - subscribed to "
            f"{fake_dds['readers']}"
        )


class TestTheExportedBridgesCheckTheirOwnFlag:
    """Both bridge classes are public API, so each checks the flag it receives."""

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS + UNDECLARED_FALSY)
    def test_the_rtps_bridge_refuses_a_non_boolean_before_touching_dds(
        self, spelling: Any, fake_dds: dict[str, list[str]]
    ) -> None:
        from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

        with pytest.raises(ValueError, match="enable_commands"):
            HardwareRtpsBridge(_robot(), enable_commands=spelling)
        assert not fake_dds["readers"] and not fake_dds["writers"], (
            f"enable_commands={spelling!r} was refused, but DDS endpoints were built first: "
            f"readers={fake_dds['readers']} writers={fake_dds['writers']}"
        )

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS + UNDECLARED_FALSY)
    def test_the_rclpy_bridge_refuses_a_non_boolean_without_a_sourced_distro(
        self, spelling: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal lands ahead of the rclpy probe and the ``ROS_DOMAIN_ID`` write.

        Same placement, and the same two reasons, as the ``spin_period`` guard
        beside it: a caller mistake reports identically whether or not a ROS 2
        distro is sourced, and a refused constructor leaves the process-wide
        environment as it found it.
        """
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
        with pytest.raises(ValueError, match="enable_commands"):
            HardwareRosBridge(_robot(), domain_id=11, enable_commands=spelling)
        assert "ROS_DOMAIN_ID" not in os.environ, (
            "a refused enable_commands still pinned the process-wide ROS_DOMAIN_ID to "
            f"{os.environ.get('ROS_DOMAIN_ID')!r}"
        )


class TestTheSecurityRefusalDoesNotInheritTheInversion:
    """A read-only request is not refused for a surface it asked to close."""

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
    def test_an_off_spelling_is_not_refused_as_an_enabled_command_bridge(
        self, spelling: Any, monkeypatch: pytest.MonkeyPatch, fake_dds: dict[str, list[str]]
    ) -> None:
        monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
        with pytest.raises(ValueError) as excinfo:
            _init_bridge(_robot(), ros2_bridge=True, ros2_commands=spelling)
        message = str(excinfo.value)
        assert "I_KNOW_THIS_IS_INSECURE" not in message, (
            f"ros2_commands={spelling!r} asks for a read-only bridge, yet the refusal reports an "
            f"enabled command bridge and advises the opt-out that opens one: {message}"
        )
        assert "ros2_commands" in message


class TestTheCheckedFlagStillSelectsBothPostures:
    """Checking the flag must not change what a boolean does."""

    def test_off_is_read_only(self, fake_dds: dict[str, list[str]]) -> None:
        bridge = _init_bridge(_robot(), ros2_bridge=True, ros2_commands=False)
        assert bridge is not None and bridge._enable_commands is False
        assert COMMAND_TOPIC not in fake_dds["readers"]

    def test_on_still_opens_the_command_surface(self, fake_dds: dict[str, list[str]]) -> None:
        bridge = _init_bridge(_robot(), ros2_bridge=True, ros2_commands=True)
        assert bridge is not None and bridge._enable_commands is True
        assert COMMAND_TOPIC in fake_dds["readers"]

    def test_a_numpy_boolean_is_a_boolean(self, fake_dds: dict[str, list[str]]) -> None:
        """The shared domain accepts ``np.bool_``; a hand-rolled ``isinstance`` would not.

        A flag read out of a numpy array of settings is a boolean the caller
        meant, so it has to select the posture it names rather than be refused.
        """
        np = pytest.importorskip("numpy")
        bridge = _init_bridge(_robot(), ros2_bridge=True, ros2_commands=np.bool_(False))
        assert bridge is not None and bridge._enable_commands is False
        assert COMMAND_TOPIC not in fake_dds["readers"]

    def test_no_bridge_is_built_when_the_flag_is_off(self, fake_dds: dict[str, list[str]]) -> None:
        built = _init_bridge(_robot(), ros2_bridge=False)
        assert built is None
        assert not fake_dds["readers"] and not fake_dds["writers"]

    def test_the_numeric_guards_keep_their_own_verdicts(self, fake_dds: dict[str, list[str]]) -> None:
        """The flag check must not displace the guards it was added beside."""
        with pytest.raises(ValueError, match="ros2_domain"):
            _robot()._init_ros_bridge(ros2_bridge=True, ros2_domain=233, ros2_transport="rtps")
        from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

        with pytest.raises(ValueError, match="poll_period"):
            HardwareRtpsBridge(_robot(), poll_period=0.0)
        with pytest.raises(ValueError, match="spin_period"):
            HardwareRosBridge(_robot(), spin_period=-1.0)
