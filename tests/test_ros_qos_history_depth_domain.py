"""The QoS history depth a ROS 2 bridge publishes with is checked where it is given.

``RosTelemetryBridge`` and its hardware subclass take a ``qos_depth`` and hand it
to ``create_publisher`` / ``create_subscription`` as rclpy's ``qos_or_depth``
argument, which becomes ``QoSProfile(depth=value, history=KEEP_LAST)``. It is the
one constructor parameter of that pair whose consumer is not reached from the
constructor: the publishers are built lazily, on the first
``publish_joint_states`` / ``publish_image`` for a robot. Every sibling parameter
is measured at the boundary - ``domain_id`` against the RTPS port map,
``spin_period`` against the positive finite domain, ``enable_commands`` against
the boolean domain, ``joint_limits`` against a finite numeric pair - and this one
was handed through unchecked.

Measured against rclpy on ROS 2 Jazzy, every unusable spelling constructed a
bridge that reported success:

* ``0`` and ``False`` were **accepted** by rclpy, which warns "A zero depth with
  KEEP_LAST doesn't make sense; no data could be stored. This will be
  interpreted as SYSTEM_DEFAULT" and builds the publisher with the middleware
  default. So the declared depth was silently not the depth in force, reported
  by a ``UserWarning`` that ``warnings.warn`` shows once per location - a second
  bridge in the same process said nothing at all. ``True`` was accepted as a
  silent depth of 1.
* ``-1`` raised ``ValueError: history depth must be greater than or equal to
  zero``, and ``2.5`` / ``"10"`` / ``None`` / ``nan`` / ``inf`` /
  ``np.int64(10)`` raised ``TypeError: Expected QoSProfile or int`` - from inside
  rclpy, naming neither the parameter nor the bridge, and on the telemetry-only
  path not until the first frame was published. On the command path
  ``create_subscription`` runs in the constructor, so the same mistake raised
  there instead, after the process-wide ``ROS_DOMAIN_ID`` write and after the
  node was created.

Nothing observed any of it. ``qos_depth`` appeared in no test, no doc and no
example, and the only rclpy double in the suite that reaches ``create_publisher``
takes the depth as ``_depth`` and discards it, so no existing test could see
which depth an endpoint was built with. The double below records it instead,
which is what lets the accepted-value control assert that a depth reaches the
transport verbatim.

``rclpy`` is optional and not installable from PyPI, so every refusal test here
runs with it absent: the guard is placed ahead of the transport probe, which is
what makes that possible and is asserted directly.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import strands_robots
import strands_robots.utils as utils_mod
from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.ros_telemetry import (
    MAX_QOS_HISTORY_DEPTH,
    RosTelemetryBridge,
    _qos_history_depth_error,
)

#: Values that cannot name a KEEP_LAST history depth, one per way of missing it.
UNUSABLE_DEPTHS: list[Any] = [
    0,  # accepted by rclpy as SYSTEM_DEFAULT: not the depth that was asked for
    -1,  # below the floor
    -10,
    True,  # int subclass: a silent depth of 1
    False,  # int subclass: the zero-depth path by another route
    2.5,  # fractional
    10.0,  # integral, still not an int the C API accepts
    np.int64(10),  # names a usable depth; rclpy refuses the type
    float("nan"),
    float("inf"),
    "10",  # a numeric string is not an int
    None,
    [10],
    MAX_QOS_HISTORY_DEPTH + 1,  # first depth the QoS profile cannot store
    2**32 - 1,
]

#: Values that do name a depth, including both ends of the range.
USABLE_DEPTHS: list[int] = [1, 2, 10, MAX_QOS_HISTORY_DEPTH]


def _refuses(build: Any, value: Any) -> bool:
    """Whether ``build(value)`` refuses ``value`` as a history depth.

    An ``ImportError`` means the value cleared the guard and the bridge then
    found ``rclpy`` missing - an install problem, not a verdict about the depth -
    so it counts as accepted.
    """
    try:
        build(value)
    except ValueError as exc:
        return "qos_depth" in str(exc)
    except ImportError:
        return False
    return False


#: Every surface that takes a history depth, with the parameter it names it by.
SURFACES: list[tuple[str, Any]] = [
    ("RosTelemetryBridge(qos_depth=)", lambda v: RosTelemetryBridge(qos_depth=v)),
    ("HardwareRosBridge(qos_depth=)", lambda v: HardwareRosBridge(qos_depth=v)),
]
SURFACE_IDS = [name.split("(")[0] for name, _ in SURFACES]


class _RecordingNode:
    """An rclpy node double that keeps the depth each endpoint was built with.

    The suite's existing double takes the depth as ``_depth`` and drops it, so a
    test driving it cannot tell a publisher built with the caller's depth from
    one built with any other. Recording it is what makes the accepted-value
    control a measurement rather than a restatement of the call.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.publisher_depths: list[Any] = []
        self.subscription_depths: list[Any] = []

    def get_clock(self) -> Any:
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))

    def create_publisher(self, _msg_type: Any, topic: str, depth: Any) -> Any:
        self.publisher_depths.append(depth)
        return SimpleNamespace(topic=topic, publish=lambda _msg: None)

    def create_subscription(self, _msg_type: Any, topic: str, callback: Any, depth: Any) -> Any:
        self.subscription_depths.append(depth)
        return SimpleNamespace(topic=topic, callback=callback)

    def destroy_node(self) -> None:
        return None


@pytest.fixture
def fake_rclpy(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingNode]:
    """Inject a depth-recording rclpy + sensor_msgs.msg; yield the nodes made."""
    nodes: list[_RecordingNode] = []

    def _create_node(name: str) -> _RecordingNode:
        node = _RecordingNode(name)
        nodes.append(node)
        return node

    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: True  # type: ignore[attr-defined]
    rclpy.init = lambda: None  # type: ignore[attr-defined]
    rclpy.shutdown = lambda: None  # type: ignore[attr-defined]
    rclpy.create_node = _create_node  # type: ignore[attr-defined]
    rclpy.spin_once = lambda _node, timeout_sec=0.0: None  # type: ignore[attr-defined]

    class _Msg:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None, frame_id="")
            self.name: list[str] = []
            self.position: list[float] = []

    sensor_pkg = ModuleType("sensor_msgs")
    sensor_msg = ModuleType("sensor_msgs.msg")
    sensor_msg.JointState = _Msg  # type: ignore[attr-defined]
    sensor_msg.Image = _Msg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_pkg)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msg)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    return nodes


class TestTheDepthDomain:
    """``_qos_history_depth_error`` decides which values name a history depth."""

    @pytest.mark.parametrize("value", UNUSABLE_DEPTHS, ids=repr)
    def test_a_value_that_cannot_name_a_depth_is_refused(self, value: Any) -> None:
        error = _qos_history_depth_error(value, "qos_depth", "Bridge")
        assert error is not None
        assert error.startswith("Bridge: qos_depth ")

    @pytest.mark.parametrize("value", USABLE_DEPTHS, ids=repr)
    def test_a_depth_the_transport_can_carry_is_accepted(self, value: int) -> None:
        assert _qos_history_depth_error(value, "qos_depth", "Bridge") is None

    def test_the_message_names_the_parameter_it_came_from(self) -> None:
        assert "some_depth" in str(_qos_history_depth_error(0, "some_depth", "Bridge"))

    def test_a_depth_over_the_ceiling_is_refused_for_the_ceiling_not_the_floor(self) -> None:
        """The two clauses report distinguishable reasons.

        A single "must be a positive integer" for an over-ceiling value would
        tell a caller who passed one that their value is not positive, which it
        is - the transport simply cannot store it.
        """
        error = str(_qos_history_depth_error(MAX_QOS_HISTORY_DEPTH + 1, "qos_depth", "Bridge"))
        assert str(MAX_QOS_HISTORY_DEPTH) in error
        assert "positive integer" in error
        assert "must be a positive integer" not in error


class TestTheCeilingIsTheTransportsNotAPolicyChoice:
    """``MAX_QOS_HISTORY_DEPTH`` is the QoS profile's storage bound, not a limit.

    The depth is stored in ``rmw_qos_profile_t`` through a pybind11 binding that
    takes a signed 32-bit integer, so the ceiling is the largest value that
    converts. Pinning the arithmetic means the constant cannot drift away from
    the reason it holds - a smaller number here would be this library refusing a
    depth the transport is happy to carry.
    """

    def test_the_ceiling_is_the_largest_signed_32_bit_value(self) -> None:
        assert MAX_QOS_HISTORY_DEPTH == 2**31 - 1
        assert MAX_QOS_HISTORY_DEPTH.bit_length() == 31

    def test_one_more_needs_a_wider_field_than_the_profile_has(self) -> None:
        assert (MAX_QOS_HISTORY_DEPTH + 1).bit_length() == 32


class TestARefusedDepthLeavesTheProcessEnvironmentAlone:
    """The refusal precedes the process-wide ``ROS_DOMAIN_ID`` write.

    That write is global to the process and lands before ``rclpy`` is imported,
    so a bridge refused for its depth must not have steered every later
    participant - and every subprocess inheriting the environment - on its way
    to the refusal.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DEPTHS, ids=repr)
    def test_a_refused_depth_does_not_touch_ros_domain_id(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        with pytest.raises(ValueError, match="qos_depth"):
            RosTelemetryBridge(domain_id=11, qos_depth=value)
        assert os.environ["ROS_DOMAIN_ID"] == "7"

    def test_a_usable_depth_still_pins_the_domain_into_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        # rclpy is optional; the pin lands before it is imported, which is
        # exactly why the guard has to run first.
        with pytest.raises(ImportError):
            RosTelemetryBridge(domain_id=11, qos_depth=10)
        assert os.environ["ROS_DOMAIN_ID"] == "11"

    def test_the_domain_is_still_answered_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both wrong: the domain is reported, because its write is the earlier harm."""
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        with pytest.raises(ValueError, match="invalid domain_id"):
            RosTelemetryBridge(domain_id=-1, qos_depth=0)
        assert os.environ["ROS_DOMAIN_ID"] == "7"


class TestARefusedDepthReachesNoTransport:
    """The guard runs before the bridge probes for its optional transport.

    Placing it there is what lets the same caller mistake report identically on
    an install with the ``[ros2]`` extra and one without it, and it means no
    rclpy context is initialised and no node created for a bridge whose
    publishers could never have been built.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DEPTHS, ids=repr)
    def test_the_bridge_refuses_before_probing_for_rclpy(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.ros_telemetry as telemetry_mod

        def _unreachable(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the rclpy probe must not be reached")

        monkeypatch.setattr(telemetry_mod, "require_optional", _unreachable)
        with pytest.raises(ValueError, match="qos_depth"):
            RosTelemetryBridge(qos_depth=value)

    def test_a_usable_depth_still_reaches_the_rclpy_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.ros_telemetry as telemetry_mod

        probed: list[str] = []

        def _record(module: str, **_kwargs: Any) -> Any:
            probed.append(module)
            raise ImportError("rclpy absent")

        monkeypatch.setattr(telemetry_mod, "require_optional", _record)
        with pytest.raises(ImportError):
            RosTelemetryBridge(qos_depth=10)
        assert probed == ["rclpy"]


class TestBothBridgesRefuseTheSameDepths:
    """Neither bridge may accept a depth the other refuses.

    The subclass forwards the value to the base constructor, so one guard covers
    both - and both are checked here so that forwarding cannot be dropped
    silently.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DEPTHS, ids=repr)
    def test_an_unusable_depth_is_refused_by_every_surface(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        refused = {name: _refuses(build, value) for name, build in SURFACES}
        assert all(refused.values()), f"accepted by {[n for n, r in refused.items() if not r]}"

    @pytest.mark.parametrize("value", USABLE_DEPTHS, ids=repr)
    def test_a_usable_depth_is_refused_by_no_surface(self, value: int, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        refused = {name: _refuses(build, value) for name, build in SURFACES}
        assert not any(refused.values()), f"refused by {[n for n, r in refused.items() if r]}"


class TestAnAcceptedDepthReachesTheTransportVerbatim:
    """A depth the guard accepts must be the depth each endpoint is built with.

    The guard must not become a coercion: the whole reason the strict-``int``
    requirement is inherited from the shared count domain is that the C API
    refuses every other spelling of the same number, so quietly converting one
    would put a depth on the wire that the caller never named.
    """

    @pytest.mark.parametrize("value", USABLE_DEPTHS, ids=repr)
    def test_every_publisher_is_built_with_the_depth_that_was_accepted(
        self, value: int, fake_rclpy: list[_RecordingNode]
    ) -> None:
        bridge = RosTelemetryBridge(qos_depth=value)
        bridge.publish_joint_states("arm", ["j0"], [0.5])
        bridge.publish_image("arm", "wrist", np.zeros((2, 2, 3), dtype=np.uint8))
        node = fake_rclpy[-1]
        assert node.publisher_depths == [value, value]
        assert all(type(depth) is int for depth in node.publisher_depths)

    def test_the_command_subscription_is_built_with_it_too(self, fake_rclpy: list[_RecordingNode]) -> None:
        # Typed ``Any``: the bridge only reads ``name`` off the robot on this
        # path, so a stand-in avoids opening a bus for a depth measurement.
        robot: Any = SimpleNamespace(name="arm", send_action=lambda _action: {"status": "success"})
        bridge = HardwareRosBridge(robot=robot, qos_depth=7, enable_commands=True)
        try:
            assert fake_rclpy[-1].subscription_depths == [7]
        finally:
            bridge.shutdown()


class TestEveryDepthSurfaceRoutesThroughTheDomain:
    """Structural guard: a depth-taking surface guards it or forwards it.

    A surface that stores a caller-supplied ``qos_depth`` without either calling
    the guard or handing the value to one that does is accepting a depth its
    endpoints may not be constructible with. Checked structurally so a third
    surface cannot ship without joining the rule.
    """

    #: Parameter names that carry a QoS history depth.
    DEPTH_PARAMS = frozenset({"qos_depth"})

    @staticmethod
    def _package_root() -> pathlib.Path:
        """The installed package directory, derived from an imported symbol."""
        return pathlib.Path(inspect.getfile(strands_robots)).parent

    @classmethod
    def _classify(cls, source: str) -> dict[str, tuple[bool, bool]]:
        """Map ``function name -> (calls the guard, forwards the parameter)``."""
        found: dict[str, tuple[bool, bool]] = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            taken = [a for a in args if a in cls.DEPTH_PARAMS]
            if not taken:
                continue
            guards = any(
                isinstance(call.func, ast.Name) and call.func.id == "_qos_history_depth_error"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
            forwards = any(
                keyword.arg in cls.DEPTH_PARAMS and isinstance(keyword.value, ast.Name) and keyword.value.id in taken
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                for keyword in call.keywords
            )
            found[node.name] = (guards, forwards)
        return found

    def _surfaces(self) -> dict[str, tuple[bool, bool]]:
        surfaces: dict[str, tuple[bool, bool]] = {}
        for path in sorted(self._package_root().rglob("*.py")):
            for name, verdict in self._classify(path.read_text()).items():
                surfaces[f"{path.relative_to(self._package_root())}::{name}"] = verdict
        return surfaces

    def test_the_scan_finds_every_known_depth_surface(self) -> None:
        """Non-vacuity: a scan rooted elsewhere would report a clean sweep."""
        assert set(self._surfaces()) == {
            "hardware_ros_bridge.py::__init__",
            "ros_telemetry.py::__init__",
        }

    def test_every_depth_surface_guards_or_forwards_the_value(self) -> None:
        adrift = {name for name, (guards, forwards) in self._surfaces().items() if not (guards or forwards)}
        assert not adrift, f"these surfaces neither validate nor forward the depth: {sorted(adrift)}"

    def test_the_scanner_detects_a_surface_that_does_neither(self) -> None:
        """A scanner that matched nothing would pass the sweep vacuously."""
        planted = "def brand_new_bridge(self, *, qos_depth: int = 10) -> None:\n    self._depth = qos_depth\n"
        assert self._classify(planted) == {"brand_new_bridge": (False, False)}
