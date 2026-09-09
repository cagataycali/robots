"""The Device Connect state RPC answers a driver that owns its motor bus.

A robot reaches its motors one of two ways. A lerobot robot is a *wrapper*:
:class:`strands_robots.hardware_robot.Robot` holds the device that owns the bus
under ``robot``, and the wrapper answers no read itself. A native driver
(:mod:`strands_robots.drivers`) owns the bus directly, so it *is* the device.
:func:`~strands_robots.bus_access.joint_read_source` is the one place that
resolves both, and ``Robot(mode="real")`` attaches Device Connect to both kinds
-- "both drivers are robots on the fleet, so both get the same two attachments".

:meth:`~strands_robots.device_connect.robot_driver.RobotDeviceDriver.getState`
resolved only the wrapper shape, ``getattr(self._robot, "robot", None)``, which
is the same resolution that left a native driver publishing no joint telemetry on
the mesh state topic (#2749) -- pinned there by
``tests/mesh/test_read_state_joints_from_a_driver_that_owns_its_bus.py``. The
reader was converted to :func:`~strands_robots.bus_access.read_joints` for the
bus-lock and dead-camera incidents (#2666) and the resolution beside it was left
as it was, so on this surface a native driver answered the RPC with no ``joints``
key at all, under the same successful status a readable arm gets.

The old gate demanded ``get_observation``, which is the capability
:func:`read_joints` treats as its *fallback*, while refusing a device carrying
only the ``bus.sync_read`` it prefers. That is the disagreement
``_answers_a_joint_read`` exists to prevent -- its docstring says keeping the
admission rule and the reader in one module is what stops a caller "refusing one
it could have read" -- so both halves of the resolution are graded here.

What this file does not assert: that ``getState`` gates on ``is_connected`` the
way the mesh snapshot does. That is a published-broadcast decision about a robot
nobody asked about; this RPC answers a caller who did, and a device that reports
positions is reported. Nor does it assert a wrapper may be read in place of the
device it wraps -- an inner device is still preferred whenever one is present.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any

import pytest

pytest.importorskip("device_connect_edge")

from strands_robots.bus_access import joint_read_source, read_joints  # noqa: E402
from strands_robots.device_connect import robot_driver  # noqa: E402
from strands_robots.device_connect.robot_driver import RobotDeviceDriver  # noqa: E402

from .test_bus_read_joints import POSITIONS, _Arm, _Bus  # noqa: E402

#: The reading as ``read_joints`` shapes it, with lerobot's own ``.pos`` suffix.
_EXPECTED = {f"{motor}.pos": value for motor, value in POSITIONS.items()}


class _NativeDriver(_Arm):
    """A native driver: owns its bus, has no ``robot``, and its camera is dead.

    ``_Arm`` is the eleven-hour-incident arm from ``test_bus_read_joints`` -- its
    ``get_observation`` raises the camera error -- so a joints answer here can
    only have come through the bus, which is the path that takes the lock.
    """

    #: Narrowed from ``_Arm``'s optional bus: this shape always has one.
    bus: _Bus

    def __init__(self) -> None:
        super().__init__(_Bus(POSITIONS))
        self._task_state = None
        self.tool_name_str = "unitree_g1"


class _BusOnlyDevice:
    """A device whose telemetry IS its motor bus: no ``get_observation`` at all.

    The shape of an SO-100/SO-101 class arm, where joint position, velocity and
    current are the whole of the telemetry.
    """

    def __init__(self) -> None:
        self.bus = _Bus(POSITIONS)
        self.config = None
        self.is_connected = True


class _Wrapper:
    """A ``hardware_robot.Robot``-shaped host holding one device under ``robot``."""

    def __init__(self, device: Any) -> None:
        self.robot = device
        self._task_state = None
        self.tool_name_str = "so101"


class _NoJointSource:
    """A host with neither an inner device nor any way to answer a joint read."""

    def __init__(self) -> None:
        self._task_state = None


def _get_state(host: Any) -> dict[str, Any]:
    return asyncio.run(RobotDeviceDriver(host).getState())


class TestADriverThatOwnsItsBusAnswersTheStateRpc:
    """The shape #2749 fixed on the mesh, now answered on this surface too."""

    def test_the_joints_key_is_present(self) -> None:
        assert "joints" in _get_state(_NativeDriver())

    def test_the_joints_are_the_positions_the_bus_reported(self) -> None:
        assert _get_state(_NativeDriver())["joints"] == pytest.approx(_EXPECTED)

    def test_the_read_went_through_the_bus_and_not_the_failing_camera(self) -> None:
        driver = _NativeDriver()
        _get_state(driver)
        assert driver.bus.calls == [{"register": "Present_Position", "num_retry": 3}]
        assert driver.observation_calls == 0


class TestADeviceCarryingOnlyABusIsNotRefused:
    """The admission rule tracks the reader rather than demanding its fallback."""

    def test_a_bus_only_device_reports_its_joints(self) -> None:
        assert _get_state(_Wrapper(_BusOnlyDevice()))["joints"] == pytest.approx(_EXPECTED)

    def test_the_reader_could_already_read_it(self) -> None:
        device = _BusOnlyDevice()
        assert not hasattr(device, "get_observation")
        assert read_joints(device) == pytest.approx(_EXPECTED)


class TestTheWrapperPathIsUnchanged:
    def test_a_lerobot_wrapper_still_reports_its_joints(self) -> None:
        assert _get_state(_Wrapper(_NativeDriver()))["joints"] == pytest.approx(_EXPECTED)

    def test_the_inner_device_is_the_resolved_source(self) -> None:
        device = _NativeDriver()
        assert joint_read_source(_Wrapper(device)) is device

    def test_a_host_that_answers_no_joint_read_reports_no_joints(self) -> None:
        # Absent rather than empty, and not an error: nothing to read is not a
        # failure, and this is what the RPC has always answered for such a host.
        assert "joints" not in _get_state(_NoJointSource())


class TestBothSurfacesResolveOneDevice:
    """The RPC and the mesh snapshot answer for the same device, per shape."""

    @pytest.mark.parametrize(
        "host_factory",
        [_NativeDriver, lambda: _Wrapper(_NativeDriver()), lambda: _Wrapper(_BusOnlyDevice())],
        ids=["native-driver", "lerobot-wrapper", "wrapper-over-bus-only-device"],
    )
    def test_the_rpc_reports_what_the_shared_resolution_can_read(self, host_factory) -> None:
        source = joint_read_source(host_factory())
        assert source is not None, "the shared resolution found no device to read"
        assert set(_get_state(host_factory())["joints"]) == set(read_joints(source))


class TestTheResolutionHasOneOwner:
    """A source-level pin, because the last conversion moved the reader alone.

    #2666 replaced this handler's ``get_observation`` call with
    :func:`read_joints` and left ``getattr(self._robot, "robot", None)`` beside
    it, so the surface kept the resolution the reader's own module had already
    replaced. Deriving the population from the module rather than naming lines
    means a third reader added here is held to the rule on arrival.
    """

    def test_the_module_resolves_no_device_itself(self) -> None:
        tree = ast.parse(inspect.getsource(robot_driver))
        own_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and ast.unparse(node.args[0]) == "self._robot"
            and len(node.args) > 1
            and getattr(node.args[1], "value", None) == "robot"
        ]
        assert own_reads == [], (
            "the device a joint read is answered from is resolved by "
            f"bus_access.joint_read_source, not read here: line(s) {[n.lineno for n in own_reads]}"
        )

    def test_the_owner_is_the_one_imported(self) -> None:
        assert robot_driver.joint_read_source is joint_read_source
