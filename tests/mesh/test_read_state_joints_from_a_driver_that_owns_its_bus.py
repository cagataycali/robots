"""A driver that owns its motor bus publishes ``joints`` on the state topic.

A robot reaches its motors one of two ways. A lerobot robot is a *wrapper*:
:class:`strands_robots.hardware_robot.Robot` holds the device that owns the bus
under ``robot``, and the wrapper answers no read itself. A native driver
(:mod:`strands_robots.drivers`) owns the bus directly, so it *is* the device.

:meth:`~strands_robots.mesh.core.Mesh._read_state` resolved only the first shape,
which left the second publishing no joint telemetry at all (#2749). Every other
section reaches the mesh through a ``getattr(robot, name, None)`` read straight
off the driver -- ``_imu``, ``_battery``, ``_temps`` and thirteen siblings -- so a
contract-complete native driver could report how hot a joint was and not where it
was, and ``missing_driver_members`` reported no problem.

The reader was never the missing piece: :func:`read_joints` reads such a driver
unchanged, preferring ``bus.sync_read`` and falling back to ``get_observation``.
Nothing handed the driver to it. ``TestTheReaderCouldAlreadyReadTheDriver``
pins that premise, because it is what makes resolving the device the whole fix.

Two shapes matter and they are not the same test. A driver whose telemetry is a
motor bus and nothing else -- an SO-100/SO-101 arm, where joint position,
velocity and current *are* the telemetry -- carries no ``get_observation`` at
all, and the old gate demanded exactly the capability ``read_joints`` treats as
its fallback while refusing the one it prefers. Its mirror image -- a driver whose
only reader *is* ``get_observation`` -- is here for the same reason: it is what
makes each half of the admission rule independently load-bearing.

What this file does not assert: that a wrapper may be read in place of the device
it wraps. An inner device is preferred whenever one is present
(``TestTheWrapperPathIsUnchanged``), so this widens which robots publish joints
and changes nothing about which device answers for a robot that already did.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from strands_robots.bus_access import joint_read_source, read_joints
from strands_robots.drivers import missing_driver_members
from strands_robots.mesh.core import Mesh

_PEER = "probe"

#: Positions the stand-in buses report, in lerobot's own ``<motor>`` keying.
_MOTORS = {"shoulder_pan": 12.5, "elbow_flex": -4.0, "gripper": 3.0}

#: The same reading as ``get_observation`` shapes it, with lerobot's ``.pos``.
_OBS = {f"{motor}.pos": value for motor, value in _MOTORS.items()}


class _Bus:
    """A motor bus, answering ``sync_read`` the way a feetech/dynamixel one does."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values = dict(_MOTORS if values is None else values)

    def sync_read(self, register: str, **kwargs: Any) -> dict[str, Any]:
        return dict(self._values)


class _ContractOnly:
    """Exactly the twelve members of ``DRIVER_SURFACE`` and nothing else.

    Deliberately not a subclass of anything in the package: the seam is
    structural, so a driver satisfies it by having the members.
    """

    tool_name_str = "so101"

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "robot"

    @property
    def tool_spec(self) -> dict[str, Any]:
        return {"name": "so101", "description": "arm", "inputSchema": {"json": {}}}

    async def stream(self, tool_use: Any, invocation_state: Any, **kwargs: Any) -> Any:
        yield {}

    def send_action(self, action: Any) -> Any:
        return action

    def get_status(self) -> dict[str, Any]:
        return {"status": "success"}

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "success"}

    def start_task(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def stop_task(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def run_policy(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def stop(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def cleanup(self) -> None:
        return None


class _InnerLerobotDevice:
    """The device a lerobot wrapper holds: it owns the bus, the wrapper does not."""

    is_connected = True
    name = "so101_follower"
    config = type("Config", (), {"cameras": {}})()

    def __init__(self) -> None:
        self.bus = _Bus()

    def get_observation(self) -> dict[str, Any]:
        return dict(_OBS)


class Wrapper(_ContractOnly):
    """A lerobot robot: the telemetry device is held under ``robot``."""

    def __init__(self) -> None:
        self.robot = _InnerLerobotDevice()


class NativeWithObservation(_ContractOnly):
    """A native driver exposing both a bus and ``get_observation``."""

    is_connected = True
    config = type("Config", (), {"cameras": {}})()

    def __init__(self) -> None:
        self.bus = _Bus()

    def get_observation(self) -> dict[str, Any]:
        return dict(_OBS)


class NativeBusOnly(_ContractOnly):
    """A native driver whose entire telemetry *is* the motor bus.

    The shape an SO-100/SO-101 driver has. It carries no ``get_observation``,
    which is the capability the old gate demanded.
    """

    is_connected = True

    def __init__(self) -> None:
        self.bus = _Bus()


class NativeObservationOnly(_ContractOnly):
    """A native driver whose only reader is ``get_observation``: no motor bus.

    The shape a network-backed or SDK-backed driver has, and the one that makes
    the fallback branch of the admission rule load-bearing rather than
    decorative -- every other native shape here also carries a bus.
    """

    is_connected = True
    config = type("Config", (), {"cameras": {}})()

    def get_observation(self) -> dict[str, Any]:
        return dict(_OBS)


class NativeNumpyBus(_ContractOnly):
    """A native bus answering with numpy scalars, as a real SDK does."""

    is_connected = True

    def __init__(self) -> None:
        self.bus = _Bus({motor: np.float32(value) for motor, value in _MOTORS.items()})


class NativeNothingToRead(_ContractOnly):
    """A native driver with no motors to report: no bus, no observation."""

    is_connected = True


class NativeNotConnected(_ContractOnly):
    """A readable native driver that says it is not live."""

    is_connected = False

    def __init__(self) -> None:
        self.bus = _Bus()


#: Every shape whose joints must reach the state topic.
_PUBLISHES = (NativeWithObservation, NativeBusOnly, NativeObservationOnly, NativeNumpyBus, Wrapper)

#: Every shape in this file, for the rules that hold across all of them.
_ALL = _PUBLISHES + (NativeNothingToRead, NativeNotConnected)


def _snapshot(cls: type) -> dict[str, Any]:
    """The state snapshot a mesh built on ``cls`` reports, never ``None``."""
    return Mesh(cls(), _PEER)._read_state() or {}


class TestADriverThatOwnsItsBusPublishesJoints:
    """The regression: a native driver's joint positions reach the state topic."""

    @pytest.mark.parametrize("cls", [NativeWithObservation, NativeBusOnly, NativeObservationOnly, NativeNumpyBus])
    def test_the_joints_section_is_present(self, cls: type) -> None:
        """Absent before: joints were reached only through an inner device."""
        assert "joints" in _snapshot(cls), f"{cls.__name__} published no joints"

    @pytest.mark.parametrize("cls", [NativeWithObservation, NativeBusOnly, NativeObservationOnly, NativeNumpyBus])
    def test_the_joints_are_the_positions_the_bus_reported(self, cls: type) -> None:
        """Not merely present: the values are the ones the device answered with."""
        joints = _snapshot(cls)["joints"]
        assert {key: pytest.approx(value) for key, value in joints.items()} == {
            key: pytest.approx(value) for key, value in _OBS.items()
        }

    @pytest.mark.parametrize("cls", [NativeWithObservation, NativeBusOnly, NativeObservationOnly, NativeNumpyBus])
    def test_the_snapshot_survives_the_encoder_that_stands_between_it_and_the_wire(self, cls: type) -> None:
        """``session._put_zenoh_directly`` runs ``json.dumps`` before ``put``.

        A payload it refuses is dropped for good, so a section that reaches the
        snapshot and not the wire is no better than an absent one. The numpy
        shape is the one that tests this: ``json.dumps`` refuses ``np.float32``.
        """
        decoded = json.loads(json.dumps(_snapshot(cls)))
        assert decoded["joints"].keys() == _OBS.keys()


class TestTheWrapperPathIsUnchanged:
    """An inner device is preferred, so a wrapper is never read in its place."""

    def test_a_lerobot_wrapper_still_publishes_its_joints(self) -> None:
        assert _snapshot(Wrapper)["joints"] == pytest.approx(_OBS)

    def test_the_inner_device_is_the_resolved_source(self) -> None:
        """Not the wrapper, even though both are shaped like a robot here."""
        robot = Wrapper()
        assert joint_read_source(robot) is robot.robot

    def test_a_wrapper_holding_an_unreadable_device_reports_no_joints(self) -> None:
        """Resolution must not escape to the wrapper when an inner device exists.

        Falling back here would read a different robot than the one the wrapper
        wraps, which is a wrong reading rather than a missing one.
        """

        class _WrapsSomethingUnreadable(_ContractOnly):
            is_connected = True

            def __init__(self) -> None:
                self.bus = _Bus()
                self.robot = object()

        assert joint_read_source(_WrapsSomethingUnreadable()) is None
        assert "joints" not in _snapshot(_WrapsSomethingUnreadable)


class TestNothingToReadIsStillNothingToRead:
    """The widening reaches drivers that can be read, and stops there."""

    def test_a_driver_with_no_motors_publishes_no_joints(self) -> None:
        assert "joints" not in _snapshot(NativeNothingToRead)

    def test_a_driver_that_says_it_is_not_live_publishes_no_joints(self) -> None:
        """``is_connected`` is the liveness gate and it still decides."""
        assert "joints" not in _snapshot(NativeNotConnected)

    def test_having_no_joints_is_not_reported_as_a_failed_probe(self) -> None:
        """ "Nothing to report" and "the probe raised" are different states.

        The ``degraded`` block exists so a peer whose joint probe is failing says
        so; a driver with no motors must not appear there.
        """
        snapshot = _snapshot(NativeNothingToRead)
        assert "hw_joints" not in snapshot.get("degraded", {})


class TestTheReaderCouldAlreadyReadTheDriver:
    """The premise: resolving the device is the whole fix.

    If ``read_joints`` needed changing too, this file would be testing a
    different repair than the one that was made.
    """

    @pytest.mark.parametrize("cls", [NativeWithObservation, NativeBusOnly, NativeObservationOnly, NativeNumpyBus])
    def test_read_joints_reads_a_native_driver_unchanged(self, cls: type) -> None:
        assert read_joints(cls()).keys() == _OBS.keys()

    def test_a_lerobot_wrapper_itself_answers_no_read(self) -> None:
        """Which is why an inner device has to be preferred rather than ignored."""
        with pytest.raises(AttributeError):
            read_joints(Wrapper())

    @pytest.mark.parametrize("cls", _ALL)
    def test_every_shape_here_satisfies_the_driver_contract(self, cls: type) -> None:
        """None of these is missing a documented member, before or after."""
        assert missing_driver_members(cls) == ()


class TestTheAdmissionRuleTracksTheReader:
    """A device is admitted exactly when the reader has a route to it.

    Derived rather than listed, so a future branch in ``read_joints`` that the
    resolver does not mirror fails here instead of raising in a mesh probe.
    """

    @pytest.mark.parametrize("cls", _ALL)
    def test_resolution_and_readability_agree(self, cls: type) -> None:
        robot = cls()
        resolved = joint_read_source(robot)
        try:
            read_joints(resolved if resolved is not None else robot)
        except AttributeError:
            readable = False
        else:
            readable = True
        assert (resolved is not None) is readable, (
            f"{cls.__name__}: resolved={resolved is not None} readable={readable}"
        )

    def test_the_two_outcomes_are_both_reached(self) -> None:
        """Non-vacuity: the rule above would pass on an all-readable set."""
        outcomes = {joint_read_source(cls()) is not None for cls in _ALL}
        assert outcomes == {True, False}
