"""Tests for :class:`strands_robots.drivers.feetech.driver.FeetechDriver`.

Grades the driver's surface, its stub behaviour, and its refusal envelopes.
Nothing here opens a serial port; every path is exercised by construction,
agent-tool invocation, and direct method calls.

The codec is graded separately in :mod:`test_feetech_protocol`. The module-
load pin against ``scservo_sdk`` lives in :mod:`test_feetech_module_load`,
and this file must stay compatible with that pin - importing anything from
:mod:`strands_robots.drivers.feetech` must not pull the vendor SDK.

Structure mirrors :mod:`test_dynamixel_driver` on purpose: the two servo-bus
stub drivers share a contract (the "not wired yet" refusal envelope), and a
reader who has read one should recognise the other's test layout on sight.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from strands.types.tools import ToolUse

from strands_robots.drivers import (
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
)
from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.drivers.feetech.driver import _NOT_WIRED, SUPPORTED_ROBOTS

# ============================================================================
# Surface.
# ============================================================================


class TestSurface:
    """Grade :class:`FeetechDriver` against :data:`DRIVER_SURFACE`.

    A driver that misses a member registered fine and then failed on the
    first agent call, one process and several minutes away from the line
    that was wrong. These tests fail at import time instead.
    """

    def test_driver_class_satisfies_the_driver_surface(self) -> None:
        """Every member :class:`HardwareDriver` requires is present."""
        assert missing_driver_members(FeetechDriver) == ()

    def test_driver_instance_satisfies_the_driver_surface(self) -> None:
        """A constructed instance also satisfies the protocol.

        ``@runtime_checkable`` :class:`HardwareDriver` grades an instance the
        same way ``missing_driver_members`` grades the class; both must agree.
        """
        driver = FeetechDriver(tool_name="so101")
        assert isinstance(driver, HardwareDriver)
        assert missing_driver_members(driver) == ()

    def test_every_supported_robot_registers_the_driver_on_package_import(self) -> None:
        """Importing :mod:`strands_robots.drivers` registers this driver.

        The registration happens through :data:`_SHIPPED_DRIVERS`, and every
        robot :data:`SUPPORTED_ROBOTS` names must resolve to
        :class:`FeetechDriver`. A missing entry surfaces as
        ``Robot("so101", mode="real", driver="strands")`` raising
        ``ValueError`` - the exact failure this driver is here to remove.
        """
        registered = list_native_drivers()
        for canonical in SUPPORTED_ROBOTS:
            cls = get_native_driver_class(canonical)
            assert cls is FeetechDriver, (
                f"canonical={canonical!r} resolved to {cls} rather than FeetechDriver; "
                f"the driver's not-wired refusal cannot reach a caller who cannot build it. "
                f"Currently registered: {registered}"
            )


# ============================================================================
# Constructor.
# ============================================================================


class TestConstructor:
    """The constructor accepts what the factory hands it and refuses only
    what the driver knows cannot work."""

    def test_construct_with_the_factory_signature(self) -> None:
        """``driver_cls(tool_name=, cameras=, data_config=, **kwargs)`` works.

        The factory builds every native driver as this signature; a driver
        that refuses one of the three named keywords is a driver the factory
        cannot build. See :class:`~strands_robots.drivers.base.HardwareDriver`
        module docstring for why the constructor contract is not in the
        Protocol itself.
        """
        driver = FeetechDriver(
            tool_name="so101",
            cameras=None,
            data_config=None,
            port="/dev/tty.usbserial-1",
            baud_rate=1_000_000,
            motor_ids=(1, 2, 3, 4, 5, 6),
        )
        assert driver.tool_name == "so101"
        assert driver.tool_type == "robot"

    def test_extras_pass_through_kwargs_are_kept(self) -> None:
        """Unknown keywords are kept for a downstream driver package.

        Refusing every unknown keyword here would refuse a valid future
        extension, and the driver factory has no way to filter them at
        construction time.
        """
        driver = FeetechDriver(tool_name="so101", weird_extension="yes")
        assert driver._extras == {"weird_extension": "yes"}

    def test_ports_multi_bus_is_refused_by_name(self) -> None:
        """A caller passing ``ports=[...]`` gets a named refusal.

        Feetech arms in :data:`SUPPORTED_ROBOTS` are single-bus; a
        multi-bus rig on this family would be a new-family PR, not a
        silent-tolerate here. The message names both keywords and the
        family so a caller reading the traceback knows which decision was
        made.
        """
        with pytest.raises(ValueError) as excinfo:
            FeetechDriver(tool_name="so101", ports=["/dev/a", "/dev/b"])
        assert "port=" in str(excinfo.value)
        assert "multi-bus" in str(excinfo.value)

    def test_baud_rate_default_is_the_feetech_default(self) -> None:
        """The STS3215's factory default baud rate is 1_000_000."""
        driver = FeetechDriver(tool_name="so101")
        assert driver._baud_rate == 1_000_000


# ============================================================================
# Motion, task and policy refusals.
# ============================================================================


class TestRefusals:
    """Every deferred verb returns the same envelope shape.

    A mesh error handler that copes with one refusal must cope with them
    all, and the shape is the shape a success would also return - which is
    the contract that lets the bus land without a caller code change.
    """

    def test_send_action_refuses_with_the_not_wired_reason(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        result = driver.send_action({"joint_1": 0.0})
        assert result == {
            "status": "error",
            "content": [{"text": f"send_action: {_NOT_WIRED}"}],
        }

    def test_start_task_refuses_with_the_not_wired_reason(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        result = driver.start_task("pick up the cube")
        assert result == {
            "status": "error",
            "content": [{"text": f"start_task: {_NOT_WIRED}"}],
        }

    def test_run_policy_refuses_with_the_not_wired_reason(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        # ``policy=None`` because building a real Policy is beyond this
        # driver's stub concern; the refusal fires before the argument is
        # inspected.
        result = driver.run_policy(policy=None)  # type: ignore[arg-type]
        assert result == {
            "status": "error",
            "content": [{"text": f"run_policy: {_NOT_WIRED}"}],
        }

    def test_get_task_status_reports_nothing_in_flight(self) -> None:
        """Polling task status must not raise; nothing is running."""
        driver = FeetechDriver(tool_name="so101")
        result = driver.get_task_status()
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload == {"in_flight": False, "reason": _NOT_WIRED}

    def test_stop_task_is_a_success_noop(self) -> None:
        """There is nothing to stop; refusing would break idempotent stops."""
        driver = FeetechDriver(tool_name="so101")
        result = driver.stop_task()
        assert result == {
            "status": "success",
            "content": [{"text": f"stop_task: {_NOT_WIRED}"}],
        }

    def test_cleanup_is_a_success_noop(self) -> None:
        """No serial port is held; ``cleanup()`` completes without raising.

        ``cleanup`` is annotated ``-> None`` (the mesh discards its return
        value), so this test pins the contract by exercising the call and
        letting the type checker confirm no return value leaks. A caller
        that ``assert``s on the returned value would be the bug this
        annotation exists to prevent - see
        https://github.com/strands-labs/robots/pull/2880#discussion for
        the pattern.
        """
        driver = FeetechDriver(tool_name="so101")
        driver.cleanup()  # idempotent; a second call must not raise either
        driver.cleanup()


# ============================================================================
# Lifecycle and status.
# ============================================================================


class TestLifecycle:
    """Connect / status / stop paths behave as the mesh expects.

    Every field the mesh reads with ``getattr(robot, name, None)`` is either
    absent (fine) or names its unwired state (also fine); nothing here
    pretends to a value that has not been measured.
    """

    def test_connect_eagerly_names_the_not_wired_reason(self) -> None:
        """A connection attempt reports why it cannot succeed.

        Returning ``None`` (which callers read as success) or raising
        (indistinguishable from a real hardware failure) are both worse
        than the named refusal.
        """
        driver = FeetechDriver(tool_name="so101")
        assert driver.connect_eagerly() == _NOT_WIRED
        assert driver._connect_error == _NOT_WIRED

    def test_get_status_reports_the_construction_state(self) -> None:
        """Status carries what the driver knows about itself.

        The port, baud rate and motor IDs the caller passed at construction
        show up here; the mesh publishes this envelope as the peer's
        presence.
        """
        driver = FeetechDriver(
            tool_name="so101",
            port="/dev/tty.usbserial-1",
            baud_rate=500_000,
            motor_ids=(1, 2, 3),
        )
        payload = asyncio.run(driver.get_status())
        assert payload["status"] == "success"
        body = payload["content"][0]["json"]
        assert body["tool_name"] == "so101"
        assert body["tool_type"] == "robot"
        assert body["connected"] is False
        assert body["port"] == "/dev/tty.usbserial-1"
        assert body["baud_rate"] == 500_000
        assert body["motor_ids"] == [1, 2, 3]
        assert body["supported_robots"] == list(SUPPORTED_ROBOTS)
        assert body["reason"] == _NOT_WIRED

    def test_stop_is_a_noop_that_does_not_raise(self) -> None:
        """``stop()`` on an unwired driver must not raise."""
        driver = FeetechDriver(tool_name="so101")
        asyncio.run(driver.stop())


# ============================================================================
# Agent tool surface (stream).
# ============================================================================


def _run_stream(driver: FeetechDriver, tool_use: ToolUse) -> dict[str, Any]:
    """Drive :meth:`FeetechDriver.stream` and return its one yielded result."""

    async def _collect() -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        async for event in driver.stream(tool_use, {}):
            results.append(event)
        assert len(results) == 1, f"expected one yielded result, got {len(results)}"
        return results[0]

    return asyncio.run(_collect())


class TestStream:
    """The agent-facing ``stream`` yields exactly one result per invocation.

    The three read-only verbs land now; the write verbs land with the bus.
    A schema that declares a verb must accept it in ``stream`` or the agent
    plans against a schema that lies.
    """

    def test_stream_status_reports_the_get_status_envelope(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        result = _run_stream(
            driver,
            {"toolUseId": "tid-1", "name": "so101", "input": {"action": "status"}},
        )
        assert result["toolUseId"] == "tid-1"
        assert result["status"] == "success"
        # The status envelope is nested inside content[0].json; the outer
        # shape and the inner shape both carry the "success" flag.
        outer_body = result["content"][0]["json"]
        assert outer_body["status"] == "success"
        assert outer_body["content"][0]["json"]["reason"] == _NOT_WIRED

    def test_stream_sensors_reports_no_joint_state_and_the_reason(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        result = _run_stream(
            driver,
            {"toolUseId": "tid-2", "name": "so101", "input": {"action": "sensors"}},
        )
        assert result["toolUseId"] == "tid-2"
        assert result["status"] == "success"
        assert result["content"][0]["json"] == {
            "joint_state": None,
            "reason": _NOT_WIRED,
        }

    def test_stream_stop_reports_the_stop_envelope(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        result = _run_stream(
            driver,
            {"toolUseId": "tid-3", "name": "so101", "input": {"action": "stop"}},
        )
        assert result["toolUseId"] == "tid-3"
        assert result["status"] == "success"
        assert result["content"][0]["text"] == f"stop: {_NOT_WIRED}"

    def test_stream_default_action_is_status(self) -> None:
        """A ``stream`` call without an ``action`` field defaults to status.

        Missing-input tolerance is deliberate: agents sometimes fire empty
        tool calls to discover the schema, and refusing that would give
        them no way in.
        """
        driver = FeetechDriver(tool_name="so101")
        result = _run_stream(
            driver,
            {"toolUseId": "tid-4", "name": "so101", "input": {}},
        )
        assert result["status"] == "success"

    def test_tool_spec_declares_only_the_verbs_stream_handles(self) -> None:
        """A verb in the schema must have a code path in ``stream``.

        An agent that plans against the schema picks a verb it sees; a
        declared verb the driver refuses is worse than one it does not
        declare at all.
        """
        driver = FeetechDriver(tool_name="so101")
        spec = driver.tool_spec
        declared = set(spec["inputSchema"]["json"]["properties"]["action"]["enum"])
        assert declared == {"status", "sensors", "stop"}
