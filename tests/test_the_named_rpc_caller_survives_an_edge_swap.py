"""The operator a session named is still the caller after a mid-test edge swap.

``named_rpc_caller`` (tests/conftest.py) runs a test as an allowlisted Device
Connect operator by binding a stub over ``get_rpc_source_device``. It can only
patch what is registered when it runs, and a stand-in is not patchable: a
``MagicMock`` edge carries no ``__file__``, so the fixture skips it and the mock
answers for itself. A test that then calls
:func:`tests._device_connect_real.use_the_real_edge` takes that mock away and
imports the real edge in its place - and the integration re-imported by the same
call binds the real ``get_rpc_source_device``, which reads the wire's
``source_device``. The allowlist still holds the fixture's operator, so the RPC
is refused before it reaches the robot and the test fails on a delegation that
never happened.

That state is not exotic: collection precedes every teardown, so a stand-in
installed by one of the two files that install one is resident for every file
collected after it. Whichever cell reaches the swap first pays for it - measured
on ``tests/test_hardware_policy_port_domain.py`` with the whole tree collected
and one test selected, where the first of four cells reported ``caller='op-1'``
rejected and the three behind it passed on the edge that cell had made real.

The fix registers the stub in
:data:`tests._device_connect_real.EDGE_REBINDERS`, so the swap carries it onto
the module the integration will read. These cells pin the outcome (the robot is
reached) and that the registration does not outlive the fixture.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests._device_connect_real import (
    EDGE_REBINDERS,
    restore_the_edge,
    use_a_mock_edge,
    use_the_real_edge,
)

pytest.importorskip("device_connect_edge", reason="needs the [device-connect] extra")


@pytest.fixture
def a_stand_in_edge_is_resident() -> Iterator[None]:
    """Put a sibling's collection-time ``MagicMock`` edge in place.

    Requested ahead of ``named_rpc_caller`` so the fixture runs against the
    stand-in, which is the ordering the session produces and a single-file run
    does not.
    """
    swap = use_a_mock_edge(
        {
            name: MagicMock()
            for name in (
                "device_connect_edge",
                "device_connect_edge.drivers",
                "device_connect_edge.types",
                "device_connect_edge.device",
            )
        }
    )
    try:
        yield
    finally:
        restore_the_edge(swap)


@pytest.fixture
def operator_over_a_stand_in(a_stand_in_edge_is_resident: None, named_rpc_caller: str) -> str:
    """The named operator, with the stand-in resident when the fixture ran."""
    return named_rpc_caller


def _reached_the_robot(source_device: str) -> tuple[list[Any], dict[str, Any]]:
    """Invoke the real ``execute`` RPC; report whether ``start_task`` was called."""
    use_the_real_edge()
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    seen: list[Any] = []

    class _Recorder:
        tool_name_str = "so100"

        def start_task(self, instruction: str, **kw: Any) -> dict[str, Any]:
            seen.append(instruction)
            return {"status": "success"}

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            RobotDeviceDriver(_Recorder()).invoke(
                "execute",
                instruction="pick up the cube",
                policy_provider="mock",
                source_device=source_device,
            )
        )
    finally:
        loop.close()
    return seen, result


def test_the_swap_hands_the_named_caller_to_the_integration_it_reimports(
    operator_over_a_stand_in: str,
) -> None:
    """The RPC reaches the robot, rather than being refused as unauthorized."""
    seen, result = _reached_the_robot(source_device="op-1")
    assert seen == ["pick up the cube"], f"the RPC was refused before the robot: {result}"


def test_the_registration_does_not_outlive_the_fixture() -> None:
    """A rebinder left behind would patch every later swap in the session."""
    assert "named_rpc_caller" not in EDGE_REBINDERS
