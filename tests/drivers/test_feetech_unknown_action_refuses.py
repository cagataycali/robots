# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin: an unrecognised action verb is refused, never silently releases torque.

Pre-fix, the ``stream`` fallthrough ``else`` branch released torque on
every action string that was not one of the declared verbs, so a
hallucinated verb (e.g. ``"home"``) de-energised a payload-holding arm
while returning a success envelope. The fix makes the ``stop`` branch
explicit and the final ``else`` an error envelope naming the declared
verbs.

Addresses review feedback on driver.py:234.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


def _make_driver() -> Any:
    """Build a ``FeetechDriver`` wired to a fake port (never touches serial)."""
    from tests.drivers.conftest import FakeServoPort

    from strands_robots.drivers.feetech.driver import FeetechDriver

    driver = FeetechDriver.__new__(FeetechDriver)
    driver._tool_name = "test_so101"
    driver._port = "/dev/fake"
    driver._baud = 1_000_000
    driver._robot_name = "so101"
    return driver


def _stream_one(driver: Any, action: str) -> dict[str, Any]:
    """Drive ``stream`` with a single tool-use and return the envelope."""

    async def _run() -> dict[str, Any]:
        tool_use = {"toolUseId": "tu_test", "input": {"action": action}}
        envelopes: list[dict[str, Any]] = []
        async for envelope in driver.stream(tool_use, {}):
            envelopes.append(envelope)
        assert len(envelopes) == 1, f"expected one envelope, got {len(envelopes)}"
        return envelopes[0]

    return asyncio.get_event_loop().run_until_complete(_run())


class TestUnknownActionRefused:
    """An unrecognised action verb returns an error, never releases torque."""

    @pytest.mark.parametrize("verb", ["home", "dance", "calibrate", "enable", ""])
    def test_unknown_verb_returns_error(self, verb: str) -> None:
        driver = _make_driver()
        result = _stream_one(driver, verb)
        assert result["status"] == "error", (
            f"unknown action {verb!r} must be refused, got status={result['status']!r}"
        )

    @pytest.mark.parametrize("verb", ["home", "dance", "calibrate"])
    def test_error_names_the_declared_verbs(self, verb: str) -> None:
        driver = _make_driver()
        result = _stream_one(driver, verb)
        text = str(result.get("content", ""))
        assert "declared verbs" in text.lower() or "unknown action" in text.lower(), (
            f"error for {verb!r} must name declared verbs, got: {text}"
        )

    def test_stop_still_works(self) -> None:
        """The explicit ``stop`` branch is unchanged."""
        from tests.drivers.conftest import FakeServoPort

        from strands_robots.drivers.feetech.bus import FeetechBus
        from strands_robots.drivers.feetech.driver import FeetechDriver

        driver = _make_driver()
        # Wire up a bus so stop can set_torque
        fake_port = FakeServoPort()
        driver._bus = FeetechBus.__new__(FeetechBus)
        driver._bus._conn = fake_port
        driver._bus.motors = {}
        driver._bus._port_path = "/dev/fake"
        driver._bus._baud = 1_000_000
        # stop with no motors is a success that lists no motors
        result = _stream_one(driver, "stop")
        assert result["status"] == "success", (
            f"stop must succeed, got status={result['status']!r}"
        )
