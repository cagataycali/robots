"""Explicitly opted-in, read-only native Reachy daemon check.

Run with REACHY_TEST_HOST=reachy-a.local:8000 and REACHY_TEST_READONLY=1.
No motion, STOP, motor-mode, audio or service writes. Never collected by the
unit suite. A daemon exposing /ws/sdk is required (verified with 1.10.0).
"""

import os
import time

import pytest
from strands import Agent

from strands_robots import Robot
from strands_robots.drivers.reachy import ReachyDriver

pytestmark = pytest.mark.skipif(
    os.environ.get("REACHY_TEST_READONLY") != "1" or not os.environ.get("REACHY_TEST_HOST"),
    reason="set REACHY_TEST_READONLY=1 and REACHY_TEST_HOST to opt into a real daemon read",
)


def test_native_factory_stream_and_agent_dispatch() -> None:
    driver = Robot("reachy_mini", mode="real", driver="strands", port=os.environ["REACHY_TEST_HOST"], mesh=False)
    assert isinstance(driver, ReachyDriver)
    try:
        assert driver.connect_eagerly() is None
        deadline_mono = time.monotonic() + 5
        while time.monotonic() < deadline_mono:
            state = driver.state_snapshot()["content"][0]["json"]
            if state["joints"] is not None:
                break
            time.sleep(0.05)
        assert state["joints"] is not None, "daemon connected but no joint frames arrived"
        assert len(state["joints"]["head_leg_deg"]) == 6
        assert len(state["joints"]["antennas_deg"]) == 2
        agent = Agent(tools=[driver], callback_handler=None)
        assert "reachy_mini" in agent.tool_names
        result = agent.tool.reachy_mini(action="sensors")
        assert result["status"] == "success"
        assert result["content"][0]["json"]["joints"] is not None
        assert driver.list_moves()["status"] == "success"
    finally:
        driver.cleanup()
