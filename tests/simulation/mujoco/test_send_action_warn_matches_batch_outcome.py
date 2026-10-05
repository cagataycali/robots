"""The WARNING `send_action` emits agrees with the envelope it returns.

Since #4486 (b0fb474) `SimEngine.send_action` refuses a whole batch when any
key fails to resolve to an actuator or joint - nothing is written and the
world does not advance. The return envelope says so plainly:

    "Nothing was applied and the world did not advance."

The WARNING log emitted on the same pre-write refusal path used to say
    "The value was dropped."
- a singular-key wording that implied the batch continued and only the typo'd
key was skipped. An operator who trusted the WARNING believed the valid keys
in the batch had landed and skipped the retry; the arm stayed still while
they debugged a controller output that never reached `ctrl`.

This test pins the WARNING to the whole-batch wording so a future refactor
cannot silently resurrect the pre-#4486 mental model.
"""
from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


class _CaptureHandler(logging.Handler):
    """Collect every WARNING record the sim's rendering module emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self.records.append(record.getMessage())


@pytest.fixture()
def captured_warnings() -> _CaptureHandler:
    """Install a handler on the rendering logger just for this test."""
    handler = _CaptureHandler()
    logger = logging.getLogger("strands_robots.simulation.mujoco.rendering")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    yield handler
    logger.removeHandler(handler)


def test_the_warning_agrees_with_the_envelope_on_a_mixed_batch(
    captured_warnings: _CaptureHandler,
) -> None:
    """A mixed valid+invalid batch's warning cannot say the value was dropped.

    The envelope on this path returns `"Nothing was applied and the world did
    not advance."` - the operator reading both messages must not see a WARNING
    telling them the sibling valid keys landed.
    """
    robot = Robot("so101")
    try:
        result = robot.send_action({"1": 0.5, "nonsense_joint": 1.0})

        assert result["status"] == "error", result
        envelope_text = result["content"][0]["text"]
        assert "Nothing was applied and the world did not advance" in envelope_text

        warnings = [m for m in captured_warnings.records if "nonsense_joint" in m]
        assert warnings, "the unresolved-key warning was not emitted"
        warning = warnings[0]

        # The pre-#4486 wording described a partial write and contradicted the
        # envelope. Pin that it is gone from this call site.
        assert "The value was dropped" not in warning, (
            f"stale pre-#4486 wording resurfaced on the batch-refused path:\n  {warning}"
        )
        # The current wording matches the envelope: nothing landed.
        assert "whole batch was refused" in warning, (
            f"expected whole-batch wording on the pre-write refusal path:\n  {warning}"
        )

        # And confirm the valid key really didn't land.
        state = robot.get_robot_state()["content"][1]["json"]["state"]
        assert abs(state["1"]["position"]) < 0.05, (
            f"joint 1 moved to {state['1']['position']}; the batch should have been refused whole"
        )
    finally:
        robot.cleanup()
