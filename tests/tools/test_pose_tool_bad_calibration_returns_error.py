# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: a malformed calibration file returns a structured error.

``_units(port, records)`` constructs a ``FeetechBus`` whose
``_resolve_calibration`` raises ``ValueError`` for a record set that is valid
JSON but not this arm's calibration (a missing motor, an ``id`` mismatch, or
``range_min >= range_max``). Before this fix, that call sat outside the
``try/except (OSError, ValueError)`` handler that wraps the calibration load,
so the ``ValueError`` escaped as an uncaught exception instead of the
structured ``{"status": "error", ...}`` result the tool contract requires
(AGENTS.md > Review Learnings (#85) > Error Handling Contracts).

Pinned for PR #3868 review thread on ``strands_robots/tools/pose_tool.py:1330``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.tools.pose_tool import pose_tool


@pytest.fixture
def _stub_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any real serial port interaction."""
    monkeypatch.setattr(
        "strands_robots.tools.pose_tool.MotorController",
        MagicMock,
    )


def _write_json(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(obj))
    return path


class TestBadCalibrationReturnsStructuredError:
    """A calibration file the bus cannot resolve returns status=error."""

    def test_missing_motor_in_calibration(self, tmp_path: Path, _stub_serial: None) -> None:
        """A record set that is short one motor raises ValueError inside the bus."""
        # Only 5 of 6 motors - missing 'gripper'
        bad = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
        }
        cal_path = _write_json(tmp_path / "bad_arm.json", bad)
        result = pose_tool(
            action="read_motor",
            port="/dev/fake-arm",
            motor_name="shoulder_pan",
            calibration=str(cal_path),
        )
        assert result["status"] == "error", (
            "A calibration missing a motor must return status=error, not raise past the tool envelope"
        )

    def test_inverted_range_in_calibration(self, tmp_path: Path, _stub_serial: None) -> None:
        """range_min >= range_max in one motor's record raises ValueError."""
        bad = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 3000, "range_max": 100},
            "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
            "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
        }
        cal_path = _write_json(tmp_path / "bad_arm.json", bad)
        result = pose_tool(
            action="read_motor",
            port="/dev/fake-arm",
            motor_name="shoulder_pan",
            calibration=str(cal_path),
        )
        assert result["status"] == "error", (
            "A calibration with range_min >= range_max must return status=error, not raise past the tool envelope"
        )

    def test_error_text_names_the_action(self, tmp_path: Path, _stub_serial: None) -> None:
        """The structured error names the action the caller invoked."""
        bad = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3000},
        }
        cal_path = _write_json(tmp_path / "bad_arm.json", bad)
        result = pose_tool(
            action="read_motor",
            port="/dev/fake-arm",
            motor_name="shoulder_pan",
            calibration=str(cal_path),
        )
        assert result["status"] == "error"
        error_text = result["content"][0]["text"]
        assert "read_motor" in error_text, "The error text must name the action so the caller knows which call failed"
