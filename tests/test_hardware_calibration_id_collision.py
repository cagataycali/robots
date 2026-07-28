# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A defaulted calibration ``id`` that reuses an existing file must be reported.

``config_data["id"] = kwargs.get("id", self.tool_name_str)`` defaults the
calibration id to the robot TYPE, but lerobot's entire purpose for
``RobotConfig.id`` is per-INSTANCE namespacing::

    calibration_fpath = calibration_dir / f"{self.id}.json"

So ``Robot("so101", mode="real", port=A)`` and ``Robot("so101", mode="real",
port=B)`` both resolve to ``so_follower/so101.json`` - two physical arms sharing
one calibration file. Verified: both configs get ``id='so101'`` and an identical
``calibration_fpath``.

Scope, per the ledger's verifier: the second arm is NOT silently driven with the
first arm's numbers - ``_connect_robot`` gates on ``is_calibrated`` and lerobot
refuses a motor-id/range mismatch. What the operator gets instead is a confusing
"not calibrated" refusal, or - if the two arms happen to be similar enough to pass
- one arm's offsets on the other's motors (the calibrations on the dev machine
differ by up to 82 degrees on ``shoulder_lift``).

The default is therefore NOT changed - a port-derived default would orphan every
existing ``<type>.json`` on disk. Instead the collision is named at construction,
with the remedy, which is what the hand-managed ids already on disk (``left_arm``,
``right_arm``, ``orange_arm``, ...) show operators end up doing anyway.

No serial port is opened.
"""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("lerobot")

from strands_robots.hardware_robot import Robot  # noqa: E402

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _config(tmp_calibration_dir=None, **kwargs):
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "so101"
    if tmp_calibration_dir is not None:
        kwargs["calibration_dir"] = str(tmp_calibration_dir)
    return hw._create_minimal_config("so101_follower", cameras=None, **kwargs)


def _seed_calibration(directory) -> None:
    """Write a plausible calibration file so the collision is real."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        motor: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 900, "range_max": 3100}
        for i, motor in enumerate(_MOTORS)
    }
    (directory / "so101.json").write_text(json.dumps(payload))


class TestTheCollisionIsReal:
    def test_two_ports_share_one_calibration_path(self):
        """The premise: nothing about the port reaches the calibration id."""
        from lerobot.robots.so_follower.so_follower import SOFollower

        first = _config(port="/dev/ttyACM0")
        second = _config(port="/dev/ttyACM1")

        assert first.id == second.id == "so101"
        assert SOFollower(first).calibration_fpath == SOFollower(second).calibration_fpath


class TestWarningOnDefaultedId:
    def test_warns_when_the_defaulted_file_already_exists(self, tmp_path, caplog):
        _seed_calibration(tmp_path)

        with caplog.at_level(logging.WARNING):
            _config(tmp_calibration_dir=tmp_path, port="/dev/ttyACM1")

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("no id= was given" in m for m in messages), messages
        # The remedy must be actionable and name the port it is talking about.
        joined = " ".join(messages)
        assert "lerobot-calibrate" in joined
        assert "/dev/ttyACM1" in joined
        assert "so101.json" in joined

    def test_silent_when_an_explicit_id_is_given(self, tmp_path, caplog):
        """An operator who namespaced the arm must not be nagged."""
        _seed_calibration(tmp_path)

        with caplog.at_level(logging.WARNING):
            config = _config(tmp_calibration_dir=tmp_path, port="/dev/ttyACM1", id="right_arm")

        assert config.id == "right_arm"
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("no id= was given" in m for m in messages), messages

    def test_silent_when_no_calibration_file_exists_yet(self, tmp_path, caplog):
        """A first-time user with nothing on disk has no collision to report."""
        tmp_path.mkdir(parents=True, exist_ok=True)

        with caplog.at_level(logging.WARNING):
            _config(tmp_calibration_dir=tmp_path, port="/dev/ttyACM0")

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("no id= was given" in m for m in messages), messages

    def test_warning_is_plain_ascii(self, tmp_path, caplog):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        _seed_calibration(tmp_path)

        with caplog.at_level(logging.WARNING):
            _config(tmp_calibration_dir=tmp_path, port="/dev/ttyACM1")

        for record in caplog.records:
            assert record.getMessage().isascii()


class TestConstructionIsUnaffected:
    def test_the_default_id_is_unchanged(self, tmp_path):
        """Deliberately NOT changed: a new default would orphan existing files."""
        _seed_calibration(tmp_path)

        assert _config(tmp_calibration_dir=tmp_path, port="/dev/ttyACM0").id == "so101"

    def test_a_diagnostic_failure_never_breaks_construction(self, tmp_path, monkeypatch):
        """The check is best-effort; a broken probe must not fail Robot()."""
        _seed_calibration(tmp_path)
        monkeypatch.setattr(
            "strands_robots.hardware_robot._lerobot_driver_dir_name",
            lambda config: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        # No calibration_dir, so the driver-dir probe is exercised and raises.
        config = _config(port="/dev/ttyACM0")

        assert config.id == "so101"


class TestDriverDirectoryResolution:
    def test_so_family_resolves_to_the_shared_driver_directory(self):
        """so101_follower and so100_follower both live in so_follower/."""
        from strands_robots.hardware_robot import _lerobot_driver_dir_name

        assert _lerobot_driver_dir_name(_config(port="/dev/ttyACM0")) == "so_follower"

    def test_an_unresolvable_config_returns_none(self):
        from strands_robots.hardware_robot import _lerobot_driver_dir_name

        assert _lerobot_driver_dir_name(object()) is None
