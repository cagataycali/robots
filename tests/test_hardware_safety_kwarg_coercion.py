# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The documented safety clamp must not brick the arm.

``max_relative_target`` was forwarded to the lerobot config verbatim. lerobot's
``ensure_safe_goal_position`` dispatches on the value's EXACT type::

    if isinstance(max_relative_target, float): ...
    elif isinstance(max_relative_target, dict): ...
    else: raise TypeError(max_relative_target)

Python's ``int`` is not a subclass of ``float``, so ``Robot('so101',
max_relative_target=10)`` made EVERY ``send_action`` raise ``TypeError: 10``
before a single servo write - the arm did not move at all. This repo's own docs
(``docs/getting-started/robot-factory.md``) present that int form, so the
documented safety configuration was the one that broke.

``SOFollowerConfig`` annotates the field ``float | dict[str, float] | None``, so
honouring the declared type belongs to the forwarding seam, which already
validates and rejects unknown kwargs.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import pytest

from strands_robots.hardware_robot import Robot, _coerce_forwarded_kwarg


class TestCoercion:
    def test_int_becomes_float(self):
        """The regression: an int must not reach lerobot's isinstance dispatch."""
        assert _coerce_forwarded_kwarg("max_relative_target", 10) == 10.0
        assert isinstance(_coerce_forwarded_kwarg("max_relative_target", 10), float)

    def test_float_passes_through(self):
        assert _coerce_forwarded_kwarg("max_relative_target", 12.5) == 12.5

    def test_none_passes_through(self):
        """None means "no clamp" and is a valid lerobot value."""
        assert _coerce_forwarded_kwarg("max_relative_target", None) is None

    def test_per_joint_dict_values_are_coerced(self):
        result = _coerce_forwarded_kwarg("max_relative_target", {"shoulder_pan": 10, "gripper": 5.5})

        assert result == {"shoulder_pan": 10.0, "gripper": 5.5}
        assert all(isinstance(v, float) for v in result.values())

    def test_control_dt_is_coerced_too(self):
        assert _coerce_forwarded_kwarg("control_dt", 1) == 1.0

    def test_unrelated_kwargs_are_untouched(self):
        """Only the numeric-dispatch kwargs are coerced; nothing else changes meaning."""
        assert _coerce_forwarded_kwarg("port", "/dev/ttyACM0") == "/dev/ttyACM0"
        assert _coerce_forwarded_kwarg("use_degrees", True) is True
        assert _coerce_forwarded_kwarg("mock", False) is False


class TestRejections:
    def test_bool_is_rejected_not_silently_coerced(self):
        """True -> 1.0 would be a 1-degree clamp that reads as a frozen arm."""
        with pytest.raises(ValueError, match="is a bool"):
            _coerce_forwarded_kwarg("max_relative_target", True)

    def test_string_is_rejected_with_the_accepted_shapes_named(self):
        with pytest.raises(ValueError) as excinfo:
            _coerce_forwarded_kwarg("max_relative_target", "10")

        message = str(excinfo.value)
        assert "unsupported type str" in message
        assert "max_relative_target=10.0" in message  # actionable example

    def test_non_numeric_dict_value_is_rejected(self):
        with pytest.raises(ValueError, match="is not a number"):
            _coerce_forwarded_kwarg("max_relative_target", {"shoulder_pan": "wide"})

    def test_bool_inside_a_dict_is_rejected(self):
        with pytest.raises(ValueError, match="is not a number"):
            _coerce_forwarded_kwarg("max_relative_target", {"shoulder_pan": True})

    def test_error_messages_are_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        for bad in (True, "10", {"j": "x"}):
            with pytest.raises(ValueError) as excinfo:
                _coerce_forwarded_kwarg("max_relative_target", bad)
            assert str(excinfo.value).isascii()


class TestThroughTheRealConfigBuilder:
    def _config(self, **kwargs):
        hw = Robot.__new__(Robot)
        hw.tool_name_str = "so101"
        return hw._create_minimal_config("so101_follower", cameras=None, port="/dev/null", **kwargs)

    def test_int_lands_on_the_config_as_a_float(self):
        pytest.importorskip("lerobot")

        config = self._config(max_relative_target=10)

        assert config.max_relative_target == 10.0
        assert isinstance(config.max_relative_target, float)

    def test_the_coerced_value_actually_clamps_in_lerobot(self):
        """End to end against the INSTALLED lerobot: pre-fix this raised TypeError."""
        pytest.importorskip("lerobot")
        from lerobot.robots.utils import ensure_safe_goal_position

        config = self._config(max_relative_target=10)

        # A commanded 15-degree move from 10.0 must clamp to 20.0, not raise.
        safe = ensure_safe_goal_position({"shoulder_pan": (25.0, 10.0)}, config.max_relative_target)

        assert safe == {"shoulder_pan": 20.0}

    def test_raw_int_still_breaks_lerobot_proving_the_coercion_is_required(self):
        """Pins the upstream behaviour this fix exists for, so it cannot silently change."""
        pytest.importorskip("lerobot")
        from lerobot.robots.utils import ensure_safe_goal_position

        with pytest.raises(TypeError):
            ensure_safe_goal_position({"shoulder_pan": (25.0, 10.0)}, 10)

    def test_bool_is_rejected_at_construction_time(self):
        """Fail next to the caller, not at the first send_action."""
        pytest.importorskip("lerobot")

        with pytest.raises(ValueError, match="is a bool"):
            self._config(max_relative_target=True)
