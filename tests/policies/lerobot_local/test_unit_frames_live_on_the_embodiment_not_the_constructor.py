"""``state_units`` / ``action_units`` passed to the constructor are refused, not dropped.

Issue #4164: the lerobot_local page told readers to set ``state_units`` or
``action_units`` to ``degrees`` or ``radians`` next to ``processor_overrides``.
Neither is a constructor keyword, so the pass-through rule dropped both with a
WARNING and ``run_policy`` reported ``success`` with the unit frame unchanged;
and ``radians`` is not a unit frame at all (``UNIT_FRAMES`` is ``native`` and
``degrees``, where ``native`` already means what the robot emits, radians in
MuJoCo). The constructor now refuses the two names with the place they belong
and the frames they take, before any checkpoint is read.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.policies.lerobot_local.embodiment import UNIT_FRAMES
from strands_robots.policies.lerobot_local.policy import (
    EMBODIMENT_UNIT_FIELDS,
    LerobotLocalPolicy,
    embodiment_units_kwarg_error,
)


@pytest.mark.parametrize("name", EMBODIMENT_UNIT_FIELDS)
@pytest.mark.parametrize("value", ["radians", "degrees", "native"])
def test_the_constructor_refuses_a_unit_frame_keyword(name: str, value: str) -> None:
    """Whatever the value, the name is in the wrong place; the message says where it goes."""
    kwargs: dict[str, Any] = {name: value}
    with pytest.raises(TypeError) as excinfo:
        LerobotLocalPolicy(**kwargs)
    message = str(excinfo.value)
    assert f"{name}={value!r}" in message
    assert "embodiment" in message
    for frame in UNIT_FRAMES:
        assert f"'{frame}'" in message
    assert "'radians' is not a frame" in message


def test_both_names_are_reported_together() -> None:
    """One refusal names both misplaced keywords rather than the first found."""
    message = embodiment_units_kwarg_error({"state_units": "degrees", "action_units": "degrees", "rtc": True})
    assert message is not None
    assert "state_units='degrees'" in message and "action_units='degrees'" in message
    assert "rtc" not in message


def test_a_bag_without_the_two_names_is_not_this_refusal() -> None:
    """The pass-through rule for other unknown names stands."""
    assert embodiment_units_kwarg_error({"rtc": True}) is None
    assert embodiment_units_kwarg_error({}) is None


def test_the_refusal_renders_an_unprintable_value() -> None:
    """A caller value whose repr raises is described, not re-raised."""

    class Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    message = embodiment_units_kwarg_error({"state_units": Unprintable()})
    assert message is not None and "state_units=" in message


def test_create_policy_refuses_before_any_download() -> None:
    """The factory path a ``policy_config`` takes reaches the same refusal."""
    from strands_robots.policies import create_policy

    with pytest.raises(TypeError, match="units live on the embodiment"):
        create_policy("lerobot_local", embodiment="so101", state_units="radians")


def test_the_embodiment_field_still_takes_the_two_frames() -> None:
    """Where the field belongs, both frames build; the third spelling is refused there."""
    from strands_robots.policies.lerobot_local.embodiment import _require_unit_frame

    for frame in UNIT_FRAMES:
        _require_unit_frame(frame, field_name="state_units", owner="test")
    with pytest.raises(ValueError, match="not a unit frame"):
        _require_unit_frame("radians", field_name="state_units", owner="test")


def test_the_page_names_the_two_frames_and_where_they_go() -> None:
    """The docs sentence that sent readers to ``radians`` now names the embodiment and its two frames."""
    from pathlib import Path

    page = Path(__file__).resolve().parents[3] / "docs" / "learn" / "policies" / "lerobot-local.md"
    text = page.read_text(encoding="utf-8")
    assert "the embodiment's `state_units` / `action_units` (`degrees` or `native`)" in text
    assert "`radians`" not in text, "radians is not a unit frame; the page may only explain it as what native means"
    assert "`native` is what the robot emits" in text
