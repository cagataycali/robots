"""Contract tests for PlannerCommand / PlannerUpdate value objects."""

from __future__ import annotations

import json

import pytest

from strands_robots.planning import STYLES, PlannerCommand, PlannerUpdate


def test_default_command_is_neutral() -> None:
    cmd = PlannerCommand()
    assert cmd.root_vel == (0.0, 0.0, 0.0)
    assert cmd.style == "run"
    assert cmd.style in STYLES


def test_velocity_accessors_match_triple() -> None:
    cmd = PlannerCommand(root_vel=(0.5, -0.2, 0.3))
    assert (cmd.vx, cmd.vy, cmd.omega) == (0.5, -0.2, 0.3)


def test_to_policy_kwargs_maps_locomotion_goal_keys() -> None:
    cmd = PlannerCommand(root_vel=(0.5, 0.0, 0.1), height=0.7, style="stealth")
    kwargs = cmd.to_policy_kwargs()
    assert kwargs == {
        "target_velocity": [0.5, 0.0, 0.1],
        "target_height": 0.7,
        "locomotion_style": "stealth",
    }


def test_json_round_trip_is_lossless() -> None:
    cmd = PlannerCommand(root_vel=(0.3, 0.1, -0.2), height=0.65, style="boxing")
    restored = PlannerCommand.from_dict(json.loads(json.dumps(cmd.to_dict())))
    assert restored == cmd


def test_unknown_style_rejected() -> None:
    with pytest.raises(ValueError, match="unknown style"):
        PlannerCommand(style="moonwalk")


def test_non_finite_velocity_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        PlannerCommand(root_vel=(float("inf"), 0.0, 0.0))


def test_wrong_length_velocity_rejected() -> None:
    with pytest.raises(ValueError, match="triple"):
        PlannerCommand(root_vel=(0.0, 0.0))  # type: ignore[arg-type]


def test_from_dict_requires_root_vel() -> None:
    with pytest.raises(ValueError, match="root_vel"):
        PlannerCommand.from_dict({"style": "run"})


def test_command_is_immutable() -> None:
    cmd = PlannerCommand()
    with pytest.raises(AttributeError):
        cmd.style = "happy"  # type: ignore[misc]


def test_update_is_empty_detects_no_change() -> None:
    assert PlannerUpdate().is_empty()
    assert not PlannerUpdate(style="run").is_empty()
    assert not PlannerUpdate(stop=True).is_empty()
    assert not PlannerUpdate(root_vel=(0.0, 0.0, 0.0)).is_empty()
