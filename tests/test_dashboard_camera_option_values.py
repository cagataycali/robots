"""An enumerated camera option's VALUE is refused before the arm is despawned.

The dashboard already refused an unknown option NAME here, because a reconfigure despawns the working
robot first and the child's ValueError arrives in a log ring buffer after the robot is gone. The value
half of that promise was missing: hardware_robot coerces neither color_mode nor rotation, so a typo'd
spelling reached lerobot verbatim and killed the respawn.
"""
from __future__ import annotations

from strands_robots.dashboard.device_manager import (
    _CAMERA_ENUM_VALUES,
    _camera_option_values,
    camera_option_value_problem,
)


def test_the_admitted_spellings_are_read_from_lerobots_own_enums_not_copied():
    """Drift, not assumption: wherever lerobot is importable its enums ARE the domain."""
    try:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
    except Exception:  # pragma: no cover - exercised on machines with no robot stack
        assert _camera_option_values() == dict(_CAMERA_ENUM_VALUES)
        return
    table = _camera_option_values()
    assert set(table["color_mode"]) == {str(m.value) for m in ColorMode}
    assert set(table["rotation"]) == {str(r.value) for r in Cv2Rotation}
    # And the frozen fallback must not have drifted from them either, or a machine without lerobot
    # would refuse a spelling the child accepts.
    assert set(_CAMERA_ENUM_VALUES["color_mode"]) == set(table["color_mode"])
    assert set(_CAMERA_ENUM_VALUES["rotation"]) == set(table["rotation"])


def test_rotation_admits_minus_90_and_refuses_the_obvious_270():
    """MEASURED: lerobot's Cv2Rotation is {0, 90, 180, -90}. 270 is the intuitive answer and wrong."""
    assert camera_option_value_problem("rotation", -90) is None
    assert camera_option_value_problem("rotation", "-90") is None
    problem = camera_option_value_problem("rotation", 270)
    assert problem and "270" in problem and "-90" in problem


def test_an_int_and_its_string_spelling_are_both_admitted():
    """The form sends strings and a remembered profile round-trips through JSON."""
    assert camera_option_value_problem("rotation", 90) is None
    assert camera_option_value_problem("rotation", "90") is None
    assert camera_option_value_problem("color_mode", "rgb") is None


def test_the_uppercase_spelling_is_refused_with_the_lowercase_hint():
    """ColorMode('RGB') raises in lerobot, so admitting it here would only move the death later."""
    problem = camera_option_value_problem("color_mode", "RGB")
    assert problem and "rgb" in problem
    assert "Did you mean 'rgb'?" in problem


def test_an_option_with_no_published_enumeration_is_not_this_functions_business():
    """fps/width/backend are graded by the numeric ranges beside it; silence here is deliberate."""
    for option, value in (("fps", 30), ("width", 640), ("backend", "avfoundation"), ("nonsense", "x")):
        assert camera_option_value_problem(option, value) is None


def test_a_structural_value_says_the_option_is_enumerated():
    problem = camera_option_value_problem("color_mode", {"rgb": True})
    assert problem and "enumerated" in problem
    assert camera_option_value_problem("rotation", True) is not None  # bool is not a rotation


def test_the_refusal_reaches_the_config_validator_before_any_despawn():
    """The wiring, not just the helper: validate_cameras must refuse the whole spawn."""
    from strands_robots.dashboard.device_manager import validate_cameras

    ok = validate_cameras({"main": {"index_or_path": 0, "color_mode": "rgb", "rotation": 90}})
    assert ok is None
    bad = validate_cameras({"main": {"index_or_path": 0, "color_mode": "RGB"}})
    assert bad and "main" in bad["error"] and "rgb" in bad["error"]
