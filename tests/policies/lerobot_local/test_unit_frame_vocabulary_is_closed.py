"""A unit frame an EmbodimentMap cannot convert is refused, not read as native.

``state_units`` / ``action_units`` are plain ``str`` fields, and every
conversion site compares them against ``"degrees"``. Any other spelling -
``"DEGREES"``, the one LeRobot's own ``MotorNormMode`` uses and this module's
docstrings cite it by - therefore used to mean ``"native"``: no conversion, no
warning, the sim's raw radians packed for a degrees-trained checkpoint. Its
sibling ``dim_policy`` has always refused an unknown spelling
(:func:`~strands_robots.policies.lerobot_local.embodiment.reconcile_dim`); these
two now do too, at construction.
"""

from typing import Any

import pytest

from strands_robots.policies.lerobot_local.embodiment import (
    UNIT_FRAMES,
    EmbodimentMap,
    load_embodiment,
)

_KEYS = [str(i) for i in range(1, 7)]
_SO_ARM: dict[str, Any] = {
    "name": "probe",
    "state_keys": _KEYS,
    "action_keys": _KEYS,
    "gripper_index": 5,
    "gripper_joint_range": [-0.175, 1.745],
    "joint_mids": [0.0, -90.0, 90.0, 0.0, 0.0, 0.0],
}

# Spellings a caller plausibly writes, none of which any conversion site reads.
_REJECTED = ["DEGREES", "Degrees", "deg", "degree", "radians", "rad", "nonsense", ""]


def test_the_vocabulary_is_the_two_frames_the_conversion_sites_compare_against() -> None:
    """The published set is exactly what ``!= "degrees"`` divides the world into."""
    assert sorted(UNIT_FRAMES) == ["degrees", "native"]


@pytest.mark.parametrize("attr", ["state_units", "action_units"])
@pytest.mark.parametrize("frame", _REJECTED)
def test_a_frame_outside_the_vocabulary_is_refused(attr: str, frame: str) -> None:
    """Construction fails naming the field, the value and the accepted set."""
    with pytest.raises(ValueError, match="is not a unit frame this map can convert") as exc:
        EmbodimentMap(**{**_SO_ARM, attr: frame})
    message = str(exc.value)
    assert attr in message
    assert repr(frame) in message
    assert "['degrees', 'native']" in message


@pytest.mark.parametrize("attr", ["state_units", "action_units"])
@pytest.mark.parametrize("frame", sorted(UNIT_FRAMES))
def test_a_frame_inside_the_vocabulary_is_accepted(attr: str, frame: str) -> None:
    """Both published spellings construct and are stored verbatim."""
    assert getattr(EmbodimentMap(**{**_SO_ARM, attr: frame}), attr) == frame


@pytest.mark.parametrize(
    ("frame", "state", "action"),
    [
        # "native" packs the sim's own radians; "degrees" converts both ways.
        ("native", 0.5, 0.5),
        ("degrees", 28.6478897, 0.0087266),
    ],
)
def test_an_accepted_frame_converts_as_its_name_says(frame: str, state: float, action: float) -> None:
    """The two accepted frames are the two distinct conversion behaviours."""
    embodiment = EmbodimentMap(**_SO_ARM, state_units=frame, action_units=frame)
    assert embodiment.sim_state_to_model([0.5] * 6)[0] == pytest.approx(state, abs=1e-6)
    assert embodiment.model_action_to_sim([0.5] * 6)[0] == pytest.approx(action, abs=1e-6)


def test_an_inline_dict_embodiment_is_refused_the_same_way() -> None:
    """``load_embodiment`` builds the map, so a caller's dict is graded too."""
    with pytest.raises(ValueError, match="is not a unit frame this map can convert"):
        load_embodiment({**_SO_ARM, "state_units": "DEGREES"})


def test_every_shipped_embodiment_declares_a_frame_in_the_vocabulary() -> None:
    """The registry is built at import, so a typo in embodiments.json cannot ship."""
    from strands_robots.policies.lerobot_local.embodiment import EMBODIMENT_MAP

    assert EMBODIMENT_MAP
    for name, embodiment in EMBODIMENT_MAP.items():
        assert embodiment.state_units in UNIT_FRAMES, name
        assert embodiment.action_units in UNIT_FRAMES, name
