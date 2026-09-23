"""A unit frame no conversion site can convert is refused, not read as native.

``state_units`` / ``action_units`` are plain ``str`` fields, and every
conversion site compares them against ``"degrees"``. Any other spelling -
``"DEGREES"``, the one LeRobot's own ``MotorNormMode`` uses and this module's
docstrings cite it by - therefore used to mean ``"native"``: no conversion, no
warning, the sim's raw radians packed for a degrees-trained checkpoint. Its
sibling ``dim_policy`` has always refused an unknown spelling
(:func:`~strands_robots.policies.lerobot_local.embodiment.reconcile_dim`); these
two now do too, at construction - and so does the ``strands_pack_state``
pipeline step, the second holder of a frame: LeRobot's ``from_pretrained``
rebuilds it from the ``policy_preprocessor.json`` a checkpoint ships, not from
the graded map.
"""

from typing import Any

import pytest

from strands_robots.policies.lerobot_local.embodiment import (
    UNIT_FRAMES,
    EmbodimentMap,
    load_embodiment,
    register_pack_state_step,
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

# The wording-independent half of the refusal: one message, both holders.
_REFUSAL = "is not a unit frame"

# Spellings a caller plausibly writes, none of which any conversion site reads.
_REJECTED = ["DEGREES", "Degrees", "deg", "degree", "radians", "rad", "nonsense", ""]


def test_the_vocabulary_is_the_two_frames_the_conversion_sites_compare_against() -> None:
    """The published set is exactly what ``!= "degrees"`` divides the world into."""
    assert sorted(UNIT_FRAMES) == ["degrees", "native"]


@pytest.mark.parametrize("attr", ["state_units", "action_units"])
@pytest.mark.parametrize("frame", _REJECTED)
def test_a_frame_outside_the_vocabulary_is_refused(attr: str, frame: str) -> None:
    """Construction fails naming the field, the value and the accepted set."""
    with pytest.raises(ValueError, match=_REFUSAL) as exc:
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
    with pytest.raises(ValueError, match=_REFUSAL):
        load_embodiment({**_SO_ARM, "state_units": "DEGREES"})


def test_every_shipped_embodiment_declares_a_frame_in_the_vocabulary() -> None:
    """The registry is built at import, so a typo in embodiments.json cannot ship."""
    from strands_robots.policies.lerobot_local.embodiment import EMBODIMENT_MAP

    assert EMBODIMENT_MAP
    for name, embodiment in EMBODIMENT_MAP.items():
        assert embodiment.state_units in UNIT_FRAMES, name
        assert embodiment.action_units in UNIT_FRAMES, name


# The pack-state step: the second holder of a frame


def _step(frame: str) -> Any:
    """Build the pipeline step over six SO-arm joints in the given frame."""
    step_class = register_pack_state_step()
    assert step_class is not None, "lerobot processor framework unavailable"
    return step_class(
        state_keys=_KEYS,
        expected_dim=6,
        state_units=frame,
        gripper_index=5,
        gripper_joint_range=[-0.175, 1.745],
    )


@pytest.mark.parametrize("frame", _REJECTED)
def test_the_pack_state_step_refuses_a_frame_outside_the_vocabulary(frame: str) -> None:
    """The step compares against ``"degrees"`` too, so it grades its own field.

    Before this was graded, every rejected spelling packed the sim's raw radians
    (``0.5`` rad stored as ``0.5``) for a checkpoint trained on degrees, where
    the same joints in ``"degrees"`` pack as ``28.648``.
    """
    with pytest.raises(ValueError, match=_REFUSAL) as exc:
        _step(frame)
    message = str(exc.value)
    assert "state_units" in message
    assert repr(frame) in message
    assert "['degrees', 'native']" in message


@pytest.mark.parametrize(
    ("frame", "packed"),
    [
        # The two accepted frames are the two distinct packings of the same obs.
        ("native", [0.5] * 6),
        ("degrees", [28.6478897] * 5 + [35.1562500]),
    ],
)
def test_an_accepted_frame_packs_the_state_as_its_name_says(frame: str, packed: list[float]) -> None:
    """A rejected spelling used to be indistinguishable from the first row."""
    out = _step(frame).observation({key: 0.5 for key in _KEYS})
    assert [float(x) for x in out["observation.state"].tolist()] == pytest.approx(packed, abs=1e-5)


def test_a_saved_pipeline_naming_an_unconvertible_frame_is_refused_on_reload() -> None:
    """LeRobot rebuilds the step from its own saved config, bypassing the map.

    ``from_pretrained`` looks the registry name up and calls the class with the
    saved config as kwargs, so the config - not ``EmbodimentMap`` - is what a
    reloaded checkpoint's frame comes from.
    """
    from lerobot.processor.pipeline import ProcessorStepRegistry

    config = _step("degrees").get_config()
    assert config["state_units"] == "degrees"

    step_class = ProcessorStepRegistry.get("strands_pack_state")
    assert step_class(**config).state_units == "degrees"
    with pytest.raises(ValueError, match=_REFUSAL):
        step_class(**{**config, "state_units": "DEGREES"})
