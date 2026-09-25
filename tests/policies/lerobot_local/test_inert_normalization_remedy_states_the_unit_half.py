"""The inert-normalization remedy covers units, not just stats.

A base checkpoint's stats are inert because they are keyed by the training
dataset, and the load-time diagnostic prescribes ``processor_overrides`` to
supply them. Those stats are in the units the dataset was RECORDED in: an SO-arm
dataset comes through the driver's ``MotorNormMode`` (arm joints in servo
degrees, gripper in ``RANGE_0_100``), while a MuJoCo state is radians. So the
stats half alone is not a remedy for a sim caller - it rescales nothing, the
sim's whole joint range lands inside a fraction of one standard deviation, and
``observation.state`` reaches the model as a near-constant.

And the reverse pairing is not a degraded remedy but a broken one: a declared
embodiment whose ``state_units``/``action_units`` are not native writes its
conversion into the tensor the inert normalizer then leaves alone, so the
converted value reaches the model unscaled. The load path refuses that
combination instead of prescribing the unit half a caller already declared.

These tests pin all three ends: the arithmetic that makes the unit half
load-bearing, the diagnostic plus the docs remedy naming it, and the refusal of
a unit-converting embodiment whose stats are inert.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.embodiment import EMBODIMENT_MAP
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# lerobot/smolvla_base ships its normalizer stats under the training dataset's
# key ('so100.buffer.action'), in SO-100 servo units - arm joints in degrees,
# joint 6 in RANGE_0_100. Read from
# policy_preprocessor_step_5_normalizer_processor.safetensors.
SO100_ACTION_STD = [26.3918, 52.4114, 49.8538, 36.9983, 59.36, 19.04]

# so101 MJCF jnt_range, radians (the sim's own units).
SO101_RANGE_RAD = [
    (-1.9199, 1.9199),
    (-1.7453, 1.7453),
    (-1.7453, 1.5708),
    (-1.6581, 1.6581),
    (-2.7925, 2.7925),
    (-0.1745, 1.7453),
]

_DOC = pathlib.Path(__file__).resolve().parents[3] / "docs" / "policies" / "lerobot-local-observations.md"


def _sigma_span(state_units: str) -> list[float]:
    """Sigma spanned by the full so101 joint range under degree-recorded stats."""
    embodiment = dataclasses.replace(EMBODIMENT_MAP["so101"], state_units=state_units)
    low = embodiment.sim_state_to_model([lo for lo, _ in SO101_RANGE_RAD])
    high = embodiment.sim_state_to_model([hi for _, hi in SO101_RANGE_RAD])
    return [abs(h - lo) / std for lo, h, std in zip(low, high, SO100_ACTION_STD, strict=True)]


def test_state_units_decide_whether_the_joint_range_is_visible_to_the_model():
    """Packed as radians the whole arm travel is under 0.2 sigma; as degrees, 3.8+."""
    native = _sigma_span("native")
    degrees = _sigma_span("degrees")

    # Radians against degree stats: every joint's ENTIRE travel is under a fifth
    # of a standard deviation, so the state is a near-constant to the model.
    assert max(native) < 0.2, native
    # The declared unit conversion restores several sigma of travel per joint.
    assert min(degrees) > 3.5, degrees
    # Per joint, the unit half - not the stats - is what separates the regimes.
    for index, (raw, converted) in enumerate(zip(native, degrees, strict=True), start=1):
        assert converted > 20 * raw, f"joint {index}: native {raw} vs degrees {converted}"


def test_the_inert_normalization_warning_states_the_unit_half(caplog):
    """The load-time diagnostic names the unit knob and the consequence of skipping it."""
    bridge = MagicMock(name="ProcessorBridge")
    bridge.is_active = True
    bridge.has_postprocessor = True
    bridge.mismatched_normalization_widths.return_value = []
    bridge.inert_normalization_features.return_value = ["observation.state (STATE/MEAN_STD)"]

    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
    policy._device = None

    with (
        patch.object(LerobotLocalPolicy, "_configure_embodiment"),
        patch(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        ),
        caplog.at_level(logging.WARNING),
    ):
        policy._load_processor_bridge()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    inert = [m for m in warnings if "ACTIVE normalization pipeline" in m]
    assert inert, warnings
    message = inert[0]
    # The stats half was already named; the unit half must be too, by knob name ...
    assert "processor_overrides" in message
    assert "state_units" in message, message
    # ... with the units that differ and what skipping the conversion costs, so a
    # sim caller is not left applying half a remedy.
    assert "radians" in message, message
    assert "sigma" in message, message


def test_the_docs_remedy_shows_the_unit_half_beside_the_stats_half():
    """The page that hands out processor_overrides also hands out the unit knob."""
    text = _DOC.read_text(encoding="utf-8")
    start = text.index("## Processor bridge and normalization")
    section = text[start : text.index("\n## ", start + 1)]

    assert "processor_overrides" in section
    assert "state_units" in section, "the stats remedy is documented without its unit half"
    # The knob is only actionable next to the units it converts between.
    assert "radians" in section
    assert "degrees" in section


def _load_with(embodiment, inert: list[str]) -> None:
    """Run the load-time bridge check with one configured embodiment and stats verdict.

    ``_configure_embodiment`` is patched out, so ``policy._embodiment`` stands in
    for the map it would have installed.
    """
    bridge = MagicMock(name="ProcessorBridge")
    bridge.is_active = True
    bridge.has_postprocessor = True
    bridge.mismatched_normalization_widths.return_value = []
    bridge.inert_normalization_features.return_value = list(inert)
    bridge.prefixed_stat_key_candidates.return_value = {}

    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
    policy._device = None
    policy._embodiment = embodiment

    with (
        patch.object(LerobotLocalPolicy, "_configure_embodiment"),
        patch(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        ),
    ):
        policy._load_processor_bridge()


_INERT_STATE = ["observation.state (STATE/MEAN_STD)"]
_INERT_ACTION = ["action (ACTION/MEAN_STD)"]


@pytest.mark.parametrize(
    ("embodiment_name", "inert"),
    [
        ("so101", _INERT_STATE),  # the sim map: state packed in degrees, nothing rescales it
        ("so100", _INERT_ACTION),  # and the action side, un-unnormalized before conversion
    ],
)
def test_a_unit_converting_embodiment_is_refused_when_the_stats_are_inert(embodiment_name, inert):
    """The conversion lands and nothing scales it back, so the load refuses."""
    embodiment = EMBODIMENT_MAP[embodiment_name]
    assert embodiment.converts_units, embodiment_name

    with pytest.raises(ValueError) as excinfo:
        _load_with(embodiment, inert)

    message = str(excinfo.value)
    # Both halves of the mismatch, by the names the caller passed / the checkpoint ships.
    assert embodiment_name in message, message
    assert "state_units" in message and "action_units" in message, message
    assert inert[0] in message, message
    # Both remedies: supply the stats, or drop the conversion.
    assert "processor_overrides" in message, message
    assert "set_robot_state_keys" in message, message


@pytest.mark.parametrize(
    ("embodiment_name", "inert", "expect_warning"),
    [
        # A native map converts nothing, so the passthrough is the old warning.
        ("so_real", _INERT_STATE, True),
        # Covering stats scale the conversion back: neither refusal nor warning.
        ("so101", [], False),
    ],
)
def test_the_refusal_does_not_fire_on_a_configuration_that_still_works(embodiment_name, inert, expect_warning, caplog):
    """Only the converting-and-inert pairing is refused; each half alone loads."""
    with caplog.at_level(logging.WARNING):
        _load_with(EMBODIMENT_MAP[embodiment_name], inert)

    warned = [r.getMessage() for r in caplog.records if "ACTIVE normalization pipeline" in r.getMessage()]
    assert bool(warned) is expect_warning, warned


def test_the_legacy_path_without_an_embodiment_still_only_warns(caplog):
    """No declared map means no conversion to refuse, so the diagnostic is unchanged."""
    with caplog.at_level(logging.WARNING):
        _load_with(None, _INERT_STATE)

    assert any("ACTIVE normalization pipeline" in r.getMessage() for r in caplog.records)


def test_the_numbers_the_refusal_quotes_are_the_ones_the_map_produces():
    """160.0 degrees vs 2.79 native, at the so101 joint range the message names."""
    high = [hi for _, hi in SO101_RANGE_RAD]
    degrees = max(abs(v) for v in EMBODIMENT_MAP["so101"].sim_state_to_model(high))
    native = max(
        abs(v) for v in dataclasses.replace(EMBODIMENT_MAP["so101"], state_units="native").sim_state_to_model(high)
    )

    assert degrees == pytest.approx(160.0, abs=0.01), degrees
    assert native == pytest.approx(2.79, abs=0.01), native
