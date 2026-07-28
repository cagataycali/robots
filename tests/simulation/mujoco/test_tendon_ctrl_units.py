"""Regression tests: a tendon command in the actuator's own units is honoured.

``_scale_ctrl_for_actuator`` maps a normalised ``[0, 1]`` open/close fraction
onto a tendon gripper's wide ctrlrange (the Panda / Robotiq ``[0, 255]``), which
is the conventional VLA gripper command. Deciding whether an incoming ``0.5`` is
that fraction or a literal tendon command is only safe when one command unit is
a negligible slice of travel.

The previous rule remapped for **any** ``span > 1.0``, which corrupted every
tendon whose ctrlrange is in physical units:

* Shadow Hand ``[0, 3.1415]`` rad - ``1.0`` rad became ``3.1415`` rad, and the
  mapping was discontinuous (``1.5`` rad passed through as ``1.5``).
* A finger-travel range ``[0, 0.04]`` m - ``0.02`` m became ``0.0008`` m, 25x
  short, so a half-open command closed the gripper almost fully.

These tests pin both directions: the wide-range rescue still happens, and a
physical-unit range is interpreted literally and monotonically.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402

_XML = """
<mujoco model="tendon_units">
  <worldbody>
    <body name="link">
      <joint name="finger" type="slide" axis="0 1 0" range="0 0.04"/>
      <geom type="box" size="0.01 0.01 0.01"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="grip"><joint joint="finger" coef="1"/></fixed>
  </tendon>
  <actuator>
    <position name="grip_act" tendon="grip" kp="10" ctrlrange="{lo} {hi}"/>
  </actuator>
</mujoco>
"""


def _model(lo: float, hi: float):
    return mujoco.MjModel.from_xml_string(_XML.format(lo=lo, hi=hi))


def _scale(model, value: float) -> float:
    return RenderingMixin._scale_ctrl_for_actuator(model, 0, value, mujoco)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (0.5, 127.5), (1.0, 255.0)],
)
def test_wide_range_still_maps_normalised_fraction(value: float, expected: float) -> None:
    """A [0, 255] tendon keeps the normalised-fraction rescue (issue #318)."""
    assert _scale(_model(0.0, 255.0), value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, 127.5, 255.0])
def test_wide_range_passes_through_literal_command(value: float) -> None:
    """A clearly in-range tendon-unit command is still honoured verbatim."""
    assert _scale(_model(0.0, 255.0), value) == pytest.approx(value)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 1.5, 3.1415])
def test_radian_range_is_literal(value: float) -> None:
    """A Shadow-Hand-style [0, pi] rad tendon is NOT rescaled.

    Pre-fix, 1.0 rad wrote 3.1415 rad (fully flexed) while 1.5 rad wrote 1.5 -
    a discontinuity across the 1.0 boundary.
    """
    assert _scale(_model(0.0, 3.1415), value) == pytest.approx(value)


@pytest.mark.parametrize("value", [0.0, 0.01, 0.02, 0.04])
def test_metre_range_is_literal(value: float) -> None:
    """A finger-travel [0, 0.04] m tendon is NOT rescaled (was 25x short)."""
    assert _scale(_model(0.0, 0.04), value) == pytest.approx(value)


def test_physical_range_is_monotonic() -> None:
    """No discontinuity: ctrl must be non-decreasing in the command."""
    model = _model(0.0, 3.1415)
    commands = [0.0, 0.25, 0.5, 0.75, 0.99, 1.0, 1.01, 1.5, 2.0, 3.0]
    out = [_scale(model, c) for c in commands]
    assert out == sorted(out), out


def test_out_of_range_command_is_clamped_not_wrapped() -> None:
    """A physical-unit command past ``hi`` clamps to ``hi`` (MuJoCo would too)."""
    assert _scale(_model(0.0, 0.04), 9.0) == pytest.approx(0.04)
