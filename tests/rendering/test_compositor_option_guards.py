# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``HybridCompositor`` only accepts render options it can actually honor.

Every option the compositor takes used to be coerced rather than checked, so a
value with no usable interpretation still returned a frame - just a different
one than the caller asked for:

* ``depth_epsilon`` was stored with a bare ``float(...)``. A non-finite
  threshold makes ``fg_depth > depth_epsilon`` false for every pixel, so the
  entire simulation foreground reads as empty and the composite is the
  background alone: the robot silently disappears. A negative threshold does
  the opposite, admitting the no-hit pixels (Isaac reports them as ``0``) that
  ``test_isaac_style_zero_and_inf_depth_reads_as_background`` pins as
  background.
* ``feather_pixels`` was clamped with ``max(0, int(value))``, so ``-5`` became
  ``0`` - silently disabling the seam blend the parameter exists to apply.
* ``width`` / ``height`` were read with ``width or self.default_width``, so a
  supplied ``0`` read as "not supplied" and the frame came back at the default
  size - even though the same ``0`` is refused by the backend's
  ``get_camera_params`` when it arrives as ``default_width``.

The compositor is pure numpy over the structural ``FrameSource`` protocol, so
these guards are pinned against a fake frame source with no GL / sim deps.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.rendering import HybridCompositor

from .test_compositor import ZFAR, FakeBackground, FakeSim, _square_depth

# Thresholds that are not a finite, non-negative distance in meters: the two
# non-finite floats that discard every foreground pixel, a negative distance, a
# numeric string, ``True`` (an ``int`` subclass that would act as a 1 m
# threshold) and a missing value.
UNUSABLE_DEPTH_EPSILON = [math.nan, math.inf, -math.inf, -1.0, "1e-4", True, None]

# Radii that are not a whole pixel count >= 0.
UNUSABLE_FEATHER_PIXELS = [-1, -5, 2.7, math.nan, math.inf, True, "3", None]

# Sizes no engine can render, including the ``0`` that truthiness read as absent.
UNUSABLE_SIZES = [0, -64, 2.5, math.nan, True, "64", []]


def _compositor(**kwargs) -> HybridCompositor:
    """A compositor over the fake square-geometry frame source."""
    return HybridCompositor(FakeSim(_square_depth()), background=FakeBackground(), **kwargs)


@pytest.mark.parametrize("depth_epsilon", UNUSABLE_DEPTH_EPSILON)
def test_unusable_depth_epsilon_is_refused_instead_of_discarding_the_foreground(depth_epsilon) -> None:
    """A threshold that cannot select geometry is refused, naming it and the value.

    The default threshold composites real geometry, so silently accepting one of
    these would have returned a frame with nothing simulated in it.
    """
    honored = _compositor(feather_pixels=0).render("cam")
    assert honored.foreground_mask.any(), "fixture must show foreground under the default threshold"

    with pytest.raises(ValueError, match="depth_epsilon"):
        _compositor(depth_epsilon=depth_epsilon)


def test_zero_depth_epsilon_is_accepted_as_the_exactly_zero_convention() -> None:
    """``0`` means "only an exactly-zero depth is no geometry" - a real setting."""
    frame = _compositor(depth_epsilon=0.0, feather_pixels=0).render("cam")
    assert frame.foreground_mask[5, 7]


@pytest.mark.parametrize("feather_pixels", UNUSABLE_FEATHER_PIXELS)
def test_unusable_feather_pixels_is_refused_instead_of_clamped_to_off(feather_pixels) -> None:
    """A radius with no box kernel is refused rather than silently becoming ``0``.

    Feathering is observable in the output, so the clamp turned "blend the seam"
    into "do not blend it" with no diagnostic.
    """
    hard = _compositor(feather_pixels=0).render("cam")
    soft = _compositor(feather_pixels=2).render("cam")
    assert not np.array_equal(hard.rgb, soft.rgb), "feathering must be observable for the clamp to matter"

    with pytest.raises(ValueError, match="feather_pixels"):
        _compositor(feather_pixels=feather_pixels)


@pytest.mark.parametrize("size", UNUSABLE_SIZES)
def test_unusable_render_size_is_refused_rather_than_read_as_absent(size) -> None:
    """``render(width=0)`` is refused, not silently replaced by the default size."""
    comp = _compositor(default_width=32, default_height=24, feather_pixels=0)
    with pytest.raises(ValueError, match="HybridCompositor.render: width"):
        comp.render("cam", width=size)
    with pytest.raises(ValueError, match="HybridCompositor.render: height"):
        comp.render("cam", height=size)


@pytest.mark.parametrize("size", UNUSABLE_SIZES)
def test_unusable_default_size_is_refused_by_the_constructor_that_took_it(size) -> None:
    """A bad default size names the constructor argument, not a later render call."""
    with pytest.raises(ValueError, match="HybridCompositor: default_width"):
        _compositor(default_width=size)
    with pytest.raises(ValueError, match="HybridCompositor: default_height"):
        _compositor(default_height=size)


def test_requested_size_reaches_the_frame_source_and_defaults_when_absent() -> None:
    """The size the engine is asked for is the caller's, or the stored default."""

    class RecordingSim(FakeSim):
        def __init__(self, depth):
            super().__init__(depth)
            self.requested: list[tuple[int | None, int | None]] = []

        def get_camera_params(self, camera_name="default", width=None, height=None):
            self.requested.append((width, height))
            return super().get_camera_params(camera_name, width=width, height=height)

    sim = RecordingSim(_square_depth())
    comp = HybridCompositor(sim, background=FakeBackground(), default_width=32, default_height=24, feather_pixels=0)
    comp.render("cam")
    comp.render("cam", width=8, height=6)
    assert sim.requested == [(32, 24), (8, 6)]


def test_integral_options_are_normalized_to_plain_python_numbers() -> None:
    """A NumPy size or an integral float is stored as ``int`` before it is used."""
    comp = _compositor(default_width=np.int64(32), default_height=24.0, feather_pixels=2.0)
    assert (comp.default_width, comp.default_height, comp.feather_pixels) == (32, 24, 2)
    assert all(type(v) is int for v in (comp.default_width, comp.default_height, comp.feather_pixels))
    assert comp.render("cam").camera.width == 32


def test_a_negative_threshold_would_have_admitted_isaac_no_hit_pixels() -> None:
    """The refused negative threshold is the one that breaks the no-hit convention.

    A ``0`` depth is Isaac's "ray hit nothing"; under the default threshold it
    reads as background (as its own test pins). Only a negative threshold puts
    it above the bar, painting simulation sky over the photoreal background.
    """
    depth = _square_depth()
    depth[0, 0] = 0.0
    comp = HybridCompositor(FakeSim(depth), background=FakeBackground(), feather_pixels=0)
    assert not comp.render("cam").foreground_mask[0, 0]
    assert 0.0 > -1.0, "the arithmetic the refused threshold would have inverted"
    with pytest.raises(ValueError, match="depth_epsilon"):
        HybridCompositor(FakeSim(depth), background=FakeBackground(), depth_epsilon=-1.0)


def test_a_far_clip_depth_still_reads_as_background_under_a_valid_threshold() -> None:
    """Sanity: the accepted domain does not change MuJoCo's zfar sky convention."""
    depth = _square_depth()
    assert depth[0, 0] == pytest.approx(ZFAR)
    frame = _compositor(depth_epsilon=1e-3, feather_pixels=0).render("cam")
    assert not frame.foreground_mask[0, 0]
