# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The environment-map clip planes are validated before anything is rendered.

``render_environment_map`` takes five numbers. Three of them size the pixel grid
and share one domain, checked before any face is rendered
(``test_environment_map_resolution_domain.py``). The other two -- ``znear`` and
``zfar`` -- are the view frustum every one of the six cube faces is rendered
through, and were checked nowhere. What follows is pinned here.

An inverted frustum produced a map rather than a refusal. ``znear=10.0`` with
``zfar=1.0`` returned a full-size, entirely black ``(equi_h, equi_w, 3)`` array,
reporting success, and only after paying all six background renders -- GPU-bound
for a ``GsplatBackground``. The refusal the caller eventually saw came from
``derive_key_light`` and blamed the scene: "the map is black above the horizon --
pass ``upper_hemisphere=False`` to search the full sphere". Following that advice
fails again ("the map is black"), because the frustum framed nothing at all
rather than leaving a dark upper half, so the only remedy offered was a dead end
and the clip planes the caller had asked for were named nowhere. That is the same
dead end the resolution knobs were guarded against, one axis over.

Neither rule can be left to the background. This function accepts *any*
``BackgroundRenderer`` and they do not agree about what an unusable pair means:
measured on one scene with the same arguments, ``PanoramaBackground`` ignores the
planes entirely and returns its usual map, while a ``GsplatBackground`` forwards
them to ``gsplat.rasterization`` as ``near_plane`` / ``far_plane`` and culls every
gaussian. That rasterizer is not even self-consistent -- ``znear=inf`` culls
everything, ``znear=nan`` is silently ignored, ``zfar=-1`` culls everything -- so
there is no tolerance to defer to, and which lighting a caller got was decided by
which background happened to be plugged in. Both behaviours are reproduced below
by the two test backgrounds, so the refusal is grounded in a real reading of the
planes rather than asserted.

Only the domain is pinned here, not the quality a frustum buys, matching the
resolution knobs. A very tight ordered pair is still a frustum: ``0.5``/``0.6``
frames a 10 cm depth slab, and on a room whose walls sit at 3 m that really does
come back black -- correctly, because the caller asked for a slab with nothing in
it. Where "too tight" begins is a judgement about the scene rather than a value
this module cannot use, so an ordered pair stays accepted and the tests below
assert only that the pair can frame something.

Each plane's own domain is :func:`~strands_robots.utils.positive_finite_number_error`,
the shared one for a continuous positive quantity, so what this refuses cannot
diverge from a rate or a duration elsewhere in the tree. It is also the authority
these tests parametrize over, so a value added there is covered here without an
edit. The ordering rule between the two is local to this module, per the
single-caller rule for a value domain: ``render_environment_map`` is the only
place in the tree that takes clip planes from a caller.
"""

import re
from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import (
    bake_environment_map,
    derive_key_light,
    environment_map_cache_path,
    render_environment_map,
)
from strands_robots.utils import positive_finite_number_error

# A resolution every entry point can honor, small enough to keep the test
# backgrounds cheap, and the documented default frustum.
# Typed loosely so a case may splat a value the signature rejects, which is the
# point of the test (the resolution guard's fixtures do the same).
GOOD: dict[str, Any] = {"face_size": 8, "equi_w": 32, "equi_h": 16}
GOOD_CLIP: dict[str, Any] = {"znear": 0.01, "zfar": 1e3}

# Values the shared continuous domain refuses. Parametrized over rather than
# hand-listed so the two cannot drift; each is asserted out of domain as a
# premise. ``bool`` is an ``int`` subclass whose ``True`` would act as a silent
# 1 m near plane; the huge int is past the float64 range the guard converts with.
OUT_OF_DOMAIN = [
    0,
    -1.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    None,
    "0.01",
    pytest.param(10**400, id="beyond_float64"),
]

CLIP_PARAMS = ["znear", "zfar"]

# Ordered pairs that cannot frame anything, so no depth interval is left.
EMPTY_FRUSTUMS = [
    pytest.param(10.0, 1.0, id="inverted"),
    pytest.param(1.0, 1.0, id="equal"),
    pytest.param(1e3, 0.01, id="defaults_swapped"),
]


def _kwargs(**overrides: Any) -> dict[str, Any]:
    """A usable resolution and frustum with ``overrides`` applied, as loose kwargs.

    Typed ``Any`` deliberately: half of these cases pass a value the signature
    rejects, which is the point of the test.
    """
    return {**GOOD, **GOOD_CLIP, **overrides}


class ClipIgnoringBackground:
    """A background that never reads the clip planes, like ``PanoramaBackground``.

    Paints a warm window on the ``+X`` face of a dim room, so the map carries a
    direction ``derive_key_light`` can actually find. Records how many renders it
    was asked for, so a refusal that arrives before the renders can be told from
    one that arrives after them.
    """

    name = "clip-ignoring"

    def __init__(self) -> None:
        self.renders = 0

    def _shade(self, cam) -> np.ndarray:
        w, h = int(cam.width), int(cam.height)
        fwd = -np.asarray(cam.T_world_cam, dtype=float)[:3, 2]
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, :] = (46, 54, 72)
        if fwd[0] > 0.9:  # the bake builds T_world_cam columns [right, up, -fwd]
            top = int(h * 0.30)
            lo, hi = max(0, w // 2 - w // 6), w // 2 + max(1, w // 6)
            rgb[top : max(top + 1, int(h * 0.55)), lo:hi] = (252, 206, 128)
        return rgb

    def render(self, cam):
        self.renders += 1
        rgb = self._shade(cam)
        return rgb, np.full(rgb.shape[:2], 1e3, np.float32)


class ClipHonoringBackground(ClipIgnoringBackground):
    """A background that culls outside ``[znear, zfar]``, like ``GsplatBackground``.

    ``GsplatBackground`` forwards the planes to ``gsplat.rasterization`` as
    ``near_plane`` / ``far_plane``, which drops every gaussian outside the
    interval. Its geometry sits at a fixed distance, so an interval that excludes
    that distance yields an all-black frame -- measured on a real 3DGS scene, and
    reproduced here by placing this background's geometry at ``SCENE_DEPTH``.
    """

    name = "clip-honoring"
    SCENE_DEPTH = 3.0

    def render(self, cam):
        self.renders += 1
        rgb = self._shade(cam)
        near, far = float(cam.znear), float(cam.zfar)
        if not near <= self.SCENE_DEPTH <= far:
            rgb = np.zeros_like(rgb)
        return rgb, np.full(rgb.shape[:2], 1e3, np.float32)


class TestAnEmptyFrustumIsRefusedRatherThanRendered:
    """The headline: a frustum with no depth interval must not be reported as a map."""

    @pytest.mark.parametrize(("znear", "zfar"), EMPTY_FRUSTUMS)
    def test_an_empty_frustum_is_refused_naming_both_planes(self, znear: float, zfar: float) -> None:
        bg = ClipHonoringBackground()
        try:
            env = render_environment_map(bg, **_kwargs(znear=znear, zfar=zfar))
        except ValueError as exc:
            text = str(exc)
            assert "znear" in text and "zfar" in text, f"the refusal must name both planes, got {text}"
            assert bg.renders == 0, f"refused after {bg.renders} background renders; check before rendering"
            return
        # Pre-fix: a map was returned. Report what the caller is left holding,
        # including the remedy the only refusal they ever see advertises.
        message = ""
        try:
            derive_key_light(env)
        except ValueError as exc:
            message = str(exc)
        remedy = re.search(r"pass (upper_hemisphere=\w+)", message)
        followed = "no remedy was offered"
        if remedy:
            try:
                derive_key_light(env, upper_hemisphere=False)
                followed = "the remedy worked"
            except ValueError as exc:
                followed = f"following it fails again: {exc}"
        raise AssertionError(
            f"znear={znear!r} zfar={zfar!r} returned a {env.shape} map whose brightest channel is "
            f"{int(env.max())} after {bg.renders} background renders, instead of naming the planes. "
            f"The caller's only refusal blames the scene: {message!r} -- and {followed}."
        )

    def test_the_scene_diagnosis_is_not_reachable_from_an_empty_frustum(self) -> None:
        """A clip-plane mistake must not be reported as a dark scene."""
        with pytest.raises(ValueError) as excinfo:
            render_environment_map(ClipHonoringBackground(), **_kwargs(znear=10.0, zfar=1.0))
        text = str(excinfo.value)
        assert "upper_hemisphere" not in text, f"a clip-plane refusal must not advise a search flag: {text}"
        assert "above the horizon" not in text, f"a clip-plane refusal must not blame the scene: {text}"


class TestEachClipPlaneSharesTheContinuousDomain:
    """Each plane is a distance in meters, on the shared positive-finite domain."""

    @pytest.mark.parametrize("bad", OUT_OF_DOMAIN)
    @pytest.mark.parametrize("param", CLIP_PARAMS)
    def test_render_refuses_what_the_shared_domain_refuses(self, param: str, bad: object) -> None:
        assert positive_finite_number_error(bad, param, "ctx") is not None, "premise: out of the shared domain"
        bg = ClipHonoringBackground()
        with pytest.raises(ValueError, match=re.escape(param)):
            render_environment_map(bg, **_kwargs(**{param: bad}))
        assert bg.renders == 0, f"refused after {bg.renders} renders; a GPU-bound bake must not be paid for"

    @pytest.mark.parametrize("param", CLIP_PARAMS)
    def test_the_raw_rasterizer_error_is_no_longer_the_caller_s_first_signal(self, param: str) -> None:
        """A non-numeric plane is refused here, not deep inside a rasterizer.

        Passed through, ``znear="0.01"`` reached ``gsplat``'s pybind11 binding and
        raised a ``TypeError`` naming ``fully_fused_projection_packed_fwd`` and
        dumping every gaussian tensor -- after one GPU render, and without naming
        the parameter at fault.
        """
        bg = ClipHonoringBackground()
        with pytest.raises(ValueError, match=re.escape(param)):
            render_environment_map(bg, **_kwargs(**{param: "0.01"}))
        assert bg.renders == 0


class TestTheAcceptedDomainIsUnchanged:
    """Controls: what worked before still works, and no quality floor was invented."""

    def test_the_documented_default_frustum_still_renders(self) -> None:
        bg = ClipHonoringBackground()
        env = render_environment_map(bg, **_kwargs())
        assert env.shape == (GOOD["equi_h"], GOOD["equi_w"], 3)
        assert bg.renders == 6
        assert int(env.max()) > 0, "the default frustum must frame the scene"

    @pytest.mark.parametrize(
        ("znear", "zfar"),
        [
            pytest.param(0.01, 1e3, id="defaults"),
            pytest.param(1e-6, 1e9, id="very_wide"),
            pytest.param(2.5, 3.5, id="fractional"),
            pytest.param(np.float32(0.01), np.float64(1000.0), id="numpy_reals"),
        ],
    )
    def test_an_ordered_pair_of_positive_finite_distances_is_accepted(self, znear: Any, zfar: Any) -> None:
        for value, param in ((znear, "znear"), (zfar, "zfar")):
            assert positive_finite_number_error(value, param, "ctx") is None, "premise: in the shared domain"
        bg = ClipHonoringBackground()
        env = render_environment_map(bg, **_kwargs(znear=znear, zfar=zfar))
        assert env.shape == (GOOD["equi_h"], GOOD["equi_w"], 3)
        assert bg.renders == 6

    def test_an_ordered_frustum_that_frames_nothing_is_still_accepted(self) -> None:
        """The domain is the ordering, not the quality: a narrow slab is a frustum.

        ``0.5``/``0.6`` asks for a 10 cm depth slab. On a scene whose geometry
        sits at 3 m that comes back black -- correctly, because the caller asked
        for a slab with nothing in it. Refusing it would be a judgement about the
        scene, which is the caller's to make.
        """
        bg = ClipHonoringBackground()
        env = render_environment_map(bg, **_kwargs(znear=0.5, zfar=0.6))
        assert bg.renders == 6, "an ordered pair must still reach the renders"
        assert int(env.max()) == 0, "premise: this ordered frustum really does frame nothing"

    def test_a_clip_ignoring_background_is_unaffected(self) -> None:
        """A background that never reads the planes renders the same map as before."""
        bg = ClipIgnoringBackground()
        env = render_environment_map(bg, **_kwargs())
        assert bg.renders == 6
        assert int(env.max()) > 0

    def test_a_bad_resolution_is_still_reported_before_a_bad_frustum(self) -> None:
        """Ordering control: the resolution refusal a caller already saw is unchanged."""
        with pytest.raises(ValueError, match=re.escape("equi_w")) as excinfo:
            render_environment_map(ClipHonoringBackground(), **_kwargs(equi_w=0, znear=10.0, zfar=1.0))
        assert "znear" not in str(excinfo.value)

    def test_the_bake_and_cache_path_take_no_clip_planes(self) -> None:
        """Scope control: only the renderer takes a frustum from the caller.

        ``bake_environment_map`` and ``environment_map_cache_path`` expose the
        resolutions but not the planes, so they render on the documented default
        and there is nothing for them to check or to encode in a cache key.
        """
        import inspect

        for func in (bake_environment_map, environment_map_cache_path):
            params = inspect.signature(func).parameters
            assert "znear" not in params and "zfar" not in params, f"{func.__name__} gained a clip plane"

    def test_the_bake_still_writes_a_map_on_the_default_frustum(self, tmp_path) -> None:
        out = tmp_path / "env.png"
        assert bake_environment_map(ClipHonoringBackground(), out, **GOOD) == out
        assert out.stat().st_size > 0


class TestTheRefusalIsGroundedInHowABackgroundReadsThePlanes:
    """Premise: the two behaviours the refusal exists for are both real."""

    def test_a_clip_honoring_background_returns_nothing_from_an_empty_frustum(self) -> None:
        """Without the guard this is the map the caller was handed."""
        from strands_robots.rendering.ibl import _face_camera

        bg = ClipHonoringBackground()
        cam, _, _ = _face_camera(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), 8, znear=10.0, zfar=1.0
        )
        rgb, _ = bg.render(cam)
        assert int(rgb.max()) == 0, "premise: an inverted frustum culls the whole scene"

    def test_the_two_backgrounds_disagree_about_the_same_pair(self) -> None:
        """Which map an unusable pair produced depended on the background.

        That is why the planes cannot be left to the renderer: the same call
        yielded a usable map through one background and an empty one through the
        other, and neither refused.
        """
        from strands_robots.rendering.ibl import _face_camera

        cam, _, _ = _face_camera(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), 8, znear=10.0, zfar=1.0
        )
        ignoring, _ = ClipIgnoringBackground().render(cam)
        honoring, _ = ClipHonoringBackground().render(cam)
        assert int(ignoring.max()) > 0
        assert int(honoring.max()) == 0
