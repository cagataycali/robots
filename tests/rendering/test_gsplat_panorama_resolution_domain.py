# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``bake_gsplat_panorama`` checks its resolutions before it names a file or loads splats.

``face_size``, ``equi_w`` and ``equi_h`` size the pixel grid of the panorama, and
this bake forwards all three to ``render_environment_map``. That renderer,
``bake_environment_map`` and ``environment_map_cache_path`` share one domain
(``positive_whole_number_error`` over the three names) checked before any render
and normalized with ``int()`` afterwards. ``bake_gsplat_panorama`` is the fourth
caller of the same knobs and had neither half, and two things read the values
before the renderer that owns the domain is reached.

*The default output path is composed from them, and that name is the cache key.*
The default is ``<stem>_pano_<equi_w>x<equi_h>_f<face_size>.jpg`` precisely so a
resolution cannot be silently dropped: before that, every bake of one scene
shared a single ``<stem>_pano.jpg`` and a later call asking for a different
resolution returned the first call's image. The shared domain accepts an integral
float, so ``face_size=640.0`` -- what a config float produces -- composed
``scene_pano_2048x1024_f640.0.jpg`` and fell through the short-circuit past the
warm ``scene_pano_2048x1024_f640.jpg`` beside it. Same silent no-op, reached by
spelling rather than by an ignored knob.

*The splats load before the refusal.* An unusable resolution was refused only
inside ``render_environment_map``, which runs after ``GsplatBackground._load()``.
With the ``sim-gs`` extra installed that costs the load. Without it the load does
not delay the refusal, it replaces it -- ``equi_w=0`` reported ``ImportError:
'torch' is required for 3D Gaussian Splatting backgrounds`` and advised
``pip install 'strands-robots[sim-gs]'``, advice that fixes nothing and names no
resolution.

That second fact is why every test here runs without ``torch``, ``gsplat`` or a
GPU: a refusal arriving before the splat load is a refusal reachable from a
CPU-only environment, so the assertion and the property it pins are the same
thing. ``GsplatBackground`` is doubled out to raise, which marks the boundary
between "checked the resolution" and "started paying for the pixels" -- a bad
resolution must not reach it, and a good one must.

Nothing here asserts that a refused request writes no file: the write is
downstream of the render, which is downstream of the backend, so "the backend was
not reached" already carries it.

The domain is :func:`~strands_robots.utils.positive_whole_number_error`, the
shared one for a knob counting pixels, parametrized over rather than hand-listed
so a value added there is covered here without an edit -- the same authority
``tests/rendering/test_environment_map_resolution_domain.py`` uses for the three
environment-map entry points.
"""

import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.rendering import backgrounds as bg
from strands_robots.utils import positive_whole_number_error

# A resolution the bake can honor. The values are irrelevant to every assertion
# here except that they are in domain, and they are spelled small so a name built
# from them is easy to read.
GOOD = {"face_size": 16, "equi_w": 64, "equi_h": 32}

# Values the shared domain refuses, asserted to be out of domain as a premise.
OUT_OF_DOMAIN = [0, -8, 2.5, True, False, float("nan"), float("inf"), None, "16"]

RESOLUTION_PARAMS = ["face_size", "equi_w", "equi_h"]


def _sizes(**overrides: Any) -> dict[str, Any]:
    """``GOOD`` with ``overrides`` applied, as loose kwargs.

    Typed ``Any`` deliberately: most of these cases pass a value the signature
    rejects, which is the point of the test.
    """
    return {**GOOD, **overrides}


@pytest.fixture
def ply(tmp_path: Path) -> Path:
    """A ``.ply`` that exists and is never parsed.

    Nothing here reaches the loader: either the resolution is refused first, or
    the cache short-circuits, or ``GsplatBackground`` is doubled out.
    """
    path = tmp_path / "scene.ply"
    path.write_bytes(b"not-a-real-ply")
    return path


@pytest.fixture
def no_splat_backend(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Fail the bake at ``GsplatBackground`` construction; return the call log."""
    calls: list[dict[str, Any]] = []

    def _refuse(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise AssertionError("reached the splat backend")

    monkeypatch.setattr(bg, "GsplatBackground", _refuse)
    return calls


class TestResolutionDomain:
    """An unusable resolution is refused, naming the knob, before anything is paid for."""

    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    @pytest.mark.parametrize("value", OUT_OF_DOMAIN)
    def test_out_of_domain_resolution_is_refused(
        self, ply: Path, no_splat_backend: list[dict[str, Any]], param: str, value: Any
    ) -> None:
        # Premise: the shared domain is what refuses this value, so the case list
        # above cannot drift from the domain it claims to cover.
        assert positive_whole_number_error(value, param, "premise") is not None

        with pytest.raises(ValueError, match=re.escape(param)):
            bg.bake_gsplat_panorama(ply, **_sizes(**{param: value}))

        # The ordering claim: refused before the splats are touched. Before this
        # fix a missing ``sim-gs`` extra answered instead, with an ImportError
        # advising an install that leaves the resolution exactly as unusable.
        assert no_splat_backend == []

    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    @pytest.mark.parametrize("value", OUT_OF_DOMAIN)
    def test_out_of_domain_resolution_is_refused_with_an_explicit_out_path(
        self, ply: Path, tmp_path: Path, no_splat_backend: list[dict[str, Any]], param: str, value: Any
    ) -> None:
        # An explicit ``out_path`` is honored verbatim, so it bypasses the name
        # composed from the resolutions -- but not the pixels those resolutions
        # size, which is the other thing the check protects. Covered separately
        # because it is the request shape where the cache-key argument for
        # checking early does not apply and the refusal still has to.
        with pytest.raises(ValueError, match=re.escape(param)):
            bg.bake_gsplat_panorama(ply, out_path=tmp_path / "explicit.jpg", **_sizes(**{param: value}))

        assert no_splat_backend == []

    def test_the_refusal_names_the_bake(self, ply: Path, no_splat_backend: list[dict[str, Any]]) -> None:
        # The shared domain quotes its caller, so a message arriving from a
        # six-render bake is attributable to it rather than to the renderer it
        # delegates to.
        with pytest.raises(ValueError, match=r"bake_gsplat_panorama"):
            bg.bake_gsplat_panorama(ply, **_sizes(equi_w=0))

    def test_a_usable_resolution_reaches_the_bake(self, ply: Path, no_splat_backend: list[dict[str, Any]]) -> None:
        # The complement of the cases above: the check refuses a domain, not a
        # request. ``face_size=1`` is coarse rather than unusable, and where "too
        # coarse" begins is the caller's judgement (the domain's own docstring).
        with pytest.raises(AssertionError, match="reached the splat backend"):
            bg.bake_gsplat_panorama(ply, **_sizes(face_size=1))

        assert no_splat_backend == [{"ply_path": ply, "device": "cuda"}]


class TestIntegralFloatSharesOneCacheEntry:
    """``16.0`` and ``16`` are one request, so they name one file."""

    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    def test_integral_float_hits_the_warm_default_path(
        self, ply: Path, no_splat_backend: list[dict[str, Any]], param: str
    ) -> None:
        warm = ply.with_name(f"{ply.stem}_pano_{GOOD['equi_w']}x{GOOD['equi_h']}_f{GOOD['face_size']}.jpg")
        warm.write_bytes(b"cached-panorama")

        result = bg.bake_gsplat_panorama(ply, **_sizes(**{param: float(GOOD[param])}))

        # Before this fix the float composed a second name (``..._f16.0.jpg``),
        # missed the warm file and went on to re-bake pixels already on disk.
        assert result == warm
        assert result.read_bytes() == b"cached-panorama"
        assert no_splat_backend == []
        assert sorted(p.name for p in ply.parent.iterdir()) == sorted([ply.name, warm.name])
