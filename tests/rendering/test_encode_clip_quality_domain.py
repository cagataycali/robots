# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``encode_clip`` only accepts a ``quality`` the clip encoder can honor.

``quality`` was the one knob in ``encode_clip``'s signature with no domain, and
the dependency's own enforcement is not a substitute for one:

* ``imageio-ffmpeg`` bounds the knob with a bare ``assert 1 <= quality <= 10``,
  so the refusal is an ``AssertionError`` -- absent from the documented
  ``Raises:`` set -- and ``python -O`` strips it. On an optimized interpreter
  ``quality=-5`` and ``quality=0`` encoded silently (a real but different clip),
  ``nan`` and ``"8"`` leaked raw arithmetic errors out of the bitrate
  computation, and ``500`` surfaced only as "the encoder wrote no clip".
* The docstring advertised ``0-10`` while the writer asserts ``1 <= quality``,
  so ``0`` was a documented value the code refused.
* ``True`` was accepted as a silent quality of ``1`` -- the lowest the encoder
  offers -- which is the same substitution ``_jpeg_quality_error`` already
  rejects for the streaming encoder in this module.
* A NumPy real such as ``np.int64(8)`` was refused by the writer's
  ``isinstance(quality, (float, int))`` gate even though it names a perfectly
  usable quality.

These tests pin the domain, its independence from ``-O``, and that a usable
quality still reaches the encoder unchanged.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys

import numpy as np
import pytest

from strands_robots.rendering import encode_clip
from strands_robots.rendering.video import (
    _MAX_CLIP_QUALITY,
    _MAX_JPEG_QUALITY,
    _MIN_CLIP_QUALITY,
    _MIN_JPEG_QUALITY,
    _clip_quality_error,
    _jpeg_quality_error,
)

# Values the clip encoder cannot turn into the requested quality: below and
# above the range it accepts (``0`` is the one the docstring used to advertise),
# non-finite floats, both bools (``True`` silently acted as a quality of 1), a
# numeric string, a missing value, a sequence, and an integer too large to
# convert to a float.
OUT_OF_RANGE = [0, -5, 11, 500]
NOT_A_NUMBER = [math.nan, math.inf, -math.inf, True, False, np.True_, "8", None, [8], 10**400]
UNUSABLE_QUALITY = [*OUT_OF_RANGE, *NOT_A_NUMBER]

# Qualities inside the range, in every spelling the guard accepts: the bounds,
# a plain int, a fractional value the encoder maps onto a bitrate, an integral
# float, and NumPy reals that the writer's own type gate would refuse.
USABLE_QUALITY = [1, 10, 8, 2.7, 8.0, np.int64(8), np.float32(8.0), np.float64(8.0)]


def _frames(n: int = 6, w: int = 48, h: int = 32) -> list[np.ndarray]:
    """``n`` distinct RGB frames of one shape (a gradient, so decode is visible)."""
    return [np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8) for i in range(n)]


@pytest.mark.parametrize("suffix", [".gif", ".mp4"])
@pytest.mark.parametrize("quality", UNUSABLE_QUALITY)
def test_unusable_quality_is_refused_before_any_file_is_written(tmp_path, suffix, quality) -> None:
    """A quality the encoder cannot honor is a ValueError naming the parameter."""
    out = tmp_path / f"clip{suffix}"
    with pytest.raises(ValueError, match="quality must be"):
        encode_clip(_frames(), out, fps=10, quality=quality)
    assert not out.exists(), f"encode_clip(quality={quality!r}) left a file behind at {out}"


def test_the_lower_bound_is_one_not_the_zero_the_docstring_advertised() -> None:
    """``0`` is refused: the encoder asserts ``1 <= quality``, so 0-10 was wrong."""
    assert _MIN_CLIP_QUALITY == 1
    assert _MAX_CLIP_QUALITY == 10
    assert _clip_quality_error(0) is not None
    assert _clip_quality_error(_MIN_CLIP_QUALITY) is None


@pytest.mark.parametrize("suffix", [".gif", ".mp4"])
@pytest.mark.parametrize("quality", USABLE_QUALITY)
def test_a_usable_quality_still_encodes_a_clip(tmp_path, suffix, quality) -> None:
    """Every accepted spelling encodes, so the guard is not narrower than the encoder."""
    pytest.importorskip("imageio.v2")
    out = encode_clip(_frames(), tmp_path / f"clip{suffix}", fps=10, quality=quality)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("quality", [np.int64(8), np.float32(8.0), np.float64(8.0), 8.0])
def test_a_numpy_real_encodes_identically_to_the_plain_int(tmp_path, quality) -> None:
    """The ``float()`` at the writer is load-bearing: NumPy reals are honored.

    The ffmpeg writer gates the knob on ``isinstance(quality, (float, int))``,
    which ``np.int64`` and ``np.float32`` fail. Asserting byte equality with the
    plain ``int`` pins both halves: the value reaches the encoder, and the
    conversion does not change the requested quality.
    """
    pytest.importorskip("imageio.v2")
    plain = encode_clip(_frames(), tmp_path / "plain.mp4", fps=10, quality=8)
    coerced = encode_clip(_frames(), tmp_path / "coerced.mp4", fps=10, quality=quality)
    assert hashlib.md5(coerced.read_bytes()).digest() == hashlib.md5(plain.read_bytes()).digest()


def test_a_fractional_quality_is_honored_rather_than_rounded(tmp_path) -> None:
    """``2.7`` is a real quality here, unlike Pillow's whole-number JPEG scale."""
    pytest.importorskip("imageio.v2")
    fractional = encode_clip(_frames(), tmp_path / "frac.mp4", fps=10, quality=2.7)
    whole = encode_clip(_frames(), tmp_path / "whole.mp4", fps=10, quality=3)
    assert fractional.read_bytes() != whole.read_bytes()


def test_true_would_have_been_a_silent_quality_of_one(tmp_path) -> None:
    """``True`` is refused, and the substitution it would have made is not a no-op.

    ``bool`` is an ``int`` subclass, so ``True`` reached the encoder as a quality
    of ``1`` -- the lowest offered. Encoding at ``1`` and at the default ``8``
    produces different clips, so the substitution silently changed the artifact
    rather than being harmless.
    """
    pytest.importorskip("imageio.v2")
    with pytest.raises(ValueError, match="quality must be"):
        encode_clip(_frames(), tmp_path / "bool.mp4", fps=10, quality=True)
    lowest = encode_clip(_frames(), tmp_path / "one.mp4", fps=10, quality=1)
    default = encode_clip(_frames(), tmp_path / "eight.mp4", fps=10, quality=8)
    assert lowest.read_bytes() != default.read_bytes()


@pytest.mark.parametrize("quality", [0, -5, 500, True])
def test_quality_is_refused_before_the_optional_encoder_is_probed(tmp_path, monkeypatch, quality) -> None:
    """The same mistake reports identically whether or not imageio is installed."""
    from strands_robots.rendering import video

    def _no_imageio(*args, **kwargs):
        raise AssertionError("require_optional must not be reached for a refused quality")

    monkeypatch.setattr(video, "require_optional", _no_imageio)
    with pytest.raises(ValueError, match="quality must be"):
        encode_clip(_frames(), tmp_path / "clip.mp4", fps=10, quality=quality)


@pytest.mark.parametrize("quality", [0, 500, True, "8"])
def test_the_domain_does_not_depend_on_the_output_container(tmp_path, quality) -> None:
    """One signature, one domain: changing the extension cannot make a call valid.

    ``quality`` is read only by the MP4 writer, but it is refused for a ``.gif``
    target too - otherwise the same call would be valid or invalid depending on
    a string, and a caller would learn that an out-of-range quality is fine.
    """
    messages = []
    for suffix in (".gif", ".mp4"):
        with pytest.raises(ValueError, match="quality must be") as excinfo:
            encode_clip(_frames(), tmp_path / f"clip{suffix}", fps=10, quality=quality)
        messages.append(str(excinfo.value))
    assert messages[0] == messages[1], messages


def test_the_refusal_does_not_depend_on_assertions_being_enabled(tmp_path) -> None:
    """``python -O`` strips the encoder's own ``assert``, so the guard must not be one.

    Run in a child interpreter with ``-O``: the dependency's bound disappears
    there, and pre-fix ``quality=-5`` encoded a clip rather than being refused.
    """
    code = (
        "import numpy as np\n"
        "from strands_robots.rendering import encode_clip\n"
        "frames = [np.zeros((32, 48, 3), np.uint8) for _ in range(4)]\n"
        "for bad in (-5, 0, 500, True):\n"
        "    try:\n"
        f"        encode_clip(frames, {str(tmp_path / 'child.mp4')!r}, fps=10, quality=bad)\n"
        "    except ValueError as exc:\n"
        "        if 'quality must be' not in str(exc):\n"
        "            print('WRONG-REASON', bad, exc)\n"
        "            raise SystemExit(1)\n"
        "    else:\n"
        "        print('ACCEPTED', bad)\n"
        "        raise SystemExit(1)\n"
        "print('ALL-REFUSED')\n"
    )
    proc = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr[-800:]!r}"
    assert "ALL-REFUSED" in proc.stdout, proc.stdout


@pytest.mark.parametrize("quality", NOT_A_NUMBER)
def test_both_quality_knobs_in_this_module_refuse_the_same_non_numeric_values(quality) -> None:
    """The clip and JPEG quality guards agree on what is not a quality at all.

    The two encoders accept different ranges, but "is this a number a quality
    can be read from" is one question, so ``bool``, a numeric string, a
    non-finite float and a missing value are refused by both.
    """
    assert _clip_quality_error(quality) is not None
    assert _jpeg_quality_error(quality) is not None


def test_the_two_quality_ranges_differ_because_the_encoders_do() -> None:
    """Each range is the one its own encoder honors, so they are not interchangeable.

    Pinned as a deliberate divergence rather than left to convention. libx264
    takes ``[1, 10]`` and maps the knob onto a bitrate by arithmetic, so ``2.7``
    is a real quality; Pillow's JPEG quality is a whole-number ``[1, 95]`` scale,
    so ``11`` is usable there and not for a clip.
    """
    assert (_MIN_CLIP_QUALITY, _MAX_CLIP_QUALITY) == (1, 10)
    assert (_MIN_JPEG_QUALITY, _MAX_JPEG_QUALITY) == (1, 95)
    # A fractional quality: usable for the clip encoder, not for the JPEG scale.
    assert _clip_quality_error(2.7) is None
    assert _jpeg_quality_error(2.7) is not None
    # A whole number above the clip range but inside the JPEG one.
    assert _clip_quality_error(11) is not None
    assert _jpeg_quality_error(11) is None
