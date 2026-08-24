# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``encode_clip`` only accepts a ``macro_block_size`` the clip encoder can honor.

``macro_block_size`` is the third of ``encode_clip``'s three encoder knobs, and
the only one left with no domain: ``fps`` resolves through
:func:`~strands_robots.utils.positive_whole_number_error` and ``quality``
through ``_clip_quality_error``, while the rounding step was handed to the
writer unchecked. The accepted set was wrong in both directions.

* **Nonsense in.** ``imageio``'s ffmpeg plugin normalizes a falsy value
  (``macro_block_size = macro_block_size or 1``) and only rounds when the value
  is ``> 1``, so ``0``, ``-4``, ``None``, ``False`` and ``True`` all encoded as
  "no rounding" -- byte-identical to ``macro_block_size=1`` -- while the caller
  was told the clip was written. ``True`` is the same ``bool`` substitution
  ``_clip_quality_error`` already rejects for this module's other quality knob.
* **The dependency's check is a bare assert.** ``imageio-ffmpeg`` guards the
  knob with ``assert isinstance(macro_block_size, int)``, so the refusal was an
  ``AssertionError`` -- absent from the documented ``Raises:`` set -- and
  ``python -O`` strips it. On an optimized interpreter ``nan`` encoded that same
  byte-identical clip and reported success, ``2.5`` and ``inf`` surfaced only as
  "the encoder wrote no clip", and ``"8"`` leaked a raw ``TypeError`` out of the
  writer's ``macro_block_size > 1`` comparison. The verdict for one call
  depended on an interpreter flag.
* **Usable out.** ``8.0``, ``np.int64(8)`` and ``np.float64(8.0)`` each name the
  same block size as the ``8`` that encodes, and the writer's ``isinstance(...,
  int)`` gate refused all three -- the mirror of the ``float(quality)``
  conversion this module already documents as load-bearing.

These tests pin the domain, its independence from ``-O``, that every usable
spelling still reaches the writer, and that the codec's own refusal of a
*legitimate* block size stays the post-encode ``RuntimeError`` it already was.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys

import numpy as np
import pytest

from strands_robots.rendering import encode_clip
from strands_robots.utils import positive_whole_number_error

# Values that name no macro-block size: zero and negative sizes, a missing
# value, both bools (each acted as "no rounding"), a NumPy bool, a fractional
# size, non-finite floats, a numeric string, a sequence, and an integer too
# large to convert to a float.
UNUSABLE_BLOCK_SIZE = [0, -4, None, False, True, np.True_, 2.5, math.nan, math.inf, -math.inf, "8", [8], 10**400]

# Every spelling of a usable block size: the recorders' ``1``, the examples'
# ``8``, the writer's own default ``16``, an integral float, and NumPy integers
# and reals that the writer's type gate would refuse on its own.
USABLE_BLOCK_SIZE = [1, 8, 16, 8.0, np.int64(8), np.float64(8.0)]

# Values that were silently read as "no rounding" rather than refused. Kept
# apart from the rest of ``UNUSABLE_BLOCK_SIZE`` because these are the ones the
# caller was told had been honored.
SILENTLY_IGNORED = [0, -4, None, False, True]


def _frames(n: int = 6, w: int = 48, h: int = 32) -> list[np.ndarray]:
    """``n`` distinct RGB frames of one shape (a gradient, so decode is visible).

    ``48x32`` is divisible by 16, so every block size in
    :data:`USABLE_BLOCK_SIZE` rounds it to itself and the codec accepts the
    result -- the rounding's own effect is measured separately, on a frame size
    chosen for it.
    """
    return [np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8) for i in range(n)]


def _odd_frames(n: int = 4) -> list[np.ndarray]:
    """Frames at ``60x40``, divisible by neither 8 nor 16, so rounding is visible."""
    return [np.full((40, 60, 3), 30 + i * 30, dtype=np.uint8) for i in range(n)]


class TestAnUnusableMacroBlockSizeIsRefused:
    """A value that names no block size is a ValueError, before any I/O."""

    @pytest.mark.parametrize("suffix", [".gif", ".mp4"])
    @pytest.mark.parametrize("macro_block_size", UNUSABLE_BLOCK_SIZE)
    def test_it_is_refused_before_any_file_is_written(self, tmp_path, suffix, macro_block_size) -> None:
        """The refusal names the parameter and leaves no artifact behind."""
        out = tmp_path / f"clip{suffix}"
        with pytest.raises(ValueError, match="macro_block_size must be"):
            encode_clip(_frames(), out, fps=10, macro_block_size=macro_block_size)
        assert not out.exists(), f"encode_clip(macro_block_size={macro_block_size!r}) left a file at {out}"

    @pytest.mark.parametrize("macro_block_size", UNUSABLE_BLOCK_SIZE)
    def test_the_refusal_is_the_shared_whole_number_domains_own_text(self, tmp_path, macro_block_size) -> None:
        """One domain, one message: the rounding step shares ``fps``'s guard.

        Compared against the helper rather than a restated string, so a block
        size refused when it names a frame rate cannot be accepted when it names
        the rounding of that same clip.
        """
        with pytest.raises(ValueError) as excinfo:
            encode_clip(_frames(), tmp_path / "clip.mp4", fps=10, macro_block_size=macro_block_size)
        assert str(excinfo.value) == positive_whole_number_error(macro_block_size, "macro_block_size", "encode_clip")

    @pytest.mark.parametrize("macro_block_size", [0, None, True, 2.5, "8"])
    def test_it_is_refused_before_the_optional_encoder_is_probed(self, tmp_path, monkeypatch, macro_block_size) -> None:
        """The same mistake reports identically whether or not imageio is installed."""
        from strands_robots.rendering import video

        def _no_imageio(*args, **kwargs):
            raise AssertionError("require_optional must not be reached for a refused block size")

        monkeypatch.setattr(video, "require_optional", _no_imageio)
        with pytest.raises(ValueError, match="macro_block_size must be"):
            encode_clip(_frames(), tmp_path / "clip.mp4", fps=10, macro_block_size=macro_block_size)

    @pytest.mark.parametrize("macro_block_size", [0, None, True, 2.5])
    def test_the_domain_does_not_depend_on_the_output_container(self, tmp_path, macro_block_size) -> None:
        """GIF has no macro blocks, but changing the extension cannot make a call valid.

        Otherwise the same call would be accepted or refused depending on a
        string, and a caller would learn that a block size of ``0`` is fine.
        """
        messages = []
        for suffix in (".gif", ".mp4"):
            with pytest.raises(ValueError, match="macro_block_size must be") as excinfo:
                encode_clip(_frames(), tmp_path / f"clip{suffix}", fps=10, macro_block_size=macro_block_size)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1], messages


class TestTheIgnoredRequestWasInvisible:
    """The values that were accepted encoded as if ``1`` had been asked for."""

    @pytest.mark.parametrize("macro_block_size", SILENTLY_IGNORED)
    def test_a_block_size_that_would_have_been_ignored_is_refused(self, tmp_path, macro_block_size) -> None:
        """Nothing reported the knob had been dropped, so the refusal has to."""
        out = tmp_path / "ignored.mp4"
        with pytest.raises(ValueError, match="macro_block_size must be"):
            encode_clip(_odd_frames(), out, fps=10, macro_block_size=macro_block_size)
        assert not out.exists()

    def test_the_substitution_it_would_have_made_is_not_a_no_op(self, tmp_path) -> None:
        """Rounding changes the encoded clip, so being read as ``1`` mattered.

        At ``60x40`` a block size of ``8`` pads the frame to ``64x40`` and ``16``
        to ``64x48``, so a request that was silently read as "no rounding"
        produced a clip at dimensions other than the ones asked for -- this is
        the premise that makes the refusal above worth having.
        """
        pytest.importorskip("imageio.v2")
        pytest.importorskip("imageio_ffmpeg")
        imageio = pytest.importorskip("imageio.v2")
        sizes = {}
        for block in (1, 8, 16):
            path = encode_clip(_odd_frames(), tmp_path / f"b{block}.mp4", fps=10, macro_block_size=block)
            reader = imageio.get_reader(path)
            try:
                sizes[block] = tuple(reader.get_meta_data()["size"])
            finally:
                reader.close()
        assert sizes[1] == (60, 40), sizes
        assert len(set(sizes.values())) == 3, f"rounding must change the encoded size: {sizes}"


class TestTheRefusalDoesNotDependOnAssertionsBeingEnabled:
    """``python -O`` strips the writer's ``assert``, so the guard must not be one."""

    def test_the_writers_own_check_is_gone_under_dash_o(self, tmp_path) -> None:
        """Premise: with ``-O`` the dependency accepts a block size it otherwise asserts on.

        Measured against the writer directly, bypassing ``encode_clip``, so this
        states the dependency's behaviour rather than this module's.
        """
        pytest.importorskip("imageio.v2")
        pytest.importorskip("imageio_ffmpeg")
        code = (
            "import math\n"
            "import imageio.v2 as imageio\n"
            f"writer = imageio.get_writer({str(tmp_path / 'raw.mp4')!r}, fps=10, macro_block_size=math.nan)\n"
            "writer.close()\n"
            "print('NO-ASSERTION')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=300, check=False
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr[-800:]!r}"
        assert "NO-ASSERTION" in proc.stdout, proc.stdout

    def test_every_unusable_block_size_is_still_refused_under_dash_o(self, tmp_path) -> None:
        """Pre-fix, ``nan`` encoded a clip here and reported success."""
        code = (
            "import math\n"
            "import numpy as np\n"
            "from strands_robots.rendering import encode_clip\n"
            "frames = [np.zeros((32, 48, 3), np.uint8) for _ in range(4)]\n"
            "for bad in (0, -4, None, False, True, 2.5, math.nan, math.inf, '8'):\n"
            "    try:\n"
            f"        encode_clip(frames, {str(tmp_path / 'child.mp4')!r}, fps=10, macro_block_size=bad)\n"
            "    except ValueError as exc:\n"
            "        if 'macro_block_size must be' not in str(exc):\n"
            "            print('WRONG-REASON', bad, exc)\n"
            "            raise SystemExit(1)\n"
            "    else:\n"
            "        print('ACCEPTED', bad)\n"
            "        raise SystemExit(1)\n"
            "print('ALL-REFUSED')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=300, check=False
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr[-800:]!r}"
        assert "ALL-REFUSED" in proc.stdout, proc.stdout


class TestAUsableBlockSizeReachesTheWriter:
    """Every spelling the domain accepts encodes, so the guard is not too narrow."""

    @pytest.mark.parametrize("suffix", [".gif", ".mp4"])
    @pytest.mark.parametrize("macro_block_size", USABLE_BLOCK_SIZE)
    def test_every_accepted_spelling_encodes_a_clip(self, tmp_path, suffix, macro_block_size) -> None:
        """A NumPy integer or an integral float is honored, not refused by the writer."""
        pytest.importorskip("imageio.v2")
        out = encode_clip(_frames(), tmp_path / f"clip{suffix}", fps=10, macro_block_size=macro_block_size)
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.parametrize("macro_block_size", [8.0, np.int64(8), np.float64(8.0)])
    def test_a_numpy_or_float_spelling_encodes_identically_to_the_plain_int(self, tmp_path, macro_block_size) -> None:
        """The ``int()`` at the writer is load-bearing, and does not change the request.

        The writer gates the knob on ``isinstance(macro_block_size, int)``, which
        ``np.int64(8)`` and ``8.0`` fail even though each names the same block
        size as ``8``. Byte equality with the plain ``int`` pins both halves: the
        value reaches the encoder, and the conversion is not a different request.
        """
        pytest.importorskip("imageio.v2")
        pytest.importorskip("imageio_ffmpeg")
        plain = encode_clip(_odd_frames(), tmp_path / "plain.mp4", fps=10, macro_block_size=8)
        coerced = encode_clip(_odd_frames(), tmp_path / "coerced.mp4", fps=10, macro_block_size=macro_block_size)
        assert hashlib.md5(coerced.read_bytes()).digest() == hashlib.md5(plain.read_bytes()).digest()


class TestTheCodecRefusalBoundaryIsUnchanged:
    """A legitimate block size the codec refuses stays the post-encode error."""

    def test_a_whole_block_size_the_codec_refuses_is_still_reported_after_the_encode(self, tmp_path) -> None:
        """``7`` is a positive whole number, so the domain accepts it and libx264 does not.

        Deliberately not folded into the pre-flight guard: this function cannot
        know which rounded sizes a codec accepts, so the verdict belongs to the
        encoder and reaches the caller as the existing ``RuntimeError``.
        """
        pytest.importorskip("imageio.v2")
        pytest.importorskip("imageio_ffmpeg")
        assert positive_whole_number_error(7, "macro_block_size", "encode_clip") is None
        out = tmp_path / "refused.mp4"
        with pytest.raises(RuntimeError, match="wrote no clip"):
            encode_clip(_frames(), out, fps=20, macro_block_size=7)
        assert not out.exists() or out.stat().st_size == 0

    def test_the_domain_applies_no_ceiling_of_its_own(self) -> None:
        """An outsized block size is the codec's refusal to give, not the domain's.

        ``10**6`` rounds a 48x32 frame to a megapixel-scale size libx264 will
        not encode, and that is reported by the same ``RuntimeError`` as ``7``.
        Pinned so a later ceiling is a decision rather than an accident.
        """
        assert positive_whole_number_error(10**6, "macro_block_size", "encode_clip") is None

    def test_the_other_two_knobs_keep_their_own_domains(self, tmp_path) -> None:
        """The rounding guard is added beside ``fps`` and ``quality``, not over them."""
        with pytest.raises(ValueError, match="fps must be a positive whole number"):
            encode_clip(_frames(), tmp_path / "a.mp4", fps=0, macro_block_size=1)
        with pytest.raises(ValueError, match="quality must be"):
            encode_clip(_frames(), tmp_path / "b.mp4", fps=10, quality=0, macro_block_size=1)
