# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``encode_clip`` only accepts clip options it can actually encode.

``fps`` is the clip's timebase, so a value the encoders cannot honor has no
usable interpretation: the GIF writer used to clamp it
(``duration=1000.0 / max(1, int(fps))``, turning ``0`` or ``-5`` into a 1 fps
clip), the MP4 writer let ffmpeg substitute its own default rate for ``0`` and
refused a negative rate without writing a file at all - and in every one of
those cases ``encode_clip`` still returned the output path. These tests pin the
contract that the returned path names a clip that exists and plays at the
requested rate, on both containers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.rendering import encode_clip
from strands_robots.simulation.recording import dataset_recording_option_error
from strands_robots.utils import positive_whole_number_error

# Values no encoder can turn into a frame rate: zero and negative rates, a
# fractional rate, non-finite floats, a numeric string, ``True`` (an ``int``
# subclass that would silently act as 1 fps) and a missing value.
UNUSABLE_FPS = [0, -5, 2.7, math.nan, math.inf, "20", True, None]


def _frames(n: int = 6, w: int = 48, h: int = 32) -> list[np.ndarray]:
    """``n`` distinct RGB frames of one shape (a gradient, so decode is visible)."""
    return [np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8) for i in range(n)]


@pytest.mark.parametrize("suffix", [".gif", ".mp4"])
@pytest.mark.parametrize("fps", UNUSABLE_FPS)
def test_unusable_fps_is_refused_before_any_file_is_written(tmp_path, suffix, fps) -> None:
    """Every container refuses the same unusable rates, naming fps and the value."""
    out = tmp_path / f"clip{suffix}"
    with pytest.raises(ValueError, match="fps must be a positive whole number"):
        encode_clip(_frames(), out, fps=fps)
    assert not out.exists(), f"encode_clip({fps!r}) left a file behind at {out}"


@pytest.mark.parametrize("fps", [1, 25, 30.0, np.int64(24)])
def test_usable_fps_encodes_a_gif_at_the_requested_rate(tmp_path, fps) -> None:
    """A whole-number rate (including a float or NumPy integer) sets the GIF timing."""
    pytest.importorskip("imageio.v2")
    pil_image = pytest.importorskip("PIL.Image")
    out = encode_clip(_frames(), tmp_path / "clip.gif", fps=fps)
    with pil_image.open(out) as im:
        assert im.n_frames == 6
        # Pillow stores per-frame duration in milliseconds, and the GIF
        # format quantizes it to centiseconds - hence the 10 ms tolerance.
        assert im.info["duration"] == pytest.approx(1000.0 / int(fps), abs=10.0)


def test_mp4_is_encoded_at_the_requested_rate(tmp_path) -> None:
    """The MP4's own metadata reports the requested rate, not an ffmpeg default."""
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    imageio_v3 = pytest.importorskip("imageio.v3")
    out = encode_clip(_frames(n=12), tmp_path / "clip.mp4", fps=3)
    assert out.stat().st_size > 0
    assert float(imageio_v3.immeta(out, plugin="pyav")["fps"]) == pytest.approx(3.0)


def test_encoder_refusal_raises_instead_of_returning_a_path_to_nothing(tmp_path) -> None:
    """A frame size the codec refuses is reported, not returned as a written clip.

    ``macro_block_size=7`` rounds a 48x32 frame up to 49x35, which libx264
    refuses; the writer then closes having written nothing. The failure has to
    reach the caller, because the return value is the caller's only handle on
    the artifact.
    """
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "refused.mp4"
    with pytest.raises(RuntimeError, match="wrote no clip"):
        encode_clip(_frames(), out, fps=20, macro_block_size=7)
    assert not out.exists() or out.stat().st_size == 0


def test_fps_is_refused_before_the_optional_encoder_is_probed(tmp_path, monkeypatch) -> None:
    """The parameter guard runs first, so the error is the same on any install.

    Without imageio present the caller would otherwise be told to install an
    extra when the real problem is the value they passed.
    """

    def _no_imageio(*args, **kwargs):
        raise ImportError("imageio is not installed")

    monkeypatch.setattr("strands_robots.rendering.video.require_optional", _no_imageio)
    with pytest.raises(ValueError, match="fps must be a positive whole number"):
        encode_clip(_frames(), tmp_path / "clip.mp4", fps=0)


@pytest.mark.parametrize("fps", UNUSABLE_FPS)
def test_one_frame_rate_domain_across_the_media_surfaces(fps) -> None:
    """encode_clip, the recorders and the run_policy video dict share one domain.

    A rate rejected when recording a dataset cannot be accepted when encoding
    the clip of that same rollout, so both surfaces resolve the value through
    the same helper and only the message prefix differs.
    """
    text = positive_whole_number_error(fps, "fps", "encode_clip")
    assert text == f"encode_clip: fps must be a positive whole number, got {fps!r}."
    assert dataset_recording_option_error("start_recording", fps) is not None
