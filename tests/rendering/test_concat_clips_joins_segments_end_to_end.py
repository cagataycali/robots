"""``concat_clips`` joins recorded segments into one clip.

Every writer in the package opens its output fresh - ``encode_clip`` and the
rollout MP4 behind ``run_policy(video=...)`` alike - so a caller who records a
sequence as several rollouts and hands each the same path keeps only the last
one. ``examples/locomotion/scripted_g1.py`` did exactly that: its four-segment
schedule left a two-second clip of the robot halting and called it the demo
artifact. ``concat_clips`` is the join such a caller needs; these cells pin
what it accepts and what it refuses, by decoding what it wrote.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strands_robots.rendering import concat_clips, encode_clip


def _frames(n: int, w: int = 32, h: int = 24, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


def _count(path: Path) -> tuple[int, float]:
    imageio = pytest.importorskip("imageio.v2")
    reader = imageio.get_reader(str(path))
    try:
        return sum(1 for _ in reader), float(reader.get_meta_data()["fps"])
    finally:
        reader.close()


@pytest.fixture
def mp4s(tmp_path: Path) -> list[Path]:
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    paths = [tmp_path / f"seg{i}.mp4" for i in range(3)]
    for i, (path, n) in enumerate(zip(paths, (3, 5, 2), strict=True)):
        encode_clip(_frames(n, seed=i), path, fps=10)
    return paths


class TestTheJoin:
    def test_every_segment_frame_lands_in_order_at_the_segments_rate(self, tmp_path: Path, mp4s: list[Path]) -> None:
        out = concat_clips(mp4s, tmp_path / "joined.mp4")

        assert out == tmp_path / "joined.mp4"
        frames, fps = _count(out)
        assert frames == 3 + 5 + 2
        assert fps == 10.0

    def test_an_explicit_rate_wins_over_the_header(self, tmp_path: Path, mp4s: list[Path]) -> None:
        out = concat_clips(mp4s, tmp_path / "joined.mp4", fps=25)

        assert _count(out)[1] == 25.0

    def test_a_gif_join_takes_the_same_door(self, tmp_path: Path, mp4s: list[Path]) -> None:
        imageio = pytest.importorskip("imageio.v2")
        out = concat_clips(mp4s, tmp_path / "joined.gif")

        assert len(imageio.mimread(out)) == 10

    def test_a_single_clip_round_trips(self, tmp_path: Path, mp4s: list[Path]) -> None:
        out = concat_clips(mp4s[:1], tmp_path / "one.mp4")

        assert _count(out)[0] == 3


class TestWhatIsRefused:
    def test_no_clips(self, tmp_path: Path) -> None:
        pytest.importorskip("imageio.v2")
        with pytest.raises(ValueError, match="no clips to join"):
            concat_clips([], tmp_path / "joined.mp4")

    def test_a_missing_clip_is_named_before_anything_is_read(self, tmp_path: Path, mp4s: list[Path]) -> None:
        gone = tmp_path / "seg9.mp4"
        with pytest.raises(ValueError, match="clip not found") as excinfo:
            concat_clips([*mp4s, gone], tmp_path / "joined.mp4")
        assert "seg9.mp4" in str(excinfo.value)
        assert not (tmp_path / "joined.mp4").exists()

    def test_a_segment_of_another_frame_size_is_named(self, tmp_path: Path, mp4s: list[Path]) -> None:
        odd = tmp_path / "odd.mp4"
        encode_clip(_frames(2, w=48, h=24), odd, fps=10)
        with pytest.raises(ValueError, match="every segment must share one frame size") as excinfo:
            concat_clips([*mp4s, odd], tmp_path / "joined.mp4")
        assert "odd.mp4" in str(excinfo.value)

    def test_a_missing_encoder_is_the_shared_refusal(
        self, tmp_path: Path, mp4s: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strands_robots.rendering import video

        def _absent(path: str | Path, purpose: str = "") -> None:
            raise ImportError(f"'imageio_ffmpeg' is required for {purpose}", name="imageio_ffmpeg")

        monkeypatch.setattr(video, "require_clip_encoder", _absent)
        with pytest.raises(ImportError, match="concat_clips"):
            video.concat_clips(mp4s, tmp_path / "joined.mp4")
