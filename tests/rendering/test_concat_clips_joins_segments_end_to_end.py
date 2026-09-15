"""``concat_clips`` joins recorded segments into one clip.

Every writer in the package opens its output fresh - ``encode_clip`` and the
rollout MP4 behind ``run_policy(video=...)`` alike - so a caller who records a
sequence as several rollouts and hands each the same path keeps only the last
one. ``examples/locomotion/scripted_g1.py`` did exactly that: its four-segment
schedule left a two-second clip of the robot halting and called it the demo
artifact. ``concat_clips`` is the join such a caller needs; these cells pin
what it accepts and what it refuses, by decoding what it wrote.

The clips it joins are the clips this package writes, so the round trip
``encode_clip`` -> ``concat_clips`` has to close for both containers
``encode_clip`` supports. A GIF stores a per-frame delay rather than a rate, and
that delay is what ``encode_clip`` writes, so reading a rate back means going
through the inverse of the same conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strands_robots.rendering import concat_clips, encode_clip
from strands_robots.rendering.video import _declared_rate


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


def _gif_delay_ms(path: Path) -> float:
    imageio = pytest.importorskip("imageio.v2")
    reader = imageio.get_reader(str(path))
    try:
        return float(reader.get_meta_data()["duration"])
    finally:
        reader.close()


class TestTheRateTheSegmentsDeclare:
    """The rate of the join is read from the segments, through their own header."""

    @pytest.mark.parametrize("fps", [10, 30])
    def test_gif_segments_this_package_wrote_join_at_the_delay_they_declare(self, tmp_path: Path, fps: int) -> None:
        """A GIF declares a per-frame delay, not a rate, and the join preserves it.

        ``encode_clip`` writes a GIF through Pillow, which takes the delay
        (``duration=1000/fps``) - so a header read for ``fps`` alone finds
        nothing and the join of clips this package itself wrote is refused. The
        delay survives the join, which is the timing a GIF actually plays at;
        the rate it implies need not be the ``fps`` the segments were written at,
        because the format quantises the delay to 10 ms.
        """
        pytest.importorskip("imageio.v2")
        segments = [tmp_path / f"seg{i}.gif" for i in range(2)]
        for i, path in enumerate(segments):
            encode_clip(_frames(3, seed=i), path, fps=fps)
        delay = _gif_delay_ms(segments[0])

        out = concat_clips(segments, tmp_path / "joined.gif")

        imageio = pytest.importorskip("imageio.v2")
        assert len(imageio.mimread(out)) == 6
        assert _gif_delay_ms(out) == delay

    def test_the_rate_is_the_first_segments_not_the_last(self, tmp_path: Path) -> None:
        """The join plays at the rate of the clip it starts with."""
        first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
        encode_clip(_frames(3), first, fps=10)
        encode_clip(_frames(3, seed=1), second, fps=25)

        out = concat_clips([first, second], tmp_path / "joined.mp4")

        assert _count(out) == (6, 10.0)

    def test_a_clip_declaring_no_delay_is_refused_and_named(self, tmp_path: Path) -> None:
        """A GIF whose header carries no delay leaves no rate to read."""
        imageio = pytest.importorskip("imageio.v2")
        rateless = tmp_path / "rateless.gif"
        imageio.mimsave(str(rateless), _frames(3))

        with pytest.raises(ValueError, match="declares no frame rate") as excinfo:
            concat_clips([rateless], tmp_path / "joined.gif")
        assert "rateless.gif" in str(excinfo.value)
        assert concat_clips([rateless], tmp_path / "explicit.gif", fps=10).is_file()


class TestWhatAHeaderCounts:
    """``duration`` means different things per container, so the suffix decides.

    An MP4 header declares ``fps`` and spends ``duration`` on the length of the
    whole clip in seconds; a GIF declares no rate and spends ``duration`` on the
    delay of one frame in milliseconds. Reading an MP4's ``duration`` as a delay
    would turn a 0.4 s clip into 2500 fps, so only a GIF's is read that way.
    """

    @pytest.mark.parametrize(
        ("name", "meta", "expected"),
        [
            ("clip.mp4", {"fps": 30.0, "duration": 0.4}, 30.0),
            ("clip.gif", {"duration": 100}, 10.0),
            ("clip.GIF", {"duration": 50}, 20.0),
            ("clip.mp4", {"duration": 0.4}, None),
            ("clip.gif", {"duration": 0}, None),
            ("clip.gif", {"duration": "100"}, None),
            ("clip.gif", {}, None),
        ],
    )
    def test_the_rate_a_header_declares(self, name: str, meta: dict, expected: float | None) -> None:
        assert _declared_rate(Path(name), meta) == expected


class TestEveryClipIsProbedForADecoder:
    def test_an_input_without_a_decoder_is_refused_too(
        self, tmp_path: Path, mp4s: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The plugin that writes a container is the one that reads it.

        Probing only the output would let a join open a segment it cannot decode
        and fail inside the reader, so the segments are probed as well.
        """
        from strands_robots.rendering import video

        real = video.require_clip_encoder

        def _absent_for_the_second_segment(path: str | Path, purpose: str = "") -> None:
            if Path(path) == mp4s[1]:
                raise ImportError(f"'imageio_ffmpeg' is required for {purpose}", name="imageio_ffmpeg")
            real(path, purpose=purpose)

        monkeypatch.setattr(video, "require_clip_encoder", _absent_for_the_second_segment)
        with pytest.raises(ImportError, match="concat_clips"):
            video.concat_clips(mp4s, tmp_path / "joined.mp4")
        assert not (tmp_path / "joined.mp4").exists()
