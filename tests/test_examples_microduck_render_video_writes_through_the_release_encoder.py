"""``examples/microduck/render_video.py`` writes its clips through ``encode_clip``.

The release's showcase example encoded its MP4 and GIF with a private
``imageio.mimwrite`` call - its own codec, quality, pixel format and
macro-block choices - while ``run_policy(video=...)`` and every recorder in the
package write through :func:`strands_robots.rendering.encode_clip`. Two encoders
for one release means the showcase clip and the library clip are different
bytes for the same frames, and a fix to the shared encoder (its domain checks,
its missing-plugin refusal) never reaches the example.

These cells load the example as a module and run its encoder helpers against a
handful of synthetic frames: what they write is decoded back, and the call is
observed to go through the package encoder. No MuJoCo, no ONNX.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "microduck" / "render_video.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_video_under_test", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(n: int = 6, w: int = 64, h: int = 36) -> list:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


def _count(path: Path) -> tuple[int, float]:
    imageio = pytest.importorskip("imageio.v2")
    reader = imageio.get_reader(str(path))
    try:
        return sum(1 for _ in reader), float(reader.get_meta_data().get("fps", 0))
    finally:
        reader.close()


class TestTheEncoderIsTheReleases:
    def test_the_mp4_is_written_by_encode_clip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.rendering as rendering

        seen: list[dict] = []
        real = rendering.encode_clip

        def _spy(frames, path, **kwargs):
            seen.append({"path": Path(path), **kwargs})
            return real(frames, path, **kwargs)

        pytest.importorskip("imageio_ffmpeg")
        monkeypatch.setattr(rendering, "encode_clip", _spy)
        out = tmp_path / "duck.mp4"

        _load()._encode(_frames(), str(out), 25)

        assert [s["path"] for s in seen] == [out]
        assert seen[0]["fps"] == 25
        frames, fps = _count(out)
        assert (frames, fps) == (6, 25.0)

    def test_the_gif_is_written_by_encode_clip_at_the_asked_width(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import strands_robots.rendering as rendering

        imageio = pytest.importorskip("imageio.v2")
        pytest.importorskip("imageio_ffmpeg")
        seen: list[Path] = []
        real = rendering.encode_clip

        def _spy(frames, path, **kwargs):
            seen.append(Path(path))
            return real(frames, path, **kwargs)

        monkeypatch.setattr(rendering, "encode_clip", _spy)
        out, gif = tmp_path / "duck.mp4", tmp_path / "duck.gif"

        _load()._encode(_frames(w=64, h=36), str(out), 25, gif_path=str(gif), gif_fps=10, gif_width=32)

        assert seen == [out, gif]
        decoded = imageio.mimread(gif)
        assert len(decoded) == 6
        assert decoded[0].shape[:2] == (18, 32)

    def test_the_gif_the_example_writes_loops(self, tmp_path: Path) -> None:
        """The showcase GIF repeats, which is what the helper's name promises.

        The private ``imageio.mimwrite`` call this helper used passed ``loop=0``
        by hand. Routing through the package encoder keeps that only because
        the encoder writes the looping block itself, so the release's showcase
        GIF is an animation rather than a still of its last frame.
        """
        pytest.importorskip("imageio.v2")
        Image = pytest.importorskip("PIL.Image")
        gif = tmp_path / "duck.gif"

        _load()._encode_gif(_frames(w=64, h=36), str(gif), 10, 32)

        with Image.open(gif) as image:
            loop = image.info.get("loop")
        assert loop == 0

    def test_a_frame_rate_the_encoder_refuses_is_named_up_front(self, tmp_path: Path) -> None:
        pytest.importorskip("imageio.v2")
        with pytest.raises(ValueError, match="fps"):
            _load()._encode(_frames(), str(tmp_path / "duck.mp4"), 0)

    def test_the_example_holds_no_private_encoder(self) -> None:
        text = _EXAMPLE.read_text(encoding="utf-8")
        assert not re.search(r"^\s*import imageio", text, re.M)
        assert "mimwrite" not in text
        assert "encode_clip" in text

    def test_the_rate_flags_take_whole_numbers(self) -> None:
        module = _load()
        import argparse

        captured: dict[str, argparse.ArgumentParser] = {}
        original = argparse.ArgumentParser.parse_args

        def _capture(self, args=None, namespace=None):
            captured["parser"] = self
            raise SystemExit(0)

        argparse.ArgumentParser.parse_args = _capture  # type: ignore[method-assign]
        try:
            with pytest.raises(SystemExit):
                module.main()
        finally:
            argparse.ArgumentParser.parse_args = original  # type: ignore[method-assign]
        by_flag = {a.option_strings[0]: a for a in captured["parser"]._actions if a.option_strings}
        assert by_flag["--fps"].type is int and by_flag["--fps"].default == 50
        assert by_flag["--gif-fps"].type is int and by_flag["--gif-fps"].default == 13
