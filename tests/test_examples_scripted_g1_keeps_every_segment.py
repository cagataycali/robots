"""``examples/locomotion/scripted_g1.py`` delivers every segment it records.

The example drives a four-segment locomotion schedule as four ``run_policy``
calls and said, in its docstring and beside the call, that it would "append
each segment to one MP4". ``run_policy(video={"path": ...})`` opens the file
fresh on every call, so after the schedule the MP4 held only the fourth segment
- the two-second halt (60 frames, 51 KB, measured rendering the v0.5.2 assets
on an L40S; segments 0-2 were 251 KB and 188 KB and gone). The "reproducible
demo artifact" was a video of a robot standing still.

The example now records each segment to ``<stem>.seg<i>.mp4`` and joins them
into ``--mp4`` with :func:`strands_robots.rendering.concat_clips`. These cells
run it against a stand-in robot whose ``run_policy`` writes a real clip of a
known length to whatever path it is handed, then decode the join - no WBC
checkpoint, no MuJoCo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import encode_clip

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "locomotion" / "scripted_g1.py"
_FRAMES_PER_SEGMENT = (4, 6, 3, 2)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scripted_g1_under_test", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Robot:
    """Answers ``run_policy`` by writing a clip of a known length to the video path."""

    add_camera_status = "success"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.video_paths: list[str] = []
        self.cameras: list[dict[str, Any]] = []
        self.recorded_from: list[str] = []

    def add_camera(self, **kwargs: Any) -> dict[str, Any]:
        self.cameras.append(kwargs)
        return {"status": self.add_camera_status, "content": [{"text": f"camera {kwargs.get('name')!r}"}]}

    def run_policy(self, **kwargs: Any) -> dict[str, Any]:
        path = kwargs["video"]["path"]
        self.video_paths.append(path)
        self.recorded_from.append(kwargs["video"]["camera"])
        n = _FRAMES_PER_SEGMENT[len(self.video_paths) - 1]
        rng = np.random.default_rng(len(self.video_paths))
        encode_clip([rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8) for _ in range(n)], path, fps=30)
        return {"status": "success", "content": [{"text": f"{n} frames"}]}


def _count(path: Path) -> int:
    imageio = pytest.importorskip("imageio.v2")
    reader = imageio.get_reader(str(path))
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[_Robot]]:
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    module = _load()
    robots: list[_Robot] = []

    def _make(*args: Any, **kwargs: Any) -> _Robot:
        robots.append(_Robot(*args, **kwargs))
        return robots[-1]

    monkeypatch.setattr(module, "Robot", _make)
    return module, robots


class TestTheArtifact:
    def test_the_joined_mp4_holds_every_segment_frame(self, tmp_path: Path, example) -> None:
        module, robots = example
        out = tmp_path / "g1.mp4"

        assert module.main(["--checkpoint", "/nonexistent", "--mp4", str(out)]) == 0

        assert len(robots) == 1
        assert _count(out) == sum(_FRAMES_PER_SEGMENT)

    def test_each_segment_records_to_its_own_path(self, tmp_path: Path, example) -> None:
        module, robots = example
        out = tmp_path / "g1.mp4"

        module.main(["--checkpoint", "/nonexistent", "--mp4", str(out)])

        paths = robots[0].video_paths
        assert len(paths) == len(module.SCHEDULE)
        assert len(set(paths)) == len(paths), "a shared path keeps only the last segment"
        assert str(out) not in paths
        assert paths == [str(module.segment_path(out, i)) for i in range(len(module.SCHEDULE))]

    def test_segments_are_removed_after_the_join_unless_asked_to_keep(self, tmp_path: Path, example) -> None:
        module, robots = example
        out = tmp_path / "g1.mp4"

        module.main(["--checkpoint", "/nonexistent", "--mp4", str(out)])
        assert not any(Path(p).exists() for p in robots[0].video_paths)

        kept = tmp_path / "kept.mp4"
        module.main(["--checkpoint", "/nonexistent", "--mp4", str(kept), "--keep-segments"])
        assert all(Path(p).exists() for p in robots[1].video_paths)
        assert _count(kept) == sum(_FRAMES_PER_SEGMENT)

    def test_a_failed_segment_still_joins_what_was_recorded(self, tmp_path: Path, example, monkeypatch) -> None:
        module, robots = example
        out = tmp_path / "g1.mp4"
        original = _Robot.run_policy

        def _fail_third(self: _Robot, **kwargs: Any) -> dict[str, Any]:
            result = original(self, **kwargs)
            if len(self.video_paths) == 3:
                return {"status": "error", "content": [{"text": "policy refused"}]}
            return result

        monkeypatch.setattr(_Robot, "run_policy", _fail_third)

        assert module.main(["--checkpoint", "/nonexistent", "--mp4", str(out)]) == 1
        assert _count(out) == sum(_FRAMES_PER_SEGMENT[:2])

    def test_the_docstring_no_longer_promises_an_append(self) -> None:
        text = _EXAMPLE.read_text(encoding="utf-8")
        assert "append each segment to one MP4" not in text
        assert "concat_clips" in text


class TestTheCameraRidesWithTheRobot:
    """The ``default`` camera frames the origin; a walking G1 leaves it within ~2 s."""

    def test_a_pelvis_mounted_camera_is_added_before_the_first_rollout(self, example, tmp_path: Path) -> None:
        module, robots = example
        out = tmp_path / "walk.mp4"

        assert module.main(["--checkpoint", "/ckpt", "--mp4", str(out)]) == 0

        (robot,) = robots
        (camera,) = robot.cameras
        assert camera["parent_body"] == "unitree_g1/pelvis"
        assert camera["name"] == module.FOLLOW_CAMERA["name"] != "default"
        assert camera["width"] == 640 and camera["height"] == 480

    def test_every_segment_is_recorded_from_it(self, example, tmp_path: Path) -> None:
        module, robots = example

        module.main(["--checkpoint", "/ckpt", "--mp4", str(tmp_path / "walk.mp4")])

        (robot,) = robots
        assert robot.recorded_from == [module.FOLLOW_CAMERA["name"]] * len(module.SCHEDULE)

    def test_a_refused_camera_stops_the_run_before_any_rollout(
        self, example, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module, robots = example
        monkeypatch.setattr(_Robot, "add_camera_status", "error")

        assert module.main(["--checkpoint", "/ckpt", "--mp4", str(tmp_path / "walk.mp4")]) == 1

        (robot,) = robots
        assert robot.video_paths == []
        assert "add_camera" in capsys.readouterr().out

    def test_the_docstring_promise_matches_the_code(self) -> None:
        text = _EXAMPLE.read_text(encoding="utf-8")
        assert "add_camera(parent_body=...)" in text
        assert '"camera": "default"' not in text
