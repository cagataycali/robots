"""docs/learn/data/record.md states ``DatasetRecorder.add_frame``'s real contract.

The page used to say ``create`` refuses a frame whose shape disagrees with the
schema with ``RecordingFrameError`` and names an action key it cannot record.
Neither held (#4149): a missing declared column is a ``ValueError``
(``RecordingFrameError`` subclasses ``RuntimeError``, so an ``except`` written
from the page missed it), ``RecordingFrameError`` is the failed-write error,
and an undeclared action key is dropped without a word. This grader reads the
page sentence, the ``add_frame`` docstring and the recorder itself, so the
three cannot drift apart again.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from strands_robots.dataset_recorder import DatasetRecorder
from strands_robots.recording_errors import RecordingFrameError

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "learn" / "data" / "record.md"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _sentence() -> str:
    text = PAGE.read_text(encoding="utf-8")
    hits = [line for line in text.splitlines() if "`add_frame`" in line and "RecordingFrameError" in line]
    assert len(hits) == 1, f"record.md must state add_frame's contract in exactly one sentence, found {len(hits)}"
    return hits[0]


class TestThePageAndTheDocstringAgree:
    def test_the_page_pairs_each_error_with_its_cause(self) -> None:
        s = _sentence()
        assert re.search(r"missing a declared column \(`ValueError`\)", s), s
        assert re.search(r"failed write \(`RecordingFrameError`\)", s), s
        assert "drops undeclared action keys" in s, s
        assert "`create`" in s and "vcodec" in s, s

    def test_the_docstring_names_the_same_roles(self) -> None:
        doc = inspect.getdoc(DatasetRecorder.add_frame) or ""
        raises = doc.split("Raises:", 1)[1]
        value_error, frame_error = raises.split("RecordingFrameError:", 1)
        assert "ValueError:" in value_error and "absent" in value_error, value_error
        assert "write failed" in frame_error, frame_error
        assert issubclass(RecordingFrameError, RuntimeError) and not issubclass(RecordingFrameError, ValueError)


class TestTheRecorderDoesWhatThePageSays:
    @pytest.fixture
    def recorder(self, tmp_path: Path) -> Iterator[DatasetRecorder]:
        rec = DatasetRecorder.create(
            repo_id="grader/record_page",
            fps=30,
            robot_type="so101",
            joint_names=NAMES,
            camera_keys=["front"],
            camera_dims={"front": (48, 64)},
            task="grade the page",
            root=str(tmp_path / "ds"),
            vcodec="libx264",
        )
        yield rec
        rec.finalize()

    def test_a_missing_declared_column_is_a_value_error_not_a_frame_error(self, recorder: DatasetRecorder) -> None:
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc:
            recorder.add_frame({**{n: 0.0 for n in NAMES[:3]}, "front": img}, {n: 0.0 for n in NAMES}, task="x")
        assert not isinstance(exc.value, RecordingFrameError)
        assert "wrist_flex" in str(exc.value)

    def test_an_undeclared_action_key_is_dropped_and_the_declared_ones_recorded(
        self, recorder: DatasetRecorder
    ) -> None:
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        recorder.add_frame(
            {**{n: 0.1 for n in NAMES}, "front": img}, {**{n: 0.2 for n in NAMES}, "bogus_key": 1.0}, task="x"
        )
        names = recorder.dataset.features["action"]["names"]
        assert "bogus_key" not in names and list(names) == NAMES

    def test_create_normalises_the_codec_name(self, recorder: DatasetRecorder) -> None:
        # ``create(vcodec="libx264")`` was accepted above (the fixture); the mapping that
        # normalises it is the one every LeRobot codec surface is fed from.
        from strands_robots.dataset_recorder import _codec_create_kwargs

        assert _codec_create_kwargs({"vcodec"}, "libx264") == {"vcodec": "h264"}
        assert _codec_create_kwargs({"vcodec"}, "h264") == {"vcodec": "h264"}
