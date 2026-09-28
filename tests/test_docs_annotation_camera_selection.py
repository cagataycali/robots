"""Keep the labelling guide honest about which camera the labels come from.

The old guide documented ``lerobot-annotate``, whose ``plan`` and
``interjections`` modules read exactly one video stream, the dataset's first
video key, so a multi-camera dataset got every label derived from one view, and
a gripper-mounted first camera produced image evidence for "the object moved"
exactly when the object did not. That pipeline is no longer documented; the
judge on ``docs/learn/data/label-and-judge.md`` is the package's own
``strands_robots.tools.episode_judge``, and its ``sample_frames`` tool answers
the same question the other way: it has no camera selector and decodes every
camera at every sampled position, cameras in sorted key order, because a judge
that sees one view cannot tell an occlusion from a failure.

So the claim to grade moved. The guide must still say which view the judge's
labels come from, and the answer it gives must be the one the tool implements:
the page's ``sample_frames`` row names the tool's real signature (no camera
parameter, read from the function) and says the images cover every camera.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC = _REPO_ROOT / "docs" / "learn" / "data" / "label-and-judge.md"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _sample_frames_row() -> str:
    """The judge-tool table row for ``sample_frames``."""
    for line in _doc_text().splitlines():
        if line.startswith("| `sample_frames("):
            return line
    raise AssertionError("docs/learn/data/label-and-judge.md has no table row for sample_frames")


class TestTheGuideNamesTheStreamTheJudgeReads:
    """The guide must say which view the labels come from, and the tool must agree."""

    def test_the_tool_has_no_camera_selector(self) -> None:
        """The premise: every camera is the only honest answer because none can be chosen."""
        from strands_robots.tools.episode_judge import sample_frames

        params = inspect.signature(sample_frames).parameters
        assert not [p for p in params if "camera" in p], (
            f"sample_frames now takes {sorted(params)}; a camera selector means the guide must "
            "document which view is the default, not that every view is read"
        )
        doc = inspect.getdoc(sample_frames) or ""
        assert "Every camera" in doc and "sorted key order" in doc, (
            "the tool's own docstring no longer states the every-camera contract"
        )

    def test_the_row_states_the_real_signature(self) -> None:
        from strands_robots.tools.episode_judge import sample_frames

        row = _sample_frames_row()
        shown = re.search(r"`sample_frames\(([^)]*)\)`", row)
        assert shown is not None, row
        listed = [p.split("=")[0].strip() for p in shown.group(1).split(",") if p.strip()]
        real = list(inspect.signature(sample_frames).parameters)
        assert listed == real, f"the row shows sample_frames({', '.join(listed)}); the tool takes ({', '.join(real)})"

    def test_the_row_says_the_images_cover_every_camera(self) -> None:
        row = _sample_frames_row().lower()
        assert re.search(r"\b(every|all|each) camera", row) or "per camera" in row, (
            "docs/learn/data/label-and-judge.md's sample_frames row says images are decoded "
            "'when asked' but not from which view; the tool reads every camera at every "
            "sampled position, and a reader with a gripper-mounted camera needs to know "
            "the judge is not labelling from that view alone."
        )
