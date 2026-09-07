# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``capture_batch`` reads its camera selection by membership, never by truthiness.

``camera_ids`` selects the cameras one ``lerobot_camera(action="capture_batch")``
call opens. ``None`` is the documented spelling of "the default robot cameras",
and the branch used to resolve it with ``if not camera_ids:`` - so every other
falsy value took the same branch, and every truthy non-list was iterated as one.
Measured on the pre-fix tree with the camera factory recording what it was asked
to open:

* ``camera_ids=[]`` - the selection a filter that matched nothing produces -
  opened ``0`` and ``/dev/video4``, wrote two files, and reported
  ``Success: 2/2 cameras`` under ``status="success"``;
* ``camera_ids="/dev/video4"`` opened eleven one-character cameras on eleven
  threads - ``/``, ``d``, ``e``, ``v`` ... - and reported eleven verdicts about
  devices the caller never named rather than one about the parameter;
* ``camera_ids=[0, 0]`` opened one device twice concurrently and wrote two files
  for it, reported as two cameras;
* ``camera_ids={"0": 1}`` was iterated over its keys, discarding the values.

Each of those is refused now, by value, before the save directory is created,
the thread pool is sized or a camera is opened. The controls pin what must not
move: ``None`` still selects the two defaults, a well-formed list opens exactly
the cameras it names, and an action that never reads ``camera_ids`` is not
refused for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# The module object is needed for the factory seam the recorder replaces, so the
# tool is reached through this alias rather than off the tools package.
import strands_robots.tools.lerobot_camera as cam_mod


class _Camera:
    """A camera stand-in that answers a frame and nothing else."""

    width = 8
    height = 6
    fps = 30

    def connect(self, warmup: bool = True) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return self.read()


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace the camera factory and record every camera id it is asked to open."""
    ids: list[Any] = []

    def _create(camera_type: str, camera_id: Any, *selectors: Any) -> _Camera:
        ids.append(camera_id)
        return _Camera()

    monkeypatch.setattr(cam_mod, "_create_camera", _create)
    return ids


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _capture_batch(save_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with agent-shaped values.

    The values under test are deliberately outside the ``list[int | str] | None``
    annotation - a bare string, a mapping, an iterator - so the one ``**kwargs:
    Any`` funnel states that once rather than suppressing it per call.
    """
    return cam_mod.lerobot_camera(
        action="capture_batch",
        save_path=str(save_path),
        async_mode=False,
        **kwargs,
    )


# What ``None`` selected before and after: the documented default robot cameras.
DEFAULT_ROBOT_CAMERAS: list[int | str] = [0, "/dev/video4"]

# Every spelling the selection cannot honor as written, with the reason each is
# refused for. ``0`` is a bare int: not a sequence, so it is refused for its shape
# rather than iterated.
UNHONORABLE: list[tuple[str, Any]] = [
    ("an empty selection", []),
    ("an empty tuple", ()),
    ("a single path as a bare string", "/dev/video4"),
    ("a single index as a bare string", "0"),
    ("bytes", b"/dev/video0"),
    ("a mapping", {"0": 1}),
    ("a one-shot iterator", iter([0])),
    ("a bare int", 0),
    ("a repeated index", [0, 0]),
    ("a repeated path", ["/dev/video0", "/dev/video0"]),
    ("a bool entry", [True]),
    ("a float entry", [1.5]),
    ("a None entry", [None]),
    ("a blank path entry", [""]),
    ("a whitespace path entry", ["   "]),
]


class TestAnUnhonorableSelectionIsRefusedBeforeAnyCameraOpens:
    @pytest.mark.parametrize(("reason", "camera_ids"), UNHONORABLE, ids=[r for r, _ in UNHONORABLE])
    def test_refused_by_value(self, opened: list[Any], tmp_path: Path, reason: str, camera_ids: Any) -> None:
        save_path = tmp_path / "captures"

        result = _capture_batch(save_path, camera_ids=camera_ids)

        assert result["status"] == "error", reason
        text = _text(result)
        assert "camera_ids" in text, text
        assert "capture_batch" in text, text
        # The refusal has no partial effect: no camera was opened and the save
        # directory the batch would have created does not exist.
        assert opened == [], (reason, opened)
        assert not save_path.exists(), reason

    def test_the_empty_selection_is_not_widened_to_the_defaults(self, opened: list[Any], tmp_path: Path) -> None:
        result = _capture_batch(tmp_path / "captures", camera_ids=[])

        text = _text(result)
        assert "selects no camera" in text, text
        # The remedy names the spelling that does select the defaults.
        assert "Omit camera_ids" in text, text
        assert opened == []

    def test_a_bare_string_is_not_read_one_camera_per_character(self, opened: list[Any], tmp_path: Path) -> None:
        result = _capture_batch(tmp_path / "captures", camera_ids="/dev/video4")

        text = _text(result)
        assert "one camera per character" in text, text
        assert "['/dev/video4']" in text, text
        assert opened == []

    def test_a_repeat_names_the_id_repeated(self, opened: list[Any], tmp_path: Path) -> None:
        result = _capture_batch(tmp_path / "captures", camera_ids=[2, "/dev/video0", 2])

        text = _text(result)
        assert "more than once" in text, text
        assert "2" in text
        assert opened == []


class TestTheHonoredSpellingsAreUnchanged:
    def test_none_selects_the_default_robot_cameras(self, opened: list[Any], tmp_path: Path) -> None:
        result = _capture_batch(tmp_path / "captures", camera_ids=None)

        assert result["status"] == "success", _text(result)
        # The literal rather than the module constant: this is a control, so it
        # must hold on the pre-fix tree too, where the constant did not exist.
        assert sorted(map(str, opened)) == sorted(map(str, DEFAULT_ROBOT_CAMERAS))

    def test_omitting_the_selector_is_the_same_as_none(self, opened: list[Any], tmp_path: Path) -> None:
        result = _capture_batch(tmp_path / "captures")

        assert result["status"] == "success", _text(result)
        assert sorted(map(str, opened)) == sorted(map(str, DEFAULT_ROBOT_CAMERAS))

    @pytest.mark.parametrize(
        "camera_ids",
        [
            [0],
            ["/dev/video0"],
            [0, "/dev/video4"],
            (3, 5),
            [2, "/dev/video2"],
        ],
        ids=["one index", "one path", "the defaults spelled out", "a tuple", "an index and a path"],
    )
    def test_a_well_formed_selection_opens_exactly_the_cameras_it_names(
        self, opened: list[Any], tmp_path: Path, camera_ids: Any
    ) -> None:
        result = _capture_batch(tmp_path / "captures", camera_ids=camera_ids)

        assert result["status"] == "success", _text(result)
        # Captures run on a thread pool, so the open order is not the list order;
        # the SET of cameras opened, each once, is the claim.
        assert sorted(map(str, opened)) == sorted(map(str, camera_ids))
        assert f"Success: {len(camera_ids)}/{len(camera_ids)} cameras" in _text(result)

    def test_the_refusal_quotes_the_defaults_the_tool_resolves(self) -> None:
        """The message for ``[]`` names the cameras ``None`` selects, read from the one constant."""
        assert list(cam_mod._DEFAULT_BATCH_CAMERA_IDS) == DEFAULT_ROBOT_CAMERAS

        text = cam_mod._camera_ids_error([])

        assert text is not None
        assert repr(DEFAULT_ROBOT_CAMERAS) in text


class TestOnlyTheActionThatReadsTheSelectorIsRefusedForIt:
    def test_capture_ignores_camera_ids(self, opened: list[Any], tmp_path: Path) -> None:
        """``capture`` reads ``camera_id``; an unusable ``camera_ids`` beside it is not its business."""
        result = cam_mod.lerobot_camera(
            action="capture",
            camera_id=0,
            camera_ids=[],
            save_path=str(tmp_path / "captures"),
            async_mode=False,
        )

        assert result["status"] == "success", _text(result)
        assert opened == [0]

    def test_list_ignores_camera_ids(self, opened: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cam_mod, "_list_camera_details", lambda camera_type, camera_id: {"status": "success", "content": []}
        )

        # Off-annotation on purpose: the bare string capture_batch refuses.
        unusable: Any = "/dev/video4"
        result = cam_mod.lerobot_camera(action="list", camera_ids=unusable)

        assert result["status"] == "success"
        assert opened == []
