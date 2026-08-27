"""Every camera in a transformed dataset keeps the dtype its source declared.

A LeRobot dataset declares each camera's storage independently: one
``observation.images.<cam>`` feature can be ``dtype="video"`` (frames encoded
into an MP4 under ``videos/``) while another is ``dtype="image"`` (frames stored
in the data parquet). ``LeRobotDataset.create`` accepts such a mixed
declaration and reads both streams back, so it is a legal source - the premise
class below records one and asserts it.

The output side has one knob for the whole dataset: ``DatasetRecorder.create``
takes a single ``use_videos`` flag that decides every camera's dtype.
``_SourceDataset`` derived that flag by assigning it once per camera inside the
loop that reads the camera shapes, so it ended up holding whichever camera the
feature dict happened to end on. A two-camera source declaring one video stream
and one image stream therefore transformed with ``status="success"`` while the
output silently re-encoded a camera, and the direction depended only on the
order the features were declared in:

=========================================  ==============================  =========
source cameras (declaration order)         pre-fix output cameras          MP4 files
=========================================  ==============================  =========
``top=video``, ``wrist=image``             both ``image``                  1 -> 0
``wrist=image``, ``top=video``             both ``video``                  1 -> 2
=========================================  ==============================  =========

Both directions lose the source's schema:
:meth:`~strands_robots.transforms.base._SourceDataset.create_output_recorder`
documents itself as "the SOURCE schema (parity by construction)", schema parity
is the first acceptance criterion of ``tests/transforms/test_round_trip.py``,
and ``docs/data/transforms.md`` contract item 1 promises a generated episode is
"the *same trajectory* rendered differently". A video stream flattened into
still frames is not that.

The same ``open()`` already refuses a source declaring a feature the
pass-through cannot preserve, precisely so nothing is silently altered - so the
fix follows that convention rather than inventing a second one: the flag is
derived from the whole camera set, and a source whose cameras disagree is
refused with a message naming each camera and its dtype. Homogeneous sources
(every shipped recording, whichever dtype) are unaffected, which the over-reach
control class holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
pytest.importorskip("lerobot.datasets.lerobot_dataset")

from strands_robots.transforms import TransformSpec, create_transform  # noqa: E402
from strands_robots.transforms.base import _SourceDataset  # noqa: E402

#: One camera declaration: ``(camera key suffix, dtype, (channels, height, width))``.
_TOP = ("top", "video", (3, 32, 48))
_WRIST = ("wrist", "image", (3, 24, 40))

#: The two orders a mixed source can declare its cameras in. Pre-fix these
#: produced opposite output schemas, which is the signature of a flag written
#: once per camera rather than derived from the set.
_MIXED_ORDERS = [
    pytest.param([_TOP, _WRIST], id="video-first"),
    pytest.param([_WRIST, _TOP], id="image-first"),
]


def _record_source(root: Path, cameras: list[tuple[str, str, tuple[int, int, int]]], frames: int = 3) -> Path:
    """Record a real source dataset declaring ``cameras`` in the order given.

    Built through ``LeRobotDataset.create`` rather than ``DatasetRecorder``
    because the recorder's own ``use_videos`` flag cannot express a mixed
    declaration - the asymmetry this file is about.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
    }
    for cam, dtype, shape in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": dtype,
            "shape": shape,
            "names": ["channels", "height", "width"],
        }
    dataset = LeRobotDataset.create(
        repo_id="local/source", fps=10, root=root, robot_type="probe", features=features, use_videos=True
    )
    rng = np.random.default_rng(0)
    for i in range(frames):
        frame: dict[str, Any] = {
            "observation.state": np.array([0.125 + i, -0.25 - i], dtype=np.float32),
            "action": np.array([0.5 + i, 1.5 + i], dtype=np.float32),
            "task": "probe task",
        }
        for cam, _dtype, shape in cameras:
            frame[f"observation.images.{cam}"] = rng.integers(0, 255, (shape[1], shape[2], 3), dtype=np.uint8)
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()
    return root


def _camera_dtypes(root: Path) -> dict[str, str]:
    """The dtype each ``observation.images.*`` feature declares on disk."""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return {k: v.get("dtype") for k, v in info["features"].items() if k.startswith("observation.images.")}


def _mp4_names(root: Path) -> list[str]:
    """Camera feature names that have an encoded MP4 under ``videos/``.

    Derived from the component that follows ``videos`` rather than a fixed
    depth, so a chunking-layout change surfaces as an empty set rather than as
    a set of directory names that happens to compare equal on both sides.
    """
    names: set[str] = set()
    for mp4 in root.rglob("*.mp4"):
        parts = mp4.relative_to(root).parts
        if "videos" in parts:
            index = parts.index("videos")
            if index + 1 < len(parts):
                names.add(parts[index + 1])
    return sorted(names)


def _transform(source_root: Path, output_root: Path) -> Any:
    """Run the reference backend over one source, one variant per episode."""
    return create_transform("mock").transform(
        TransformSpec(source_root=str(source_root), output_root=str(output_root), variants_per_episode=1, seed=7)
    )


def _reader(cameras: list[tuple[str, str, tuple[int, int, int]]]) -> _SourceDataset:
    """A ``_SourceDataset`` over a stand-in whose ``meta`` declares ``cameras``.

    The flag derivation reads nothing but ``meta.features``, so a stand-in
    exercises the whole camera grid without recording a dataset per row.
    """
    features: dict[str, Any] = {}
    for cam, dtype, shape in cameras:
        features[f"observation.images.{cam}"] = {"dtype": dtype, "shape": shape}
    meta = SimpleNamespace(features=features, fps=10, robot_type="probe", total_episodes=1)
    spec = TransformSpec(source_root="unused", output_root="unused")
    return _SourceDataset(SimpleNamespace(meta=meta), spec)


class TestAMixedDtypeSourceIsALegalLeRobotDataset:
    """Premise: the refused source is a dataset LeRobot itself writes and reads.

    Without this the refusal could be dismissed as covering an input nothing
    can produce.
    """

    @pytest.fixture(scope="class")
    def mixed_root(self, tmp_path_factory) -> Path:
        return _record_source(tmp_path_factory.mktemp("legal") / "source", [_TOP, _WRIST])

    def test_each_camera_keeps_its_own_declared_dtype(self, mixed_root: Path) -> None:
        assert _camera_dtypes(mixed_root) == {
            "observation.images.top": "video",
            "observation.images.wrist": "image",
        }

    def test_only_the_video_camera_is_encoded_to_an_mp4(self, mixed_root: Path) -> None:
        assert _mp4_names(mixed_root) == ["observation.images.top"]

    def test_both_streams_read_back_as_frames(self, mixed_root: Path) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        frame = LeRobotDataset(repo_id="local/source", root=mixed_root)[0]
        assert tuple(frame["observation.images.top"].shape) == (3, 32, 48)
        assert tuple(frame["observation.images.wrist"].shape) == (3, 24, 40)


class TestASourceWhoseCamerasDisagreeIsRefused:
    """Regression: neither declaration order may be transformed silently."""

    @pytest.mark.parametrize("cameras", _MIXED_ORDERS)
    def test_the_transform_reports_an_error_envelope(self, cameras, tmp_path: Path) -> None:
        source = _record_source(tmp_path / "source", cameras)
        result = _transform(source, tmp_path / "out")
        assert result.status == "error", result.message

    @pytest.mark.parametrize("cameras", _MIXED_ORDERS)
    def test_the_refusal_names_every_camera_stream(self, cameras, tmp_path: Path) -> None:
        source = _record_source(tmp_path / "source", cameras)
        message = _transform(source, tmp_path / "out").message or ""
        for cam, _dtype, _shape in cameras:
            assert f"observation.images.{cam}" in message, message

    @pytest.mark.parametrize("cameras", _MIXED_ORDERS)
    def test_the_refusal_pairs_each_camera_with_the_dtype_it_declared(self, cameras, tmp_path: Path) -> None:
        """Which camera holds which dtype is what an operator has to convert."""
        source = _record_source(tmp_path / "source", cameras)
        message = _transform(source, tmp_path / "out").message or ""
        for cam, dtype, _shape in cameras:
            assert f"observation.images.{cam}={dtype!r}" in message, message

    @pytest.mark.parametrize("cameras", _MIXED_ORDERS)
    def test_nothing_is_written_to_the_output_root(self, cameras, tmp_path: Path) -> None:
        source = _record_source(tmp_path / "source", cameras)
        output = tmp_path / "out"
        assert _transform(source, output).status == "error"
        assert not output.exists() or not any(output.rglob("*"))

    def test_a_refused_source_is_read_before_any_episode_is(self, tmp_path: Path) -> None:
        """The refusal is a source-side one, so no episode is reported read."""
        source = _record_source(tmp_path / "source", [_TOP, _WRIST])
        result = _transform(source, tmp_path / "out")
        assert (result.episodes_read, result.episodes_written) == (0, 0)


class TestAHomogeneousSourceStillTransforms:
    """Over-reach control: every dtype the fix leaves alone.

    Each expectation here is one the pre-fix code also met - a source whose
    cameras agree already derived the right flag, whichever camera the loop
    ended on.
    """

    @pytest.mark.parametrize("dtype", ["video", "image"])
    def test_two_cameras_transform_and_keep_the_source_dtype(self, dtype: str, tmp_path: Path) -> None:
        cameras = [("top", dtype, (3, 32, 48)), ("wrist", dtype, (3, 24, 40))]
        source = _record_source(tmp_path / "source", cameras)
        output = tmp_path / "out"
        result = _transform(source, output)
        assert result.status == "success", result.message
        assert _camera_dtypes(output) == _camera_dtypes(source)

    @pytest.mark.parametrize("dtype", ["video", "image"])
    def test_the_output_encodes_exactly_the_streams_the_source_did(self, dtype: str, tmp_path: Path) -> None:
        cameras = [("top", dtype, (3, 32, 48)), ("wrist", dtype, (3, 24, 40))]
        source = _record_source(tmp_path / "source", cameras)
        output = tmp_path / "out"
        assert _transform(source, output).status == "success"
        assert _mp4_names(output) == _mp4_names(source)

    def test_a_recorded_single_camera_dataset_still_transforms(self, record_source_dataset, tmp_path: Path) -> None:
        """The shipped ``DatasetRecorder`` path, which declares one camera."""
        source = Path(record_source_dataset([40]))
        output = tmp_path / "out"
        result = _transform(source, output)
        assert result.status == "success", result.message
        assert _camera_dtypes(output) == _camera_dtypes(source)


class TestTheDtypeFlagIsAPropertyOfTheWholeCameraSet:
    """The derivation, graded over a camera grid rather than one example."""

    @pytest.mark.parametrize(
        ("dtypes", "expected"),
        [
            pytest.param(["video"], True, id="one-video"),
            pytest.param(["image"], False, id="one-image"),
            pytest.param(["video", "video"], True, id="all-video"),
            pytest.param(["image", "image"], False, id="all-image"),
            pytest.param(["video", "image"], False, id="video-then-image"),
            pytest.param(["image", "video"], False, id="image-then-video"),
            pytest.param(["video", "video", "image"], False, id="one-dissenter-last"),
            pytest.param(["image", "video", "video"], False, id="one-dissenter-first"),
        ],
    )
    def test_videos_are_used_only_when_every_camera_declares_one(self, dtypes: list[str], expected: bool) -> None:
        cameras = [(f"cam{i}", dtype, (3, 8, 8)) for i, dtype in enumerate(dtypes)]
        assert _reader(cameras).use_videos is expected

    @pytest.mark.parametrize(
        "dtypes",
        [
            pytest.param(["video", "video"], id="all-video"),
            pytest.param(["image", "image"], id="all-image"),
            pytest.param(["video", "image"], id="mixed"),
        ],
    )
    def test_reversing_the_declaration_order_does_not_change_the_flag(self, dtypes: list[str]) -> None:
        forward = [(f"cam{i}", dtype, (3, 8, 8)) for i, dtype in enumerate(dtypes)]
        assert _reader(forward).use_videos is _reader(forward[::-1]).use_videos

    def test_every_camera_s_dtype_is_recorded_not_just_the_last(self) -> None:
        cameras = [("top", "video", (3, 8, 8)), ("wrist", "image", (3, 8, 8))]
        assert _reader(cameras).camera_dtypes == {"top": "video", "wrist": "image"}

    def test_a_dtype_the_recorder_does_not_write_is_reported_as_declared(self) -> None:
        """An unknown dtype is not a video, and is reported rather than mapped."""
        reader = _reader([("depth", "uint16", (1, 8, 8))])
        assert (reader.camera_dtypes, reader.use_videos) == ({"depth": "uint16"}, False)
