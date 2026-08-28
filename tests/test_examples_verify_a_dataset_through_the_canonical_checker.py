"""A recording example verifies its dataset with the library's own checker.

An example that records N episodes and then re-reads the parquet itself to
confirm the count is printing a verdict about a document it opened by name.
LeRobot v3 does not guarantee that name. ``data/`` is written in files capped at
``lerobot.datasets.utils.DEFAULT_DATA_FILE_SIZE_IN_MB`` (100 MB), so a recording
that outgrows the cap spills into ``file-001.parquet`` while
``meta/episodes/chunk-000/file-000.parquet`` - two orders of magnitude smaller -
still holds every row. Opening only ``data/chunk-000/file-000.parquet`` then
reads the leading episodes and nothing else.

Measured on ``731e8235``, before the call site this module guards was paired. A
healthy twenty-episode dataset whose ``data/`` had spilled at episode twelve::

    strands_robots.verify_dataset.verify_dataset(root, expected=20)
        -> status='success'  total_episodes=20

    examples/kimodo/kimodo_g1_dataset_headcam.py::_verify_parquet_truth(root, 20)
        -> SystemExit: parquet truth FAIL: unique episode_index=[0, ... 11],
                       expected [0, ... 19]

The example's own docstring calls that check "the same contract the autonomous
harness uses", and a head-cam dataset - two camera streams, twenty episodes -
passes 100 MB well before it finishes. So the failure direction is the damaging
one: the check exists to catch the mega-episode corruption (N episodes buffered
into ``episode_index=0``) and instead refuses the successful runs, which trains
a reader to disbelieve it.

``verify_dataset`` globs ``meta/episodes/**/*.parquet`` and takes that as the
ground truth, so it is size-independent by construction. It also checks two
corruptions a count-only read cannot see: a per-episode MP4 that is absent,
empty or truncated, and an ``action`` / ``observation.state`` column written
identically zero across an episode.

What this module does NOT claim: delegating does not add detection of a corrupt
file under ``data/``. ``meta/episodes`` is the declared ground truth and
``verify_dataset`` reports success for a dataset whose ``data/`` tail is
unreadable - it did so before this pairing too. The change here removes a false
refusal; it does not widen what is refused.

The structural half is scoped to this one call site rather than swept over
``examples/``: exactly one example reads a dataset root today, and ``tests/``
legitimately names ``chunk-000`` because a fixture *writes* those files at a
path it chooses. Naming a path you are creating is not the same act as trusting
a path you are reading, so the rule is graded against constructed exemplars.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO_ROOT / "examples" / "kimodo" / "kimodo_g1_dataset_headcam.py"

_EPISODES = 20
_SPILL_AT = 12
_FRAMES_PER_EPISODE = 10
_CAMERA_FEATURES = ("observation.images.head", "observation.images.front")

# A dataset path written with the chunk/file index spelled as a literal. Reading
# one is what this module refuses; a fixture *writing* one is not the same act.
_CHUNK_BY_NAME = re.compile(r"""["']chunk-\d+["']""")


def _verifier_ast() -> ast.FunctionDef:
    """The example's verifier, parsed."""
    source = _EXAMPLE.read_text(encoding="utf-8")
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_verify_parquet_truth"
    )


def _load_example() -> Any:
    """Import the example module without running its ``main``."""
    spec = importlib.util.spec_from_file_location("_headcam_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_headcam_example"] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(
    root: Path,
    *,
    episodes: int = _EPISODES,
    spill_at: int | None = _SPILL_AT,
    features: tuple[str, ...] = _CAMERA_FEATURES,
) -> None:
    """Write a healthy LeRobot v3 dataset, optionally spilling ``data/``.

    ``spill_at`` splits ``data/`` across ``file-000`` and ``file-001`` the way
    the 100 MB cap does on a real recording, leaving the small
    ``meta/episodes`` parquet in a single file - the on-disk shape a head-cam
    run produces.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    indices = list(range(episodes))
    lengths = [_FRAMES_PER_EPISODE] * episodes

    pq.write_table(
        pa.table({"episode_index": indices, "length": lengths}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    def _frames(start: int, stop: int) -> pa.Table:
        rows: list[int] = []
        for episode in indices[start:stop]:
            rows += [episode] * _FRAMES_PER_EPISODE
        return pa.table({"episode_index": rows, "frame_index": list(range(len(rows)))})

    spans = [(0, episodes)] if spill_at is None else [(0, spill_at), (spill_at, episodes)]
    for file_index, (start, stop) in enumerate(spans):
        pq.write_table(_frames(start, stop), root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet")

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": episodes,
                "total_frames": sum(lengths),
                "features": {name: {"dtype": "image"} for name in features} | {"action": {"dtype": "float32"}},
            }
        )
    )


def _verify(root: Path, episodes: int = _EPISODES) -> str:
    """Run the example's verifier, returning what it printed."""
    captured = io.StringIO()
    with redirect_stdout(captured):
        _load_example()._verify_parquet_truth(root, episodes)
    return captured.getvalue()


class TestASpilledDatasetIsNotReadAsTruncated:
    """The regression: the cap is reached, every episode is present."""

    def test_a_dataset_whose_data_spilled_is_accepted(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path)
        spilled = sorted(p.name for p in (tmp_path / "data" / "chunk-000").glob("*.parquet"))
        assert spilled == ["file-000.parquet", "file-001.parquet"], spilled

        printed = _verify(tmp_path)

        assert "parquet truth PASS" in printed
        assert f"{_EPISODES} eps" in printed

    def test_the_leading_file_alone_does_not_hold_every_episode(self, tmp_path: Path) -> None:
        """Premise: the spill is what a by-name read would have missed."""
        _write_dataset(tmp_path)
        leading = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        present = sorted(set(leading.column("episode_index").to_pylist()))

        assert present == list(range(_SPILL_AT))
        assert len(present) < _EPISODES


class TestTheCheckStillRefusesWhatItIsFor:
    """Over-reach controls: delegating must not accept a broken dataset."""

    def test_a_short_recording_is_still_refused(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, episodes=13, spill_at=None)
        with pytest.raises(SystemExit) as excinfo:
            _verify(tmp_path)
        assert "parquet truth FAIL" in str(excinfo.value)

    def test_a_mega_episode_recording_is_still_refused(self, tmp_path: Path) -> None:
        """Every frame buffered into one episode - the corruption class."""
        _write_dataset(tmp_path, episodes=1, spill_at=None)
        with pytest.raises(SystemExit) as excinfo:
            _verify(tmp_path)
        assert "parquet truth FAIL" in str(excinfo.value)

    def test_a_missing_head_camera_is_still_refused(self, tmp_path: Path) -> None:
        """The script's own contract, which the shared checker does not own.

        Unspilled deliberately: the point of this control is the feature check,
        and a spilled fixture would make the pre-pairing code refuse on the
        count before it ever read ``features``.
        """
        _write_dataset(tmp_path, spill_at=None, features=("observation.images.front",))
        with pytest.raises(SystemExit) as excinfo:
            _verify(tmp_path)
        assert "observation.images.head" in str(excinfo.value)

    def test_a_healthy_unspilled_dataset_is_accepted(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, spill_at=None)
        assert "parquet truth PASS" in _verify(tmp_path)


class TestTheExampleReadsTheDatasetThroughTheSharedChecker:
    """Structural: the verdict is the library's, not a local re-read."""

    def test_the_verifier_calls_verify_dataset(self) -> None:
        source = _EXAMPLE.read_text(encoding="utf-8")
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_verify_parquet_truth"
        )
        called = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "verify_dataset" in called, sorted(called)

    def test_the_verifier_names_no_chunk_file(self) -> None:
        """The shape rule of :class:`TestTheRuleIsNotVacuous`, applied here."""
        assert _CHUNK_BY_NAME.search(ast.unparse(_verifier_ast())) is None

    def test_the_verifier_opens_no_parquet_of_its_own(self) -> None:
        source = _EXAMPLE.read_text(encoding="utf-8")
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_verify_parquet_truth"
        )
        body = ast.unparse(function)
        assert "read_table" not in body
        assert "pyarrow" not in body


class TestTheRuleIsNotVacuous:
    """The by-name read is what is refused, and it is refused by shape."""

    @pytest.mark.parametrize(
        ("label", "snippet", "flagged"),
        [
            pytest.param(
                "reads-a-named-chunk",
                'pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")',
                True,
                id="reads-a-named-chunk",
            ),
            pytest.param(
                "globs-every-chunk",
                'sorted((root / "data").glob("**/*.parquet"))',
                False,
                id="globs-every-chunk",
            ),
            pytest.param(
                "delegates",
                "verify_dataset(root, expected=n_episodes)",
                False,
                id="delegates",
            ),
        ],
    )
    def test_the_shape_rule_grades_constructed_exemplars(self, label: str, snippet: str, flagged: bool) -> None:
        assert bool(_CHUNK_BY_NAME.search(snippet)) is flagged, label

    def test_both_outcomes_are_reachable(self) -> None:
        """Non-vacuity: the rule accepts and refuses something."""
        outcomes = {
            bool(_CHUNK_BY_NAME.search(snippet))
            for snippet in (
                'pq.read_table(root / "chunk-000" / "file-000.parquet")',
                "verify_dataset(root)",
            )
        }
        assert outcomes == {True, False}
