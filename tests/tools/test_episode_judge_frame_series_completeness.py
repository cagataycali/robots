"""An episode whose frames are not all readable is refused, not summarised.

:func:`strands_robots.tools.episode_judge.sample_frames` reports two motion
statistics its own docstring says the judge grades motion by: ``max_state_delta``
("for spotting discontinuities and teleports") and ``rms_state_jerk`` ("so a
text-only judge can ground ``jerky_motion`` from state alone"). Both are
computed over consecutive rows of the episode's frame series.

The reader behind them, ``_episode_frame_rows``, walked ``data/**/*.parquet``
and skipped any shard it could not read, with a comment saying the skip
"mirrors read_dataset_episode_indices". It does not.
:func:`strands_robots.dataset_recorder.read_dataset_episode_indices` tolerates
an unreadable shard *and reports it* - it returns every damaged file in
``unreadable_files`` and its docstring states that "any non-empty
``unreadable_files`` means the totals are a lower bound and the dataset must
not be certified as complete". ``_episode_frame_rows`` took the tolerance
without the report, so the surviving rows were handed to the statistics as
though they were consecutive.

The consequence is fabricated evidence for exactly the failure modes the judge
grades. On a recording that moves a uniform 0.01 per step - no discontinuity
anywhere - losing one middle shard of three took ``max_state_delta`` from
0.010 to 0.070 and ``rms_state_jerk`` from 0.000 to 6123.7, reported as
``status="success"``. ``jerky_motion`` and ``drift`` are both in
:data:`strands_robots.episode_labels.FAILURE_MODES`, and ``write_label``
persists the judge's grade into the sidecar that filters training data, so a
truncated download - "the usual outcome of an interrupted sync or hub
download", in the sibling reader's own words - silently produces a wrong label
on a good episode.

It was also visible as two tools disagreeing: on the same dataset,
``load_episode`` (which reads the episode metadata) reported the true length
12 while ``sample_frames`` reported 6, both ``success``, neither naming a
dropped shard.

Covers:

* the refusal names the damaged shard, how many shards were damaged, and why a
  hole invalidates the summary;
* every damaged shard is named, not just the first one read;
* a fully unreadable episode reports the damage rather than "has no frames",
  which would name the wrong fault for frames that are present but unreadable;
* the two tools no longer disagree about the episode length - the one that
  cannot answer completely says so;
* the premise, measured on the fixture rather than assumed: the healthy series
  really is smooth, and the surviving rows really do read as a discontinuity;
* the controls that scope the refusal - a healthy dataset is unchanged, a
  parquet carrying no ``episode_index`` column is still skipped quietly (that
  is a non-frame table, not damage), and a dataset with no data parquet at all
  still reports the emptiness it always did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.episode_judge as M

pq = pytest.importorskip("pyarrow.parquet", reason="the synthetic dataset fixture writes parquet")
import pyarrow as pa  # noqa: E402

_sample_frames = getattr(M.sample_frames, "__wrapped__", None) or M.sample_frames
_load_episode = getattr(M.load_episode, "__wrapped__", None) or M.load_episode

#: Frames per shard, and the number of shards. Three shards so one of them can
#: be a *middle* shard: losing the tail only shortens the series, while losing
#: the middle leaves the surviving rows reading as consecutive across a gap.
_PER_SHARD = 6
_SHARDS = 3
_LENGTH = _PER_SHARD * _SHARDS

#: The recording moves this much per step, uniformly, on its single dimension.
#: A uniform ramp has zero third difference, so the healthy jerk is exactly 0
#: and any non-zero reading is manufactured.
_STEP = 0.01

_FPS = 50.0


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def _text(result: dict[str, Any]) -> str:
    return " ".join(c.get("text", "") for c in result.get("content", []) if "text" in c)


def _write_dataset(root: Path) -> list[Path]:
    """Write a LeRobot-v3-shaped single-episode dataset across ``_SHARDS`` shards.

    Returns the data shard paths in frame order, so a test can damage a chosen
    one by position.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": _FPS,
                "total_episodes": 1,
                "total_frames": _LENGTH,
                "features": {"observation.state": {"dtype": "float32", "shape": [1], "names": ["pan"]}},
            }
        )
    )
    pq.write_table(
        pa.table({"episode_index": [0], "length": [_LENGTH]}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    shards: list[Path] = []
    for index in range(_SHARDS):
        frames = list(range(index * _PER_SHARD, (index + 1) * _PER_SHARD))
        path = root / "data" / "chunk-000" / f"file-{index:03d}.parquet"
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0] * len(frames),
                    "frame_index": frames,
                    "timestamp": [frame / _FPS for frame in frames],
                    "observation.state": [[_STEP * frame] for frame in frames],
                }
            ),
            path,
        )
        shards.append(path)
    return shards


@pytest.fixture
def dataset(tmp_path):
    """A healthy three-shard dataset plus its shard paths in frame order."""
    root = tmp_path / "dataset"
    root.mkdir()
    return root, _write_dataset(root)


def _damage(path: Path) -> None:
    """Make a shard unreadable the way a truncated download does."""
    path.write_bytes(b"not a parquet file")


class TestThePremise:
    """The fixture really carries the properties the refusal is argued from."""

    def test_the_episode_spans_several_shards(self, dataset):
        root, shards = dataset
        assert len(shards) == _SHARDS >= 3
        assert len(list((root / "data").glob("**/*.parquet"))) == _SHARDS

    def test_the_healthy_recording_is_perfectly_smooth(self, dataset):
        root, _ = dataset
        payload = _json_payload(_sample_frames(str(root), 0, n_frames=4))
        assert payload["length"] == _LENGTH
        assert payload["max_state_delta"] == pytest.approx(_STEP)
        assert payload["rms_state_jerk"] == pytest.approx(0.0, abs=1e-6)

    def test_dropping_the_middle_shard_would_read_as_a_discontinuity(self, dataset):
        """Why a hole cannot be summarised: the seam is a jump, not a step.

        Measured on the fixture's own numbers rather than through the tool, so
        it holds whichever way the reader disposes of the damage.
        """
        surviving = list(range(0, _PER_SHARD)) + list(range(2 * _PER_SHARD, _LENGTH))
        seam = max(abs(_STEP * b - _STEP * a) for a, b in zip(surviving, surviving[1:], strict=False))
        assert seam == pytest.approx(_STEP * (_PER_SHARD + 1))
        assert seam > 5 * _STEP


class TestAnIncompleteFrameSeriesIsRefused:
    """The regression: damage is reported, never summarised over."""

    def test_a_damaged_middle_shard_is_refused(self, dataset):
        root, shards = dataset
        _damage(shards[1])
        result = _sample_frames(str(root), 0, n_frames=4)
        assert result["status"] == "error"

    def test_the_refusal_arrives_as_an_envelope_not_a_raise(self, dataset):
        """The module promises every tool returns the envelope and never raises."""
        root, shards = dataset
        _damage(shards[1])
        result = _sample_frames(str(root), 0, n_frames=4)
        assert set(result) == {"status", "content"}
        assert result["status"] == "error"

    def test_the_refusal_names_the_damaged_shard(self, dataset):
        root, shards = dataset
        _damage(shards[1])
        assert shards[1].name in _text(_sample_frames(str(root), 0, n_frames=4))

    def test_the_refusal_counts_the_damage_against_the_whole_episode(self, dataset):
        root, shards = dataset
        _damage(shards[1])
        assert f"1 of {_SHARDS} data shard(s)" in _text(_sample_frames(str(root), 0, n_frames=4))

    def test_the_refusal_says_why_a_hole_invalidates_the_summary(self, dataset):
        root, shards = dataset
        _damage(shards[1])
        assert "hole" in _text(_sample_frames(str(root), 0, n_frames=4))

    def test_every_damaged_shard_is_named_not_only_the_first(self, dataset):
        root, shards = dataset
        _damage(shards[0])
        _damage(shards[2])
        text = _text(_sample_frames(str(root), 0, n_frames=4))
        assert shards[0].name in text
        assert shards[2].name in text
        assert f"2 of {_SHARDS} data shard(s)" in text

    def test_a_wholly_unreadable_episode_reports_the_damage(self, dataset):
        """Not "has no frames": the frames are on disk, they cannot be read."""
        root, shards = dataset
        for shard in shards:
            _damage(shard)
        text = _text(_sample_frames(str(root), 0, n_frames=4))
        assert f"{_SHARDS} of {_SHARDS} data shard(s)" in text
        assert "has no frames" not in text

    def test_no_fabricated_statistic_reaches_the_judge(self, dataset):
        root, shards = dataset
        _damage(shards[1])
        result = _sample_frames(str(root), 0, n_frames=4)
        assert _json_payload(result) == {}

    def test_the_two_tools_no_longer_disagree_about_the_length(self, dataset):
        """``load_episode`` reads the metadata, so it still answers truthfully.

        The tool that cannot answer completely now says so instead of reporting
        a shorter episode with the same ``success`` envelope.
        """
        root, shards = dataset
        _damage(shards[1])
        described = _json_payload(_load_episode(str(root), 0))
        assert described["length"] == _LENGTH
        assert _sample_frames(str(root), 0, n_frames=4)["status"] == "error"


class TestWhatIsUnchanged:
    """Controls: every expectation here is one the pre-fix reader also met."""

    def test_a_healthy_dataset_is_still_sampled(self, dataset):
        root, _ = dataset
        result = _sample_frames(str(root), 0, n_frames=4)
        assert result["status"] == "success", _text(result)
        assert _json_payload(result)["length"] == _LENGTH

    def test_a_parquet_with_no_episode_index_column_is_still_skipped(self, dataset):
        """A non-frame table under ``data/`` is not damage and is not refused."""
        root, _ = dataset
        pq.write_table(pa.table({"unrelated": [1, 2, 3]}), root / "data" / "chunk-000" / "sidecar.parquet")
        result = _sample_frames(str(root), 0, n_frames=4)
        assert result["status"] == "success", _text(result)
        assert _json_payload(result)["length"] == _LENGTH

    def test_a_dataset_with_no_data_parquet_still_reports_emptiness(self, tmp_path):
        root = tmp_path / "empty"
        (root / "data").mkdir(parents=True)
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"fps": _FPS, "features": {}}))
        result = _sample_frames(str(root), 0, n_frames=4)
        assert result["status"] == "error"
        assert "No data parquet" in _text(result)

    def test_an_absent_episode_still_reports_having_no_frames(self, dataset):
        root, _ = dataset
        result = _sample_frames(str(root), 7, n_frames=4)
        assert result["status"] == "error"
        assert "has no frames" in _text(result)


class TestTheSiblingReaderStillTolerates:
    """The scope line: the reader whose product *is* the damage report is untouched."""

    def test_the_sibling_reports_an_unreadable_metadata_shard_rather_than_refusing(self, dataset):
        from strands_robots.dataset_recorder import read_dataset_episode_indices

        root, _ = dataset
        chunk = root / "meta" / "episodes" / "chunk-000"
        pq.write_table(pa.table({"episode_index": [1], "length": [3]}), chunk / "file-001.parquet")
        _damage(chunk / "file-001.parquet")
        described = read_dataset_episode_indices(root)
        assert described["unreadable_files"], "the sibling reports damage in its own field"
        assert described["total_episodes"] == 1
