"""The recorder's frame counters must describe frames that reached disk.

``DatasetRecorder.add_frame`` counts a frame into ``frame_count`` (cumulative)
and ``episode_frame_count`` (current, unsaved episode) at BUFFER time, while
frames only reach disk when ``save_episode`` flushes the episode. Every counter
consumer treats those numbers as on-disk truth: ``save_episode`` returns
``frame_count`` as ``total_frames``, ``push_to_hub`` reports it as ``frames`` and
refuses an empty dataset by asking it, ``stop_recording`` asks it whether
anything was ever captured, and ``resume`` seeds it from ``meta.total_frames``.

``clear_episode_buffer`` is the abort path - both ``run_multi_policy``
implementations call it from their ``finally`` when a rollout bails mid-episode -
so the two have to be reconciled there, in whichever direction the discard
actually went:

  * the discard happened  -> the buffered frames can never reach disk, so they
    come back out of the cumulative total;
  * the discard did NOT happen -> the frames are still queued for the next
    ``save_episode``, so both counters already describe them and are left alone.

These tests drive the real ``add_frame``/``save_episode``/``clear_episode_buffer``
against a dataset that distinguishes buffered from written frames, so a counter
is only ever compared against what actually landed.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dataset_recorder import DatasetRecorder

_FEATURES: dict[str, Any] = {
    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j"]},
    "action": {"dtype": "float32", "shape": (1,), "names": ["j"]},
}


class _BufferedDataset:
    """Dataset double that separates the open buffer from written frames.

    ``add_frame`` only buffers; ``save_episode`` is what moves buffered frames
    to ``disk_frames`` and appends the episode length to ``episodes``. That
    split is the whole point: it makes "frames the recorder claims" and "frames
    a parquet row would account for" two independently observable numbers.
    """

    def __init__(self, *, can_clear: bool = True, clear_raises: bool = False):
        self.repo_id = "local/counters"
        self.root = None
        self.features = _FEATURES
        self.meta = type("_Meta", (), {"total_frames": 0, "total_episodes": 0, "features": _FEATURES})()
        self.buffered = 0
        self.disk_frames = 0
        self.episodes: list[int] = []
        self._clear_raises = clear_raises
        if can_clear:
            self.clear_episode_buffer = self._clear  # type: ignore[method-assign]

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.buffered += 1

    def save_episode(self) -> None:
        self.episodes.append(self.buffered)
        self.disk_frames += self.buffered
        self.buffered = 0

    def finalize(self) -> None:
        pass

    def _clear(self) -> None:
        if self._clear_raises:
            raise RuntimeError("buffer is wedged")
        self.buffered = 0


def _record(recorder: DatasetRecorder, n: int) -> None:
    """Feed ``n`` frames through the real ``add_frame``."""
    for i in range(n):
        recorder.add_frame({"j": 0.1 * i}, {"j": 0.2 * i})


def _recorder(**kwargs: Any) -> DatasetRecorder:
    return DatasetRecorder(dataset=_BufferedDataset(**kwargs), task="counters", strict=True)


class TestADiscardedEpisodeIsNotCounted:
    """A successful discard un-counts the frames it threw away."""

    def test_the_reported_total_matches_the_frames_on_disk(self):
        rec = _recorder()
        _record(rec, 10)
        assert rec.clear_episode_buffer() is True  # premise: the discard happened
        assert rec.dataset.disk_frames == 0, "premise: nothing was written before the abort"
        _record(rec, 5)

        result = rec.save_episode()

        assert result["total_frames"] == rec.dataset.disk_frames, (
            f"save_episode reported total_frames={result['total_frames']} but only "
            f"{rec.dataset.disk_frames} frame(s) reached disk; the "
            f"{10} frame(s) discarded by clear_episode_buffer are still counted"
        )

    def test_each_abort_does_not_add_more_drift(self):
        """Three aborts must not accumulate three aborts' worth of frames."""
        rec = _recorder()
        for _ in range(3):
            _record(rec, 7)
            assert rec.clear_episode_buffer() is True
        _record(rec, 4)

        result = rec.save_episode()

        assert result["total_frames"] == rec.dataset.disk_frames == 4

    def test_the_counter_never_goes_negative(self):
        rec = _recorder()
        _record(rec, 6)
        assert rec.clear_episode_buffer() is True
        assert rec.frame_count == 0
        assert rec.episode_frame_count == 0

    def test_a_resumed_recorder_stays_on_disk_truth_across_an_abort(self):
        """resume() seeds frame_count from disk, so an abort must not drift it."""
        rec = _recorder()
        _record(rec, 8)
        rec.save_episode()
        rec.frame_count = rec.dataset.disk_frames  # what resume() seeds
        _record(rec, 9)
        assert rec.clear_episode_buffer() is True

        assert rec.frame_count == rec.dataset.disk_frames == 8


class TestAFailedDiscardLeavesTheFramesCounted:
    """When nothing was discarded the frames are still queued, so still counted.

    These are the boundary of the fix. The frames survive in the open episode
    and the warning tells the caller to drain them with
    ``save_episode``/``stop_recording``, which writes them - so deducting them
    here would under-report the very episode about to be flushed.
    """

    @pytest.mark.parametrize(
        "kwargs, why",
        [
            ({"can_clear": False}, "no clear surface on this LeRobot version"),
            ({"clear_raises": True}, "the dataset raised mid-clear"),
        ],
    )
    def test_the_queued_frames_are_still_counted(self, kwargs, why):
        rec = _recorder(**kwargs)
        _record(rec, 6)

        assert rec.clear_episode_buffer() is False, f"premise: {why}"
        assert rec.dataset.buffered == 6, "premise: the frames are still buffered"
        assert rec.frame_count == 6
        assert rec.episode_frame_count == 6

    @pytest.mark.parametrize("kwargs", [{"can_clear": False}, {"clear_raises": True}])
    def test_draining_them_reports_the_episode_it_really_wrote(self, kwargs):
        rec = _recorder(**kwargs)
        _record(rec, 6)
        rec.clear_episode_buffer()  # failed discard: the 6 frames survive

        result = rec.save_episode()

        assert result["episode_frames"] == rec.dataset.episodes[-1] == 6
        assert result["total_frames"] == rec.dataset.disk_frames == 6


class TestCountersWithoutAnAbort:
    """Controls: the counters already agreed with disk on the normal path."""

    def test_a_saved_episode_reports_its_own_length_and_the_total(self):
        rec = _recorder()
        _record(rec, 4)
        first = rec.save_episode()
        _record(rec, 3)
        second = rec.save_episode()

        assert (first["episode_frames"], first["total_frames"]) == (4, 4)
        assert (second["episode_frames"], second["total_frames"]) == (3, 7)
        assert rec.dataset.disk_frames == 7
        assert rec.dataset.episodes == [4, 3]

    def test_clearing_an_untouched_buffer_is_a_no_op(self):
        rec = _recorder()
        assert rec.clear_episode_buffer() is True
        assert rec.frame_count == 0
        assert rec.episode_frame_count == 0
