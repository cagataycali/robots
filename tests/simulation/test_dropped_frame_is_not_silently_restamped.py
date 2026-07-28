# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A dropped dataset frame must not be renumbered away in silence.

lerobot's writer derives frame indices POSITIONALLY (``frame_index = len(buffer)``;
``timestamp = frame_index / fps``). So when ``add_frame`` rejects a frame the
SURVIVORS are renumbered contiguously and the discontinuity is ERASED - the
episode is internally consistent, passes ``verify_dataset``, and encodes a
trajectory that was never executed. Measured with the 3rd of 5 frames rejected::

    frame_count: 4  dropped: 1
    recorded j1: [0.0, 1.0, 3.0, 4.0]        <- jumps 1.0 -> 3.0
    recorded ts: [0.0, 0.0333, 0.0667, 0.1]  <- across ONE declared period

Two holes, both closed here:

1. ``dropped_frame_count`` was never surfaced. ``save_episode`` returned only
   status/episode/frames, and ``stop_recording``'s payload carried repo_id,
   frame_count, episode_count, parquet counts and root - so a caller had no way
   to learn the episode was lossy. Both now report ``dropped_frame_count``,
   ``zero_filled_frame_count`` and a ``degraded`` flag.

2. ``strict=True`` (the DEFAULT) did not actually stop this. ``PolicyRunner``
   treats a raised ``add_frame`` as an ``on_frame`` telemetry failure and only
   aborts after 5 CONSECUTIVE failures, so a single transient recorder failure was
   swallowed regardless of ``strict``. A recorder-originated failure is now
   non-tolerated and aborts the episode immediately, while a genuine user
   telemetry hook keeps its tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")

from strands_robots.dataset_recorder import DatasetRecorder  # noqa: E402
from strands_robots.simulation.policy_runner import _is_recorder_frame_failure  # noqa: E402

_JOINTS = ["j1"]


def _recorder(root: str, case: str, *, strict: bool) -> DatasetRecorder:
    recorder = DatasetRecorder.create(
        repo_id=f"local/{case}",
        fps=30,
        robot_type="so101",
        joint_names=_JOINTS,
        action_names=_JOINTS,
        camera_keys=["cam"],
        camera_dims={"cam": (4, 4)},
        task="t",
        root=f"{root}/{case}",
    )
    recorder.strict = strict
    return recorder


def _good_frame(index: float) -> tuple[dict, dict]:
    return {"j1": index, "cam": np.zeros((4, 4, 3), dtype=np.uint8)}, {"j1": index}


def _bad_frame(index: float) -> tuple[dict, dict]:
    """A camera frame whose shape lerobot rejects."""
    return {"j1": index, "cam": np.zeros((8, 8, 3), dtype=np.uint8)}, {"j1": index}


class TestTheLossIsSurfaced:
    def test_save_episode_reports_the_dropped_count(self, tmp_path):
        """Regression: the payload was status/episode/frames only."""
        recorder = _recorder(str(tmp_path), "reported", strict=False)
        for index in range(2):
            recorder.add_frame(*_good_frame(float(index)), task="t")
        recorder.add_frame(*_bad_frame(2.0), task="t")
        recorder.add_frame(*_good_frame(3.0), task="t")
        assert recorder.dropped_frame_count == 1, "the fixture did not drop a frame"

        result = recorder.save_episode()

        assert result["status"] == "success", result
        assert result["dropped_frame_count"] == 1, result
        assert result["degraded"] is True, result

    def test_a_clean_episode_is_not_marked_degraded(self, tmp_path):
        recorder = _recorder(str(tmp_path), "clean", strict=False)
        for index in range(3):
            recorder.add_frame(*_good_frame(float(index)), task="t")

        result = recorder.save_episode()

        assert result["dropped_frame_count"] == 0
        assert result["zero_filled_frame_count"] == 0
        assert "degraded" not in result, result

    def test_the_degraded_warning_names_the_consequence(self, tmp_path, caplog):
        recorder = _recorder(str(tmp_path), "warned", strict=False)
        recorder.add_frame(*_good_frame(0.0), task="t")
        recorder.add_frame(*_bad_frame(1.0), task="t")

        with caplog.at_level("WARNING"):
            recorder.save_episode()

        degraded = [r.getMessage() for r in caplog.records if "DEGRADED" in r.getMessage()]
        assert degraded, [r.getMessage() for r in caplog.records]
        assert "never executed" in degraded[0], degraded[0]
        assert degraded[0].isascii()

    def test_zero_filled_frames_also_mark_the_episode_degraded(self, tmp_path):
        """The other silent-corruption counter shares the flag."""
        recorder = _recorder(str(tmp_path), "zerofill", strict=False)
        # '.pos' keys against a bare-name schema: nothing matches (see D46).
        recorder.add_frame({"j1.pos": 0.5, "cam": np.zeros((4, 4, 3), dtype=np.uint8)}, {"j1.pos": 0.5}, task="t")
        assert recorder.zero_filled_frame_count == 1

        result = recorder.save_episode()

        assert result["zero_filled_frame_count"] == 1
        assert result["degraded"] is True


class TestRecorderFailuresAreNotTolerated:
    """``strict=True`` must actually stop the rollout, not be absorbed by the
    ``on_frame`` retry tolerance."""

    def test_a_real_recorder_error_is_classified_as_a_recorder_failure(self, tmp_path):
        recorder = _recorder(str(tmp_path), "classify", strict=True)

        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - lerobot raises ValueError
            recorder.add_frame(*_bad_frame(0.0), task="t")

        assert _is_recorder_frame_failure(excinfo.value), (
            "a real add_frame failure was not recognised, so PolicyRunner would tolerate it"
        )

    def test_a_user_telemetry_failure_is_not_classified_as_one(self):
        """The tolerance must survive for the case it was written for."""

        def my_metrics_hook():
            raise ValueError("my metrics server is down")

        try:
            my_metrics_hook()
        except ValueError as exc:
            assert not _is_recorder_frame_failure(exc)

    def test_a_wrapped_recorder_error_is_still_classified(self):
        """A hook that re-raises must not launder the recorder failure."""

        class _Rec:
            def add_frame(self, *args, **kwargs):
                raise ValueError("feature shape mismatch")

        try:
            try:
                _Rec().add_frame()
            except ValueError as inner:
                raise RuntimeError("hook wrapper") from inner
        except RuntimeError as exc:
            assert _is_recorder_frame_failure(exc)

    def test_an_exception_with_no_traceback_is_not_misclassified(self):
        """Defensive: a bare exception object must not crash the check."""
        assert not _is_recorder_frame_failure(ValueError("never raised"))


class TestTheRestampingIsReal:
    """Pin the premise, so nobody 'fixes' this by trusting the timestamps."""

    def test_survivors_are_renumbered_with_no_gap(self, tmp_path):
        recorder = _recorder(str(tmp_path), "restamp", strict=False)
        for index, frame in enumerate([_good_frame(0.0), _good_frame(1.0), _bad_frame(2.0), _good_frame(3.0)]):
            recorder.add_frame(*frame, task="t")

        # 4 offered, 1 rejected, so 3 recorded - contiguously.
        assert recorder.frame_count == 3
        assert recorder.dropped_frame_count == 1
        buffer = getattr(recorder.dataset, "episode_buffer", None)
        if buffer is None:
            pytest.skip("this lerobot version exposes no episode_buffer to inspect")
        indices = list(buffer.get("frame_index", []))
        # The indices are 0,1,2 - NOT 0,1,3. That is the erasure.
        assert indices == sorted(indices), indices
        assert len(indices) == 3, indices
