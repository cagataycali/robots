"""Behavior tests for ``DatasetRecordingMixin`` lifecycle accessors + ``save_episode``.

These cover the engine-independent recording accessors the base ``run_policy``
loop reads (``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root``)
and the ``save_episode`` error/edge contracts, exercised directly on the mixin
with a stub world rather than through a real MuJoCo backend or an on-disk
``LeRobotDataset``.

``_active_dataset_root`` is what :meth:`SimEngine.verify_dataset_episodes` calls
to locate the parquet ground truth AFTER ``stop_recording`` has finalized and
dropped the live recorder, so its prefer-live / fall-back-to-last / world-gone
branches must be exact: a wrong root silently turns the dataset-integrity gate
into a no-op. Likewise ``save_episode`` must drop a recorder whose flush failed
so callers never append into a poisoned (closed) LeRobot episode buffer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from strands_robots.simulation.recording import DatasetRecordingMixin


class _Engine(DatasetRecordingMixin):
    """Minimal concrete mixin host exposing a settable stub world."""

    def __init__(self, world: Any = None) -> None:
        self._world = world


def _world(**backend_state: Any) -> SimpleNamespace:
    """A stub SimWorld carrying just the ``_backend_state`` dict the mixin reads."""
    return SimpleNamespace(_backend_state=dict(backend_state))


# --- _is_recording ---------------------------------------------------------


def test_is_recording_false_when_no_world() -> None:
    assert _Engine(world=None)._is_recording() is False


def test_is_recording_reflects_backend_flag() -> None:
    assert _Engine(_world())._is_recording() is False
    assert _Engine(_world(recording=False))._is_recording() is False
    assert _Engine(_world(recording=True))._is_recording() is True


# --- _active_recorder ------------------------------------------------------


def test_active_recorder_none_when_no_world() -> None:
    assert _Engine(world=None)._active_recorder() is None


def test_active_recorder_returns_stashed_recorder() -> None:
    recorder = object()
    assert _Engine(_world(dataset_recorder=recorder))._active_recorder() is recorder


# --- _active_dataset_root --------------------------------------------------


def test_active_dataset_root_prefers_live_recorder_root() -> None:
    recorder = SimpleNamespace(root="/data/live_session")
    engine = _Engine(_world(dataset_recorder=recorder, last_dataset_root="/data/stale"))
    assert engine._active_dataset_root() == "/data/live_session"


def test_active_dataset_root_falls_back_when_recorder_has_no_root() -> None:
    # A recorder whose ``.root`` access raises AttributeError must not abort the
    # lookup: fall back to the root stashed at start_recording.
    class _NoRoot:
        @property
        def root(self) -> str:
            raise AttributeError("root not set yet")

    engine = _Engine(_world(dataset_recorder=_NoRoot(), last_dataset_root="/data/finalized"))
    assert engine._active_dataset_root() == "/data/finalized"


def test_active_dataset_root_uses_last_root_after_recorder_dropped() -> None:
    # The post-stop_recording state: recorder is gone but last_dataset_root
    # remains so verify_dataset_episodes can still find the finalized parquet.
    engine = _Engine(_world(last_dataset_root="/data/finalized"))
    assert engine._active_dataset_root() == "/data/finalized"


def test_active_dataset_root_none_when_no_recorder_and_no_last() -> None:
    assert _Engine(_world())._active_dataset_root() is None


def test_active_dataset_root_none_when_no_world() -> None:
    assert _Engine(world=None)._active_dataset_root() is None


# --- save_episode ----------------------------------------------------------


def test_save_episode_errors_when_not_recording() -> None:
    result = _Engine(world=None).save_episode()
    assert result["status"] == "error"
    assert "not recording" in result["content"][0]["text"]


def test_save_episode_errors_when_recording_but_no_recorder() -> None:
    result = _Engine(_world(recording=True)).save_episode()
    assert result["status"] == "error"
    assert "No dataset recorder active." in result["content"][0]["text"]


def test_save_episode_no_frames_to_flush_is_idempotent_success() -> None:
    recorder = SimpleNamespace(episode_frame_count=0)
    result = _Engine(_world(recording=True, dataset_recorder=recorder)).save_episode()
    assert result["status"] == "success"
    assert "no frames to flush" in result["content"][0]["text"]


def test_save_episode_drops_poisoned_recorder_on_failed_flush() -> None:
    # When the underlying flush fails, the LeRobot episode buffer is in an
    # undefined state; the mixin must close recording and drop the recorder so
    # callers never append into a poisoned buffer.
    recorder = SimpleNamespace(
        episode_frame_count=5,
        save_episode=lambda: {"status": "error", "message": "buffer corrupt"},
    )
    world = _world(recording=True, dataset_recorder=recorder, trajectory=[{"x": 1}])
    result = _Engine(world).save_episode()

    assert result["status"] == "error"
    assert "buffer corrupt" in result["content"][0]["text"]
    assert world._backend_state["recording"] is False
    assert world._backend_state["dataset_recorder"] is None
    assert world._backend_state["trajectory"] == []


def test_save_episode_success_resets_trajectory_and_reports_counts() -> None:
    recorder = SimpleNamespace(
        episode_frame_count=12,
        save_episode=lambda: {
            "status": "success",
            "episode": 3,
            "episode_frames": 12,
            "total_frames": 48,
        },
    )
    world = _world(recording=True, dataset_recorder=recorder, trajectory=[{"x": 1}])
    result = _Engine(world).save_episode()

    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "Episode 3 saved" in text
    assert "12 frames" in text
    assert "48 total" in text
    # In-memory trajectory mirror reset so the next episode reports from zero.
    assert world._backend_state["trajectory"] == []
