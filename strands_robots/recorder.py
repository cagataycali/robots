"""The recording contract: control-loop frames in, a lerobot-format dataset out.

:class:`Recorder` is what every recording path writes through - the simulation
backends' dataset hooks, :class:`~strands_robots.simulation.policy_runner.PolicyRunner`'s
per-episode flush and the dashboard's record worker. The one shipped
implementation is :class:`strands_robots.dataset_recorder.DatasetRecorder`; a
test double subclasses this class, so a double that drifts from the contract
fails at construction instead of passing against a surface nothing ships.

The lifecycle is ``add_frame`` per control step, ``save_episode`` at each
episode boundary (``clear_episode_buffer`` to discard an aborted one instead),
and ``finalize`` once. A recorder whose ``finalize`` ran, or whose
``save_episode`` failed, is :attr:`Recorder.closed` for good, and a closed
recorder refuses every later frame: it raises
:class:`~strands_robots.recording_errors.RecordingFrameError`, or counts the
frame in ``dropped_frame_count`` when it was built best-effort. A frame is never
accepted and then thrown away without a trace.

Nothing internal is imported here beyond the standard library, so a caller can
name the contract without paying for numpy or LeRobot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class Recorder(ABC):
    """Episodes of control-loop frames, written as one dataset.

    Attributes:
        frame_count: Frames written, or buffered for the open episode, across
            the whole dataset.
        episode_frame_count: Frames buffered for the open (unsaved) episode.
        episode_count: Episodes saved.
    """

    frame_count: int
    episode_frame_count: int
    episode_count: int
    _closed: bool = False

    @property
    def closed(self) -> bool:
        """Whether this recorder refuses frames: ``finalize`` ran or a save failed."""
        return self._closed

    @abstractmethod
    def add_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        task: str | None = None,
        camera_keys: list[str] | None = None,
        required_action_keys: Sequence[str] | None = None,
    ) -> None:
        """Write one control step's observation and the action it was answered with."""

    @abstractmethod
    def save_episode(self) -> dict[str, Any]:
        """Close the open episode; a result whose ``status`` is not ``"success"`` closes the recorder."""

    @abstractmethod
    def clear_episode_buffer(self) -> bool:
        """Discard the open episode's frames; False when this recorder cannot."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush the dataset and close the recorder."""
