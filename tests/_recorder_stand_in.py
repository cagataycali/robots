"""One stand-in for the dataset recorder a recording path writes through.

Twelve test modules grew their own recorder double, each shaped like the part
of :class:`~strands_robots.dataset_recorder.DatasetRecorder` its test touched and
none bound to anything: a double could keep a verb the recorder no longer has,
or skip one it gained, and every test using it stayed green.
:class:`RecorderStandIn` subclasses :class:`~strands_robots.recorder.Recorder`,
so it must implement the whole contract to be constructed at all, and it keeps
the contract's one behavioural rule: a closed recorder refuses frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from strands_robots.recorder import Recorder
from strands_robots.recording_errors import RecordingFrameError


class RecorderStandIn(Recorder):
    """Keeps every call in memory and answers like the shipped recorder.

    Args:
        pending: Frames already buffered for the open episode.
        save_result: What every ``save_episode`` returns, verbatim and without
            touching the counters, instead of the success envelope; a result
            whose ``status`` is not ``"success"`` closes the stand-in, as a
            failed save closes the real recorder.

    Attributes:
        frames: The arguments of every accepted ``add_frame``, the two dicts copied.
        saves: Every ``save_episode`` result, in order.
        calls: The contract verbs called, in order.
    """

    def __init__(self, *, pending: int = 0, save_result: dict[str, Any] | None = None) -> None:
        self.frame_count = self.episode_frame_count = pending
        self.episode_count = 0
        self.dropped_frame_count = 0
        self.frames: list[dict[str, Any]] = []
        self.saves: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.repo_id = "local/recorder_stand_in"
        self.root = "/tmp/recorder_stand_in"
        self._save_result = save_result

    def add_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        task: str | None = None,
        camera_keys: list[str] | None = None,
        required_action_keys: Sequence[str] | None = None,
    ) -> None:
        """Record the frame, or refuse it once closed."""
        self.calls.append("add_frame")
        if self._closed:
            raise RecordingFrameError("the recorder is closed; nothing more is written")
        self.frames.append(
            {
                "observation": dict(observation),
                "action": dict(action),
                "task": task,
                "camera_keys": camera_keys,
                "required_action_keys": required_action_keys,
            }
        )
        self.frame_count += 1
        self.episode_frame_count += 1

    def save_episode(self) -> dict[str, Any]:
        """Close the open episode, answering ``save_result`` when one was given."""
        self.calls.append("save_episode")
        result = self._save_result
        if result is None:
            self.episode_count += 1
            result = {
                "status": "success",
                "episode": self.episode_count,
                "episode_frames": self.episode_frame_count,
                "total_frames": self.frame_count,
            }
            self.episode_frame_count = 0
        elif result.get("status") != "success":
            self._closed = True
        self.saves.append(result)
        return result

    def clear_episode_buffer(self) -> bool:
        """Discard the open episode's frames."""
        self.calls.append("clear_episode_buffer")
        self.frame_count -= self.episode_frame_count
        self.episode_frame_count = 0
        return True

    def finalize(self) -> None:
        """Close the stand-in."""
        self.calls.append("finalize")
        self._closed = True
