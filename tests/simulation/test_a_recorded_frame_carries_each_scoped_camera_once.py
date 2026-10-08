"""A recorded frame carries each declared camera once, under its column, rendered at most once.

Newton's recording hook rendered every declared camera on every recorded frame
even when the rollout's observation already carried that frame (the runner asks
for pixels whenever a recording keeps cameras), and copied every array of the
observation into the frame - so a recorded step paid two renders per camera and
wrote an out-of-scope camera under its scene name beside the schema. Isaac and
mjlab applied the rule (rename to the column, drop what is out of scope, render
only what is missing) from their own copies. :func:`split_recorded_observation`
is that rule, once.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.newton.recording import NewtonRecordingMixin
from strands_robots.simulation.recording import split_recorded_observation

_SCOPE = [("front", "front", 4, 3), ("arm/wrist", "arm__wrist", 4, 3)]
_FRONT = np.full((3, 4, 3), 7, dtype=np.uint8)
_WRIST = np.full((3, 4, 3), 9, dtype=np.uint8)
_RENDERED = np.full((3, 4, 3), 1, dtype=np.uint8)
_OTHER = np.zeros((3, 4, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("observation", "render", "state", "images", "renders"),
    [
        pytest.param(
            {"j0": 0.5, "front": _FRONT, "arm/wrist": _WRIST},
            True,
            {"j0": 0.5},
            {"front": _FRONT, "arm__wrist": _WRIST},
            [],
            id="carried-cameras-are-renamed-not-rendered",
        ),
        pytest.param(
            {"j0": 0.5, "front": _FRONT, "top": _OTHER},
            None,
            {"j0": 0.5},
            {"front": _FRONT},
            [],
            id="out-of-scope-camera-is-dropped-missing-one-left-absent",
        ),
        pytest.param(
            {"j0": 0.5, "front": _FRONT},
            True,
            {"j0": 0.5},
            {"front": _FRONT, "arm__wrist": _RENDERED},
            [("arm/wrist", 4, 3)],
            id="only-the-missing-camera-is-rendered",
        ),
        pytest.param(
            {"base_pose": np.zeros(7), "j0": 0.5},
            True,
            {"base_pose": "vector", "j0": 0.5},
            {"front": _RENDERED, "arm__wrist": _RENDERED},
            [("front", 4, 3), ("arm/wrist", 4, 3)],
            id="a-vector-is-state-not-an-image",
        ),
    ],
)
def test_the_split(observation: dict, render: bool | None, state: dict, images: dict, renders: list) -> None:
    calls: list[tuple[str, int, int]] = []

    def _render(source: str, width: int, height: int) -> np.ndarray:
        calls.append((source, width, height))
        return _RENDERED

    got_state, got_images = split_recorded_observation(observation, _SCOPE, _render if render else None)
    assert {k: "vector" if isinstance(v, np.ndarray) else v for k, v in got_state.items()} == state
    assert got_images.keys() == images.keys()
    assert all(got_images[k] is images[k] for k in images)
    assert calls == renders


class _NewtonEngine(NewtonRecordingMixin):
    """The attributes Newton's recording hook reads, on the mixin itself."""

    def __init__(self) -> None:
        self.recorder = SimpleNamespace(frames=[], episode_frame_count=0)
        self.recorder.add_frame = lambda **frame: self.recorder.frames.append(frame)
        state = {"recording": True, "dataset_recorder": self.recorder, "recording_cameras": _SCOPE}
        self._world = SimpleNamespace(robots={"arm": object()}, _backend_state=state)  # type: ignore[assignment]
        self.renders: list[str] = []

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return ["j0"]

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        from PIL import Image

        self.renders.append(camera_name)
        buf = io.BytesIO()
        Image.fromarray(_RENDERED).save(buf, format="PNG")
        source = {"bytes": base64.b64encode(buf.getvalue()).decode()}
        return {"status": "success", "content": [{"image": {"format": "png", "source": source}}]}


def test_newton_renders_only_the_camera_the_observation_lacks() -> None:
    engine = _NewtonEngine()
    hook = engine._make_recording_on_frame("arm", "probe")
    hook(0, {"j0": 0.5, "front": _FRONT, "top": _OTHER}, {"j0": 0.1})

    assert engine.renders == ["arm/wrist"], "a camera the observation carries was rendered again"
    (frame,) = engine.recorder.frames
    assert frame["observation"]["j0"] == 0.5
    assert frame["observation"]["front"] is _FRONT
    assert np.array_equal(frame["observation"]["arm__wrist"], _RENDERED)
    assert "top" not in frame["observation"], "an out-of-scope camera was written beside the schema"
