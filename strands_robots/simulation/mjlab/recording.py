"""mjlab recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset``) lives in
:class:`~strands_robots.simulation.recording.DatasetRecordingMixin`. This
subclass supplies the mjlab-specific parts, following the Isaac backend's
shape (an engine-owned state dict rather than a ``SimWorld``):

* :meth:`_recording_state` - the state seam (``self._recording_state_dict``).
* :meth:`_collect_recording_schema` - declares the dataset schema from the
  live scene for the one shared ``start_recording``: joint names from every
  robot, actuator-ordered action names, the named look-at cameras, and the
  floating-base columns for a free-base robot.
* :meth:`_make_run_policy_hook` - the ``on_frame`` closure the shared
  run-policy loop calls every control step; it records world 0.

The recorder is engine-independent, so an mjlab recording produces the same
LeRobot v3 layout (``meta/info.json`` + per-episode parquet + per-camera MP4)
as the MuJoCo, Newton and Isaac backends.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.models import registered
from strands_robots.simulation.recording import (
    DatasetRecordingMixin,
    RecordedFrame,
    RecordingSchema,
    floating_base_state_specs,
)
from strands_robots.utils import camera_schema_key

if TYPE_CHECKING:
    import threading

logger = logging.getLogger(__name__)


class MjlabRecordingMixin(DatasetRecordingMixin):
    """mjlab dataset recording mixed into :class:`MjlabEngine`."""

    if TYPE_CHECKING:
        _world_created: bool
        _robots: dict[str, Any]
        _cameras: dict[str, dict[str, Any]]
        _lock: threading.RLock
        _recording_state_dict: dict[str, Any]
        default_width: int
        default_height: int

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Actuator-ordered action keys (concrete on the engine)."""

        def _render_rgb(
            self, camera_name: str | None = ..., width: int | None = ..., height: int | None = ..., env_id: int = ...
        ) -> Any:
            """Type-only stub for the engine-provided pixel-array render."""

    def _recording_state(self) -> dict[str, Any] | None:
        """Engine-owned recording-state dict; ``None`` before ``create_world``."""
        if not getattr(self, "_world_created", False):
            return None
        state = getattr(self, "_recording_state_dict", None)
        if state is None:
            state = self._recording_state_dict = {}
        return state

    _RECORDING_VIDEO_HINT = "For plain MP4 video, pass video={'path': ...} to run_policy instead."
    _RECORDING_REPLY_LABEL = "Recording mjlab scene (world 0) to LeRobotDataset"

    def _recording_start_error(self, state: dict[str, Any] | None) -> dict[str, Any] | None:
        """Refuse without a world, and without a robot to declare a schema from."""
        if state is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if not self._robots:
            return {"status": "error", "content": [{"text": "No robot in the scene. Call add_robot first."}]}
        return None

    def _recording_scene_cameras(self) -> list[str]:
        """The look-at cameras registered via ``add_camera``."""
        return list(self._cameras)

    def _recording_start_lock(self) -> threading.RLock:
        """The engine lock, held while the session is armed."""
        return self._lock

    def _collect_recording_schema(self, probe: Any = None) -> RecordingSchema:
        """Schema from the live scene (joints, actions, cameras, base columns).

        Joint names come from every robot spec (namespaced ``robot__joint`` when
        more than one robot exists), action columns from :meth:`robot_action_keys`
        (actuator order, the same authority ``send_action`` resolves), cameras
        from the ``add_camera`` registry at their requested render size, and a
        free-base robot adds the
        :func:`~strands_robots.simulation.recording.floating_base_state_specs`
        columns. ``probe`` is unused: mjlab renders at exactly the requested
        size, so nothing has to be measured before the schema is declared.
        """
        joint_names: list[str] = []
        action_names: list[str] = []
        robot_type = "unknown"
        multi_robot = len(self._robots) > 1
        base_state_specs: list[tuple[str, list[str]]] = []
        for rname, spec in self._robots.items():
            names = list(spec.joint_names)
            acts = self.robot_action_keys(rname)
            if multi_robot:
                joint_names.extend(f"{rname}__{jn}" for jn in names)
                action_names.extend(f"{rname}__{ak}" for ak in acts)
            else:
                joint_names.extend(names)
                action_names.extend(acts)
            robot_type = rname
            if getattr(spec, "free_base", False):
                base_state_specs.extend(floating_base_state_specs(f"{rname}__" if multi_robot else ""))
        recording_cameras: list[tuple[str, str, int, int]] = []
        for cam_name, cfg in self._cameras.items():
            recording_cameras.append((cam_name, camera_schema_key(cam_name), int(cfg["width"]), int(cfg["height"])))
        return RecordingSchema(
            joint_names,
            action_names,
            base_state_specs,
            recording_cameras,
            robot_type,
            (int(self.default_width), int(self.default_height)),
        )

    def _make_recording_on_frame(self, robot_name: str, instruction: str) -> Any:
        """``on_frame(step, observation, action)`` that appends world-0 frames to the recorder."""
        from strands_robots.simulation.models import TrajectoryStep

        state = self._recording_state()
        if state is None or not registered(self._robots, robot_name):
            return None
        frame = RecordedFrame(self, (robot_name,), self._robots)

        def _record(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            if not state.get("recording", False):
                return
            rec = state.get("dataset_recorder")
            if rec is None:
                return
            raw_to_safe = {src: safe for src, safe, _w, _h in state.get("recording_cameras", [])}
            scalars: dict[str, Any] = {}
            images: dict[str, Any] = {}
            for k, v in observation.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    safe = raw_to_safe.get(k)
                    if safe is not None:
                        images[safe] = v
                else:
                    scalars[k] = v
            # The runner may hand over a proprio-only observation (fast_mode /
            # skip_images); render the declared cameras here, as the Newton hook does.
            for src, safe, w, h in state.get("recording_cameras", []):
                if safe not in images:
                    images[safe] = self._render_rgb(src, width=w, height=h)
            dt = getattr(self, "_timestep", None) or 0.0
            state["trajectory"].append(
                TrajectoryStep(
                    timestamp=time.time(),
                    sim_time=float(getattr(self, "_step_count", 0)) * float(dt),
                    robot_name=robot_name,
                    observation=scalars,
                    action=action,
                    instruction=instruction,
                )
            )
            frame.write(rec, {robot_name: scalars}, {robot_name: action}, images, instruction)

        return _record

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Mark the robot as driven and forward every frame to the recorder."""
        state = self._recording_state()
        if state is None or not registered(self._robots, robot_name):
            return None
        driven = state.setdefault("driven", {})
        driven[robot_name] = {"instruction": instruction, "steps": 0}
        record_frame = self._make_recording_on_frame(robot_name, instruction)

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            driven[robot_name]["steps"] = step + 1
            if record_frame is not None:
                record_frame(step, observation, action)

        return _hook

    def _release_run_policy_hook(self, robot_name: str) -> None:
        """Lower the driven flag raised by :meth:`_make_run_policy_hook`."""
        state = self._recording_state()
        if state is None:
            return
        state.get("driven", {}).pop(robot_name, None)
