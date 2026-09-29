"""mjlab recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset``) lives in
:class:`~strands_robots.simulation.recording.DatasetRecordingMixin`. This
subclass supplies the mjlab-specific parts, following the Isaac backend's
shape (an engine-owned state dict rather than a ``SimWorld``):

* :meth:`_recording_state` - the state seam (``self._recording_state_dict``).
* :meth:`start_recording` - declares the dataset schema from the live scene:
  joint names from every robot, actuator-ordered action names, the named
  look-at cameras, and the floating-base columns for a free-base robot.
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
    camera_schema_key_collision_error,
    dataset_recording_option_error,
    dataset_recording_posture_error,
    recorded_cameras_line,
    undriven_robot_state,
)
from strands_robots.utils import camera_schema_key, name_list_error

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

        def render(self, camera_name: str | None = ..., width: int | None = ..., height: int | None = ...) -> Any:
            """Type-only stub for the engine-provided render method."""

    def _recording_state(self) -> dict[str, Any] | None:
        """Engine-owned recording-state dict; ``None`` before ``create_world``."""
        if not getattr(self, "_world_created", False):
            return None
        state = getattr(self, "_recording_state_dict", None)
        if state is None:
            state = self._recording_state_dict = {}
        return state

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str | None = None,
        push_to_hub: bool = False,
        vcodec: str = "h264",
        overwrite: bool = False,
        cameras: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start recording world 0 of the mjlab scene to LeRobotDataset format.

        Same arguments and refusal order as the MuJoCo / Newton / Isaac
        backends: option and posture checks, camera-list checks, the
        lerobot-extra probe, the camera schema-key collision check and the
        double-start check all happen before anything touches the disk.
        """
        state = self._recording_state()
        if state is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if not self._robots:
            return {"status": "error", "content": [{"text": "No robot in the scene. Call add_robot first."}]}
        if error := dataset_recording_option_error("start_recording", fps):
            return error
        for _flag, _value in (("push_to_hub", push_to_hub), ("overwrite", overwrite)):
            if error := dataset_recording_posture_error("start_recording", _flag, _value):
                return error
        if cameras and (text := name_list_error(cameras, "cameras", "start_recording")):
            return {"status": "error", "content": [{"text": text}]}
        if error := self._validate_recording_start_rate(fps, "start_recording"):
            return error
        _DatasetRecorder, refusal = self._dataset_recorder_or_refusal(
            "For plain MP4 video, pass video={'path': ...} to run_policy instead.",
        )
        if refusal is not None:
            return refusal
        if error := camera_schema_key_collision_error("start_recording", list(self._cameras)):
            return error
        if error := self._already_recording_error("start_recording", repo_id):
            return error

        with self._lock:
            state["recording"] = True
            state["trajectory"] = []
            state["push_to_hub"] = push_to_hub
            dataset_dir = self._stash_dataset_target(repo_id, root)
            try:
                (
                    joint_names,
                    action_names,
                    camera_keys,
                    camera_dims,
                    robot_type,
                    recording_cameras,
                    base_state_specs,
                ) = self._collect_recording_schema()
                recorded_cameras = {src: safe for src, safe, _w, _h in recording_cameras}
                scene_cameras = list(self._cameras)
                if cameras is not None:
                    raw_to_safe = {src: safe for src, safe, _w, _h in recording_cameras}
                    safe_to_raw = {safe: src for src, safe in raw_to_safe.items()}
                    selected_safe: list[str] = []
                    selected_raw: set[str] = set()
                    unknown: list[str] = []
                    for requested in cameras:
                        if requested in raw_to_safe:
                            raw, safe = requested, raw_to_safe[requested]
                        elif requested in safe_to_raw:
                            raw, safe = safe_to_raw[requested], requested
                        else:
                            unknown.append(requested)
                            continue
                        if safe not in selected_safe:
                            selected_safe.append(safe)
                            selected_raw.add(raw)
                    if unknown:
                        state["recording"] = False
                        available = sorted(raw_to_safe)
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": (
                                        f"start_recording: unknown camera(s) {unknown} in cameras=. "
                                        f"Available scene cameras: {available}. Add them with "
                                        "add_camera(...) before recording, or omit cameras= to "
                                        "record all of them."
                                    )
                                }
                            ],
                        }
                    camera_keys = selected_safe
                    camera_dims = {safe: camera_dims[safe] for safe in selected_safe}
                    recording_cameras = [tpl for tpl in recording_cameras if tpl[0] in selected_raw]
                    recorded_cameras = {safe_to_raw[safe]: safe for safe in selected_safe}
                state["recording_cameras"] = recording_cameras
                resume_existing = self._prepare_dataset_target(dataset_dir, overwrite)
                if resume_existing:
                    logger.info("Resuming existing dataset for append: %s", dataset_dir)
                    resumed = _DatasetRecorder.resume(
                        repo_id=repo_id,
                        root=root,
                        task=task,
                        vcodec=vcodec,
                        joint_names=joint_names,
                        extra_state_specs=base_state_specs,
                    )
                    state_names_full = list(joint_names) + [
                        f"{src}.{comp}" for src, comps in base_state_specs for comp in comps
                    ]
                    self._verify_resume_schema(
                        resumed, state_names_full, camera_keys, camera_dims, action_names, fps=fps
                    )
                    recorder = resumed
                else:
                    recorder = _DatasetRecorder.create(
                        repo_id=repo_id,
                        fps=fps,
                        robot_type=robot_type,
                        joint_names=joint_names,
                        action_names=action_names,
                        extra_state_specs=base_state_specs,
                        camera_keys=camera_keys,
                        camera_dims=camera_dims,
                        task=task,
                        root=root,
                        vcodec=vcodec,
                        video_width=int(self.default_width),
                        video_height=int(self.default_height),
                    )
                resumed_line = self._arm_dataset_recorder(state, recorder, resumed=resume_existing)
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Recording mjlab scene (world 0) to LeRobotDataset: {repo_id}\n"
                                f"{resumed_line}"
                                f"{recorded_cameras_line(joint_names, recorded_cameras, scene_cameras, cameras, fps)}"
                                f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                                "Run policies to capture frames, then stop_recording to save the episode"
                            )
                        }
                    ],
                }
            except Exception as e:
                state["recording"] = False
                logger.error("Dataset recorder init failed: %s", e)
                return {"status": "error", "content": [{"text": f"Dataset init failed: {e}"}]}

    def _collect_recording_schema(
        self,
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        dict[str, tuple[int, int]],
        str,
        list[tuple[str, str, int, int]],
        list[tuple[str, list[str]]],
    ]:
        """Schema from the live scene (joints, actions, cameras, base columns)."""
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
                prefix = f"{rname}__" if multi_robot else ""
                base_state_specs.append((f"{prefix}base_pos", ["x", "y", "z"]))
                base_state_specs.append((f"{prefix}base_quat", ["w", "x", "y", "z"]))
                base_state_specs.append((f"{prefix}base_lin_vel", ["x", "y", "z"]))
                base_state_specs.append((f"{prefix}base_ang_vel", ["x", "y", "z"]))
        camera_keys: list[str] = []
        camera_dims: dict[str, tuple[int, int]] = {}
        recording_cameras: list[tuple[str, str, int, int]] = []
        for cam_name, cfg in self._cameras.items():
            safe = camera_schema_key(cam_name)
            width, height = int(cfg["width"]), int(cfg["height"])
            camera_keys.append(safe)
            camera_dims[safe] = (height, width)
            recording_cameras.append((cam_name, safe, width, height))
        return joint_names, action_names, camera_keys, camera_dims, robot_type, recording_cameras, base_state_specs

    def _make_recording_on_frame(self, robot_name: str, instruction: str) -> Any:
        """``on_frame(step, observation, action)`` that appends world-0 frames to the recorder."""
        from strands_robots.simulation.models import TrajectoryStep

        state = self._recording_state()
        if state is None or not registered(self._robots, robot_name):
            return None
        multi_robot = len(self._robots) > 1
        action_key_cache: dict[bool, list[str]] = {}

        def _required_action_keys(prefixed: bool) -> list[str]:
            cached = action_key_cache.get(prefixed)
            if cached is None:
                keys = self.robot_action_keys(robot_name)
                cached = [f"{robot_name}__{key}" for key in keys] if prefixed else list(keys)
                action_key_cache[prefixed] = cached
            return cached

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
                    images[safe] = self.render(src, width=w, height=h)
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
            if multi_robot:
                obs = undriven_robot_state(self, (robot_name,), self._robots)
                obs.update({f"{robot_name}__{k}": v for k, v in scalars.items()})
                obs.update(images)
                act = {f"{robot_name}__{k}": v for k, v in action.items()}
                rec.add_frame(
                    observation=obs, action=act, task=instruction, required_action_keys=_required_action_keys(True)
                )
            else:
                obs = dict(scalars)
                obs.update(images)
                rec.add_frame(
                    observation=obs, action=action, task=instruction, required_action_keys=_required_action_keys(False)
                )

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
