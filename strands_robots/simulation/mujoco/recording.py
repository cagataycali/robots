"""Recording mixin - start/stop trajectory recording to LeRobotDataset."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


class RecordingMixin:
    """Trajectory recording mixed into ``Simulation``.

    Writes per-step observations + actions + instruction to a LeRobotDataset
    via ``start_recording`` / ``stop_recording`` and the ``on_frame`` hook
    in ``PolicyRunner``. Separately from that, ``start_cameras_recording``
    dumps raw per-camera MP4s.

    **Coupling** (see simulation.py top-level docstring): mixin reaches
    into ``self._world`` (trajectory buffer + dataset_recorder live in
    ``_world._backend_state``). ``TYPE_CHECKING`` stub below exists so mypy
    accepts the ``_world`` lookup; it is a documentary contract, not an
    enforceable protocol.
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: "SimWorld | None"
        default_width: int
        default_height: int

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str | None = None,
        push_to_hub: bool = False,
        vcodec: str = "libsvtav1",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Start recording to LeRobotDataset format (parquet + per-camera MP4).

        Requires the ``lerobot`` extra for the dataset schema. If you only
        need plain MP4 video (no dataset schema, no policy-training metadata),
        use :meth:`start_cameras_recording` - it runs under the
        ``[sim-mujoco]`` extra alone (imageio-ffmpeg backend).

        Raises:
            Friendly error when ``lerobot`` is not installed, directing the
            caller to :meth:`start_cameras_recording` or to install the
            optional extra.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        _DatasetRecorder: Any = None
        _has_lerobot = False
        try:
            from strands_robots.dataset_recorder import DatasetRecorder as _DatasetRecorder
            from strands_robots.dataset_recorder import has_lerobot_dataset as _check_lerobot

            _has_lerobot = _check_lerobot()
        except ImportError:
            pass

        if not _has_lerobot or _DatasetRecorder is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "start_recording produces a LeRobotDataset (parquet + video) and "
                            "requires the lerobot extra. For plain MP4 video under the "
                            "[sim-mujoco] extra alone, use start_cameras_recording instead.\n"
                            "\n"
                            "  - Dataset + policy training data:  pip install 'strands-robots[lerobot]'\n"
                            "  - Plain MP4 only:                  start_cameras_recording(cameras=..., output_dir=...)"
                        )
                    }
                ],
            }

        self._world._backend_state["recording"] = True
        self._world._backend_state["trajectory"] = []
        self._world._backend_state["push_to_hub"] = push_to_hub

        try:
            if overwrite:
                if root:
                    dataset_dir = Path(root)
                elif "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("./"):
                    dataset_dir = Path(repo_id)
                else:
                    dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
                if dataset_dir.exists() and dataset_dir.is_dir():
                    shutil.rmtree(dataset_dir)
                    logger.info("Removed existing dataset dir: %s", dataset_dir)

            # Collect joint names from every robot. When the scene contains
            # more than one robot (e.g. multi-agent dual-task recording), prefix
            # each joint with the robot's instance name (``alice__shoulder_pan``)
            # so the dataset schema has unique joint ids per agent. Single-robot
            # scenes keep the clean ``shoulder_pan`` names for backwards compat.
            joint_names: list[str] = []
            camera_keys: list[str] = []
            robot_type = "unknown"
            multi_robot = len(self._world.robots) > 1
            for rname, robot in self._world.robots.items():
                if multi_robot:
                    joint_names.extend(f"{rname}__{jn}" for jn in robot.joint_names)
                else:
                    joint_names.extend(robot.joint_names)
                robot_type = robot.data_config or rname

            mj = _ensure_mujoco()
            for i in range(self._world._model.ncam):
                cam_name = mj.mj_id2name(self._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
                if not cam_name:
                    continue
                # LeRobot feature names can't contain '/' (reserved for
                # nested-feature addressing). When a robot injects a
                # namespaced camera (e.g. ``arm0/wrist_cam``), collapse
                # the separator to ``__`` for the dataset schema.
                safe_name = cam_name.replace("/", "__")
                camera_keys.append(safe_name)

            assert _DatasetRecorder is not None  # checked above
            self._world._backend_state["dataset_recorder"] = _DatasetRecorder.create(
                repo_id=repo_id,
                fps=fps,
                robot_type=robot_type,
                joint_names=joint_names,
                camera_keys=camera_keys,
                task=task,
                root=root,
                vcodec=vcodec,
                video_width=self.default_width,
                video_height=self.default_height,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Recording to LeRobotDataset: {repo_id}\n"
                            f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                            f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                            f"Run policies to capture frames, then stop_recording to save episode"
                        )
                    }
                ],
            }
        except Exception as e:
            self._world._backend_state["recording"] = False
            logger.error("Dataset recorder init failed: %s", e)
            return {"status": "error", "content": [{"text": f"Dataset init failed: {e}"}]}

    def stop_recording(self, output_path: str | None = None) -> dict[str, Any]:
        """Stop recording and save episode to LeRobotDataset.

        idempotent - calling when not recording succeeds with a
        'Was not recording' message so callers can safely call it unconditionally.
        """
        if self._world is None or not self._world._backend_state.get("recording", False):
            return {"status": "success", "content": [{"text": "Was not recording."}]}

        self._world._backend_state["recording"] = False
        recorder = self._world._backend_state.get("dataset_recorder", None)

        if recorder is None:
            return {"status": "error", "content": [{"text": "No dataset recorder active."}]}

        recorder.save_episode()
        push_result = None
        if self._world._backend_state.get("push_to_hub", False):
            push_result = recorder.push_to_hub(tags=["strands-robots", "sim"])

        repo_id = recorder.repo_id
        frame_count = recorder.frame_count
        episode_count = recorder.episode_count
        root = recorder.root

        recorder.finalize()
        self._world._backend_state["dataset_recorder"] = None
        self._world._backend_state["trajectory"] = []

        text = (
            f"Episode saved to LeRobotDataset\n"
            f"{repo_id} -- {frame_count} frames, {episode_count} episode(s)\n"
            f"Local: {root}"
        )
        if push_result and push_result.get("status") == "success":
            text += "\nPushed to HuggingFace Hub"

        return {"status": "success", "content": [{"text": text}]}

    def get_recording_status(self) -> dict[str, Any]:
        """Returns success in every lifecycle state (no world / not
        recording / recording) with a distinguishing message so callers can
        poll it unconditionally without try/except."""
        if self._world is None:
            return {
                "status": "success",
                "content": [{"text": "⚪ No world - call create_world to start recording."}],
            }

        recording = self._world._backend_state.get("recording", False)
        steps = len(self._world._backend_state.get("trajectory", []))

        if recording:
            text = f"🔴 Recording: {steps} steps captured"
        else:
            text = f"⚪ Not recording (last episode: {steps} steps)"

        return {
            "status": "success",
            "content": [{"text": text}],
        }
