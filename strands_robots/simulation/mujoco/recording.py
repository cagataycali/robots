"""Recording mixin — start/stop trajectory recording to LeRobotDataset."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


class RecordingMixin:
    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: "SimWorld | None"

    """Trajectory recording for Simulation. Expects self._world."""

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
        """Start recording to LeRobotDataset format (parquet + video)."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world."}]}

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
                        "text": "lerobot not installed. Install with: pip install lerobot\nRequired for dataset recording."
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

            joint_names = []
            camera_keys = []
            robot_type = "unknown"
            for rname, robot in self._world.robots.items():
                joint_names.extend(robot.joint_names)
                robot_type = robot.data_config or rname

            mj = _ensure_mujoco()
            for i in range(self._world._model.ncam):
                cam_name = mj.mj_id2name(self._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
                if cam_name:
                    camera_keys.append(cam_name)

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
        """Stop recording and save episode to LeRobotDataset."""
        if self._world is None or not self._world._backend_state.get("recording", False):
            return {"status": "error", "content": [{"text": "Not recording."}]}

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
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}

        recording = self._world._backend_state.get("recording", False)
        steps = len(self._world._backend_state.get("trajectory", []))

        return {
            "status": "success",
            "content": [{"text": f"{'🔴 Recording' if recording else '⚪ Not recording'}: {steps} steps captured"}],
        }
