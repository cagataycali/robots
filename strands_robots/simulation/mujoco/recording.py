"""MuJoCo recording mixin - schema declaration + raw camera MP4 capture.

The engine-independent recording lifecycle (stop/save/status/stream and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`.
and so does ``start_recording``; this subclass supplies the MuJoCo schema
(joints and cameras enumerated from the live ``MjModel``) and the offscreen
render refusal.
"""

import logging
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.models import registry_entry
from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _can_render, _ensure_mujoco, mj_name_to_id
from strands_robots.simulation.recording import DatasetRecordingMixin, RecordingSchema, floating_base_state_specs
from strands_robots.utils import camera_schema_key

logger = logging.getLogger(__name__)


class RecordingMixin(DatasetRecordingMixin):
    """MuJoCo trajectory recording mixed into ``Simulation``.

    Inherits the engine-independent lifecycle from
    :class:`DatasetRecordingMixin` and adds the MuJoCo schema declaration:
    :meth:`_collect_recording_schema` reads the live ``MjModel`` to enumerate
    joints and cameras (with their real render resolutions). Per-step frames are fed by the ``on_frame`` hook built in
    :mod:`simulation`. Separately, ``start_cameras_recording`` dumps raw
    per-camera MP4s.

    **Coupling** (see the :mod:`simulation` top-level docstring): mixin reaches
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

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Actuator-ordered action keys for ``robot_name`` (concrete on SimEngine)."""

        def _robot_free_base_joint_id(self, model: Any, robot: Any) -> int:
            """Free-base joint id for ``robot`` or -1 (concrete on RenderingMixin)."""

    _RECORDING_VIDEO_HINT = (
        "For plain MP4 video under the [sim-mujoco] extra alone, use "
        "start_cameras_recording(cameras=..., output_dir=...) instead."
    )
    _RECORDING_REPLY_TAIL = (
        "Frames are captured by a policy rollout - run_policy (one rollout; "
        "it closes NO episode, so call reset between rollouts or pass "
        "n_episodes=N in one call, else consecutive rollouts merge into one "
        "episode), start_policy (async), eval_policy / evaluate_benchmark "
        "(one dataset episode per evaluation episode) or run_multi_policy "
        "(several robots into one merged frame) - or by stepping a scripted motion: "
        "set_joint_positions(hold=True) + step records one frame per 1/{fps}s "
        "of sim time, and teleoperate steps the world the same way, so a "
        "teleoperated session records too. replay_episode does not feed the "
        "recorder. Then stop_recording to save the open episode"
    )

    def _recording_start_error(self, state: dict[str, Any] | None) -> dict[str, Any] | None:
        """Refuse without a compiled model: the schema is read from ``MjModel``."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        return None

    def _recording_scene_cameras(self) -> list[str]:
        """Every ``MjModel`` camera name, in model order."""
        assert self._world is not None
        mj, model = _ensure_mujoco(), self._world._model
        return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]

    def _collect_recording_schema(self, probe: Any = None) -> RecordingSchema:
        """The dataset schema declared from the live ``MjModel``.

        Action columns are keyed by the ACTUATORS the rollout loops emit
        (``robot_action_keys``), not the joint names: the two diverge for a
        passive/mimic joint or a tendon-driven gripper. A floating base's free
        joint is excluded from the scalar joints - its state is recorded as the
        structured ``base_*`` columns. Each camera is declared at the size it
        renders at (its ``add_camera`` width/height, else the sim default).
        """
        world = self._world
        assert world is not None
        mj, model = _ensure_mujoco(), world._model
        joint_names: list[str] = []
        action_names: list[str] = []
        base_state_specs: list[tuple[str, list[str]]] = []
        robot_type = "unknown"
        multi_robot = len(world.robots) > 1
        for rname, robot in world.robots.items():
            pfx = robot.namespace or ""
            scalar_joint_names: list[str] = []
            for jn in robot.joint_names:
                jid = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, (pfx + jn) if pfx else jn)
                if jid < 0 and pfx:
                    jid = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0 and model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                    continue
                scalar_joint_names.append(jn)
            prefix = f"{rname}__" if multi_robot else ""
            joint_names.extend(f"{prefix}{jn}" for jn in scalar_joint_names)
            action_names.extend(f"{prefix}{ak}" for ak in self.robot_action_keys(rname))
            robot_type = robot.data_config or rname
            if self._robot_free_base_joint_id(model, robot) >= 0:
                base_state_specs.extend(floating_base_state_specs(prefix))

        cameras: list[tuple[str, str, int, int]] = []
        for cam_name in self._recording_scene_cameras():
            if not cam_name:
                continue
            safe_name = camera_schema_key(cam_name)
            info = registry_entry(world.cameras, cam_name) or registry_entry(world.cameras, safe_name)
            if info is not None:
                cameras.append((cam_name, safe_name, int(info.width), int(info.height)))
            else:
                cameras.append((cam_name, safe_name, int(self.default_width), int(self.default_height)))
        return RecordingSchema(
            joint_names,
            action_names,
            base_state_specs,
            cameras,
            robot_type,
            (self.default_width, self.default_height),
        )

    def _recording_cameras_scope(
        self, cameras: list[tuple[str, str, int, int]], selected: set[str] | None
    ) -> set[str] | None:
        """The scoped RAW camera names the frame hook keeps (``None``: all)."""
        return selected

    def _recording_cameras_refusal(self, camera_keys: list[str], cameras: list[str] | None) -> dict[str, Any] | None:
        """Refuse camera columns no frame can carry, and warn on the ``default`` view.

        ``get_observation`` skips every camera frame when offscreen rendering is
        unavailable (headless without EGL/OSMesa), so a declared camera column
        would make the first ``add_frame`` fail. On the record-all path the
        implicit ``default`` overview camera is swept in beside real sensors - a
        view no policy declares - so it is recorded, but not silently.
        """
        if camera_keys and not _can_render():
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"start_recording: {len(camera_keys)} camera(s) {camera_keys} would be "
                            "declared in the dataset schema, but MuJoCo offscreen rendering is "
                            "unavailable on this machine (headless without libEGL.so.1 / "
                            "libOSMesa.so), so no frame will carry them and the first add_frame "
                            "would fail. Pass cameras=[] to record joint state and actions only, "
                            "or install an offscreen GL library (libegl1 / libosmesa6) and "
                            "restart to record camera frames."
                        )
                    }
                ],
            }
        if cameras is None and "default" in camera_keys and len(camera_keys) > 1:
            sensor_cams = [c for c in camera_keys if c != "default"]
            logger.warning(
                "start_recording: recording the implicit 'default' overview "
                "camera into observation.images.default alongside %d sensor "
                "camera(s) %s. The 'default' view is not a sensor any policy "
                "declares; it bloats the dataset and will not match a policy's "
                "input_features. Pass cameras=%r to record only your sensors.",
                len(sensor_cams),
                sensor_cams,
                sensor_cams,
            )
        return None
