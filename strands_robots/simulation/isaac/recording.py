"""Isaac recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset`` and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`,
which is backend-agnostic. This subclass adds the Isaac-specific parts:

* :meth:`_collect_recording_schema` declares the dataset schema from the live Isaac
  scene - joint names from every robot (namespaced for multi-robot scenes) and
  the RTX cameras registered via ``add_camera``, each at the resolution its
  render product actually produces (probed from one ``get_observation`` call,
  because RTX cameras render at a DLSS-safe native size that can differ from
  the requested output size).
* :meth:`_make_run_policy_hook` returns the ``on_frame`` closure the shared
  :class:`~strands_robots.simulation.base.SimEngine` run-policy loop calls
  every control step. Unlike the MuJoCo/Newton hooks it does not render:
  Isaac's ``get_observation`` already carries a fresh RGB frame per camera
  (refreshing every RTX render product first when more than one camera exists,
  so multi-cam recordings never capture a stale secondary product), and
  ``IsaacSimulation.get_observation`` forces images on while a recording is
  active even when the driving policy sets ``requires_images = False``.
* :meth:`_release_run_policy_hook` lowers the ``policy_running`` flag the hook
  builder raised, when the rollout that hook served ends. The claim and its
  release are one contract, and only the shared facade knows where the rollout
  ends, so it calls this in a ``finally`` around every rollout it drives.

**State seam**: Isaac's ``self._world`` is the Isaac Sim ``World`` handle, not
the :class:`~strands_robots.simulation.models.SimWorld` the shared mixin's
default accessor expects, so :meth:`_recording_state` overrides the seam to
return the engine-owned ``self._recording_state_dict`` instead. Everything
else in the shared lifecycle runs unchanged.

**Pacing**: the recorded ``fps`` is dataset metadata; the actual frame cadence
is ``run_policy(control_frequency=...)`` (one recorded frame per control
step). The RTX renderer produces new frames at ``IsaacConfig.rendering_dt``
(default 1/30 s), so a control frequency above ``1 / rendering_dt`` records
duplicate frames from the same render product. For distinct per-step images
keep ``control_frequency <= 1 / rendering_dt`` and set ``fps`` to match the
control frequency.

**Threading**: schema declaration probes ``get_observation`` once. When the
main-thread pump (:meth:`IsaacSimulation.run_pump_forever`) owns the renderer
and ``start_recording`` is called from a worker thread (a Gradio-style host),
that probe is routed through :meth:`IsaacSimulation.run_on_main` so the RTX
render-product refresh never runs off the owning thread. Per-step capture
rides the ``run_policy`` thread's own ``get_observation`` calls, which hosts
already route via ``run_on_main`` when driving rollouts from worker threads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.models import registered, registry_entry
from strands_robots.simulation.recording import (
    DatasetRecordingMixin,
    RecordedFrame,
    RecordingSchema,
    floating_base_state_specs,
)
from strands_robots.utils import camera_schema_key

if TYPE_CHECKING:
    import threading

    from strands_robots.simulation.isaac.config import IsaacConfig

logger = logging.getLogger(__name__)


class IsaacRecordingMixin(DatasetRecordingMixin):
    """Isaac dataset recording mixed into :class:`IsaacSimulation`.

    Inherits the engine-independent lifecycle from
    :class:`DatasetRecordingMixin` and supplies the Isaac-specific state seam
    (:meth:`_recording_state`), schema declaration (:meth:`_collect_recording_schema`)
    and per-step capture hook (:meth:`_make_run_policy_hook`).
    """

    if TYPE_CHECKING:
        _world_created: bool
        _robots: dict[str, Any]
        _cameras: dict[str, Any]
        _config: IsaacConfig
        _lock: threading.RLock
        _sim_time: float
        _pump_running: bool
        _recording_state_dict: dict[str, Any]

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Actuator-ordered action keys (concrete on SimEngine)."""

        def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
            """Type-only stub for the engine-provided observation method."""

        def run_on_main(self, fn: Any, timeout: float | None = None) -> Any:
            """Type-only stub for the engine-provided main-thread executor."""

        def _on_main_thread(self) -> bool:
            """Type-only stub for the engine-provided thread check."""

    def _recording_state(self) -> dict[str, Any] | None:
        """Engine-owned recording-state dict (the Isaac state seam).

        Overrides :meth:`DatasetRecordingMixin._recording_state`: Isaac's
        ``self._world`` is the Isaac Sim ``World`` handle (no
        ``_backend_state``), so the recording flag / trajectory mirror /
        recorder handle live in ``self._recording_state_dict`` instead
        (initialised in ``IsaacSimulation.__init__`` and reset by
        ``destroy``). Returns ``None`` before ``create_world`` so the shared
        lifecycle reports the documented no-world responses.
        """
        if not getattr(self, "_world_created", False):
            return None
        return self._recording_state_dict

    _RECORDING_REPLY_LABEL = "Recording Isaac scene to LeRobotDataset"

    def _recording_start_error(self, state: dict[str, Any] | None) -> dict[str, Any] | None:
        """Refuse without a world, and without a robot to probe the cameras through."""
        if state is None:
            return {"status": "error", "content": [{"text": "No world created. Call create_world() first."}]}
        if not self._robots:
            return {
                "status": "error",
                "content": [{"text": "No robots in the world. Call add_robot() before start_recording()."}],
            }
        return None

    def _recording_scene_cameras(self) -> list[str]:
        """The RTX cameras registered via ``add_camera`` - the scene truth, which
        ``render_mode='headless'`` does not change even though it records none."""
        return list(self._cameras)

    def _recording_unknown_cameras_refusal(self, unknown: list[str]) -> dict[str, Any] | None:
        """Name ``render_mode`` when the requested cameras exist but record nothing.

        In ``render_mode="headless"`` the schema probe returns no images, so a
        camera just added with ``add_camera`` is absent from the recordable set
        and the generic refusal told the caller to add it again.
        """
        if getattr(self._config, "render_mode", None) != "headless":
            return None
        if not set(unknown) <= set(self._recording_scene_cameras()):
            return None
        from strands_robots.simulation.isaac.simulation import _HEADLESS_RENDER_REMEDY

        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"start_recording: camera(s) {unknown} are in the scene but record no "
                        f"frames: {_HEADLESS_RENDER_REMEDY}."
                    )
                }
            ],
        }

    def _probe_recording_scene(self) -> dict[str, Any]:
        """One observation, taken BEFORE the start lock: routed to the pump
        thread, holding ``self._lock`` across that handoff would deadlock."""
        return self._probe_recording_observation()

    def _recording_start_lock(self) -> threading.RLock:
        """The engine lock, held while the session is armed."""
        return self._lock

    def _probe_recording_observation(self) -> dict[str, Any]:
        """One ``get_observation`` probe used to size the camera schema.

        RTX cameras render at a DLSS-safe native resolution that can exceed
        the requested output size (see ``add_camera``), and ``get_observation``
        returns frames at that native size - so the schema must declare the
        shape the observation stream actually produces, not the requested one.
        Routed through :meth:`run_on_main` when the main-thread pump owns the
        renderer and the caller is a worker thread (Gradio-style hosts), since
        the multi-camera render-product refresh may only run on the owning
        thread. In headless render mode the probe returns no images and the
        schema falls back to proprio-only.
        """
        first_robot = next(iter(self._robots))
        if getattr(self, "_pump_running", False) and not self._on_main_thread():
            return self.run_on_main(lambda: self.get_observation(robot_name=first_robot))
        return self.get_observation(robot_name=first_robot)

    def _collect_recording_schema(self, probe: Any = None) -> RecordingSchema:
        """Build the dataset schema from the live Isaac scene.

        Args:
            probe: One observation from :meth:`_probe_recording_observation`,
                used to size each camera at the resolution its render product
                actually emits.

        A floating-base robot (``fixed_base`` False, the field the MJCF
        free-joint read records - this backend has no compiled model to ask)
        declares the :func:`~strands_robots.simulation.recording.floating_base_state_specs`
        columns; a fixed-base arm leaves its schema unchanged.
        """
        probe_obs = probe or {}
        joint_names: list[str] = []
        action_names: list[str] = []
        robot_type = "unknown"
        multi_robot = len(self._robots) > 1
        base_state_specs: list[tuple[str, list[str]]] = []
        for rname, robot in self._robots.items():
            if multi_robot:
                joint_names.extend(f"{rname}__{jn}" for jn in robot.joint_names)
                action_names.extend(f"{rname}__{ak}" for ak in self.robot_action_keys(rname))
            else:
                joint_names.extend(robot.joint_names)
                action_names.extend(self.robot_action_keys(rname))
            robot_type = getattr(robot, "data_config", None) or rname
            if not getattr(robot, "fixed_base", True):
                base_state_specs.extend(floating_base_state_specs(f"{rname}__" if multi_robot else ""))

        recording_cameras: list[tuple[str, str, int, int]] = []
        video_size = (int(self._config.camera_width), int(self._config.camera_height))
        if self._config.render_mode == "headless" and self._cameras:
            # get_observation never emits frames in headless render mode, so a
            # camera column would record nothing. Warn loudly and record a
            # proprio-only dataset rather than silently declaring dead columns.
            logger.warning(
                "start_recording: %d camera(s) %s are registered but render_mode='headless' "
                "produces no RTX frames - recording a proprio-only dataset. Construct the "
                "sim with render_mode='rtx_realtime' to record camera columns.",
                len(self._cameras),
                sorted(self._cameras),
            )
            return RecordingSchema(
                joint_names, action_names, base_state_specs, recording_cameras, robot_type, video_size
            )

        for cam_name, cam in self._cameras.items():
            safe_name = camera_schema_key(cam_name)
            frame = probe_obs.get(cam_name)
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                height, width = int(frame.shape[0]), int(frame.shape[1])
            else:
                # Probe frame unavailable (render product not warmed yet).
                # Fall back to the camera's native render size, which is what
                # get_observation emits once the product accumulates a frame.
                width, height = int(cam.width), int(cam.height)
                logger.warning(
                    "start_recording: probe observation carried no frame for camera %r; "
                    "declaring its native render size %dx%d. If recorded frames arrive at "
                    "a different resolution, add_frame will reject them - step the sim a "
                    "few times before start_recording to warm the render product.",
                    cam_name,
                    width,
                    height,
                )
            recording_cameras.append((cam_name, safe_name, width, height))
        return RecordingSchema(joint_names, action_names, base_state_specs, recording_cameras, robot_type, video_size)

    def _make_recording_on_frame(self, robot_name: str, instruction: str) -> Any:
        """The recording half of the per-step ``on_frame`` hook for Isaac.

        Returns an ``on_frame(step, observation, action)`` closure that, while
        a recording session is active, appends a step to the trajectory mirror
        and forwards the frame to the active
        :class:`~strands_robots.dataset_recorder.DatasetRecorder`. Camera
        frames are consumed from the observation itself (Isaac's
        ``get_observation`` carries a fresh RGB frame per camera and refreshes
        every RTX render product when more than one camera exists, so
        multi-cam recordings never duplicate a stale secondary product) and
        renamed to their schema-safe names; cameras outside the
        ``start_recording(cameras=...)`` scope are dropped. In multi-robot
        scenes scalar observation/action keys are namespaced
        (``robot__joint``) to match the declared schema.

        Returns ``None`` when there is no world or the robot is unknown. No
        rollout claim is made here: :meth:`_make_run_policy_hook` layers that
        on top, and the evaluation facades (``eval_policy``,
        ``evaluate_benchmark``) install this hook alone when a recording is
        open and the caller passed no ``on_frame``.
        """
        import time

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

            # Split the observation: camera ndarrays are renamed raw -> safe
            # and scoped to the declared recording cameras; scalars feed
            # observation.state. Resolved per step (not captured at hook build
            # time) so a start_recording issued after run_policy launched
            # still scopes correctly.
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

            if _opens_on_an_unrendered_frame(rec, state, images):
                return

            state["trajectory"].append(
                TrajectoryStep(
                    timestamp=time.time(),
                    sim_time=self._sim_time,
                    robot_name=robot_name,
                    observation=scalars,
                    action=action,
                    instruction=instruction,
                )
            )

            frame.write(rec, {robot_name: scalars}, {robot_name: action}, images, instruction)
            state["unrendered_skips"] = 0

        return _record

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Build the per-step ``on_frame`` hook for a rollout: claim + recording.

        Marks the robot as driven (``policy_running`` / ``policy_instruction`` /
        ``policy_steps``, released by :meth:`_release_run_policy_hook`) and
        forwards every frame to :meth:`_make_recording_on_frame`. ``None``
        when there is no world or the robot is unknown, so the base
        run-policy loop runs without recording.
        """
        state = self._recording_state()
        if state is None or not registered(self._robots, robot_name):
            return None
        robot = self._robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0
        record_frame = self._make_recording_on_frame(robot_name, instruction)

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            robot.policy_steps = step + 1
            if record_frame is not None:
                record_frame(step, observation, action)

        return _hook

    def _release_run_policy_hook(self, robot_name: str) -> None:
        """Lower the ``policy_running`` flag :meth:`_make_run_policy_hook` raised.

        The hook builder marks the robot as driven so a motion primitive and
        the rollout cannot race on the same articulation's PD targets, and
        :meth:`~strands_robots.simulation.isaac.motion_primitives.IsaacMotionPrimitivesMixin._primitive_resolve_robot`
        refuses on exactly that flag. Nothing lowered it, so a recorded rollout
        left the robot marked driven for the rest of the session: every
        primitive answered ``Cannot 'set_gripper' on 'so100' while its policy
        is running ... wait for the rollout to finish``, and
        :meth:`~strands_robots.simulation.isaac.simulation.IsaacSimulation.run_multi_policy`
        answered ``policy already running ... Stop it first`` - two remedies
        for a rollout that had already ended, and at the time neither was even
        reachable, because Isaac exposed no ``stop_policy``. It inherits one now
        (:meth:`~strands_robots.simulation.base.SimEngine.stop_policy`), which
        on this backend states why it cannot help - the per-robot record here
        carries a bare ``policy_running`` flag and not the durable claim
        ``SimRobot.request_policy_stop`` writes - rather than raising
        ``AttributeError``. The release below is what makes the remedy
        unnecessary in the first place.

        ``run_multi_policy`` lowers the flag in its own ``finally`` for the
        loop it owns; this is the same release for the rollout the shared
        facade owns, so the flag means "a loop is driving this robot" on both
        paths. ``policy_instruction`` / ``policy_steps`` stay as the last
        rollout's record, as they do on the MuJoCo backend.

        Args:
            robot_name: The robot whose rollout has ended. A robot removed
                mid-rollout has nothing to release.
        """
        robot = registry_entry(self._robots, robot_name)
        if robot is not None:
            robot.policy_running = False


#: Frames an episode may skip at its start while a recorded camera has not
#: rendered yet. Measured on one L40S (Isaac Sim 6.1): an so101 wrist camera
#: returned all-zero frames for the first two rollout steps of every episode,
#: and render-only ticks did not light it - only a physics step did.
_MAX_UNRENDERED_SKIPS = 8


def _opens_on_an_unrendered_frame(rec: Any, state: dict[str, Any], images: dict[str, Any]) -> bool:
    """Whether this frame would open the episode with a camera that has not rendered.

    An episode's first frames used to be written with an all-zero (black) image
    for such a camera, so a policy trained on the dataset saw black inputs at
    every episode start (MuJoCo: never). Until the episode has its first frame,
    a frame whose declared camera image is missing or all zeros is skipped - up
    to :data:`_MAX_UNRENDERED_SKIPS` in a row, after which it is written as it
    is (a camera that really sees black). A frame after the episode's first is
    always written, black or not.
    """
    if int(getattr(rec, "episode_frame_count", 1) or 0) > 0:
        return False
    declared = [safe for _src, safe, _w, _h in state.get("recording_cameras", [])]
    if not declared:
        return False
    unrendered = [
        name
        for name in declared
        if name not in images or not np.asarray(images[name]).size or not np.any(np.asarray(images[name])[..., :3])
    ]
    if not unrendered:
        return False
    skips = int(state.get("unrendered_skips", 0) or 0)
    if skips >= _MAX_UNRENDERED_SKIPS:
        logger.warning(
            "recording: camera(s) %s still black after %d skipped frames at the episode start; recording them as black",
            unrendered,
            skips,
        )
        return False
    state["unrendered_skips"] = skips + 1
    logger.debug("recording: skipped an episode-opening frame, camera(s) %s not rendered yet", unrendered)
    return True
