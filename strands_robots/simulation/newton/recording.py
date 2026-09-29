"""Newton recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset`` and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`,
which is backend-agnostic. This subclass adds the Newton-specific parts:

* :meth:`_collect_recording_schema` declares the dataset schema from the live
  Newton scene - joint names from every robot (namespaced for multi-robot
  scenes) and the named cameras registered on the world.
* :meth:`_make_run_policy_hook` returns the ``on_frame`` closure the shared
  :class:`~strands_robots.simulation.base.SimEngine` run-policy loop calls every
  control step. It feeds joint state + action + rendered camera frames to the
  active :class:`~strands_robots.dataset_recorder.DatasetRecorder`.
* :meth:`_release_run_policy_hook` lowers the ``policy_running`` flag the hook
  builder raised, when the rollout that hook served ends. The claim and its
  release are one contract, and only the shared facade knows where the rollout
  ends, so it calls this in a ``finally`` around every rollout it drives.

The recorder, episode-boundary flushing (``save_episode``), and the canonical
parquet-correctness contract are identical to the MuJoCo backend - the
``DatasetRecorder`` is engine-independent, so a Newton recording produces the
same LeRobot v3 dataset layout (``meta/info.json`` + per-episode parquet +
per-camera MP4).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.models import registered
from strands_robots.simulation.recording import (
    DatasetRecordingMixin,
    RecordingSchema,
    floating_base_state_specs,
    undriven_robot_state,
)
from strands_robots.utils import camera_schema_key

if TYPE_CHECKING:
    from strands_robots.simulation.models import SimWorld

logger = logging.getLogger(__name__)


class NewtonRecordingMixin(DatasetRecordingMixin):
    """Newton dataset recording mixed into :class:`NewtonSimEngine`.

    Inherits the engine-independent lifecycle from
    :class:`DatasetRecordingMixin` and supplies the Newton-specific schema
    declaration (:meth:`_collect_recording_schema`) and per-step capture hook
    (:meth:`_make_run_policy_hook`).
    """

    if TYPE_CHECKING:
        _world: SimWorld | None
        _model: Any
        _robot_free_base_joint: dict[str, str]
        default_width: int
        default_height: int

        def render(self, camera_name: str = ..., width: int | None = ..., height: int | None = ...) -> dict[str, Any]:
            """Type-only stub for the engine-provided render method."""

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Type-only stub for the engine-provided action-key accessor."""
            return []

    _RECORDING_VIDEO_HINT = "For plain MP4 video, pass video={'path': ...} to run_policy instead."
    _RECORDING_REPLY_LABEL = "Recording Newton scene to LeRobotDataset"

    def _recording_start_error(self, state: dict[str, Any] | None) -> dict[str, Any] | None:
        """Refuse without a finalized Newton model."""
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        return None

    def _recording_scene_cameras(self) -> list[str]:
        """The named cameras registered on ``world.cameras``."""
        assert self._world is not None
        return list(self._world.cameras)

    def _collect_recording_schema(self, probe: Any = None) -> RecordingSchema:
        """Build the dataset schema from the live Newton scene.

        Joint names come from every robot (namespaced ``robot__joint`` when more
        than one robot exists) and cameras from ``world.cameras`` at their real
        render resolutions. Action columns are taken from
        :meth:`robot_action_keys` - the same authority ``send_action`` and the
        recording hook resolve, so a declared column is always a key
        ``add_frame`` can receive.
        """
        world = self._world
        assert world is not None  # guarded by start_recording
        joint_names: list[str] = []
        action_names: list[str] = []
        base_state_specs: list[tuple[str, list[str]]] = []
        robot_type = "unknown"
        multi_robot = len(world.robots) > 1
        free_base = getattr(self, "_robot_free_base_joint", {})
        for rname, robot in world.robots.items():
            # Exclude the floating base's free joint from the scalar joint
            # schema: its 6-DoF state is recorded as the structured base_*
            # columns below and get_observation no longer emits it as a scalar,
            # so a floating_base_joint scalar column would be dead/degenerate.
            # Mirrors get_observation / get_robot_state.
            free_short = free_base.get(rname)
            scalar_jn = [jn for jn in robot.joint_names if jn != free_short]
            # Action columns come from ``robot_action_keys``, not from the
            # scalar joint list computed above, even though the two agree for
            # every robot this backend can build today. They agree only because
            # both apply the free-base exclusion, and each applied it from its
            # own copy of the rule; declaring the action schema from the joint
            # list makes that agreement load-bearing. A column declared under a
            # name the hook never emits is not a mismatch the recorder can
            # report - ``add_frame`` reads the action dict by declared name, so
            # an unmatched column takes the ``0.0`` fill and the episode records
            # a command nobody issued under a success result (#1715).
            act_keys = self.robot_action_keys(rname)
            if multi_robot:
                joint_names.extend(f"{rname}__{jn}" for jn in scalar_jn)
                action_names.extend(f"{rname}__{ak}" for ak in act_keys)
            else:
                joint_names.extend(scalar_jn)
                action_names.extend(act_keys)
            robot_type = robot.data_config or rname
            if free_short:
                base_state_specs.extend(floating_base_state_specs(f"{rname}__" if multi_robot else ""))

        recording_cameras: list[tuple[str, str, int, int]] = []
        for cam_name, cam in world.cameras.items():
            safe_name = camera_schema_key(cam_name)
            width = int(getattr(cam, "width", self.default_width))
            height = int(getattr(cam, "height", self.default_height))
            recording_cameras.append((cam_name, safe_name, width, height))
        return RecordingSchema(
            joint_names,
            action_names,
            base_state_specs,
            recording_cameras,
            robot_type,
            (self.default_width, self.default_height),
        )

    def _make_recording_on_frame(self, robot_name: str, instruction: str) -> Any:
        """The recording half of the per-step ``on_frame`` hook for Newton.

        Returns an ``on_frame(step, observation, action)`` closure that, while a
        recording session is active, augments the joint-state observation with a
        rendered frame for each declared camera and forwards the frame to the
        active :class:`DatasetRecorder`. In multi-robot scenes scalar
        observation/action keys are namespaced (``robot__joint``) to match the
        schema declared in :meth:`start_recording`; camera ndarrays keep their
        sanitized names.

        Returns ``None`` when there is no world or the robot is unknown. No
        rollout claim is made here: :meth:`_make_run_policy_hook` layers that
        on top, and the evaluation facades (``eval_policy``,
        ``evaluate_benchmark``) install this hook alone when a recording is
        open and the caller passed no ``on_frame``.
        """
        from strands_robots.simulation.policy_runner import _extract_frame_ndarray

        world = self._world
        if world is None or not registered(world.robots, robot_name):
            return None

        multi_robot = len(world.robots) > 1

        # Action columns this rollout is responsible for: the driven robot's own
        # actuators. A declared column the policy never produced cannot be written
        # as a placeholder without persisting a command nobody issued, so
        # ``add_frame`` refuses it.
        #
        # Resolved on the first recorded frame and cached, rather than up front:
        # ``robot_action_keys`` is explicitly best-effort for the runner's
        # fail-fast probe (a backend quirk or a mid-rollout teardown may make it
        # raise, and that must not mask the primary "robot has not moved" signal),
        # so the hook must not call it for a rollout that is not recording. Where a
        # recording IS attached the keys are load-bearing - without them the frame
        # cannot be checked - so a raise there correctly fails the recording.
        action_key_cache: dict[bool, list[str]] = {}

        def _required_action_keys(prefixed: bool) -> list[str]:
            """Action columns this frame owes the recorder, resolved once."""
            cached = action_key_cache.get(prefixed)
            if cached is None:
                keys = self.robot_action_keys(robot_name)
                cached = [f"{robot_name}__{key}" for key in keys] if prefixed else list(keys)
                action_key_cache[prefixed] = cached
            return cached

        def _record(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            if not world._backend_state.get("recording", False):
                return
            rec = world._backend_state.get("dataset_recorder")
            if rec is None:
                return

            obs: dict[str, Any] = dict(observation)
            for source_name, safe_name, width, height in world._backend_state.get("recording_cameras", []):
                render_result = self.render(camera_name=source_name, width=width, height=height)
                img = _extract_frame_ndarray(render_result)
                if img is not None:
                    obs[safe_name] = img

            if multi_robot:
                import numpy as np

                # The schema declares a state column for every robot in the
                # scene, and this frame carries only the driven robot's. An
                # undriven robot's columns are a readable measurement, so they
                # are filled from the engine at this step rather than left to
                # add_frame's 0.0 fill, which records them as a zero pose the
                # robot is not in. Driven keys win any collision.
                driven = {(k if isinstance(v, np.ndarray) else f"{robot_name}__{k}"): v for k, v in obs.items()}
                obs = undriven_robot_state(self, (robot_name,), world.robots)
                obs.update(driven)
                act = {f"{robot_name}__{k}": v for k, v in action.items()}
                rec.add_frame(
                    observation=obs,
                    action=act,
                    task=instruction,
                    required_action_keys=_required_action_keys(True),
                )
            else:
                rec.add_frame(
                    observation=obs,
                    action=action,
                    task=instruction,
                    required_action_keys=_required_action_keys(False),
                )

        return _record

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Build the per-step ``on_frame`` hook for a rollout: claim + recording.

        Marks the robot as driven (``policy_running`` / ``policy_instruction`` /
        ``policy_steps``, released by :meth:`_release_run_policy_hook`) and
        forwards every frame to :meth:`_make_recording_on_frame`. ``None``
        when there is no world or the robot is unknown, so the base
        run-policy loop runs without recording.
        """
        world = self._world
        if world is None or not registered(world.robots, robot_name):
            return None
        robot = world.robots[robot_name]
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

        Nothing lowered it, so a recorded rollout left the robot marked as
        driven for the rest of the session. On this backend the flag is what
        :meth:`~strands_robots.simulation.models.SimRobot.request_policy_stop`
        reports as ``was_running``, and that answer is the whole verdict every
        stop path here reports - :meth:`_request_policy_stop` hands it to
        :meth:`~strands_robots.simulation.base.SimEngine.stop_policy`, and the
        Device Connect ``stop`` RPC reads that - so a flag left raised made an
        idle simulation report a halted rollout that had finished on its own.

        Args:
            robot_name: The robot whose rollout has ended. A robot removed
                mid-rollout, or a world torn down under it, has nothing to
                release.
        """
        world = self._world
        if world is not None and registered(world.robots, robot_name):
            world.robots[robot_name].policy_running = False
