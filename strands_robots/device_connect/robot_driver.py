"""RobotDeviceDriver - Device Connect DeviceDriver adapter wrapping a strands-robots Robot.

Exposes the Robot's task execution, status, and observation methods as
structured RPCs and events via Device Connect's DeviceDriver interface.
"""

import asyncio
import logging
from typing import Any

from device_connect_edge.drivers import (
    DeviceDriver,
    emit,
    get_rpc_source_device,
    on,
    periodic,
    rpc,
)
from device_connect_edge.types import DeviceIdentity, DeviceStatus

from strands_robots.bus_access import read_joints
from strands_robots.device_connect._authz import authz_error, is_authorized_caller
from strands_robots.mesh.security import is_safe_policy_provider

logger = logging.getLogger(__name__)


class RobotDeviceDriver(DeviceDriver):
    """Device Connect device driver wrapping a strands-robots Robot instance."""

    device_type = "strands_robot"

    def __init__(self, robot):
        super().__init__()
        self._robot = robot

    @property
    def identity(self) -> DeviceIdentity:
        """Static Device Connect identity for the wrapped robot.

        Returns a :class:`~device_connect_edge.types.DeviceIdentity` reporting
        ``device_type="strands_robot"``, the ``strands-robots`` manufacturer, and
        the robot's ``tool_name_str`` as the model (falling back to ``"robot"``).
        """
        return DeviceIdentity(
            device_type="strands_robot",
            manufacturer="strands-robots",
            model=getattr(self._robot, "tool_name_str", "robot"),
            description="Strands Robots LeRobot-based robot arm",
        )

    @property
    def status(self) -> DeviceStatus:
        """Live availability of the wrapped robot.

        Returns a :class:`~device_connect_edge.types.DeviceStatus` that is
        ``"busy"`` (``busy_score`` 1.0) while a task is running and ``"idle"``
        (``busy_score`` 0.0) otherwise, derived from the robot's task state.
        """
        task = getattr(self._robot, "_task_state", None)
        is_busy = task is not None and hasattr(task, "status") and getattr(task.status, "value", "idle") == "running"
        return DeviceStatus(
            availability="busy" if is_busy else "idle",
            busy_score=1.0 if is_busy else 0.0,
        )

    async def connect(self) -> None:
        """No-op - the Robot manages its own hardware connection."""
        pass

    async def disconnect(self) -> None:
        """No-op - the Robot manages its own hardware shutdown."""
        pass

    # ── RPCs ──────────────────────────────────────────────────

    @rpc()
    async def execute(
        self,
        instruction: str,
        policy_provider: str = "mock",
        duration: float = 30.0,
        policy_port: int = 0,
    ) -> dict[str, Any]:
        """Execute a VLA task instruction on the robot.

        Args:
            instruction: Natural language task instruction
            policy_provider: Policy backend (groot, mock, lerobot_local, ...)
            duration: Maximum task duration in seconds
            policy_port: Policy server port (0 for default)
        """
        # Security hardening: authorize the calling device before mutating
        # physical robot state.
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "execute")

        # Security hardening: restrict policy_provider to the vetted allowlist
        # so a caller cannot steer inference to an arbitrary network endpoint.
        if not is_safe_policy_provider(policy_provider):
            return {"status": "error", "reason": f"policy_provider not allowed: {policy_provider!r}"}

        # Call by keyword: HardwareRobot.start_task is
        # (instruction, policy_port, policy_host, policy_provider, duration).
        # A positional call here silently misroutes the arguments (the provider
        # string lands in policy_port, the port in policy_host, and "localhost"
        # in policy_provider), so bind them explicitly to their target fields.
        return self._robot.start_task(
            instruction,
            policy_port=policy_port or None,
            policy_host="localhost",
            policy_provider=policy_provider,
            duration=duration,
        )

    @rpc()
    async def stop(self) -> dict[str, Any]:
        """Stop the currently running task."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "stop")
        return self._robot.stop_task()

    @rpc()
    async def getStatus(self) -> dict[str, Any]:
        """Get current task execution status."""
        return self._robot.get_task_status()

    @rpc()
    async def getFeatures(self) -> dict[str, Any]:
        """Get robot observation and action features."""
        get_features = getattr(self._robot, "get_features", None)
        if callable(get_features):
            return get_features()
        # Main's HardwareRobot does not expose get_features(); degrade gracefully.
        return {"features": {}, "note": "get_features unavailable on this robot"}

    @rpc()
    async def getState(self) -> dict[str, Any]:
        """Get current robot state (joints, task info).

        Returns joint positions and task state if a task is running.

        The joints are read through the shared motor-bus lock, so this RPC
        waits its turn behind an in-flight rollout, teleop write or mesh probe
        instead of colliding with it, and reports the joints even when a
        camera on the same driver is failing.
        """
        result = {}
        task = getattr(self._robot, "_task_state", None)
        if task:
            result["task_status"] = getattr(task.status, "value", "unknown")
            result["instruction"] = task.instruction
            result["step_count"] = task.step_count

        # Read the joint half through the shared motor-bus lock
        # (:func:`strands_robots.bus_access.read_joints`) rather than calling the
        # driver's ``get_observation`` directly. Both failures that replaces were
        # reported as an absent ``joints`` key under a successful RPC, because the
        # handler below logs at debug and returns what it has. A read that barged in
        # on one of the five converted readers already holding the bus collided
        # ("Port is in use!"), and lerobot's ``get_observation`` sync-reads the
        # motors FIRST and only then loops the cameras, so one dead USB camera threw
        # away the joint positions already in hand. The frame filter below still
        # matters: a driver exposing no readable motor bus falls back to the full
        # observation, frames included.
        inner = getattr(self._robot, "robot", None)
        if inner and hasattr(inner, "get_observation"):
            try:
                obs = await asyncio.to_thread(read_joints, inner)
                # Filter out camera frames (numpy arrays) - only include scalars
                result["joints"] = {k: float(v) for k, v in obs.items() if not hasattr(v, "shape")}
            except Exception as e:
                logger.debug("Could not read observation: %s", e)

        return result

    # ── Events ────────────────────────────────────────────────

    @emit()
    async def taskStarted(self, instruction: str, policy_provider: str):
        """Emitted when a VLA task begins execution.

        Args:
            instruction: The task instruction
            policy_provider: The policy backend used
        """
        pass

    @emit()
    async def taskComplete(self, instruction: str, steps: int, duration: float):
        """Emitted when a VLA task finishes.

        Args:
            instruction: The task instruction
            steps: Total steps executed
            duration: Total execution time in seconds
        """
        pass

    @emit()
    async def streamStep(self, step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
        """Emitted for each VLA inference step (high frequency).

        Args:
            step: Step number
            observation: Observation dict (joints only, no camera frames)
            action: Action dict
        """
        pass

    @emit()
    async def emergencyStop(self, reason: str = ""):
        """Emitted when this device triggers an emergency stop.

        Args:
            reason: Why the emergency stop was triggered
        """
        pass

    @on(event_name="emergencyStop")
    async def onEmergencyStop(self, device_id: str, event_name: str, payload: dict[str, Any]) -> None:
        """React to emergencyStop from an authorized safety controller.

        Security hardening: only act on emergency-stop events whose source is
        in the emergency-stop allowlist, so a spoofed event from an arbitrary
        device cannot interrupt operations.
        """
        if not is_authorized_caller(device_id, scope="estop"):
            logger.warning("Ignoring emergencyStop from unauthorized source %s", device_id)
            return
        logger.warning("Emergency stop received from %s - stopping task", device_id)
        self._robot.stop_task()

    # ── Periodic state publishing ─────────────────────────────

    @periodic(interval=0.1, wait_for_completion=True)
    async def _publishState(self):
        """Publish robot state at 10Hz."""
        task = getattr(self._robot, "_task_state", None)
        if task and getattr(task.status, "value", "idle") == "running":
            await self.stateUpdate(
                task_status="running",
                instruction=task.instruction,
                step_count=task.step_count,
            )

    @emit()
    async def stateUpdate(self, task_status: str = "", instruction: str = "", step_count: int = 0):
        """Periodic state update.

        Args:
            task_status: Current task status
            instruction: Current task instruction
            step_count: Steps completed so far
        """
        pass
