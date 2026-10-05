"""Native driver for the Boston Dynamics Spot (with or without the arm), over ``bosdyn-client``.

``Robot("spot", mode="real", port="192.168.80.3")`` builds one of these.
lerobot registers no Spot robot type, so before this driver the robot was
simulation-only and ``mode="real"`` refused it by name.

Spot is driven the way Boston Dynamics' own examples drive it: authenticate,
time-sync, hold the body lease with a ``LeaseKeepAlive``, and send
``RobotCommand`` messages through the robot command service. The legs belong to
Spot's locomotion controller, so this driver commands what the public API
commands:

* the base, as a body-frame twist (:meth:`SpotDriver.set_twist`), sent with a
  short end time so the robot stops on its own when commands stop arriving;
* the arm's six joints and the claw (:meth:`SpotDriver.send_action`), named as
  the MuJoCo asset's actuators are (``arm_sh0`` ... ``arm_wr1``, ``arm_f1x``) and
  in the same radians the SDK uses;
* standing up (:meth:`SpotDriver.stand`), which powers the motors on first.

It refuses before it writes: any command while the robot is E-stopped (an
E-Stop endpoint must be registered and released, by the tablet or
``estop_gui``) or its motors are off; a leg joint, which no public command
reaches; an arm command on a Spot without an arm; a joint target outside the
arm's published limits; and a non-finite value. ``cleanup`` stops motion, then
powers the motors off safely (the robot sits first) and returns the lease.

Credentials come from ``BOSDYN_CLIENT_USERNAME`` and ``BOSDYN_CLIENT_PASSWORD``,
the variables the SDK's own ``bosdyn.client.util.authenticate`` reads. The SDK
is imported inside :meth:`SpotDriver.connect_eagerly` only, so the module
imports on any machine and every test runs against a fake robot.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
import types
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from strands_robots.drivers.base import policy_step, refuse, undeclared_verb_error
from strands_robots.drivers.rollout import PolicyRollout, policy_from_provider
from strands_robots.utils import finite_number_error, positive_count_error, positive_finite_number_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver serves, read by ``strands_robots.drivers._SHIPPED_DRIVERS``.
SUPPORTED_ROBOTS: tuple[str, ...] = ("spot",)

#: Leg joints, read only: Spot's locomotion controller owns them.
LEG_JOINTS: tuple[str, ...] = tuple(f"{leg}_{part}" for leg in ("fl", "fr", "hl", "hr") for part in ("hx", "hy", "kn"))
#: Arm joints in ``RobotCommandBuilder.arm_joint_command`` argument order.
ARM_JOINTS: tuple[str, ...] = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
#: The claw's finger joint; ``claw_gripper_open_angle_command`` takes the same radians.
GRIPPER_JOINT = "arm_f1x"
#: Every joint the driver reads, named as the MuJoCo asset's actuators are.
JOINT_NAMES: tuple[str, ...] = LEG_JOINTS + ARM_JOINTS + (GRIPPER_JOINT,)

#: Joint limits, radians: Boston Dynamics' published arm limits (the asset carries the same ranges).
#: The claw runs from -1.5708 (fully open) to 0 (closed).
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "arm_sh0": (-2.61799, 3.14159),
    "arm_sh1": (-3.14159, 0.523599),
    "arm_el0": (0.0, 3.14159),
    "arm_el1": (-2.79253, 2.79253),
    "arm_wr0": (-1.8326, 1.8326),
    "arm_wr1": (-2.87979, 2.87979),
    GRIPPER_JOINT: (-1.5708, 0.0),
}

#: Seconds a twist stays in force; the robot stops when it expires without a new one.
VELOCITY_COMMAND_DURATION: float = 0.6
#: Default rollout cadence, hertz.
DEFAULT_CONTROL_FREQUENCY: float = 10.0
#: Seconds to wait for power-on, stand and safe power-off.
POWER_TIMEOUT: float = 20.0


def sdk_joint_name(name: str) -> str:
    """The robot-state name of an asset joint: ``fl_hx`` -> ``fl.hx``, ``arm_sh0`` -> ``arm0.sh0``."""
    prefix, part = name.split("_", 1)
    return f"{'arm0' if prefix == 'arm' else prefix}.{part}"


def _resolve_sdk() -> Any:
    """Return the ``bosdyn.client`` modules the driver uses, or a reason naming the install."""
    try:
        modules = {
            key: importlib.import_module(f"bosdyn.client{suffix}")
            for key, suffix in (
                ("client", ""),
                ("command", ".robot_command"),
                ("lease", ".lease"),
                ("state", ".robot_state"),
                ("exceptions", ".exceptions"),
            )
        }
    except ImportError as exc:
        return f"the Spot SDK is not importable ({exc}). Install it with: pip install 'strands-robots[spot]'"
    return types.SimpleNamespace(**modules)


class SpotDriver:
    """Native driver for the Boston Dynamics Spot.

    Args:
        tool_name: Name the agent invokes the driver by.
        cameras: Accepted for the factory contract; this driver opens none.
        data_config: Accepted for the factory contract; unused.
        port: The robot's hostname or IP (``192.168.80.3`` on its own Wi-Fi).
        control_frequency: Rollout cadence in hertz.

    Raises:
        ValueError: If ``control_frequency`` is not a positive finite number.
    """

    def __init__(
        self,
        tool_name: str = "spot",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
    ) -> None:
        del cameras, data_config
        if reason := positive_finite_number_error(control_frequency, "control_frequency", "SpotDriver"):
            raise ValueError(reason)
        self._tool_name = tool_name
        self._host = str(port) if port else ""
        self._control_frequency = float(control_frequency)
        self._lock = threading.Lock()
        self._sdk: Any = None
        self._robot: Any = None
        self._state_client: Any = None
        self._command_client: Any = None
        self._keepalive: Any = None
        self._has_arm = False
        self._errors: tuple[type[BaseException], ...] = (OSError, RuntimeError)
        self._connect_error: str | None = None
        self._joints: dict[str, float] = {}
        self._halt_epoch = 0
        self._rollout: PolicyRollout | None = None
        self._task_admission = threading.Lock()

    # Agent tool surface.

    @property
    def tool_name(self) -> str:
        """The name the agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Tool kind reported to the agent runtime."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether the driver is authenticated and holds the body lease."""
        return self._robot is not None

    @property
    def tool_spec(self) -> ToolSpec:
        """Read state, report status, stop. Motion goes through ``send_action``, ``set_twist`` and ``stand``."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Boston Dynamics Spot native driver: reads the leg, arm and claw joints, reports E-stop, "
                    "motor power, lease and battery, and stops motion."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joint positions and velocities; "
                                    "status: E-stop, power, lease and battery; stop: halt the robot"
                                ),
                                "enum": ["state", "status", "stop"],
                                "default": "state",
                            }
                        },
                        "required": ["action"],
                    }
                },
            },
        )

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result."""
        del kwargs, invocation_state
        action = (tool_use.get("input") or {}).get("action", "state")
        if action == "state":
            envelope = self.state()
        elif action == "status":
            envelope = await self.get_status()
        elif action == "stop":
            envelope = self.stop_task()
        else:
            envelope = undeclared_verb_error(self, action)
        yield {"toolUseId": tool_use.get("toolUseId", ""), **envelope}

    # Lifecycle.

    def connect_eagerly(self) -> str | None:
        """Authenticate, time-sync and take the body lease; the motors are left as they are.

        Returns:
            ``None`` once connected, or a reason; the driver stays usable
            either way (reads report the reason, writes refuse).
        """
        if self.is_connected:
            return None
        if not self._host:
            self._connect_error = (
                "SpotDriver: no robot address - pass port='<hostname or IP>' (192.168.80.3 on its Wi-Fi)"
            )
            return self._connect_error
        username = os.environ.get("BOSDYN_CLIENT_USERNAME")
        password = os.environ.get("BOSDYN_CLIENT_PASSWORD")
        if not username or not password:
            self._connect_error = (
                "SpotDriver: set BOSDYN_CLIENT_USERNAME and BOSDYN_CLIENT_PASSWORD to the robot's credentials"
            )
            return self._connect_error
        sdk = _resolve_sdk()
        if isinstance(sdk, str):
            self._connect_error = sdk
            return sdk
        errors = (OSError, RuntimeError, sdk.exceptions.Error)
        robot = None
        try:
            robot = sdk.client.create_standard_sdk("strands-robots").create_robot(self._host)
            robot.authenticate(username, password)
            robot.time_sync.wait_for_sync()
            state_client = robot.ensure_client(sdk.state.RobotStateClient.default_service_name)
            command_client = robot.ensure_client(sdk.command.RobotCommandClient.default_service_name)
            lease_client = robot.ensure_client(sdk.lease.LeaseClient.default_service_name)
            has_arm = bool(robot.has_arm())
            keepalive = sdk.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
        except sdk.lease.ResourceAlreadyClaimedError:
            self._stop_time_sync(robot)
            self._connect_error = (
                f"SpotDriver: another client holds {self._host}'s body lease (the tablet, usually); "
                "release control there and connect again"
            )
            return self._connect_error
        except errors as exc:
            self._stop_time_sync(robot)
            self._connect_error = f"SpotDriver: could not connect to {self._host}: {type(exc).__name__}: {exc}"
            return self._connect_error
        with self._lock:
            self._sdk, self._errors, self._has_arm = sdk, errors, has_arm
            self._state_client, self._command_client, self._keepalive = state_client, command_client, keepalive
            self._robot = robot
        self._connect_error = None
        self.state()
        return None

    @staticmethod
    def _stop_time_sync(robot: Any) -> None:
        if robot is not None:
            try:
                robot.time_sync.stop()
            except (AttributeError, RuntimeError) as exc:
                logger.debug("SpotDriver: time_sync.stop() raised %s", exc)

    def _ready(self, verb: str, robot: Any) -> str | None:
        """The reason ``verb`` may not command the robot now, or ``None``."""
        try:
            if robot.is_estopped():
                return (
                    f"{verb}: the robot is E-stopped; register and release an E-Stop endpoint "
                    "(the tablet or estop_gui) before commanding"
                )
            if not robot.is_powered_on():
                return f"{verb}: the motors are off; call stand() to power on and stand up"
        except self._errors as exc:
            return f"{verb}: the robot's state could not be read: {exc}"
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the link, E-stop, motor power, lease, battery and arm."""
        robot = self._robot
        estopped = powered = battery = lease = None
        errors: tuple[type[BaseException], ...] = (*self._errors, AttributeError, IndexError)
        if robot is not None:
            try:
                estopped = bool(robot.is_estopped())
                powered = bool(robot.is_powered_on())
                lease = bool(self._keepalive.is_alive())
                batteries = self._state_client.get_robot_state().battery_states
                battery = float(batteries[0].charge_percentage.value) if batteries else None
            except errors as exc:
                logger.debug("SpotDriver.get_status(): %s", exc)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "estopped": estopped,
                        "powered_on": powered,
                        "lease_held": lease,
                        "battery_pct": battery,
                        "has_arm": self._has_arm,
                        "task_running": self._rollout is not None and self._rollout.is_running,
                    }
                }
            ],
        }

    def _halt(self, robot: Any) -> str | None:
        """Send Spot's stop command (minimal-motion stop in place) when the motors are on."""
        self._begin_halt()
        try:
            if robot.is_powered_on():
                self._command_client.robot_command(self._sdk.command.RobotCommandBuilder.stop_command())
        except self._errors as exc:
            return f"the halt failed: {exc}"
        return None

    async def stop(self) -> None:
        """Halt motion, leaving the motors on and the lease held."""
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        robot = self._robot
        if robot is not None and (reason := self._halt(robot)) is not None:
            logger.warning("%s.stop(): %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Stop the rollout, halt, power off safely (the robot sits first) and return the lease. Idempotent."""
        self._begin_halt()
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
            rollout.join()
        with self._lock:
            robot, self._robot = self._robot, None
        if robot is None:
            return
        if (reason := self._halt(robot)) is not None:
            logger.debug("SpotDriver.cleanup(): %s", reason)
        try:
            if robot.is_powered_on():
                robot.power_off(cut_immediately=False, timeout_sec=POWER_TIMEOUT)
        except self._errors as exc:
            logger.warning("SpotDriver.cleanup(): safe power-off failed: %s", exc)
        self._keepalive.shutdown()
        self._stop_time_sync(robot)

    # Reads.

    def _read(self) -> tuple[dict[str, float], dict[str, float], str | None]:
        errors: tuple[type[BaseException], ...] = (*self._errors, AttributeError, TypeError, ValueError)
        try:
            measured = self._state_client.get_robot_state().kinematic_state.joint_states
            by_name = {joint.name: joint for joint in measured}
            present = [name for name in JOINT_NAMES if sdk_joint_name(name) in by_name]
            joints = {name: float(by_name[sdk_joint_name(name)].position.value) for name in present}
            velocities = {name: float(by_name[sdk_joint_name(name)].velocity.value) for name in present}
        except errors as exc:
            return {}, {}, f"the robot's state read failed: {exc}"
        with self._lock:
            self._joints = dict(joints)
        return joints, velocities, None

    def state(self) -> dict[str, Any]:
        """Read every joint the robot reports, in asset names and radians.

        Returns:
            ``joints`` and ``joint_velocities`` (the legs, and the arm and claw
            when fitted), or a refusal naming the failed read.
        """
        if self._robot is None:
            return refuse(f"state: not connected - {self._connect_error or 'call connect_eagerly() first'}")
        joints, velocities, reason = self._read()
        if reason is not None:
            return refuse(f"state: {reason}")
        return {
            "status": "success",
            "content": [{"json": {"robot": self._tool_name, "joints": joints, "joint_velocities": velocities}}],
        }

    def get_observation(self) -> dict[str, float]:
        """The measured joint positions (name -> rad), for the mesh and the rollout."""
        if self._robot is not None:
            self._read()
        with self._lock:
            return dict(self._joints)

    # Command path.

    def _begin_halt(self) -> None:
        with self._lock:
            self._halt_epoch += 1

    def _targets(self, action: Any) -> tuple[dict[str, float], str | None]:
        """Validate an arm/claw action against the joint table and the arm's limits."""
        if not isinstance(action, dict) or not action:
            return {}, f"nothing to command - the action names none of {[*ARM_JOINTS, GRIPPER_JOINT]}"
        legs = sorted(set(action) & set(LEG_JOINTS))
        if legs:
            return {}, (
                f"{legs} are leg joints, which Spot's locomotion controller owns; "
                "drive the base with set_twist(vx, vy, wz)"
            )
        unknown = sorted(set(action) - set(JOINT_LIMITS))
        if unknown:
            return {}, f"{unknown} name no Spot joint; expected any of {list(JOINT_LIMITS)}"
        if not self._has_arm:
            return {}, "this Spot has no arm"
        targets: dict[str, float] = {}
        for name, value in action.items():
            if (reason := finite_number_error(value, name, "send_action")) is not None:
                return {}, reason
            lower, upper = JOINT_LIMITS[name]
            if not lower <= float(value) <= upper:
                return {}, f"{name}={float(value):.4f} rad is outside the arm's limits [{lower:.4f}, {upper:.4f}]"
            targets[name] = float(value)
        return targets, None

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Move the arm's joints and the claw to the given angles in one command.

        An arm joint the action omits keeps its measured angle; the robot plans
        the move with its own velocity and acceleration limits.

        Args:
            action: Angles in radians keyed by :data:`ARM_JOINTS` and
                :data:`GRIPPER_JOINT` (-1.5708 fully open, 0 closed).
            robot_name: ``None`` or this driver's own name.

        Returns:
            A success envelope carrying the targets sent, or a refusal.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            robot, halted_at = self._robot, self._halt_epoch
        if robot is None:
            return refuse("send_action: not connected - call connect_eagerly() first")
        targets, reason = self._targets(action)
        if reason is not None:
            return refuse(f"send_action: {reason}")
        if (reason := self._ready("send_action", robot)) is not None:
            return refuse(reason)
        builder = self._sdk.command.RobotCommandBuilder
        command = None
        if set(targets) & set(ARM_JOINTS):
            measured, _, reason = self._read()
            if reason is not None:
                return refuse(f"send_action: {reason}")
            command = builder.arm_joint_command(*(targets.get(name, measured[name]) for name in ARM_JOINTS))
        if GRIPPER_JOINT in targets:
            command = builder.claw_gripper_open_angle_command(targets[GRIPPER_JOINT], build_on_command=command)
        with self._lock:
            superseded = self._halt_epoch != halted_at
        if superseded:
            return refuse("send_action: the robot was halted while this command was being prepared; not sent")
        try:
            self._command_client.robot_command(command)
        except self._errors as exc:
            return refuse(f"send_action: the command failed: {exc}")
        return {"status": "success", "content": [{"json": {"robot": self._tool_name, "joints": targets}}]}

    def set_twist(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> dict[str, Any]:
        """Walk at a body-frame twist for :data:`VELOCITY_COMMAND_DURATION` seconds.

        The command carries an end time, so Spot stops when twists stop
        arriving; re-send it to keep walking. Spot applies its own speed limits.

        Args:
            vx: Forward velocity, m/s.
            vy: Left velocity, m/s.
            wz: Yaw rate, rad/s, positive left.

        Returns:
            A success envelope carrying the twist, or a refusal.
        """
        for value, name in ((vx, "vx"), (vy, "vy"), (wz, "wz")):
            if (reason := finite_number_error(value, name, "set_twist")) is not None:
                return refuse(reason)
        robot = self._robot
        if robot is None:
            return refuse("set_twist: not connected - call connect_eagerly() first")
        if (reason := self._ready("set_twist", robot)) is not None:
            return refuse(reason)
        command = self._sdk.command.RobotCommandBuilder.synchro_velocity_command(
            v_x=float(vx), v_y=float(vy), v_rot=float(wz)
        )
        try:
            self._command_client.robot_command(command, end_time_secs=time.time() + VELOCITY_COMMAND_DURATION)
        except self._errors as exc:
            return refuse(f"set_twist: the command failed: {exc}")
        twist = {"vx": float(vx), "vy": float(vy), "wz": float(wz)}
        return {"status": "success", "content": [{"json": {"robot": self._tool_name, "twist": twist}}]}

    def stand(self) -> dict[str, Any]:
        """Power the motors on if they are off, then stand (``blocking_stand``).

        Returns:
            A success envelope, or a refusal naming why the robot did not stand.
        """
        robot = self._robot
        if robot is None:
            return refuse("stand: not connected - call connect_eagerly() first")
        try:
            if robot.is_estopped():
                return refuse("stand: the robot is E-stopped; release the E-Stop endpoint first")
            if not robot.is_powered_on():
                robot.power_on(timeout_sec=POWER_TIMEOUT)
            self._sdk.command.blocking_stand(self._command_client, timeout_sec=POWER_TIMEOUT)
        except self._errors as exc:
            return refuse(f"stand: {type(exc).__name__}: {exc}")
        return {"status": "success", "content": [{"json": {"robot": self._tool_name, "standing": True}}]}

    # Task and policy path.

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "lerobot_local",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Build a policy from the provider registry and roll it out (see :meth:`run_policy`)."""
        kwargs: dict[str, Any] = {"host": policy_host, **policy_kwargs}
        if policy_port is not None:
            kwargs["port"] = policy_port
        policy, reason = policy_from_provider(
            policy_provider,
            kwargs,
            "start_task",
            "the rollout would start on a live robot and fail at its first action",
            self.get_observation,
        )
        if reason is not None:
            return refuse(reason)
        return self.run_policy(policy, instruction=instruction, duration=duration)

    def run_policy(
        self,
        policy_object: Policy | Callable[[dict[str, Any]], dict[str, Any]],
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Roll a built policy out on a background thread; poll :meth:`get_task_status`.

        Every step writes through :meth:`send_action`, so an E-stop mid-rollout
        ends it with that refusal as the exit reason.
        """
        if err := positive_finite_number_error(duration, "duration", "run_policy"):
            return refuse(err)
        if n_steps is not None and (err := positive_count_error(n_steps, "n_steps", "run_policy")):
            return refuse(err)
        if policy_object is None or not policy_step(policy_object, instruction):
            return refuse("run_policy: policy_object must be callable or expose get_actions_sync() or step()")
        if not self.is_connected:
            return refuse("run_policy: not connected - call connect_eagerly() first")
        rollout = PolicyRollout(
            name=f"spot-rollout-{self._tool_name}",
            policy=policy_object,
            instruction=instruction,
            duration=float(duration),
            n_steps=n_steps,
            period=1.0 / self._control_frequency,
            observe=self.get_observation,
            act=self.send_action,
        )
        with self._task_admission:
            if self._rollout is not None and self._rollout.is_running:
                return refuse("run_policy: a task is already running; call stop_task first")
            self._rollout = rollout
            rollout.start()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "instruction": instruction,
                        "control_frequency": self._control_frequency,
                        "duration": duration,
                        "n_steps": n_steps,
                    }
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
        """Report the rollout's progress, or that none has run."""
        rollout = self._rollout
        if rollout is None:
            return {"status": "success", "content": [{"json": {"running": False, "steps": 0}}]}
        return {"status": "success", "content": [{"json": rollout.snapshot()}]}

    def stop_task(self) -> dict[str, Any]:
        """Stop the rollout and halt the robot.

        Returns:
            Success when the rollout left its loop and the robot took the halt;
            a refusal naming the failure otherwise; an error carrying
            ``stopped=False`` when the rollout thread did not join.
        """
        self._begin_halt()
        rollout = self._rollout
        joined = True
        if rollout is not None and rollout.is_running:
            rollout.request_stop()
            joined = rollout.join()
        robot = self._robot
        if robot is None:
            return refuse("stop_task: not connected")
        if (reason := self._halt(robot)) is not None:
            return refuse(f"stop_task: {reason}")
        steps = 0 if rollout is None else rollout.steps
        if not joined and rollout is not None:
            snapshot = {**rollout.snapshot(), "stopped": False, "robot": self._tool_name}
            snapshot["reason"] = (
                "stop_task: the rollout thread did not join; the robot is halted, the task still holds it"
            )
            return {"status": "error", "content": [{"json": snapshot}]}
        return {"status": "success", "content": [{"json": {"stopped": True, "steps": steps, "robot": self._tool_name}}]}
