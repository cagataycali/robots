"""Native driver for the Rainbow Robotics RB-Y1 (model A), over the vendor's ``rby1-sdk``.

``Robot("rby1", mode="real", port="192.168.30.1:50051")`` builds one of these.
lerobot registers no RB-Y1 robot type, so before this driver the robot was
simulation-only and ``mode="real"`` refused it by name.

The robot's own computer runs the control manager and speaks gRPC;
``rby1_sdk.create_robot_a`` is the vendor's client for it. This driver follows
the SDK's documented bring-up - ``connect()``, ``power_on(".*")``,
``servo_on(".*")``, ``enable_control_manager()`` - and streams each action as
one joint-position command for the body (torso and both arms, 20 joints) and
the head (2 joints) through a command stream.

Three things are gated here because they decide whether a write is safe to send:

* the emergency stop and the control manager - a pressed e-stop or a control
  manager in ``MinorFault``/``MajorFault`` refuses the write and names the
  state (clearing a fault with ``reset_fault_control_manager()`` is an
  operator's decision, not this driver's);
* joint range and step size, both read from the robot's own dynamics model
  (``get_dynamics()``, the URDF the robot serves) at connect: a target outside
  ``[q_lower, q_upper]`` is refused, and so is a joint asked to travel further
  in one control period than its ``qdot_upper`` allows. The step is measured
  from the last *commanded* setpoint while a stream is in progress, for the
  reason :meth:`~strands_robots.drivers.ur.URDriver._reference_pose` gives;
* a halt that lands while a setpoint is being prepared - a counter bumped by
  every halt verb is re-read just before the write.

Deliberately absent: the wheels and the grippers. The base is velocity
commanded (``MobilityCommandBuilder``), not a joint target, and the grippers
are Dynamixel servos on the tool flanges outside the 24-joint model, so an
action naming either is refused rather than mapped by guess. Every read still
reports all 24 joints the robot measures, wheels included.

``rby1_sdk`` is imported inside :meth:`RBY1Driver.connect_eagerly` only, so the
module imports on any machine and every test runs against a fake robot.
"""

from __future__ import annotations

import importlib
import logging
import threading
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
SUPPORTED_ROBOTS: tuple[str, ...] = ("rby1",)

#: The joints this driver commands, in the SDK's body-then-head order. The
#: MuJoCo asset names its joints the same way, so an action recorded in
#: simulation needs no remap.
BODY_JOINTS: tuple[str, ...] = (
    *(f"torso_{index}" for index in range(6)),
    *(f"right_arm_{index}" for index in range(7)),
    *(f"left_arm_{index}" for index in range(7)),
)
HEAD_JOINTS: tuple[str, ...] = ("head_0", "head_1")
JOINT_NAMES: tuple[str, ...] = BODY_JOINTS + HEAD_JOINTS

#: Joints the robot measures but this driver does not command.
WHEEL_JOINTS: tuple[str, ...] = ("right_wheel", "left_wheel")

#: Default rollout cadence, hertz. Also the period the step gate sizes against
#: and each streamed command's ``minimum_time``.
DEFAULT_CONTROL_FREQUENCY: float = 50.0

#: How long the control manager keeps tracking a streamed command after it
#: stops arriving, seconds. Short, so a rollout that dies stops the robot soon.
CONTROL_HOLD_TIME: float = 1.0

#: ``send_command`` timeout, milliseconds.
COMMAND_TIMEOUT_MS: int = 1000

#: Control manager states that execute no command.
FAULT_STATES: frozenset[str] = frozenset({"MinorFault", "MajorFault"})


def _resolve_sdk() -> Any:
    """Return the ``rby1_sdk`` module, or a reason naming the install."""
    try:
        return importlib.import_module("rby1_sdk")
    except ImportError as exc:
        return f"the RB-Y1 SDK is not importable ({exc}). Install it with: pip install 'strands-robots[rby1]'"


def _enum_name(value: Any) -> str:
    """The member name of a vendor enum value (``State.Enabled`` -> ``"Enabled"``)."""
    return str(getattr(value, "name", value))


def targets_from_action(
    action: dict[str, Any],
    reference: list[float],
    *,
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    max_step: list[float] | None = None,
) -> tuple[list[float], str | None]:
    """Turn a joint-name-keyed action into one ordered 22-joint setpoint.

    A joint the action omits holds its reference value: a component command
    carries every joint of the component, and a zero there would fold the arm.

    Args:
        action: Joint targets in radians keyed by :data:`JOINT_NAMES` members.
        reference: The pose the step is measured from, in :data:`JOINT_NAMES` order.
        lower: Per-joint lower position limit, radians. ``None`` skips the range gate.
        upper: Per-joint upper position limit, radians.
        max_step: Largest travel each joint may be asked for, radians. ``None``
            skips the step gate.

    Returns:
        ``(targets, None)`` or ``([], reason)``.
    """
    if len(reference) != len(JOINT_NAMES):
        return [], f"reference holds {len(reference)} joint positions, expected {len(JOINT_NAMES)}"
    if not isinstance(action, dict) or not action:
        return [], f"nothing to command - the action names none of {list(JOINT_NAMES)}"
    unknown = sorted(set(action) - set(JOINT_NAMES))
    if unknown:
        return [], (
            f"{unknown} name no commanded RB-Y1 joint; expected any of {list(JOINT_NAMES)}. "
            "The wheels and the grippers are not driven by this driver."
        )
    targets = list(reference)
    for index, name in enumerate(JOINT_NAMES):
        if name not in action:
            continue
        if (reason := finite_number_error(action[name], name, "send_action")) is not None:
            return [], reason
        target = float(action[name])
        if lower is not None and upper is not None and not lower[index] <= target <= upper[index]:
            return [], f"{name}={target:.4f} rad is outside the robot's range [{lower[index]:.4f}, {upper[index]:.4f}]"
        step = abs(target - reference[index])
        if max_step is not None and step > max_step[index]:
            return [], (
                f"{name} asks for {step:.4f} rad in one control period, more than the {max_step[index]:.4f} rad "
                "its velocity limit allows. Slow the policy or lower control_frequency."
            )
        targets[index] = target
    return targets, None


class RBY1Driver:
    """Native driver for the Rainbow Robotics RB-Y1 (model A).

    Args:
        tool_name: Name the agent invokes the driver by.
        cameras: Accepted for the factory contract; this driver opens none.
        data_config: Accepted for the factory contract; unused.
        port: The robot's gRPC address, ``"<ip>:<port>"``.
        control_frequency: Rollout cadence in hertz, and the period the step
            gate sizes a setpoint against.
        priority: Command-stream priority; a higher-priority client preempts.

    Raises:
        ValueError: If ``control_frequency`` is not a positive finite number or
            ``priority`` is not a positive count.
    """

    def __init__(
        self,
        tool_name: str = "rby1",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
        priority: int = 1,
    ) -> None:
        del cameras, data_config
        if reason := positive_finite_number_error(control_frequency, "control_frequency", "RBY1Driver"):
            raise ValueError(reason)
        if reason := positive_count_error(priority, "priority", "RBY1Driver"):
            raise ValueError(reason)
        self._tool_name = tool_name
        self._address = str(port) if port else ""
        self._control_frequency = float(control_frequency)
        self._priority = int(priority)
        self._lock = threading.Lock()
        self._sdk: Any = None
        self._robot: Any = None
        self._stream: Any = None
        self._measured_names: list[str] = []
        self._connect_error: str | None = None
        self._lower: list[float] | None = None
        self._upper: list[float] | None = None
        self._max_step: list[float] | None = None
        self._joints: dict[str, float] = {}
        self._commanded: list[float] | None = None
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
        """Whether the gRPC link to the robot is live."""
        robot = self._robot
        if robot is None:
            return False
        try:
            return bool(robot.is_connected())
        except (OSError, RuntimeError):
            return False

    @property
    def tool_spec(self) -> ToolSpec:
        """Read state, report status, stop. Motion goes through ``send_action``."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Rainbow RB-Y1 native driver: reads the 24 joint positions, velocities and torques, "
                    "reports the control manager, e-stop and battery, and stops motion."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joints, velocities and torques; "
                                    "status: link, control manager, e-stop and battery; stop: halt the robot"
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
        """Connect, refuse a pressed e-stop or a faulted control manager, and enable control.

        The vendor's bring-up: ``connect()``, ``power_on(".*")``,
        ``servo_on(".*")``, ``enable_control_manager()``. The e-stop and the
        control manager are checked before anything is powered, so connecting
        never energises a robot in a fault. The range and velocity limits the
        gates use are read from ``get_dynamics()`` here.

        Returns:
            ``None`` once control is enabled, or a reason; the driver stays
            usable either way (reads report the reason, writes refuse).
        """
        if self.is_connected:
            return None
        if not self._address:
            self._connect_error = (
                'RBY1Driver: no robot address - pass port="<ip>:<port>", e.g. '
                'Robot("rby1", mode="real", port="192.168.30.1:50051")'
            )
            return self._connect_error
        sdk = _resolve_sdk()
        if isinstance(sdk, str):
            self._connect_error = sdk
            return sdk
        try:
            robot = sdk.create_robot_a(self._address)
            connected = robot.connect()
        except (OSError, RuntimeError) as exc:
            self._connect_error = f"RBY1Driver: robot at {self._address!r} did not answer: {exc}"
            return self._connect_error
        if not connected:
            self._connect_error = f"RBY1Driver: robot at {self._address!r} did not answer"
            return self._connect_error
        reason = self._bring_up(robot)
        if reason is not None:
            self._release(robot)
            self._connect_error = f"RBY1Driver: {reason}"
            return self._connect_error
        with self._lock:
            self._sdk, self._robot, self._stream, self._commanded = sdk, robot, None, None
        self._connect_error = None
        self.state()
        return None

    def _bring_up(self, robot: Any) -> str | None:
        """Gate, power, servo and enable; read the limits. Returns a reason or ``None``."""
        try:
            if (reason := self._fault_refusal(robot)) is not None:
                return reason
            for call, args in (("power_on", (".*",)), ("servo_on", (".*",)), ("enable_control_manager", ())):
                if not getattr(robot, call)(*args):
                    return f"{call}{args} returned False"
            manager = _enum_name(robot.get_control_manager_state().state)
            if manager != "Enabled":
                return f"the control manager is {manager} after enable_control_manager()"
            names = list(robot.model().robot_joint_names)
            missing = sorted(set(JOINT_NAMES + WHEEL_JOINTS) - set(names))
            if missing:
                return f"the robot's model has no joints {missing}; this driver serves the RB-Y1 model A"
            dynamics = robot.get_dynamics()
            limits = dynamics.make_state(["base"], list(JOINT_NAMES))
            width = len(JOINT_NAMES)
            # The limit vectors follow the state's joint order, then append the
            # joints the state left out (the wheels), so the head is ours.
            lower = [float(value) for value in dynamics.get_limit_q_lower(limits)][:width]
            upper = [float(value) for value in dynamics.get_limit_q_upper(limits)][:width]
            speed = [float(value) for value in dynamics.get_limit_qdot_upper(limits)][:width]
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            return f"bring-up failed: {exc}"
        if min(len(lower), len(upper), len(speed)) != width:
            return f"the dynamics model reported limits for fewer than {width} joints"
        period = 1.0 / self._control_frequency
        with self._lock:
            self._measured_names = names
            self._lower, self._upper = lower, upper
            self._max_step = [value * period for value in speed]
        return None

    @staticmethod
    def _release(robot: Any) -> None:
        try:
            robot.disconnect()
        except (OSError, RuntimeError) as exc:
            logger.debug("RBY1Driver: disconnect raised %s", exc)

    def _fault_refusal(self, robot: Any) -> str | None:
        """Name a pressed e-stop or a faulted control manager, or ``None`` when clear."""
        try:
            pressed = [
                index for index, emo in enumerate(robot.get_state().emo_states) if _enum_name(emo.state) == "Pressed"
            ]
            manager = _enum_name(robot.get_control_manager_state().state)
        except (OSError, RuntimeError, AttributeError) as exc:
            return f"the robot's state read failed: {exc}"
        if pressed:
            return f"robot at {self._address!r} has its emergency stop pressed (EMO {pressed}); release it first"
        if manager in FAULT_STATES:
            return (
                f"robot at {self._address!r} has its control manager in {manager}. Clear it with "
                "reset_fault_control_manager() once the cause is fixed; a command now moves nothing."
            )
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the link, the control manager, the e-stops and the battery."""
        robot = self._robot
        manager = emo = battery = None
        if robot is not None:
            try:
                manager = _enum_name(robot.get_control_manager_state().state)
                state = robot.get_state()
                emo = [_enum_name(item.state) for item in state.emo_states]
                battery = float(state.battery_state.level_percent)
            except (OSError, RuntimeError, AttributeError) as exc:
                logger.debug("RBY1Driver.get_status(): %s", exc)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "address": self._address,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "control_manager": manager,
                        "emo_states": emo,
                        "task_running": self._rollout is not None and self._rollout.is_running,
                        "battery_pct": battery,
                    }
                }
            ],
        }

    def _halt(self, robot: Any) -> str | None:
        """Cancel the control in flight and drop the stream; the next write opens a new one."""
        self._begin_halt()
        with self._lock:
            stream, self._stream, self._commanded = self._stream, None, None
        if stream is not None:
            try:
                stream.cancel()
            except (OSError, RuntimeError) as exc:
                logger.debug("RBY1Driver: stream.cancel() raised %s", exc)
        try:
            cancelled = robot.cancel_control()
        except (OSError, RuntimeError) as exc:
            return f"cancel_control() raised {exc}"
        return None if cancelled else "cancel_control() returned False"

    async def stop(self) -> None:
        """Halt motion, leaving the link open."""
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        robot = self._robot
        if robot is not None and (reason := self._halt(robot)) is not None:
            logger.warning("%s.stop(): %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Stop the rollout, halt the robot, disable control and release the link. Idempotent."""
        self._begin_halt()
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
            rollout.join()
        robot = self._robot
        if robot is None:
            return
        if (reason := self._halt(robot)) is not None:
            logger.debug("RBY1Driver.cleanup(): %s", reason)
        with self._lock:
            self._robot = None
        try:
            robot.disable_control_manager()
        except (OSError, RuntimeError) as exc:
            logger.debug("RBY1Driver.cleanup(): disable_control_manager() raised %s", exc)
        self._release(robot)

    # Reads.

    def _read(self, robot: Any) -> tuple[list[dict[str, float]], str | None]:
        """Position, velocity and torque, each keyed by the robot's 24 joint names."""
        try:
            state = robot.get_state()
            vectors = [[float(value) for value in getattr(state, key)] for key in ("position", "velocity", "torque")]
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            return [], f"the robot's state read failed: {exc}"
        names = self._measured_names
        for vector in vectors:
            if len(vector) != len(names):
                return [], f"the robot reported {len(vector)} values, its model names {len(names)} joints"
        named = [dict(zip(names, vector, strict=True)) for vector in vectors]
        with self._lock:
            self._joints = dict(named[0])
        return named, None

    def state(self) -> dict[str, Any]:
        """Read joints, velocities and torques in one envelope.

        Returns:
            ``joints``/``joint_velocities``/``joint_efforts`` keyed by the
            robot's 24 joint names (radians, rad/s, Nm) and the control
            manager state; or a refusal naming the failed read.
        """
        robot = self._robot
        if robot is None:
            return refuse("state: not connected - call connect_eagerly() first")
        named, reason = self._read(robot)
        if reason is not None:
            return refuse(f"state: {reason}")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": named[0],
                        "joint_velocities": named[1],
                        "joint_efforts": named[2],
                    }
                }
            ],
        }

    def get_observation(self) -> dict[str, float]:
        """The 24 measured joint positions (name -> radians), for the mesh and the rollout."""
        robot = self._robot
        if robot is not None:
            self._read(robot)
        with self._lock:
            return dict(self._joints)

    # Command path.

    def _begin_halt(self) -> None:
        with self._lock:
            self._halt_epoch += 1

    def _drop_anchor(self) -> None:
        with self._lock:
            self._commanded = None

    def _command(self, targets: list[float]) -> Any:
        """Build the body + head joint-position command for one setpoint."""
        import numpy as np  # the driver seam is imported eagerly and stays light

        sdk, period = self._sdk, 1.0 / self._control_frequency

        def joint_position(values: list[float]) -> Any:
            return (
                sdk.JointPositionCommandBuilder()
                .set_command_header(sdk.CommandHeaderBuilder().set_control_hold_time(CONTROL_HOLD_TIME))
                .set_minimum_time(period)
                .set_position(np.asarray(values, dtype=np.float64))
            )

        width = len(BODY_JOINTS)
        component = (
            sdk.ComponentBasedCommandBuilder()
            .set_body_command(sdk.BodyCommandBuilder().set_command(joint_position(targets[:width])))
            .set_head_command(sdk.HeadCommandBuilder().set_command(joint_position(targets[width:])))
        )
        return sdk.RobotCommandBuilder().set_command(component)

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Stream one body + head joint-position setpoint.

        Gates, in order: this driver fronts ``robot_name``; the link is live; no
        e-stop is pressed and the control manager is not faulted; the action
        names only commanded joints, with finite values inside the robot's
        range and no step past its velocity limit; no halt landed while those
        gates ran. The stream's feedback decides the verdict.

        Args:
            action: Joint targets in radians keyed by :data:`JOINT_NAMES`. An
                omitted joint holds its value.
            robot_name: ``None`` or this driver's own name.

        Returns:
            A success envelope carrying the setpoint written, or a refusal.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            robot, halted_at, commanded, stream = self._robot, self._halt_epoch, self._commanded, self._stream
        if robot is None or not self.is_connected:
            return refuse("send_action: not connected - call connect_eagerly() first")
        if (reason := self._fault_refusal(robot)) is not None:
            self._drop_anchor()
            return refuse(f"send_action: {reason}")
        if stream is not None and stream.is_done():
            stream = commanded = None
        if commanded is not None:
            reference = list(commanded)
        else:
            named, reason = self._read(robot)
            if reason is not None:
                return refuse(f"send_action: {reason}")
            reference = [named[0][name] for name in JOINT_NAMES]
        targets, reason = targets_from_action(
            action, reference, lower=self._lower, upper=self._upper, max_step=self._max_step
        )
        if reason is not None:
            return refuse(f"send_action: {reason}")
        with self._lock:
            superseded = self._halt_epoch != halted_at
        if superseded:
            return refuse(
                "send_action: the robot was halted while this setpoint was being prepared; it was not written"
            )
        try:
            if stream is None:
                stream = robot.create_command_stream(self._priority)
            feedback = stream.send_command(self._command(targets), COMMAND_TIMEOUT_MS)
        except (OSError, RuntimeError) as exc:
            with self._lock:
                self._stream, self._commanded = None, None
            return refuse(f"send_action: the command stream failed: {exc}")
        if _enum_name(feedback.status) == "Finished" and _enum_name(feedback.finish_code) != "Ok":
            with self._lock:
                self._stream, self._commanded = None, None
            return refuse(f"send_action: the robot finished the command with {_enum_name(feedback.finish_code)}")
        with self._lock:
            self._stream, self._commanded = stream, list(targets)
        return {
            "status": "success",
            "content": [{"json": {"robot": self._tool_name, "joints": dict(zip(JOINT_NAMES, targets, strict=True))}}],
        }

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

        Every step writes through :meth:`send_action`, so an e-stop or a fault
        mid-rollout ends it with that refusal as the exit reason.
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
            name=f"rby1-rollout-{self._tool_name}",
            policy=policy_object,
            instruction=instruction,
            duration=float(duration),
            n_steps=n_steps,
            period=1.0 / self._control_frequency,
            observe=self.get_observation,
            act=self.send_action,
            on_finish=self._drop_anchor,
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
            Success when the rollout left its loop and the robot cancelled its
            control; a refusal naming the failure otherwise; an error carrying
            ``stopped=False`` when the rollout thread did not join (the robot is
            halted, the task still holds it).
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
            return refuse(f"stop_task: the robot refused the halt: {reason}")
        steps = 0 if rollout is None else rollout.steps
        if not joined and rollout is not None:
            snapshot = {**rollout.snapshot(), "stopped": False, "robot": self._tool_name}
            snapshot["reason"] = (
                "stop_task: the rollout thread did not join; the robot is halted, the task still holds it"
            )
            return {"status": "error", "content": [{"json": snapshot}]}
        return {"status": "success", "content": [{"json": {"stopped": True, "steps": steps, "robot": self._tool_name}}]}
