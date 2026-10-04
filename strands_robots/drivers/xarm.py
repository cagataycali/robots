"""Native driver for the UFactory xArm 7, over the vendor's ``xarm-python-sdk``.

``Robot("xarm7", mode="real", port="192.168.1.185")`` builds one of these.
lerobot registers no xArm robot type, so before this driver the arm was
simulation-only and ``mode="real"`` refused it by name.

The controller owns the joints and speaks a TCP protocol on the arm's own
network; ``xarm.wrapper.XArmAPI`` is the vendor's client for it. This driver
holds the arm in **servo motion mode** (``set_mode(1)``), the controller's
streaming-setpoint posture, and writes each action with ``set_servo_angle_j`` -
the same role ``servoJ`` plays for :class:`~strands_robots.drivers.ur.URDriver`.

Three things are gated here because the controller does not gate them:

* the controller's own error word - a controller holding an error code accepts
  a servo write and moves nothing, so a non-zero ``error_code`` refuses the
  write and names the code (clearing it is an operator's decision, not this
  driver's);
* the step size - servo mode tracks a setpoint as fast as the joint can, so a
  joint asked to travel further in one control period than the controller's own
  reported ``joint_speed_limit`` allows is refused. The step is measured from
  the last *commanded* setpoint while a stream is in progress, for the reason
  :meth:`~strands_robots.drivers.ur.URDriver._reference_pose` gives;
* a halt that lands while a setpoint is being prepared - a counter bumped by
  every halt verb is re-read just before the write.

Joint range is not restated: the SDK refuses an out-of-range servo target
itself (``APIState.OUT_OF_RANGE``), and every non-zero SDK code is reported
verbatim.

Deliberately absent: the xArm Gripper. Its pulse scale has not been measured
against the simulated gripper's ``0..255`` actuator, so an action naming
``gripper`` is refused rather than mapped by guess. No Cartesian control and no
kinematics: the action space is joint space, which is what the policies here
emit.

``xarm`` is imported inside :meth:`XArmDriver.connect_eagerly` only, so the
module imports on any machine and every test runs against a fake controller.
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
SUPPORTED_ROBOTS: tuple[str, ...] = ("xarm7",)

#: Joint order on the wire, base to flange. ``get_joint_states`` and
#: ``set_servo_angle_j`` both use it, and the MuJoCo asset names its joints the
#: same way, so an action recorded in simulation needs no remap.
JOINT_NAMES: tuple[str, ...] = tuple(f"joint{index}" for index in range(1, 8))

#: Default rollout cadence, hertz. Also the period the step gate sizes against.
DEFAULT_CONTROL_FREQUENCY: float = 100.0

#: ``set_mode`` value for servo motion, the only mode ``set_servo_angle_j`` moves in.
SERVO_MODE: int = 1

#: ``set_state`` values: ``0`` ready to move, ``4`` stop.
STATE_READY: int = 0
STATE_STOP: int = 4

#: ``arm.state`` values that execute no motion: suspended and stopping.
STATES_THAT_DO_NOT_MOVE: frozenset[int] = frozenset({3, 4})


def _resolve_sdk() -> Any:
    """Return ``xarm.wrapper.XArmAPI``, or a reason naming the install."""
    try:
        return importlib.import_module("xarm.wrapper").XArmAPI
    except (ImportError, AttributeError) as exc:
        return f"the xArm SDK is not importable ({exc}). Install it with: pip install 'strands-robots[xarm]'"


def targets_from_action(
    action: dict[str, Any],
    reference: list[float],
    *,
    max_step: float | None = None,
) -> tuple[list[float], str | None]:
    """Turn a joint-name-keyed action into one ordered servo vector.

    A joint the action omits holds its reference value: ``set_servo_angle_j``
    takes a whole-arm setpoint, and a zero there would fold the arm.

    Args:
        action: Joint targets in radians keyed by :data:`JOINT_NAMES` members.
        reference: The pose the step is measured from, in :data:`JOINT_NAMES` order.
        max_step: Largest travel one joint may be asked for, radians. ``None``
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
            f"{unknown} name no xArm 7 joint; expected any of {list(JOINT_NAMES)}. "
            "The xArm Gripper is not driven by this driver yet."
        )
    targets = list(reference)
    for index, name in enumerate(JOINT_NAMES):
        if name not in action:
            continue
        if (reason := finite_number_error(action[name], name, "send_action")) is not None:
            return [], reason
        target = float(action[name])
        step = abs(target - reference[index])
        if max_step is not None and step > max_step:
            return [], (
                f"{name} asks for {step:.4f} rad in one control period, more than the {max_step:.4f} rad "
                "the controller's joint speed limit allows. Servo mode would jump there; slow the policy "
                "or lower control_frequency."
            )
        targets[index] = target
    return targets, None


class XArmDriver:
    """Native driver for the UFactory xArm 7.

    Args:
        tool_name: Name the agent invokes the driver by.
        cameras: Accepted for the factory contract; this driver opens none.
        data_config: Accepted for the factory contract; unused.
        port: The controller's IP address.
        control_frequency: Rollout cadence in hertz, and the period the step
            gate sizes a setpoint against.

    Raises:
        ValueError: If ``control_frequency`` is not a positive finite number.
    """

    def __init__(
        self,
        tool_name: str = "xarm7",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
    ) -> None:
        del cameras, data_config
        if reason := positive_finite_number_error(control_frequency, "control_frequency", "XArmDriver"):
            raise ValueError(reason)
        self._tool_name = tool_name
        self._host = str(port) if port else ""
        self._control_frequency = float(control_frequency)
        self._lock = threading.Lock()
        self._arm: Any = None
        self._connect_error: str | None = None
        self._max_speed: float | None = None
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
        """Whether the controller link is live."""
        arm = self._arm
        return arm is not None and bool(getattr(arm, "connected", False))

    @property
    def tool_spec(self) -> ToolSpec:
        """Read state, report status, stop. Motion goes through ``send_action``."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "UFactory xArm 7 native driver: reads joint positions, velocities, efforts and the TCP "
                    "pose, reports the controller's state and error word, and stops motion."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joints, velocities, efforts and TCP pose; "
                                    "status: link, controller state, mode and error codes; stop: halt the arm"
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
        """Connect, refuse a controller holding an error, and enter servo mode.

        The vendor's own servo recipe: ``motion_enable(True)``, ``set_mode(1)``,
        ``set_state(0)``. A controller already holding an error code is released
        before any of them, so connecting never energises an arm in a fault.

        Returns:
            ``None`` once the arm is in servo mode, or a reason; the driver stays
            usable either way (reads report the reason, writes refuse).
        """
        if self.is_connected:
            return None
        if not self._host:
            self._connect_error = (
                'XArmDriver: no controller address - pass port="<controller IP>", e.g. '
                'Robot("xarm7", mode="real", port="192.168.1.185")'
            )
            return self._connect_error
        api = _resolve_sdk()
        if isinstance(api, str):
            self._connect_error = api
            return api
        try:
            arm = api(self._host, is_radian=True)
        except Exception as exc:  # the SDK raises a bare Exception("connect socket failed") for an unreachable host
            self._connect_error = f"XArmDriver: controller at {self._host!r} did not answer: {exc}"
            return self._connect_error
        if not getattr(arm, "connected", False):
            self._connect_error = f"XArmDriver: controller at {self._host!r} did not answer"
            return self._connect_error
        if (reason := self._connect_refusal(arm)) is not None:
            self._release(arm)
            self._connect_error = reason
            return reason
        for call, args in (("motion_enable", (True,)), ("set_mode", (SERVO_MODE,)), ("set_state", (STATE_READY,))):
            code = getattr(arm, call)(*args)
            if code != 0:
                self._release(arm)
                self._connect_error = f"XArmDriver: {call}{args} returned code {code}"
                return self._connect_error
        limit = getattr(arm, "joint_speed_limit", None)
        with self._lock:
            self._arm = arm
            self._max_speed = float(limit[1]) if limit else None
            self._commanded = None
        self._connect_error = None
        self.state()
        return None

    @staticmethod
    def _release(arm: Any) -> None:
        try:
            arm.disconnect()
        except (OSError, RuntimeError) as exc:
            logger.debug("XArmDriver: disconnect raised %s", exc)

    def _connect_refusal(self, arm: Any) -> str | None:
        """Fetch the controller's error word with a round-trip and refuse a fault.

        ``arm.error_code`` and ``arm.state`` are report-stream caches that start
        at ``0`` and ``4`` and are not filled until the first report packet,
        which ``XArmAPI`` does not wait for. Right after connecting they say
        nothing about the controller, so the words are fetched here.
        ``state`` is not gated: ``4`` is how a controller boots, and the servo
        recipe that follows is what clears it.
        """
        try:
            code, words = arm.get_err_warn_code()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return f"XArmDriver: reading the controller's error word failed: {exc}"
        if code != 0:
            return f"XArmDriver: get_err_warn_code returned code {code}"
        return self._fault_reason(words[0])

    def _fault_reason(self, code: int) -> str | None:
        """Name a non-zero controller error word, or ``None`` for ``0``."""
        if not code:
            return None
        return (
            f"XArmDriver: controller at {self._host!r} holds error code {code}. Clear it from "
            "UFactory Studio or with clean_error() once the cause is fixed; a write now moves nothing."
        )

    def _error_refusal(self, arm: Any) -> str | None:
        """Name the error word or a non-moving state from the report-stream caches."""
        if (reason := self._fault_reason(getattr(arm, "error_code", 0))) is not None:
            return reason
        state = getattr(arm, "state", None)
        if state in STATES_THAT_DO_NOT_MOVE:
            return f"XArmDriver: controller at {self._host!r} is in state {state} (suspended or stopping)"
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the link and the controller's state, mode, error and warn words."""
        arm = self._arm
        words = {key: getattr(arm, key, None) for key in ("state", "mode", "error_code", "warn_code")}
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "host": self._host,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        **words,
                        "joint_speed_limit": self._max_speed,
                        "task_running": self._rollout is not None and self._rollout.is_running,
                        "battery_pct": None,
                    }
                }
            ],
        }

    def _halt(self, arm: Any) -> str | None:
        """Stop the arm, then re-arm servo mode so the next write is admitted.

        ``set_state(4)`` drops the motion in flight; the controller then needs
        ``set_mode(1)`` and ``set_state(0)`` before it moves again, and the
        stream re-anchors on the measured pose.
        """
        self._begin_halt()
        self._drop_anchor()
        for call, args in (("set_state", (STATE_STOP,)), ("set_mode", (SERVO_MODE,)), ("set_state", (STATE_READY,))):
            try:
                code = getattr(arm, call)(*args)
            except (OSError, RuntimeError) as exc:
                return f"{call}{args} raised {exc}"
            if code != 0:
                return f"{call}{args} returned code {code}"
        return None

    async def stop(self) -> None:
        """Halt motion, leaving the link open."""
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        arm = self._arm
        if arm is not None and (reason := self._halt(arm)) is not None:
            logger.warning("%s.stop(): %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Stop the rollout, halt the arm and release the link. Idempotent."""
        self._begin_halt()
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
            rollout.join()
        with self._lock:
            arm, self._arm = self._arm, None
        if arm is None:
            return
        try:
            arm.set_state(STATE_STOP)
        except (OSError, RuntimeError) as exc:
            logger.debug("XArmDriver.cleanup(): set_state(4) raised %s", exc)
        self._release(arm)

    # Reads.

    def state(self) -> dict[str, Any]:
        """Read joints, velocities, efforts and the TCP pose in one envelope.

        Returns:
            ``joints``/``joint_velocities``/``joint_efforts`` keyed by
            :data:`JOINT_NAMES` (radians, rad/s, Nm), ``tcp_pose`` as the SDK
            reports it (x, y, z in mm, roll, pitch, yaw in rad), and the
            controller words; or a refusal naming the SDK code.
        """
        arm = self._arm
        if arm is None:
            return refuse("state: not connected - call connect_eagerly() first")
        try:
            code, vectors = arm.get_joint_states(is_radian=True)
            pose_code, pose = arm.get_position(is_radian=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return refuse(f"state: the controller read failed: {exc}")
        if code != 0 or pose_code != 0:
            return refuse(f"state: get_joint_states returned code {code}, get_position code {pose_code}")
        named = []
        for vector in vectors[:3]:
            values = [float(value) for value in vector]
            if len(values) != len(JOINT_NAMES):
                return refuse(f"state: the controller reported {len(values)} values, expected {len(JOINT_NAMES)}")
            named.append(dict(zip(JOINT_NAMES, values, strict=True)))
        if len(named) != 3:
            return refuse(f"state: get_joint_states returned {len(named)} vectors, expected position/velocity/effort")
        with self._lock:
            self._joints = dict(named[0])
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": named[0],
                        "joint_velocities": named[1],
                        "joint_efforts": named[2],
                        "tcp_pose": [float(value) for value in pose],
                        "state": getattr(arm, "state", None),
                        "error_code": getattr(arm, "error_code", None),
                    }
                }
            ],
        }

    def _read_joints(self, arm: Any) -> tuple[list[float], str | None]:
        try:
            code, angles = arm.get_servo_angle(is_radian=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return [], f"the controller's joint read failed: {exc}"
        if code != 0:
            return [], f"get_servo_angle returned code {code}"
        joints = [float(value) for value in angles]
        if len(joints) != len(JOINT_NAMES):
            return [], f"the controller reported {len(joints)} joint positions, expected {len(JOINT_NAMES)}"
        return joints, None

    def get_observation(self) -> dict[str, float]:
        """The seven joint positions (name -> radians), for the mesh and the rollout."""
        arm = self._arm
        if arm is not None:
            joints, reason = self._read_joints(arm)
            if reason is None:
                with self._lock:
                    self._joints = dict(zip(JOINT_NAMES, joints, strict=True))
        with self._lock:
            return dict(self._joints)

    # Command path.

    def _begin_halt(self) -> None:
        with self._lock:
            self._halt_epoch += 1

    def _drop_anchor(self) -> None:
        with self._lock:
            self._commanded = None

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command one joint-space setpoint through ``set_servo_angle_j``.

        Gates, in order: this driver fronts ``robot_name``; the link is live; the
        controller holds no error and is not stopped; the action names only xArm
        joints with finite values and no step past the speed limit; no halt
        landed while those gates ran. The SDK's return code decides the verdict.

        Args:
            action: Joint targets in radians keyed by :data:`JOINT_NAMES`. An
                omitted joint holds its value.
            robot_name: ``None`` or this driver's own name.

        Returns:
            A success envelope carrying the vector written, or a refusal.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            arm, halted_at, commanded = self._arm, self._halt_epoch, self._commanded
        if arm is None or not self.is_connected:
            return refuse("send_action: not connected - call connect_eagerly() first")
        if (reason := self._error_refusal(arm)) is not None:
            self._drop_anchor()
            return refuse(f"send_action: {reason}")
        if commanded is not None:
            reference = list(commanded)
        else:
            reference, reason = self._read_joints(arm)
            if reason is not None:
                return refuse(f"send_action: {reason}")
        max_step = None if self._max_speed is None else self._max_speed / self._control_frequency
        targets, reason = targets_from_action(action, reference, max_step=max_step)
        if reason is not None:
            return refuse(f"send_action: {reason}")
        with self._lock:
            superseded = self._halt_epoch != halted_at
        if superseded:
            return refuse("send_action: the arm was halted while this setpoint was being prepared; it was not written")
        try:
            code = arm.set_servo_angle_j(targets, is_radian=True)
        except (OSError, RuntimeError) as exc:
            return refuse(f"send_action: set_servo_angle_j failed: {exc}")
        if code != 0:
            return refuse(f"send_action: the controller refused set_servo_angle_j({targets}) with code {code}")
        with self._lock:
            self._commanded = list(targets)
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
            "the rollout would start on a live arm and fail at its first action",
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

        Every step writes through :meth:`send_action`, so a controller that
        faults mid-rollout ends it with that refusal as the exit reason.
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
            name=f"xarm-rollout-{self._tool_name}",
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
        """Stop the rollout and halt the arm.

        Returns:
            Success when the rollout left its loop and the controller accepted
            the halt; a refusal naming the SDK code otherwise; an error carrying
            ``stopped=False`` when the rollout thread did not join (the arm is
            halted, the task still holds it).
        """
        self._begin_halt()
        rollout = self._rollout
        joined = True
        if rollout is not None and rollout.is_running:
            rollout.request_stop()
            joined = rollout.join()
        arm = self._arm
        if arm is None:
            return refuse("stop_task: not connected")
        if (reason := self._halt(arm)) is not None:
            return refuse(f"stop_task: the controller refused the halt: {reason}")
        steps = 0 if rollout is None else rollout.steps
        if not joined and rollout is not None:
            snapshot = {**rollout.snapshot(), "stopped": False, "robot": self._tool_name}
            snapshot["reason"] = (
                "stop_task: the rollout thread did not join; the arm is halted, the task still holds it"
            )
            return {"status": "error", "content": [{"json": snapshot}]}
        return {"status": "success", "content": [{"json": {"stopped": True, "steps": steps, "robot": self._tool_name}}]}
