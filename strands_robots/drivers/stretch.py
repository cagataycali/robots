"""Native driver for the Hello Robot Stretch (RE1, RE2 and Stretch 3), over ``stretch_body``.

``Robot("stretch3", mode="real")`` builds one of these on the robot's own
computer. lerobot registers no Stretch robot type, so before this driver the
robot was simulation-only and ``mode="real"`` refused it by name.

``stretch_body.robot.Robot`` is the vendor's Python interface: it opens the
robot's USB devices (``/dev/hello-*``), runs its status threads and holds a
file lock so one process owns the body. This driver follows its documented
use - ``startup()``, queue joint goals with ``move_to``, ``push_command()`` -
and adds the base twist a mobile manipulator needs (:meth:`StretchDriver.set_twist`).

The vendor's calls are lenient where a caller should be told. ``move_to``
clips a goal to the joint's soft limits, drops a ``NaN``, does nothing for an
unhomed joint or while the runstop is latched, and ``Base.set_velocity``
clamps each wheel to its speed limit - all without an error. This driver
refuses each of those instead, before anything is queued:

* a goal outside the joint's *current* soft limits (``get_soft_motion_limits()``,
  read on every write, so a limit the collision manager tightens is honoured);
* a non-finite value, or a joint the robot does not have (``wrist_pitch`` and
  ``wrist_roll`` exist only with a dex wrist);
* any joint write while the robot is not homed, or while the runstop is latched;
* a twist with a sideways component (the base is differential drive), one whose
  wheel speed would exceed the base's ``motion.max.vel_m``, or any twist when
  the wheels' firmware velocity watchdog is off - the watchdog is what stops
  the base when commands stop arriving.

Joint goals are position targets the firmware tracks with its own trapezoidal
profile, so a large step is a far goal reached at the profile speed, not a
jump. Deliberately absent: the gripper, whose unit is the SDK's unitless
percentage rather than the MuJoCo asset's metres, so an action naming it is
refused rather than mapped by guess.

``stretch_body`` is imported inside :meth:`StretchDriver.connect_eagerly` only,
so the module imports on any machine and every test runs against a fake robot.
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
SUPPORTED_ROBOTS: tuple[str, ...] = ("stretch", "stretch3")

#: Stepper-driven joints, metres: the attribute on ``stretch_body.robot.Robot``.
PRISMATIC_JOINTS: tuple[str, ...] = ("lift", "arm")
#: Dynamixel joints on the head chain, radians.
HEAD_JOINTS: tuple[str, ...] = ("head_pan", "head_tilt")
#: Dynamixel joints on the end-of-arm chain, radians; a robot carries a subset.
WRIST_JOINTS: tuple[str, ...] = ("wrist_yaw", "wrist_pitch", "wrist_roll")
#: Every joint this driver can command, named as the Stretch 3 MuJoCo actuators are.
JOINT_NAMES: tuple[str, ...] = PRISMATIC_JOINTS + WRIST_JOINTS + HEAD_JOINTS

#: Default rollout cadence, hertz: the rate of the SDK's Dynamixel status thread.
DEFAULT_CONTROL_FREQUENCY: float = 15.0

#: Base odometry keys reported by :meth:`StretchDriver.state`.
BASE_KEYS: tuple[str, ...] = ("x", "y", "theta", "x_vel", "theta_vel")


def _resolve_sdk() -> Any:
    """Return the ``stretch_body.robot`` module, or a reason naming the install.

    Importing it reads the robot's calibration at import time: it raises
    ``KeyError`` without ``HELLO_FLEET_PATH``/``HELLO_FLEET_ID`` and calls
    ``sys.exit(1)`` when the fleet YAML files are missing. Both are caught, so
    a machine that is not a Stretch gets a reason instead of losing its process.
    """
    try:
        return importlib.import_module("stretch_body.robot")
    except ImportError as exc:
        return (
            f"the Stretch SDK is not importable ({exc}). It ships on the robot; elsewhere install it "
            "with: pip install hello-robot-stretch-body"
        )
    except (KeyError, SystemExit) as exc:
        return (
            f"the Stretch SDK found no robot configuration ({type(exc).__name__}: {exc}); it needs "
            "HELLO_FLEET_PATH, HELLO_FLEET_ID and the fleet's YAML files, which exist on the robot"
        )


def wheel_speeds(vx: float, wz: float, separation: float) -> tuple[float, float]:
    """Left and right wheel ground speeds, m/s, for a unicycle twist (the SDK's own model)."""
    return vx - wz * separation / 2.0, vx + wz * separation / 2.0


class StretchDriver:
    """Native driver for the Hello Robot Stretch.

    Args:
        tool_name: Name the agent invokes the driver by.
        cameras: Accepted for the factory contract; this driver opens none.
        data_config: Accepted for the factory contract; unused.
        port: Must be left unset. The SDK opens the robot's own ``/dev/hello-*``
            devices, so a port names nothing; :meth:`connect_eagerly` refuses
            one rather than ignore it.
        control_frequency: Rollout cadence in hertz.

    Raises:
        ValueError: If ``control_frequency`` is not a positive finite number.
    """

    def __init__(
        self,
        tool_name: str = "stretch",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
    ) -> None:
        del cameras, data_config
        if reason := positive_finite_number_error(control_frequency, "control_frequency", "StretchDriver"):
            raise ValueError(reason)
        self._tool_name = tool_name
        self._port = port
        self._control_frequency = float(control_frequency)
        self._lock = threading.Lock()
        self._robot: Any = None
        self._joints_present: tuple[str, ...] = ()
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
        """Whether ``startup()`` succeeded and the body has not been released."""
        return self._robot is not None

    @property
    def tool_spec(self) -> ToolSpec:
        """Read state, report status, stop. Motion goes through ``send_action`` and ``set_twist``."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Hello Robot Stretch native driver: reads the lift, arm, wrist and head joints and the "
                    "base odometry, reports runstop, homing and battery voltage, and stops motion."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joints and base odometry; "
                                    "status: runstop, homing and voltage; stop: halt the robot"
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
        """Construct the vendor ``Robot`` and run ``startup()``.

        ``startup()`` returns ``False`` when another process holds the body's
        file lock or a device did not answer; either is reported, and the
        partially started robot is stopped so it releases what it opened.

        Returns:
            ``None`` once the robot is up, or a reason; the driver stays usable
            either way (reads report the reason, writes refuse).
        """
        if self.is_connected:
            return None
        if self._port:
            self._connect_error = (
                f"StretchDriver: port={self._port!r} names nothing - the SDK opens the robot's own "
                "/dev/hello-* devices; run on the robot and leave port unset"
            )
            return self._connect_error
        sdk = _resolve_sdk()
        if isinstance(sdk, str):
            self._connect_error = sdk
            return sdk
        try:
            robot = sdk.Robot()
            started = robot.startup()
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            self._connect_error = f"StretchDriver: the robot did not start: {exc}"
            return self._connect_error
        if not started:
            self._release(robot)
            self._connect_error = (
                "StretchDriver: startup() returned False - another process may hold the robot "
                "(stretch_free_robot_process.py releases it), or a device did not answer"
            )
            return self._connect_error
        present = [*PRISMATIC_JOINTS]
        present += [name for name in WRIST_JOINTS if name in robot.end_of_arm.motors]
        present += [name for name in HEAD_JOINTS if name in robot.head.motors]
        with self._lock:
            self._robot, self._joints_present = robot, tuple(present)
        self._connect_error = None
        self.state()
        return None

    @staticmethod
    def _release(robot: Any) -> None:
        try:
            robot.stop()
        except (OSError, RuntimeError, AttributeError) as exc:
            logger.debug("StretchDriver: robot.stop() raised %s", exc)

    def _joint(self, robot: Any, name: str) -> Any:
        """The vendor object that measures and limits ``name``."""
        if name in PRISMATIC_JOINTS:
            return getattr(robot, name)
        chain = robot.head if name in HEAD_JOINTS else robot.end_of_arm
        return chain.motors[name]

    def _runstopped(self, robot: Any) -> bool:
        return bool(robot.pimu.status["runstop_event"])

    async def get_status(self) -> dict[str, Any]:
        """Report the link, the runstop, homing, the battery voltage and the joints present."""
        robot = self._robot
        runstop = homed = voltage = None
        if robot is not None:
            try:
                runstop = self._runstopped(robot)
                homed = bool(robot.is_homed())
                voltage = float(robot.pimu.status["voltage"])
            except (OSError, RuntimeError, AttributeError, KeyError) as exc:
                logger.debug("StretchDriver.get_status(): %s", exc)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "runstop": runstop,
                        "homed": homed,
                        # The pimu measures pack voltage, not a charge percentage.
                        "battery_pct": None,
                        "voltage": voltage,
                        "joints": list(self._joints_present),
                        "task_running": self._rollout is not None and self._rollout.is_running,
                    }
                }
            ],
        }

    def _halt(self, robot: Any) -> str | None:
        """Zero the base twist and hold every joint where it is measured now."""
        self._begin_halt()
        try:
            robot.base.set_velocity(0.0, 0.0)
            for name in self._joints_present:
                position = float(self._joint(robot, name).status["pos"])
                self._queue(robot, name, position)
            robot.push_command()
        except (OSError, RuntimeError, AttributeError, KeyError) as exc:
            return f"the halt failed: {exc}"
        return None

    async def stop(self) -> None:
        """Halt motion, leaving the robot started."""
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        robot = self._robot
        if robot is not None and (reason := self._halt(robot)) is not None:
            logger.warning("%s.stop(): %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Stop the rollout, halt the robot and release the body (``stop()``). Idempotent."""
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
            logger.debug("StretchDriver.cleanup(): %s", reason)
        self._release(robot)

    # Reads.

    def _read(self, robot: Any) -> tuple[dict[str, float], str | None]:
        try:
            joints = {name: float(self._joint(robot, name).status["pos"]) for name in self._joints_present}
        except (OSError, RuntimeError, AttributeError, KeyError, TypeError, ValueError) as exc:
            return {}, f"the robot's status read failed: {exc}"
        with self._lock:
            self._joints = dict(joints)
        return joints, None

    def state(self) -> dict[str, Any]:
        """Read the joints and the base odometry in one envelope.

        Returns:
            ``joints`` (metres for ``lift``/``arm``, radians otherwise),
            ``joint_velocities`` and ``base`` (``x``/``y`` m, ``theta`` rad and
            their rates); or a refusal naming the failed read.
        """
        robot = self._robot
        if robot is None:
            return refuse(f"state: not connected - {self._connect_error or 'call connect_eagerly() first'}")
        joints, reason = self._read(robot)
        if reason is not None:
            return refuse(f"state: {reason}")
        try:
            velocities = {name: float(self._joint(robot, name).status["vel"]) for name in self._joints_present}
            base = {key: float(robot.base.status[key]) for key in BASE_KEYS}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return refuse(f"state: the robot's status read failed: {exc}")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": joints,
                        "joint_velocities": velocities,
                        "base": base,
                    }
                }
            ],
        }

    def get_observation(self) -> dict[str, float]:
        """The measured joint positions (name -> m or rad), for the mesh and the rollout."""
        robot = self._robot
        if robot is not None:
            self._read(robot)
        with self._lock:
            return dict(self._joints)

    # Command path.

    def _begin_halt(self) -> None:
        with self._lock:
            self._halt_epoch += 1

    def _queue(self, robot: Any, name: str, target: float) -> None:
        if name in PRISMATIC_JOINTS:
            getattr(robot, name).move_to(target)
        elif name in HEAD_JOINTS:
            robot.head.move_to(name, target)
        else:
            robot.end_of_arm.move_to(name, target)

    def _targets(self, robot: Any, action: dict[str, Any]) -> tuple[dict[str, float], str | None]:
        """Validate a joint action against what the robot has and its current soft limits."""
        if not isinstance(action, dict) or not action:
            return {}, f"nothing to command - the action names none of {list(self._joints_present)}"
        unknown = sorted(set(action) - set(self._joints_present))
        if unknown:
            return {}, (
                f"{unknown} name no commanded joint on this Stretch; expected any of "
                f"{list(self._joints_present)}. The gripper is not driven by this driver."
            )
        targets: dict[str, float] = {}
        for name, value in action.items():
            if (reason := finite_number_error(value, name, "send_action")) is not None:
                return {}, reason
            lower, upper = (float(bound) for bound in self._joint(robot, name).get_soft_motion_limits())
            target = float(value)
            if not lower <= target <= upper:
                return {}, (
                    f"{name}={target:.4f} is outside its soft limits [{lower:.4f}, {upper:.4f}]; "
                    "the SDK would clip it to the limit, so it is refused"
                )
            targets[name] = target
        return targets, None

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Queue one set of joint goals and push them.

        Gates, in order: this driver fronts ``robot_name``; the robot is
        started; the runstop is not latched; the robot is homed; the action
        names only joints this robot has, with finite values inside their
        current soft limits; no halt landed while those gates ran.

        Args:
            action: Joint goals keyed by :data:`JOINT_NAMES` members (metres for
                ``lift``/``arm``, radians otherwise). An omitted joint keeps its goal.
            robot_name: ``None`` or this driver's own name.

        Returns:
            A success envelope carrying the goals written, or a refusal.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            robot, halted_at = self._robot, self._halt_epoch
        if robot is None:
            return refuse("send_action: not connected - call connect_eagerly() first")
        try:
            if self._runstopped(robot):
                return refuse("send_action: the runstop is latched; release it on the robot before commanding")
            if not robot.is_homed():
                return refuse(
                    "send_action: the robot is not homed, so the SDK would ignore lift and arm goals. "
                    "Run stretch_robot_home.py first."
                )
            targets, reason = self._targets(robot, action)
        except (OSError, RuntimeError, AttributeError, KeyError, TypeError) as exc:
            return refuse(f"send_action: the robot's status read failed: {exc}")
        if reason is not None:
            return refuse(f"send_action: {reason}")
        with self._lock:
            superseded = self._halt_epoch != halted_at
        if superseded:
            return refuse("send_action: the robot was halted while these goals were being prepared; not written")
        try:
            for name, target in targets.items():
                self._queue(robot, name, target)
            robot.push_command()
        except (OSError, RuntimeError) as exc:
            return refuse(f"send_action: the command failed: {exc}")
        return {"status": "success", "content": [{"json": {"robot": self._tool_name, "joints": targets}}]}

    def set_twist(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> dict[str, Any]:
        """Command the base's body-frame twist (``Base.set_velocity``).

        The wheels' firmware watchdog stops the base when commands stop
        arriving, so a twist must be re-sent to keep moving; a twist is refused
        when that watchdog is off.

        Args:
            vx: Forward velocity, m/s.
            vy: Left velocity, m/s; the base is differential drive, so only ``0`` is accepted.
            wz: Yaw rate, rad/s, positive left.

        Returns:
            A success envelope carrying the twist and wheel speeds, or a refusal.
        """
        for value, name in ((vx, "vx"), (vy, "vy"), (wz, "wz")):
            if (reason := finite_number_error(value, name, "set_twist")) is not None:
                return refuse(reason)
        if float(vy) != 0.0:
            return refuse(f"set_twist: vy={float(vy)} - the Stretch base is differential drive and cannot strafe")
        robot = self._robot
        if robot is None:
            return refuse("set_twist: not connected - call connect_eagerly() first")
        base = robot.base
        try:
            if self._runstopped(robot):
                return refuse("set_twist: the runstop is latched; release it on the robot before driving")
            watchdogs = [bool(wheel.gains["enable_vel_watchdog"]) for wheel in (base.left_wheel, base.right_wheel)]
            separation = float(base.params["wheel_separation_m"])
            limit = float(base.params["motion"]["max"]["vel_m"])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return refuse(f"set_twist: the base's parameters could not be read: {exc}")
        if not all(watchdogs):
            return refuse(
                "set_twist: a wheel's velocity watchdog is disabled, so the base would keep rolling if "
                "commands stopped; enable enable_vel_watchdog in the robot's parameters"
            )
        left, right = wheel_speeds(float(vx), float(wz), separation)
        if max(abs(left), abs(right)) > limit:
            return refuse(
                f"set_twist: wheel speeds ({left:.3f}, {right:.3f}) m/s exceed the base's {limit:.3f} m/s; "
                "the SDK would clamp each wheel and bend the path, so it is refused"
            )
        try:
            base.set_velocity(float(vx), float(wz))
            robot.push_command()
        except (OSError, RuntimeError) as exc:
            return refuse(f"set_twist: the command failed: {exc}")
        twist = {"vx": float(vx), "vy": 0.0, "wz": float(wz)}
        return {
            "status": "success",
            "content": [{"json": {"robot": self._tool_name, "twist": twist, "wheel_speeds": [left, right]}}],
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

        Every step writes through :meth:`send_action`, so a latched runstop
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
            name=f"stretch-rollout-{self._tool_name}",
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
