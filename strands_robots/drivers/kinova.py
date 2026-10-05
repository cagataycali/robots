"""Native driver for the Kinova Gen3 (7-DOF), over Kinova's ``kortex_api``.

``Robot("kinova_gen3", mode="real", port="192.168.1.10")`` builds one of these.
lerobot registers no Kinova robot type, so before this driver the arm was
simulation-only and ``mode="real"`` refused it by name.

The arm's base controller speaks Kortex RPC over TCP port 10000 inside an
authenticated session. This driver keeps the arm in **single-level servoing**,
the mode every high-level command runs in, and streams joint velocities with
``Base.SendJointSpeedsCommand``: each joint-position setpoint becomes the speed
that reaches it in one control period, read against the measured pose from
``BaseCyclic.RefreshFeedback``.

Four things are gated here because the controller does not gate them:

* the arm state - a base in ``ARMSTATE_IN_FAULT`` (or any state other than
  ``ARMSTATE_SERVOING_READY``) is refused at connect and at every write, and
  clearing a fault is an operator's decision, not this driver's;
* the step size - a joint asked to travel further in one control period than
  :data:`MAX_JOINT_SPEED` allows is refused, measured from the last
  *commanded* setpoint while a stream is in progress (the reason is in
  :meth:`~strands_robots.drivers.ur.URDriver._reference_pose`);
* a quiet stream - the Kortex joint-speed command has no expiry (its
  ``duration`` field is documented "not implemented yet"), so a speed keeps
  the arm moving until something replaces it. A watchdog sends ``Base.Stop``
  once no write has landed for :data:`DEADMAN_PERIODS` control periods;
* a halt that lands while a setpoint is being prepared - a counter bumped by
  every halt verb is re-read just before the write.

Joint range is not restated: the base enforces its own angular limits.
Positions are reported in radians wrapped to ``(-pi, pi]`` (the wire reports
``0..360`` degrees), and a step is measured the short way round, so the four
continuous joints cross ``pi`` without a refusal.

Deliberately absent: the Robotiq gripper the Gen3 often carries (it is a
separate interconnect device) and Cartesian control.

The vendor wheel is not on PyPI and its generated protobuf modules predate
protobuf 3.19, so they import only with the pure-Python protobuf runtime
(``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python``, set before Python starts).
``kortex_api`` is imported inside :meth:`KinovaDriver.connect_eagerly` only,
so this module imports on any machine and every test runs against a fake base.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from strands_robots._pacing import Ticker
from strands_robots.drivers.base import policy_step, refuse, undeclared_verb_error
from strands_robots.drivers.rollout import PolicyRollout, policy_from_provider
from strands_robots.utils import finite_number_error, positive_count_error, positive_finite_number_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver serves, read by ``strands_robots.drivers._SHIPPED_DRIVERS``.
SUPPORTED_ROBOTS: tuple[str, ...] = ("kinova_gen3",)

#: Joint order on the wire (``joint_identifier`` 0..6), base to flange. The
#: MuJoCo asset names its joints the same way, so a simulated action needs no remap.
JOINT_NAMES: tuple[str, ...] = tuple(f"joint_{index}" for index in range(1, 8))

#: Largest joint speed a setpoint may ask for, rad/s: the Gen3 7-DOF
#: ``maximum_velocities`` row of Kinova's ``ros2_kortex`` description
#: (``kortex_description/arms/gen3/7dof/config/joint_limits.yaml``), 50 deg/s on
#: every joint.
MAX_JOINT_SPEED: float = 0.8727

#: Default rollout cadence, hertz. Also the period the step gate and the
#: velocity a setpoint becomes are sized against.
DEFAULT_CONTROL_FREQUENCY: float = 40.0

#: Control periods without a write after which the watchdog stops the arm.
DEADMAN_PERIODS: int = 3

#: The base's Kortex RPC port.
TCP_PORT: int = 10000


def _resolve_sdk() -> SimpleNamespace | str:
    """Return the ``kortex_api`` pieces this driver speaks, or a reason naming the fix."""
    try:
        sdk = SimpleNamespace(
            TCPTransport=importlib.import_module("kortex_api.TCPTransport").TCPTransport,
            RouterClient=importlib.import_module("kortex_api.RouterClient").RouterClient,
            SessionManager=importlib.import_module("kortex_api.SessionManager").SessionManager,
            BaseClient=importlib.import_module("kortex_api.autogen.client_stubs.BaseClientRpc").BaseClient,
            BaseCyclicClient=importlib.import_module(
                "kortex_api.autogen.client_stubs.BaseCyclicClientRpc"
            ).BaseCyclicClient,
            Base_pb2=importlib.import_module("kortex_api.autogen.messages.Base_pb2"),
            Session_pb2=importlib.import_module("kortex_api.autogen.messages.Session_pb2"),
            KException=importlib.import_module("kortex_api.Exceptions.KException").KException,
        )
    except TypeError as exc:  # protobuf >= 4 refuses descriptors generated before protoc 3.19
        return (
            f"the Kortex API's protobuf modules need the pure-Python protobuf runtime ({str(exc).splitlines()[0]}). "
            "Start Python with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
        )
    except (ImportError, AttributeError) as exc:
        return (
            f"the Kortex API is not importable ({exc}). Kinova ships it as a wheel, not on PyPI; install it "
            "with pip install --no-deps kortex_api-2.6.0.post3-py3-none-any.whl (its pins would downgrade protobuf)"
        )
    return sdk


def wrap_angle(value: float) -> float:
    """Fold an angle in radians into ``(-pi, pi]``."""
    wrapped = math.remainder(value, math.tau)
    return math.pi if wrapped == -math.pi else wrapped


def targets_from_action(
    action: dict[str, Any],
    reference: list[float],
    *,
    max_step: float | None = None,
) -> tuple[list[float], str | None]:
    """Turn a joint-name-keyed action into one ordered position vector.

    A joint the action omits holds its reference value. The step is measured
    the short way round (:func:`wrap_angle`).

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
            f"{unknown} name no Gen3 joint; expected any of {list(JOINT_NAMES)}. "
            "The gripper is not driven by this driver."
        )
    targets = list(reference)
    for index, name in enumerate(JOINT_NAMES):
        if name not in action:
            continue
        if (reason := finite_number_error(action[name], name, "send_action")) is not None:
            return [], reason
        target = float(action[name])
        step = abs(wrap_angle(target - reference[index]))
        if max_step is not None and step > max_step:
            return [], (
                f"{name} asks for {step:.4f} rad in one control period, more than the {max_step:.4f} rad that "
                f"{MAX_JOINT_SPEED} rad/s allows. Slow the policy or lower control_frequency."
            )
        targets[index] = target
    return targets, None


class KinovaDriver:
    """Native driver for the Kinova Gen3 7-DOF arm.

    Args:
        tool_name: Name the agent invokes the driver by.
        cameras: Accepted for the factory contract; this driver opens none.
        data_config: Accepted for the factory contract; unused.
        port: The base's IP address.
        username: Kortex session user (the factory default is ``admin``).
        password: Kortex session password.
        control_frequency: Rollout cadence in hertz; sizes the step gate, the
            speed a setpoint becomes and the watchdog.

    Raises:
        ValueError: If ``control_frequency`` is not a positive finite number.
    """

    def __init__(
        self,
        tool_name: str = "kinova_gen3",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        username: str = "admin",
        password: str = "admin",
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
    ) -> None:
        del cameras, data_config
        if reason := positive_finite_number_error(control_frequency, "control_frequency", "KinovaDriver"):
            raise ValueError(reason)
        self._tool_name = tool_name
        self._host = str(port) if port else ""
        self._username = username
        self._password = password
        self._control_frequency = float(control_frequency)
        self._lock = threading.Lock()
        self._sdk: SimpleNamespace | None = None
        self._link: SimpleNamespace | None = None
        self._connect_error: str | None = None
        self._arm_state: str | None = None
        self._joints: dict[str, float] = {}
        self._commanded: list[float] | None = None
        self._halt_epoch = 0
        self._last_write: float | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
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
        """Whether a Kortex session is open."""
        return self._link is not None

    @property
    def tool_spec(self) -> ToolSpec:
        """Read state, report status, stop. Motion goes through ``send_action``."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Kinova Gen3 native driver: reads joint positions, velocities, torques and the tool pose, "
                    "reports the arm state, and stops motion."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joints, velocities, torques and tool pose; "
                                    "status: session and arm state; stop: halt the arm"
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
        """Open a session, refuse an arm that cannot servo, and enter single-level servoing.

        Returns:
            ``None`` once the arm is ready, or a reason; the driver stays usable
            either way (reads report the reason, writes refuse).
        """
        if self.is_connected:
            return None
        if not self._host:
            self._connect_error = (
                'KinovaDriver: no base address - pass port="<base IP>", e.g. '
                'Robot("kinova_gen3", mode="real", port="192.168.1.10")'
            )
            return self._connect_error
        sdk = _resolve_sdk()
        if isinstance(sdk, str):
            self._connect_error = f"KinovaDriver: {sdk}"
            return self._connect_error
        link, reason = self._open(sdk)
        if link is not None and (reason := self._ready_refusal(sdk, link)) is not None:
            self._close(link)
        if reason is not None:
            self._connect_error = f"KinovaDriver: {reason}"
            return self._connect_error
        with self._lock:
            self._sdk, self._link, self._commanded, self._last_write = sdk, link, None, None
        self._connect_error = None
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=self._watch, name=f"kinova-deadman-{self._tool_name}", daemon=True)
        self._watchdog.start()
        self.state()
        return None

    def _open(self, sdk: SimpleNamespace) -> tuple[SimpleNamespace | None, str | None]:
        transport = sdk.TCPTransport()
        router = sdk.RouterClient(transport, lambda exc: logger.warning("KinovaDriver: router error %s", exc))
        try:
            transport.connect(self._host, TCP_PORT)
        except OSError as exc:
            return None, f"base at {self._host!r} did not answer on port {TCP_PORT}: {exc}"
        session = sdk.SessionManager(router)
        link = SimpleNamespace(
            transport=transport, session=None, base=sdk.BaseClient(router), cyclic=sdk.BaseCyclicClient(router)
        )
        info = sdk.Session_pb2.CreateSessionInfo(
            username=self._username,
            password=self._password,
            session_inactivity_timeout=60000,
            connection_inactivity_timeout=2000,
        )
        try:
            session.CreateSession(info)
        except (sdk.KException, OSError) as exc:
            self._close(link)
            return None, f"the base at {self._host!r} refused the session for {self._username!r}: {exc}"
        link.session = session
        return link, None

    def _ready_refusal(self, sdk: SimpleNamespace, link: SimpleNamespace) -> str | None:
        """Refuse a 6-DOF arm or a base that is not ready, then select single-level servoing."""
        try:
            count = link.base.GetActuatorCount().count
            if count != len(JOINT_NAMES):
                return f"the base reports {count} actuators; this driver serves the 7-DOF Gen3"
            if (reason := self._state_refusal(sdk, link.base.GetArmState().active_state)) is not None:
                return reason
            mode = sdk.Base_pb2.ServoingModeInformation(servoing_mode=sdk.Base_pb2.SINGLE_LEVEL_SERVOING)
            link.base.SetServoingMode(mode)
        except (sdk.KException, OSError) as exc:
            return f"preparing the base failed: {exc}"
        return None

    def _state_refusal(self, sdk: SimpleNamespace, state: int) -> str | None:
        """Name an arm state that executes no joint-speed command, or ``None``."""
        name = sdk.Base_pb2.ArmState.Name(state)
        self._arm_state = name
        if state == sdk.Base_pb2.ARMSTATE_SERVOING_READY:
            return None
        if state == sdk.Base_pb2.ARMSTATE_IN_FAULT:
            return (
                f"the base at {self._host!r} is in fault. Clear it from the Kinova Web App or with "
                "Base.ClearFaults() once the cause is fixed; a write now moves nothing."
            )
        return f"the base at {self._host!r} is in {name}, not ARMSTATE_SERVOING_READY"

    def _close(self, link: SimpleNamespace) -> None:
        for close in (getattr(link.session, "CloseSession", None), link.transport.disconnect):
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # the SDK raises KException or socket errors on a dead link
                logger.debug("KinovaDriver: closing the link raised %s", exc)

    def _watch(self) -> None:
        """Stop the arm once the joint-speed stream has gone quiet."""
        period = 1.0 / self._control_frequency
        with Ticker(period / 2, self._watchdog_stop) as ticker:
            while not ticker.wait():
                with self._lock:
                    last, link, sdk = self._last_write, self._link, self._sdk
                    if last is None or link is None or sdk is None:
                        continue
                    if time.monotonic() - last < DEADMAN_PERIODS * period:
                        continue
                    self._last_write, self._commanded = None, None
                try:
                    link.base.Stop()
                except (sdk.KException, OSError) as exc:
                    logger.warning("%s: the watchdog's Stop failed: %s", self._tool_name, exc)

    async def get_status(self) -> dict[str, Any]:
        """Report the session, the arm state and the limits the gates use."""
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "host": self._host,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "arm_state": self._arm_state,
                        "max_joint_speed": MAX_JOINT_SPEED,
                        "control_frequency": self._control_frequency,
                        "task_running": self._rollout is not None and self._rollout.is_running,
                        "battery_pct": None,
                    }
                }
            ],
        }

    def _halt(self) -> str | None:
        """Stop the arm with ``Base.Stop``; the stream re-anchors on the measured pose."""
        self._begin_halt()
        with self._lock:
            link, sdk = self._link, self._sdk
            self._commanded, self._last_write = None, None
        if link is None or sdk is None:
            return "not connected"
        try:
            link.base.Stop()
        except (sdk.KException, OSError) as exc:
            return f"Base.Stop raised {exc}"
        return None

    async def stop(self) -> None:
        """Halt motion, leaving the session open."""
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        if self.is_connected and (reason := self._halt()) is not None:
            logger.warning("%s.stop(): %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Stop the rollout, halt the arm, close the session. Idempotent."""
        self._begin_halt()
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
            rollout.join()
        if self.is_connected and (reason := self._halt()) is not None:
            logger.debug("KinovaDriver.cleanup(): %s", reason)
        self._watchdog_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
        with self._lock:
            link, self._link = self._link, None
        if link is not None:
            self._close(link)

    # Reads.

    def _feedback(self) -> tuple[Any, str | None]:
        with self._lock:
            link, sdk = self._link, self._sdk
        if link is None or sdk is None:
            return None, "not connected - call connect_eagerly() first"
        try:
            feedback = link.cyclic.RefreshFeedback()
        except (sdk.KException, OSError) as exc:
            return None, f"BaseCyclic.RefreshFeedback failed: {exc}"
        if len(feedback.actuators) != len(JOINT_NAMES):
            return None, f"the base reported {len(feedback.actuators)} actuators, expected {len(JOINT_NAMES)}"
        joints = [wrap_angle(math.radians(actuator.position)) for actuator in feedback.actuators]
        with self._lock:
            self._joints = dict(zip(JOINT_NAMES, joints, strict=True))
        return feedback, None

    def state(self) -> dict[str, Any]:
        """Read joints, velocities, torques and the tool pose in one envelope.

        Returns:
            ``joints``/``joint_velocities``/``joint_efforts`` keyed by
            :data:`JOINT_NAMES` (radians, rad/s, Nm), ``tool_pose`` as the base
            reports it (x, y, z in m, theta x, y, z in degrees) and the arm
            state; or a refusal naming the failure.
        """
        feedback, reason = self._feedback()
        if reason is not None:
            return refuse(f"state: {reason}")
        sdk = cast(SimpleNamespace, self._sdk)
        self._arm_state = sdk.Base_pb2.ArmState.Name(feedback.base.active_state)
        base = feedback.base
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": dict(self._joints),
                        "joint_velocities": {
                            name: math.radians(actuator.velocity)
                            for name, actuator in zip(JOINT_NAMES, feedback.actuators, strict=True)
                        },
                        "joint_efforts": {
                            name: float(actuator.torque)
                            for name, actuator in zip(JOINT_NAMES, feedback.actuators, strict=True)
                        },
                        "tool_pose": [
                            float(getattr(base, f"tool_pose_{axis}"))
                            for axis in ("x", "y", "z", "theta_x", "theta_y", "theta_z")
                        ],
                        "arm_state": self._arm_state,
                    }
                }
            ],
        }

    def get_observation(self) -> dict[str, float]:
        """The seven joint positions (name -> radians), for the mesh and the rollout."""
        self._feedback()
        with self._lock:
            return dict(self._joints)

    # Command path.

    def _begin_halt(self) -> None:
        with self._lock:
            self._halt_epoch += 1

    def _drop_anchor(self) -> None:
        with self._lock:
            self._commanded = None

    def _end_stream(self) -> None:
        """Rollout finish hook: a stream that ends leaves no speed running."""
        if self.is_connected and (reason := self._halt()) is not None:
            logger.warning("%s: stopping after the rollout failed: %s", self._tool_name, reason)

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command one joint-space setpoint as the joint speeds that reach it in one period.

        Gates, in order: this driver fronts ``robot_name``; the session is open;
        the arm is ``ARMSTATE_SERVOING_READY``; the action names only Gen3 joints
        with finite values and no step past :data:`MAX_JOINT_SPEED`; no halt
        landed while those gates ran. The speeds written are measured against
        the pose just read and clamped to :data:`MAX_JOINT_SPEED`.

        Args:
            action: Joint targets in radians keyed by :data:`JOINT_NAMES`. An
                omitted joint holds its value.
            robot_name: ``None`` or this driver's own name.

        Returns:
            A success envelope carrying the targets and the speeds written, or a refusal.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            halted_at, commanded = self._halt_epoch, self._commanded
        feedback, reason = self._feedback()
        if reason is not None:
            return refuse(f"send_action: {reason}")
        sdk, link = cast(SimpleNamespace, self._sdk), cast(SimpleNamespace, self._link)
        if (reason := self._state_refusal(sdk, feedback.base.active_state)) is not None:
            self._drop_anchor()
            return refuse(f"send_action: {reason}")
        with self._lock:
            measured = [self._joints[name] for name in JOINT_NAMES]
        reference = list(commanded) if commanded is not None else measured
        period = 1.0 / self._control_frequency
        targets, reason = targets_from_action(action, reference, max_step=MAX_JOINT_SPEED * period)
        if reason is not None:
            return refuse(f"send_action: {reason}")
        speeds = [
            max(-MAX_JOINT_SPEED, min(MAX_JOINT_SPEED, wrap_angle(target - now) / period))
            for target, now in zip(targets, measured, strict=True)
        ]
        command = sdk.Base_pb2.JointSpeeds(
            joint_speeds=[
                sdk.Base_pb2.JointSpeed(joint_identifier=index, value=math.degrees(speed), duration=0)
                for index, speed in enumerate(speeds)
            ]
        )
        with self._lock:
            superseded = self._halt_epoch != halted_at
        if superseded:
            return refuse("send_action: the arm was halted while this setpoint was being prepared; it was not written")
        try:
            link.base.SendJointSpeedsCommand(command)
        except (sdk.KException, OSError) as exc:
            return refuse(f"send_action: the base refused SendJointSpeedsCommand: {exc}")
        with self._lock:
            self._commanded = list(targets)
            self._last_write = time.monotonic()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": dict(zip(JOINT_NAMES, targets, strict=True)),
                        "joint_speeds": dict(zip(JOINT_NAMES, speeds, strict=True)),
                    }
                }
            ],
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

        Every step writes through :meth:`send_action`, so a base that faults
        mid-rollout ends it with that refusal as the exit reason, and the arm is
        stopped when the rollout ends for any reason.
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
            name=f"kinova-rollout-{self._tool_name}",
            policy=policy_object,
            instruction=instruction,
            duration=float(duration),
            n_steps=n_steps,
            period=1.0 / self._control_frequency,
            observe=self.get_observation,
            act=self.send_action,
            on_finish=self._end_stream,
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
            Success when the rollout left its loop and the base accepted
            ``Stop``; a refusal naming the failure otherwise; an error carrying
            ``stopped=False`` when the rollout thread did not join (the arm is
            stopped, the task still holds it).
        """
        self._begin_halt()
        rollout = self._rollout
        joined = True
        if rollout is not None and rollout.is_running:
            rollout.request_stop()
            joined = rollout.join()
        if not self.is_connected:
            return refuse("stop_task: not connected")
        if (reason := self._halt()) is not None:
            return refuse(f"stop_task: the base refused the halt: {reason}")
        steps = 0 if rollout is None else rollout.steps
        if not joined and rollout is not None:
            snapshot = {**rollout.snapshot(), "stopped": False, "robot": self._tool_name}
            snapshot["reason"] = (
                "stop_task: the rollout thread did not join; the arm is stopped, the task still holds it"
            )
            return {"status": "error", "content": [{"json": snapshot}]}
        return {"status": "success", "content": [{"json": {"stopped": True, "steps": steps, "robot": self._tool_name}}]}
