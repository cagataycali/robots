"""Native driver for the KUKA LBR iiwa 14, over the Fast Robot Interface (FRI).

``Robot("kuka_iiwa", mode="real")`` builds one of these. lerobot registers no
KUKA robot type, so before this driver the arm was simulation-only and
``mode="real"`` refused it by name.

FRI is a UDP session the Sunrise controller opens towards this machine once an
FRI application (a joint-position overlay, ``ClientCommandMode.POSITION``) is
running on the pendant. The controller sends a monitoring message every sample
period (1 to 5 ms) and the client must answer each one; in the
``COMMANDING_ACTIVE`` state the answer carries the absolute joint position the
arm goes to next. The client is ``pyfri``, lbr-stack's pybind11 binding of
KUKA's FRI Client SDK.

``pyfri``'s ``ClientApplication.step()`` blocks in ``recvfrom`` **holding the
GIL** until a datagram arrives, so the session runs in its own process: a
controller that goes quiet would otherwise freeze every thread of the caller,
policy and agent included. :mod:`strands_robots.drivers.kuka_session` is that
process: it publishes the robot state over one pipe every cycle and reads the
latest target :meth:`KukaDriver.send_action` writes over another.

Gated here, at the rate each limit is defined at:

* the session - a write is refused unless the session is ``COMMANDING_ACTIVE``
  in ``POSITION`` command mode, the safety state is ``NORMAL_OPERATION``, the
  drives are ``ACTIVE`` and the connection quality is at least ``GOOD``;
* the joint range - a target outside :data:`JOINT_LIMITS` is refused, because
  the controller answers one with a stop that ends the session;
* the step - a joint asked to travel further in one control period than
  :data:`MAX_JOINT_SPEED` allows is refused, measured from the last commanded
  target while a stream is in progress (the reason is in
  :meth:`~strands_robots.drivers.ur.URDriver._reference_pose`); and, every FRI
  cycle, the session process moves the commanded position towards the target
  by at most ``MAX_JOINT_SPEED * sample time``, the per-cycle check lbr-stack's
  ``CommandGuard`` closes the connection on;
* a halt that lands while a setpoint is being prepared - a counter bumped by
  every halt verb is re-read just before the write.

A halt holds the last commanded position. FRI positions do not expire, so no
watchdog is needed: a stream that goes quiet leaves the arm where it was sent.
Holding the controller's interpolated position instead would move the arm back
to the pose the Java motion holds, which is why it is never the halt target.

Deliberately absent: torque and wrench command modes, Cartesian overlays and
the media flange I/O.

``pyfri`` is not on PyPI; it builds from source against the FRI Client SDK
(``FRI_CLIENT_VERSION=1.15 pip install .`` in a clone of lbr-stack/pyfri). It
is imported in the session process only, so this module imports on any machine
and every test runs against a fake client application.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from strands_robots.drivers import kuka_session
from strands_robots.drivers.base import policy_step, refuse, undeclared_verb_error
from strands_robots.drivers.rollout import PolicyRollout, policy_from_provider
from strands_robots.utils import finite_number_error, positive_count_error, positive_finite_number_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver serves, read by ``strands_robots.drivers._SHIPPED_DRIVERS``.
SUPPORTED_ROBOTS: tuple[str, ...] = ("kuka_iiwa",)

#: Axis order A1..A7, base to flange. The MuJoCo asset names its joints the
#: same way, so a simulated action needs no remap.
JOINT_NAMES: tuple[str, ...] = tuple(f"joint{index}" for index in range(1, 8))

#: Joint range per axis, radians: the iiwa 14 rows of lbr-stack's
#: ``lbr_description/urdf/iiwa14/joint_limits.yaml`` (170, 120, 170, 120, 170,
#: 120, 175 degrees).
JOINT_LIMITS: tuple[float, ...] = tuple(math.radians(d) for d in (170, 120, 170, 120, 170, 120, 175))

#: Largest joint speed per axis, rad/s: the ``velocity`` rows of the same file
#: (85, 85, 100, 75, 130, 135, 135 degrees per second).
MAX_JOINT_SPEED: tuple[float, ...] = kuka_session.MAX_JOINT_SPEED

#: Default rollout cadence, hertz; sizes the step gate.
DEFAULT_CONTROL_FREQUENCY: float = 50.0

#: The UDP port the FRI application sends to (``FRIConfiguration`` default).
FRI_PORT: int = 30200

#: Seconds to wait for the first monitoring message before refusing the connect.
CONNECT_TIMEOUT: float = 5.0

#: FRI enum values, as the SDK's ``ESessionState``/``ESafetyState``/``EDriveState``/
#: ``EConnectionQuality``/``EClientCommandMode`` define them.
SESSION_STATES: tuple[str, ...] = (
    "IDLE",
    "MONITORING_WAIT",
    "MONITORING_READY",
    "COMMANDING_WAIT",
    "COMMANDING_ACTIVE",
)
SAFETY_STATES: tuple[str, ...] = (
    "NORMAL_OPERATION",
    "SAFETY_STOP_LEVEL_0",
    "SAFETY_STOP_LEVEL_1",
    "SAFETY_STOP_LEVEL_2",
)
DRIVE_STATES: tuple[str, ...] = ("OFF", "TRANSITIONING", "ACTIVE")
CONNECTION_QUALITIES: tuple[str, ...] = ("POOR", "FAIR", "GOOD", "EXCELLENT")
COMMAND_MODES: tuple[str, ...] = ("NO_COMMAND_MODE", "POSITION", "WRENCH", "TORQUE")

_N = kuka_session.N_JOINTS
_VECTORS = ("measured", "ipo", "commanded", "torque", "external_torque")
_SCALARS = ("cycles", "session", "safety", "drive", "quality", "command_mode", "sample_time")


def _enum_name(names: tuple[str, ...], value: float) -> str:
    index = int(value)
    return names[index] if 0 <= index < len(names) else f"UNKNOWN({index})"


def _launch(host: str | None, fri_port: int) -> tuple[Any, int, int]:
    """Start the session process (:mod:`strands_robots.drivers.kuka_session`).

    The child is a plain script run by this interpreter, so it imports neither
    this package nor the caller's ``__main__``; it inherits this process's import
    path so it finds the same ``pyfri``.

    Returns:
        ``(process, to_session_fd, from_session_fd)``; the process answers
        ``poll``/``terminate``/``wait`` like :class:`subprocess.Popen`.
    """
    to_child, inbox = os.pipe()[::-1]
    outbox, from_child = os.pipe()[::-1]
    command = [sys.executable, "-P", kuka_session.__file__, "--inbox", str(inbox), "--outbox", str(outbox)]
    command += ["--fri-port", str(fri_port), "--parent", str(os.getpid())] + (["--host", host] if host else [])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    try:
        process = subprocess.Popen(command, pass_fds=(inbox, outbox), env=env, stdout=sys.stderr)
    finally:
        os.close(inbox)
        os.close(outbox)
    # The session drains this pipe only between FRI steps, so a quiet controller
    # stops the draining; a blocking write would then wedge every halt verb. A
    # target frame is under PIPE_BUF, so a non-blocking write is still atomic.
    os.set_blocking(to_child, False)
    return process, to_child, from_child


def _resolve_sdk() -> str | None:
    """Return ``None`` when ``pyfri`` imports, or a reason naming the fix."""
    try:
        sdk = importlib.import_module("pyfri")
        for member in ("ClientApplication", "LBRClient"):
            getattr(sdk, member)
    except (ImportError, AttributeError) as exc:
        return (
            f"pyfri is not importable ({exc}). It is not on PyPI: clone github.com/lbr-stack/pyfri "
            "with --recursive and run FRI_CLIENT_VERSION=1.15 pip install . (match your Sunrise FRI version)"
        )
    return None


def targets_from_action(
    action: dict[str, Any],
    reference: list[float],
    *,
    period: float | None = None,
) -> tuple[list[float], str | None]:
    """Turn a joint-name-keyed action into one ordered position vector.

    A joint the action omits holds its reference value.

    Args:
        action: Joint targets in radians keyed by :data:`JOINT_NAMES` members.
        reference: The pose the step is measured from, in :data:`JOINT_NAMES` order.
        period: The control period the step must fit in, seconds. ``None``
            skips the step gate.

    Returns:
        ``(targets, None)`` or ``([], reason)``.
    """
    if len(reference) != _N:
        return [], f"reference holds {len(reference)} joint positions, expected {_N}"
    if not isinstance(action, dict) or not action:
        return [], f"nothing to command - the action names none of {list(JOINT_NAMES)}"
    unknown = sorted(set(action) - set(JOINT_NAMES))
    if unknown:
        return [], f"{unknown} name no iiwa joint; expected any of {list(JOINT_NAMES)}"
    targets = list(reference)
    for index, name in enumerate(JOINT_NAMES):
        if name not in action:
            continue
        if (reason := finite_number_error(action[name], name, "send_action")) is not None:
            return [], reason
        target = float(action[name])
        if abs(target) > JOINT_LIMITS[index]:
            return [], f"{name} target {target:.4f} rad is outside +/-{JOINT_LIMITS[index]:.4f} rad"
        step = abs(target - reference[index])
        if period is not None and step > MAX_JOINT_SPEED[index] * period:
            return [], (
                f"{name} asks for {step:.4f} rad in one control period, more than the "
                f"{MAX_JOINT_SPEED[index] * period:.4f} rad that {MAX_JOINT_SPEED[index]:.4f} rad/s allows. "
                "Slow the policy or lower control_frequency."
            )
        targets[index] = target
    return targets, None


class KukaDriver:
    """Native driver for the KUKA LBR iiwa 14 over FRI joint-position overlay.

    Args:
        tool_name: Name the agent invokes the driver by.
        cameras: Accepted for the factory contract; this driver opens none.
        data_config: Accepted for the factory contract; unused.
        port: The controller's address (the KONI interface, often
            ``192.170.10.2``) to accept datagrams from; ``None`` accepts the
            first controller that sends.
        fri_port: The local UDP port the FRI application sends to.
        control_frequency: Rollout cadence in hertz; sizes the step gate.
        connect_timeout: Seconds to wait for the first monitoring message.

    Raises:
        ValueError: If ``control_frequency`` or ``connect_timeout`` is not a
            positive finite number.
    """

    def __init__(
        self,
        tool_name: str = "kuka_iiwa",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        fri_port: int = FRI_PORT,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
        connect_timeout: float = CONNECT_TIMEOUT,
    ) -> None:
        del cameras, data_config
        for value, name in ((control_frequency, "control_frequency"), (connect_timeout, "connect_timeout")):
            if reason := positive_finite_number_error(value, name, "KukaDriver"):
                raise ValueError(reason)
        if reason := positive_count_error(fri_port, "fri_port", "KukaDriver"):
            raise ValueError(reason)
        self._tool_name = tool_name
        self._host = str(port) if port else None
        self._fri_port = int(fri_port)
        self._control_frequency = float(control_frequency)
        self._connect_timeout = float(connect_timeout)
        self._lock = threading.Lock()
        self._session: Any = None
        self._to_session: int | None = None
        self._latest: tuple[float, ...] | None = None
        self._reader: threading.Thread | None = None
        self._connect_error: str | None = None
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
        """Whether the FRI session process is running."""
        return self._session is not None and self._session.poll() is None

    @property
    def tool_spec(self) -> ToolSpec:
        """Read state, report status, stop. Motion goes through ``send_action``."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "KUKA LBR iiwa 14 native driver (FRI): reads measured and commanded joint positions and "
                    "torques, reports the FRI session, safety and drive state, and holds the arm."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joints and torques; status: FRI session, safety and drive state; "
                                    "stop: hold the last commanded position"
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
        """Start the FRI session process and wait for the controller's first message.

        Returns:
            ``None`` once the controller is monitoring, or a reason; the driver
            stays usable either way (reads report the reason, writes refuse).
        """
        if self.is_connected:
            return None
        if (reason := _resolve_sdk()) is not None:
            self._connect_error = f"KukaDriver: {reason}"
            return self._connect_error
        session, to_session, from_session = _launch(self._host, self._fri_port)
        with self._lock:
            self._session, self._to_session, self._latest, self._commanded = session, to_session, None, None
        reader = threading.Thread(
            target=self._read, args=(from_session,), name=f"kuka-fri-{self._tool_name}", daemon=True
        )
        self._reader = reader
        reader.start()
        deadline = time.monotonic() + self._connect_timeout
        while self._latest is None and session.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._latest is None:
            self._end_session()
            code = session.poll()
            if code == kuka_session.EXIT_BIND_FAILED:
                reason = f"could not bind UDP port {self._fri_port} for FRI (another client holds it?)"
            elif code == kuka_session.EXIT_BROKEN_BINDING:
                reason = (
                    "pyfri reported the first joint's position and torque for all seven, so nothing was commanded. "
                    "Its vendored pybind11 2.11 misreads arrays under numpy 2: rebuild pyfri against pybind11>=2.13 "
                    "(replace add_subdirectory(pybind) with find_package(pybind11 CONFIG REQUIRED) in its CMakeLists.txt)"
                )
            else:
                source = self._host or "any controller"
                reason = (
                    f"no FRI monitoring message from {source} on UDP {self._fri_port} within "
                    f"{self._connect_timeout:g} s. Start the FRI application on the Sunrise pendant and check its "
                    "FRIConfiguration names this machine's address and port."
                )
            self._connect_error = f"KukaDriver: {reason}"
            return self._connect_error
        self._connect_error = None
        return None

    def _read(self, fd: int) -> None:
        """Keep the newest state frame the session process writes, until it exits."""
        frame, pending = kuka_session.STATE, bytearray()
        try:
            while chunk := os.read(fd, 4096):
                pending.extend(chunk)
                whole = len(pending) // frame.size
                if whole:
                    latest = frame.unpack_from(pending, (whole - 1) * frame.size)
                    del pending[: whole * frame.size]
                    with self._lock:
                        self._latest = latest
        except OSError as exc:
            logger.debug("%s: the FRI state pipe closed: %s", self._tool_name, exc)
        finally:
            os.close(fd)

    def _write(self, target: list[float] | None, *, stop: bool = False) -> str | None:
        with self._lock:
            fd = self._to_session
        if fd is None:
            return "not connected"
        frame = kuka_session.TARGET.pack(*(target or [0.0] * _N), float(target is not None), float(stop))
        try:
            os.write(fd, frame)
        except BlockingIOError:
            return "the FRI session has stopped draining setpoints; the controller went quiet (check the FRI link)"
        except OSError as exc:
            return f"the FRI session process is gone: {exc}"
        return None

    def _end_session(self) -> None:
        """Ask the session process to stop, then signal it if it is blocked on a quiet controller."""
        self._write(None, stop=True)
        with self._lock:
            session, fd, reader = self._session, self._to_session, self._reader
            self._to_session = None
        if session is not None:
            try:
                session.wait(1.0)
            except subprocess.TimeoutExpired:
                # Blocked in recvfrom on a controller that went quiet; only a signal frees it.
                session.terminate()
                session.wait(1.0)
        if fd is not None:
            os.close(fd)
        if reader is not None:
            reader.join(1.0)

    def _snapshot(self) -> tuple[dict[str, Any] | None, str | None]:
        with self._lock:
            latest = self._latest
        if latest is None or not self.is_connected:
            detail = f" ({self._connect_error})" if self._connect_error else ""
            return None, f"not connected - call connect_eagerly() first{detail}"
        snapshot: dict[str, Any] = {name: list(latest[i * _N : (i + 1) * _N]) for i, name in enumerate(_VECTORS)}
        snapshot |= dict(zip(_SCALARS, latest[len(_VECTORS) * _N :], strict=True))
        return snapshot, None

    @staticmethod
    def _states(snapshot: dict[str, Any]) -> dict[str, str]:
        return {
            "session_state": _enum_name(SESSION_STATES, snapshot["session"]),
            "safety_state": _enum_name(SAFETY_STATES, snapshot["safety"]),
            "drive_state": _enum_name(DRIVE_STATES, snapshot["drive"]),
            "connection_quality": _enum_name(CONNECTION_QUALITIES, snapshot["quality"]),
            "client_command_mode": _enum_name(COMMAND_MODES, snapshot["command_mode"]),
        }

    def _command_refusal(self, snapshot: dict[str, Any]) -> str | None:
        """Name the FRI state that would not track a joint-position command, or ``None``."""
        states = self._states(snapshot)
        expected = {
            "session_state": "COMMANDING_ACTIVE",
            "client_command_mode": "POSITION",
            "safety_state": "NORMAL_OPERATION",
            "drive_state": "ACTIVE",
        }
        for field, wanted in expected.items():
            if states[field] != wanted:
                return (
                    f"the FRI {field.replace('_', ' ')} is {states[field]}, not {wanted}. Start a joint-position "
                    "overlay FRI application on the pendant (ClientCommandMode.POSITION) with the drives enabled."
                )
        if snapshot["quality"] < CONNECTION_QUALITIES.index("GOOD"):
            return f"the FRI connection quality is {states['connection_quality']}; the controller commands only at GOOD or better"
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the session process, the FRI states and the limits the gates use."""
        snapshot, _ = self._snapshot()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "host": self._host,
                        "fri_port": self._fri_port,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        **(self._states(snapshot) if snapshot is not None else {}),
                        "sample_time": None if snapshot is None else snapshot["sample_time"],
                        "max_joint_speed": dict(zip(JOINT_NAMES, MAX_JOINT_SPEED, strict=True)),
                        "control_frequency": self._control_frequency,
                        "task_running": self._rollout is not None and self._rollout.is_running,
                        "battery_pct": None,
                    }
                }
            ],
        }

    def _halt(self) -> str | None:
        """Hold the last commanded position; the stream re-anchors on it."""
        self._begin_halt()
        with self._lock:
            self._commanded = None
        if not self.is_connected:
            return "not connected"
        return self._write(None)

    async def stop(self) -> None:
        """Halt motion, leaving the session open."""
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        if self.is_connected and (reason := self._halt()) is not None:
            logger.warning("%s.stop(): %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Stop the rollout, hold the arm, end the session process. Idempotent."""
        self._begin_halt()
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
            rollout.join()
        self._end_session()
        with self._lock:
            self._session, self._latest, self._commanded = None, None, None

    # Reads.

    def state(self) -> dict[str, Any]:
        """Read measured and commanded joints, torques and the FRI states in one envelope.

        Returns:
            ``joints`` (measured), ``commanded_joints``, ``joint_efforts`` and
            ``external_torques`` keyed by :data:`JOINT_NAMES` (radians, Nm),
            plus the session, safety and drive states; or a refusal naming the failure.
        """
        snapshot, reason = self._snapshot()
        if reason is not None:
            return refuse(f"state: {reason}")
        snapshot = cast(dict[str, Any], snapshot)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": dict(zip(JOINT_NAMES, snapshot["measured"], strict=True)),
                        "commanded_joints": dict(zip(JOINT_NAMES, snapshot["commanded"], strict=True)),
                        "joint_efforts": dict(zip(JOINT_NAMES, snapshot["torque"], strict=True)),
                        "external_torques": dict(zip(JOINT_NAMES, snapshot["external_torque"], strict=True)),
                        "sample_time": snapshot["sample_time"],
                        **self._states(snapshot),
                    }
                }
            ],
        }

    def get_observation(self) -> dict[str, float]:
        """The seven measured joint positions (name -> radians), for the mesh and the rollout."""
        snapshot, _ = self._snapshot()
        return {} if snapshot is None else dict(zip(JOINT_NAMES, snapshot["measured"], strict=True))

    # Command path.

    def _begin_halt(self) -> None:
        with self._lock:
            self._halt_epoch += 1

    def _end_stream(self) -> None:
        """Rollout finish hook: a stream that ends holds where it was sent."""
        if self.is_connected and (reason := self._halt()) is not None:
            logger.warning("%s: holding after the rollout failed: %s", self._tool_name, reason)

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Set the joint-position target the FRI session moves the arm to.

        Gates, in order: this driver fronts ``robot_name``; the session process
        is running; the FRI session is commanding in ``POSITION`` mode with
        normal safety, active drives and a good connection; the action names
        only iiwa joints with finite values inside :data:`JOINT_LIMITS` and no
        step past :data:`MAX_JOINT_SPEED`; no halt landed while those gates ran.

        Args:
            action: Joint targets in radians keyed by :data:`JOINT_NAMES`. An
                omitted joint holds its value.
            robot_name: ``None`` or this driver's own name.

        Returns:
            A success envelope carrying the targets, or a refusal.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            halted_at, commanded = self._halt_epoch, self._commanded
        snapshot, reason = self._snapshot()
        if reason is not None:
            return refuse(f"send_action: {reason}")
        snapshot = cast(dict[str, Any], snapshot)
        if (reason := self._command_refusal(snapshot)) is not None:
            with self._lock:
                self._commanded = None
            return refuse(f"send_action: {reason}")
        reference = list(commanded) if commanded is not None else list(snapshot["commanded"])
        targets, reason = targets_from_action(action, reference, period=1.0 / self._control_frequency)
        if reason is not None:
            return refuse(f"send_action: {reason}")
        with self._lock:
            superseded = self._halt_epoch != halted_at
        if superseded:
            return refuse("send_action: the arm was halted while this setpoint was being prepared; it was not written")
        if (reason := self._write(targets)) is not None:
            return refuse(f"send_action: {reason}")
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

        Every step writes through :meth:`send_action`, so a session that leaves
        ``COMMANDING_ACTIVE`` mid-rollout ends it with that refusal as the exit
        reason, and the arm holds when the rollout ends for any reason.
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
            name=f"kuka-rollout-{self._tool_name}",
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
        """Stop the rollout and hold the arm.

        Returns:
            Success when the rollout left its loop and the arm holds; a refusal
            when not connected; an error carrying ``stopped=False`` when the
            rollout thread did not join (the arm holds, the task still owns it).
        """
        self._begin_halt()
        rollout = self._rollout
        joined = True
        if rollout is not None and rollout.is_running:
            rollout.request_stop()
            joined = rollout.join()
        if (reason := self._halt()) is not None:
            return refuse(f"stop_task: {reason}")
        steps = 0 if rollout is None else rollout.steps
        if not joined and rollout is not None:
            snapshot = {**rollout.snapshot(), "stopped": False, "robot": self._tool_name}
            snapshot["reason"] = "stop_task: the rollout thread did not join; the arm holds, the task still owns it"
            return {"status": "error", "content": [{"json": snapshot}]}
        return {"status": "success", "content": [{"json": {"stopped": True, "steps": steps, "robot": self._tool_name}}]}
