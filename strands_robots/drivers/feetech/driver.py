"""Native Feetech STS/SMS-series driver satisfying :class:`HardwareDriver`.

This driver drives the arm. :mod:`~strands_robots.drivers.feetech.bus` opens
the SCS serial port, so ``send_action`` writes goal positions and the joints
reach the mesh - that is :issue:`360` scope 1, which the driver named as
deferred when it landed as a stub against the driver seam (:issue:`353` /
:pr:`2734`).

What works and what does not:

* ``send_action`` - writes the commanded joints in one SYNC_WRITE frame.
  Targets are **degrees** (``gripper`` is percent open); a key may be spelled
  ``shoulder_pan`` or ``shoulder_pan.pos``, matching lerobot's suffix - but only
  one of the two per motor, because both name the same servo.
* ``bus`` / ``is_connected`` - the pair
  :func:`strands_robots.bus_access.joint_read_source` resolves, so an SO-arm
  publishes ``joints`` on the mesh state topic without a wrapper. This is the
  documented seam for a driver that owns its motors; a ``state()`` method is
  not part of :data:`~strands_robots.drivers.base.DRIVER_SURFACE` and no
  shipped driver has one.
* ``stop`` - releases torque on every motor, reporting any that stayed driven.
* ``run_policy`` / ``start_task`` - roll a policy out on a background thread at
  ``control_frequency`` (30 Hz by default, the SO-arm teleop rate). Each step
  reads the whole arm in one sync-read, hands the policy a lerobot-shaped
  ``{"<joint>.pos": degrees}`` observation and commands its action through
  ``send_action``, so a setpoint the bus refuses ends the rollout with that
  refusal as its exit reason. ``get_task_status`` reports the live snapshot and
  ``stop_task`` halts the loop, leaving the arm energized where it stands -
  ``stop`` is still the verb that de-energizes. The loop is
  :class:`~strands_robots.drivers.rollout.PolicyRollout`, shared with the UR
  driver rather than copied.

* ``transport="twin"`` - the same driver, with the arm's MuJoCo model at the
  far end of the bus (:mod:`~strands_robots.drivers.feetech.twin`). Every verb,
  unit and refusal above is unchanged; only the seam is. ``sim=`` hands in an
  engine already carrying the arm and ``realtime=`` steps it at wall-clock
  speed. The default transport is the serial bus, and nothing about it moves.

None of this pretends. Every refusal returns an envelope of the same shape a
successful path returns, so the mesh and the agent read one shape either way.

The class is registered for every Feetech robot the package registry knows
about - see :func:`~strands_robots.drivers._register_shipped_drivers` for the
list. Registering after import (``from strands_robots.drivers.feetech import
FeetechDriver`` then :func:`register_native_driver`) is also supported and
is how an out-of-tree driver package would extend the table.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncGenerator, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

from strands_robots.bus_access import bus_lock, read_joints
from strands_robots.drivers.base import halt_failure_detail, policy_step, undeclared_verb_error
from strands_robots.drivers.feetech.bus import (
    DEFAULT_TIMEOUT_S,
    SO_ARM_MOTORS,
    FeetechBus,
    MotorCalibration,
    load_calibration,
)
from strands_robots.drivers.rollout import PolicyRollout, policy_from_provider
from strands_robots.utils import boolean_flag_error, positive_count_error, positive_finite_number_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The robots this driver serves. Every entry corresponds to a canonical name
# in ``strands_robots/registry/robots.json``. The list is deliberately narrow:
# a Feetech driver could in principle serve every arm on that bus, but the
# ones :issue:`360` names are the ones the acceptance criteria measure, and
# registering for a robot we cannot verify is a promise this driver does not
# yet keep.
#
# Mirrors :data:`~strands_robots.drivers.dynamixel.driver.SUPPORTED_ROBOTS`
# in shape and spirit: canonical names, deliberately excluding any robot the
# scope note does not name.
# ---------------------------------------------------------------------------
# ``hope_jr`` and ``open_duck_mini`` share the SCS bus but NOT the six-servo
# SO-arm layout, and no joint map for either is established in this package. A
# driver built for one of them gets :data:`SO_ARM_MOTORS` by default, which
# names joints it does not have; pass ``motor_ids=`` (or a ``motors=`` map on
# the bus) until a verified map for those two lands. Said here rather than
# discovered on the wire.
SUPPORTED_ROBOTS: tuple[str, ...] = (
    "so100",
    "so101",
    "lekiwi",
    "hope_jr",
    "open_duck_mini",
)

_TOOL_TYPE = "robot"

#: How the driver reaches its servos: the SCS serial bus, or the arm's MuJoCo
#: model answering the same bus (:class:`~strands_robots.drivers.feetech.twin.FeetechTwinBus`).
TRANSPORTS: tuple[str, ...] = ("serial", "twin")

#: Steps per second a rollout commands when the caller names no other rate. The
#: SO-arm teleop rate, and what an ACT or pi0 checkpoint trained on lerobot SO
#: data expects: a Feetech servo at its default speed reaches a goal within one
#: 33 ms period, so a faster loop commands a setpoint the arm has not arrived at.
DEFAULT_CONTROL_FREQUENCY: float = 30.0


class FeetechDriver:
    """Native Feetech driver for the arms in :data:`SUPPORTED_ROBOTS`.

    The bus writes STS/SMS-series frames: that series' two-byte word order and
    its 12-bit ``Goal_Position`` full scale both come from
    :mod:`~strands_robots.drivers.feetech.protocol`. An SCS-series servo reads
    the same two bytes in the opposite order, so a bus of that series is not
    addressed by this driver - which is why the one SCS-series robot the registry
    declares, ``hope_jr_hand``, is absent from :data:`SUPPORTED_ROBOTS`.

    Constructor contract matches :class:`~strands_robots.drivers.base.HardwareDriver`
    - the factory builds every native driver as ``driver_cls(tool_name=...,
    cameras=..., data_config=..., **kwargs)`` and the driver declares every
    further keyword it honours, so the factory refuses one it does not. The
    Feetech-specific keywords:

    * ``port`` - a serial device path (``/dev/tty.usbserial-*``) for the SCS
      bus. Optional at construction; the bus opens it on connect.
    * ``baud_rate`` - a positive integer, defaults to ``1_000_000``. The Feetech
      default for STS3215 arms; SCS-series can also run at 500_000 or below and
      a caller who knows better passes it here. Held to
      :func:`~strands_robots.utils.positive_count_error` - the domain
      :mod:`~strands_robots.tools.serial_tool` holds its own ``baudrate`` to -
      because pyserial coerces the speed rather than checking it, so a value
      that is not a count is applied instead of refused.
    * ``motor_ids`` - the servo IDs on the bus, in wire order. Optional at
      construction; the bus discovers them on connect.
    * ``calibration`` - this arm's own calibration: the path of the JSON
      ``lerobot-calibrate`` wrote (see
      :func:`~strands_robots.drivers.feetech.bus.lerobot_calibration_path`), or
      the records themselves. Omitting it reads and commands the servo's full
      travel instead of the arm's measured travel, which is off by however far
      the stops sit inside that rotation - so an arm that HAS been calibrated
      should be given the file, and ``get_status`` reports which of the two is
      in force under ``calibration_source``.
    * ``timeout`` - how long a read waits for a servo's reply, in seconds,
      defaulting to :data:`~strands_robots.drivers.feetech.bus.DEFAULT_TIMEOUT_S`.
      Forwarded to the bus rather than recorded, for the reason ``motor_ids`` is:
      a caller who lengthens the window for a slow servo, and gets the default
      window and six motors that did not answer, has been told nothing. Held to
      :func:`~strands_robots.utils.positive_finite_number_error`.
    * ``transport`` - one of :data:`TRANSPORTS`, default ``"serial"``. ``"twin"``
      builds the bus over the arm's MuJoCo model instead of a serial port
      (:mod:`~strands_robots.drivers.feetech.twin`): the same verbs, units and
      refusals, with ``sensors`` reading the model's joints. The model is the
      one the registry names for ``tool_name``, so ``tool_name`` must resolve to
      a robot in :data:`SUPPORTED_ROBOTS` with a simulation asset - refused by
      name otherwise (``hope_jr`` has no asset) - unless ``sim=`` hands in an
      engine already carrying the arm, whose robot then names the entry.
    * ``sim`` - ``twin`` only: a built sim engine (what ``Robot("so101",
      mode="sim")`` returns). ``None`` builds one on first connect; the driver
      destroys an engine it built and leaves a caller's alone.
    * ``realtime`` - ``twin`` only: step the model at wall-clock speed so a
      viewer sees the motion as the arm would make it. Default ``False``.
    """

    tool_type = _TOOL_TYPE

    def __init__(
        self,
        tool_name: str,
        cameras: Any | None = None,
        data_config: Any | None = None,
        *,
        port: str | None = None,
        ports: Any = None,
        baud_rate: int = 1_000_000,
        timeout: float = DEFAULT_TIMEOUT_S,
        calibration: str | Path | dict[str, MotorCalibration] | None = None,
        motor_ids: Sequence[int] = (),
        transport: str = "serial",
        sim: Any = None,
        realtime: bool = False,
    ) -> None:
        self._tool_name = tool_name
        # Discarded, not stored: this driver never opens a caller-supplied
        # camera, and the factory refuses a non-empty ``cameras=`` for a driver
        # that does not declare ``reads_cameras``. An attribute nothing reads
        # only suggests otherwise.
        del cameras
        self._data_config = data_config
        # A Feetech arm today is one U-shape bus. Aloha-style bimanual rigs
        # are Dynamixel not Feetech, so we accept a single ``port`` and refuse
        # ``ports`` outright rather than pretend to multi-bus a family that
        # does not need it. The keyword is still declared so a caller
        # mis-passing ``ports=[...]`` gets a named refusal rather than the
        # roster of keywords this driver does read.
        if ports is not None:
            raise ValueError(
                f"FeetechDriver({tool_name!r}): pass port= for the Feetech bus; "
                f"multi-bus rigs are not part of {SUPPORTED_ROBOTS}",
            )
        self._port: str | None = port
        # Graded, not coerced. pyserial takes the speed through its own
        # ``int()`` and refuses only a negative, so a value this constructor
        # converted was applied rather than reported: ``2.7`` opened the port at
        # 2 baud and ``0`` opened it successfully at a speed no servo answers,
        # while ``get_status`` reported the converted number as the configured
        # one. The same domain :mod:`~strands_robots.tools.serial_tool` holds
        # its ``baudrate`` to, because the two reach the same ``serial.Serial``.
        if (reason := positive_count_error(baud_rate, "baud_rate", f"FeetechDriver({tool_name!r})")) is not None:
            raise ValueError(reason)
        self._baud_rate: int = baud_rate
        # Forwarded, not recorded. The bus has this knob, so a ``timeout`` this
        # constructor accepted and kept to itself would not be an extension
        # waiting for a downstream package - it is a window the caller set and
        # the bus never saw.
        if (reason := positive_finite_number_error(timeout, "timeout", f"FeetechDriver({tool_name!r})")) is not None:
            raise ValueError(reason)
        self._timeout: float = float(timeout)
        # The arm's own measured travel, or None for the servo's full rotation.
        # Loaded here rather than in the bus so a path that is not a
        # calibration is refused while the caller still has the traceback that
        # names their keyword, and so ``get_status`` can report the source.
        self._calibration_source: str | None = None
        records: dict[str, MotorCalibration] | None = None
        if isinstance(calibration, str | Path):
            self._calibration_source = str(calibration)
            records = load_calibration(calibration)
        elif isinstance(calibration, dict):
            self._calibration_source = "caller"
            records = calibration
        elif calibration is not None:
            raise ValueError(
                f"FeetechDriver({tool_name!r}): calibration must be a path to the JSON "
                f"lerobot-calibrate wrote, or the records themselves, got {type(calibration).__name__}",
            )
        self._motor_ids: tuple[int, ...] = tuple(motor_ids)
        # ``motor_ids`` narrows the arm to a subset of SO_ARM_MOTORS. Honoured
        # rather than recorded: a keyword that changes nothing is worse than one
        # that is refused, because the caller believes the arm is configured.
        # An ID this driver has no joint name for is refused for the same
        # reason - we would otherwise command a motor we cannot name.
        motors = dict(SO_ARM_MOTORS)
        if self._motor_ids:
            known = {spec.motor_id: name for name, spec in SO_ARM_MOTORS.items()}
            if unknown := sorted(set(self._motor_ids) - set(known)):
                raise ValueError(
                    f"FeetechDriver({tool_name!r}): motor_ids {unknown} are not on an SO-arm; "
                    f"ids {sorted(known)} map to {[known[i] for i in sorted(known)]}",
                )
            motors = {known[i]: SO_ARM_MOTORS[known[i]] for i in self._motor_ids}
        # The seam. ``"serial"`` is the shipped default and is untouched by the
        # twin's knobs; ``"twin"`` builds the same bus surface over the model.
        context = f"FeetechDriver({tool_name!r})"
        if transport not in TRANSPORTS:
            raise ValueError(f"{context}: transport must be one of {list(TRANSPORTS)}, got {transport!r}")
        if sim is not None and transport != "twin":
            raise ValueError(f"{context}: sim= is the twin transport's engine; pass transport='twin' with it")
        if (reason := boolean_flag_error(realtime, "realtime", context)) is not None:
            raise ValueError(reason)
        self._transport: str = transport
        self._bus: FeetechBus
        if transport == "twin":
            from strands_robots.drivers.feetech.twin import FeetechTwinBus  # noqa: PLC0415 - imports this module

            self._bus = FeetechTwinBus(
                _twin_robot(tool_name, sim, context),
                motors=motors,
                timeout=self._timeout,
                calibration=records,
                sim=sim,
                realtime=bool(realtime),
            )
        else:
            self._bus = FeetechBus(
                port=self._port,
                baud_rate=self._baud_rate,
                motors=motors,
                timeout=self._timeout,
                calibration=records,
            )
        self._connect_error: str | None = None
        self._rollout: PolicyRollout | None = None
        # Held across the running check, the reference assignment and ``start``,
        # so two callers cannot both pass the check and then stream setpoints
        # from two rollouts onto one half-duplex bus.
        self._task_admission = threading.Lock()

    # ------------------------------------------------------------------ #
    # Tool surface.                                                       #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """Name the agent invokes this robot by."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Schema describing the actions the agent may request.

        Every verb here reaches the bus. ``home`` is deliberately absent: a
        home pose is a per-arm calibration this driver does not own, and
        refusing a verb the schema declares is worse than not declaring it - an
        agent that plans against the schema will pick a verb it sees.
        """
        return {
            "name": self._tool_name,
            "description": (
                f"Feetech-native driver for {self._tool_name} (STS/SMS series). Joint targets are degrees; "
                "gripper is percent open."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "sensors", "move_to", "set_torque", "stop"],
                            "description": (
                                "status: connection + motor list; sensors: read joint positions; "
                                "move_to: command joints (targets, degrees); "
                                "set_torque: energize or release (enabled); stop: release torque."
                            ),
                        },
                        "targets": {
                            "type": "object",
                            "description": "move_to only: joint name -> degrees (gripper: percent open).",
                            "additionalProperties": {"type": "number"},
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "set_torque only: true energizes, false releases.",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result.

        Follows the shape :class:`DynamixelDriver` uses for its own deferred
        motion path so a caller writes the same error-checking code either
        way.
        """
        del kwargs  # forward-compat only
        del invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "status")
        if action == "status":
            envelope = {
                "status": "success",
                "content": [{"json": await self.get_status()}],
            }
        elif action == "sensors":
            envelope = self._read_joints_envelope()
        elif action == "move_to":
            envelope = self.send_action((tool_use.get("input") or {}).get("targets") or {})
        elif action == "set_torque":
            enabled = (tool_use.get("input") or {}).get("enabled", True)
            # Checked before use, never coerced into one: every non-boolean an
            # agent actually emits for this field reads as the WRONG state.
            # "false", "no" and "0" are all truthy strings, so a caller asking
            # to release an arm would energize it instead while the envelope
            # reported torque_enabled=True as a success. A refusal naming the
            # field is recoverable; a silently inverted torque command on a
            # loaded arm is not.
            #
            # Through `boolean_flag_error` rather than `isinstance(x, bool)` so
            # the accepted domain matches every other posture flag in the
            # package: that helper also admits a numpy boolean, which an
            # `isinstance` check refuses. A policy or array path handing over
            # `np.bool_(False)` is a legitimate release request, and turning it
            # into a refusal would leave an operator unable to de-energize.
            # `bool()` below narrows the accepted value for the type checker;
            # it runs only after the check has already ruled the value in.
            if text := boolean_flag_error(enabled, "enabled", "set_torque"):
                envelope = _refuse(text)
            else:
                envelope = self._set_torque_envelope(bool(enabled))
        elif action == "stop":
            # The rollout is halted before the arm is released: a loop still
            # streaming setpoints would have its next write refused by a bus
            # whose torque is off, reporting a policy fault for what was an
            # operator's stop. The halt's own verdict is reported when it is not
            # a success rather than restated here - ``stop_task`` decided it.
            halt = self.stop_task()
            envelope = halt if halt["status"] != "success" else self._set_torque_envelope(False)
        else:
            envelope = undeclared_verb_error(self, action)
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Motion, task and policy paths. All refuse in the same envelope.     #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command joint positions over the SCS bus.

        Args:
            action: Joint name -> target. Degrees for every joint, percent open
                for ``gripper``. A ``.pos`` suffix is accepted and stripped, so
                a lerobot-shaped action dict works unchanged. Each motor must be
                spelled once: ``{"gripper": 0.0, "gripper.pos": 100.0}`` names one
                motor twice with two different targets and is refused rather
                than letting insertion order pick the winner.
            robot_name: Unused; this driver fronts exactly one arm.

        Returns:
            A success envelope naming the joints commanded, or an error
            envelope. Never raises: a driver is invoked as an agent tool, and
            an exception past dispatch is not something the caller can handle.
        """
        del robot_name
        if not isinstance(action, dict) or not action:
            return _refuse("send_action: pass a non-empty mapping of joint targets")
        targets, doubled = _motor_targets(action)
        if doubled is not None:
            return _refuse(doubled)
        try:
            with bus_lock(self):
                self._connect_if_needed()
                self._bus.write_goal_positions(targets)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return _refuse(f"send_action: {e}")
        body: dict[str, Any] = {"commanded": targets, "unit": "degrees (gripper: percent open)"}
        # The twin reports a target the model's travel clamped (design: "the
        # model's limits are reported"); the serial bus never sets this, so a
        # serial reply is byte for byte what it was.
        if note := getattr(self._bus, "last_clamp_note", ""):
            body["note"] = note
        return {"status": "success", "content": [{"json": body}]}

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Build a policy from the provider registry and roll it out on the arm.

        Args:
            instruction: Natural-language instruction handed to the policy.
            policy_port: Port the policy server listens on; ``None`` uses the
                provider's default.
            policy_host: Host the policy server runs on.
            policy_provider: Which provider to build, by registry name.
            duration: Wall-clock budget for the rollout, in seconds.
            **policy_kwargs: Extra provider-specific policy options.

        Returns:
            The envelope :meth:`run_policy` returns for the rollout it started,
            or a refusal naming the provider that could not be built - or the
            keyword the registry says that provider needs and did not get. A
            refusal rather than a raise, and *before* the rollout starts: the
            verb is reached as an agent tool, and a rollout that answered
            "started" and then faulted at its first action would hold an
            energized arm it can never step. See
            :func:`~strands_robots.drivers.rollout.policy_from_provider`.
        """
        kwargs: dict[str, Any] = {"host": policy_host, **policy_kwargs}
        if policy_port is not None:
            kwargs["port"] = policy_port
        policy, reason = policy_from_provider(
            policy_provider,
            kwargs,
            "start_task",
            "the rollout would start on an energized arm and fail at its first action",
        )
        if reason is not None:
            return _refuse(reason)
        return self.run_policy(policy, instruction=instruction, duration=duration)

    def run_policy(
        self,
        policy_object: Policy | Callable[[dict[str, Any]], dict[str, Any]],
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
    ) -> dict[str, Any]:
        """Roll an already-built policy out on the arm, on a background thread.

        Each step reads the whole arm in one sync-read, hands the policy that
        observation and commands the action it answers through
        :meth:`send_action` - so the rollout crosses the same target-domain and
        torque gates a hand-written setpoint does, and a frame the bus refuses
        ends the rollout with that refusal as its exit reason rather than being
        retried against a bus that just said no.

        The observation is shaped ``{"<joint>.pos": value}`` - what lerobot's own
        SO-arm robot class answers and what a checkpoint trained on lerobot SO
        data reads - in the driver's units: degrees, percent open for the
        gripper. It is read through
        :func:`~strands_robots.bus_access.read_joints`, so the rollout shares the
        one bus lock with the mesh readers instead of colliding with them.

        Args:
            policy_object: The policy to roll out. A built
                :class:`~strands_robots.policies.Policy`, an object exposing
                ``step(observation)``, or a bare callable - the set
                :func:`~strands_robots.drivers.base.policy_step` resolves, which
                is the same set the refusal below names.
            instruction: Natural-language instruction handed to the policy on
                every step.
            duration: Wall-clock budget in seconds.
            n_steps: Step budget; when given it wins over ``duration``.
            control_frequency: Steps per second, defaulting to
                :data:`DEFAULT_CONTROL_FREQUENCY`. Graded rather than coerced: a
                zero or a nan becomes the loop's period, and a period that is not
                a positive number paces the arm at whatever the float divides to.

        Returns:
            A success envelope describing the rollout that started - poll
            :meth:`get_task_status` for its progress - or a refusal. Never
            raises.
        """
        if err := positive_finite_number_error(duration, "duration", "run_policy"):
            return _refuse(err)
        if err := positive_finite_number_error(control_frequency, "control_frequency", "run_policy"):
            return _refuse(err)
        if n_steps is not None and (err := positive_count_error(n_steps, "n_steps", "run_policy")):
            return _refuse(err)
        if policy_object is None:
            return _refuse("run_policy: policy_object is required")
        if policy_step(policy_object, instruction) is None:
            return _refuse("run_policy: policy_object must be callable or expose get_actions_sync() or step()")
        # The bus opens here rather than on the worker thread, so a port that
        # cannot be opened is this verb's refusal instead of a rollout that
        # reports "started" and ends at step 0 with the same reason.
        if reason := self.connect_eagerly():
            return _refuse(f"run_policy: {reason}")

        rollout = PolicyRollout(
            name=f"feetech-rollout-{self._tool_name}",
            policy=policy_object,
            instruction=instruction,
            duration=float(duration),
            n_steps=n_steps,
            period=1.0 / float(control_frequency),
            observe=lambda: read_joints(self),
            act=self.send_action,
        )
        with self._task_admission:
            if self._rollout is not None and self._rollout.is_running:
                return _refuse("run_policy: a task is already running; call stop_task first")
            self._rollout = rollout
            rollout.start()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "instruction": instruction,
                        "control_frequency": float(control_frequency),
                        "duration": duration,
                        "n_steps": n_steps,
                        "transport": self._transport,
                    }
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
        """Report the rollout's progress, or that none has run.

        The snapshot outlives the thread, so a poller arriving after the loop
        finished still reads what ended it (``exit_reason``, and the bus's own
        refusal when that is what ended it) rather than an empty answer.
        """
        rollout = self._rollout
        if rollout is None:
            return {"status": "success", "content": [{"json": {"running": False, "steps": 0}}]}
        return {"status": "success", "content": [{"json": rollout.snapshot()}]}

    def stop_task(self) -> dict[str, Any]:
        """Halt the rollout, leaving the arm energized where it stands.

        Torque is deliberately kept: an SO arm holding a payload would drop it,
        and ``stop`` is the verb that de-energizes. What stops is the stream of
        setpoints.

        Returns:
            A success envelope naming the steps that ran when the loop left -
            including when nothing was running, because an idempotent stop is
            what a caller tearing down needs. An *error* envelope carrying
            ``stopped=False`` when the thread is still in the loop: a
            caller-supplied policy blocking on a remote inference call is the
            ordinary case, and claiming a halt that :meth:`get_task_status`
            would contradict is the one thing this verb will not do. No further
            setpoint reaches the bus either way - the loop re-reads the stop
            signal after the policy returns and before it commands.
        """
        rollout = self._rollout
        if rollout is None:
            return {"status": "success", "content": [{"json": {"stopped": True, "steps": 0, "robot": self._tool_name}}]}
        if rollout.is_running:
            rollout.request_stop()
            if not rollout.join():
                snapshot = rollout.snapshot()
                snapshot["stopped"] = False
                snapshot["robot"] = self._tool_name
                snapshot["reason"] = (
                    "stop_task: the rollout thread did not join within its budget; the policy is "
                    "likely blocking - the loop commands no further setpoint, but the task is "
                    "still holding the arm"
                )
                return {"status": "error", "content": [{"json": snapshot}]}
        return {
            "status": "success",
            "content": [{"json": {"stopped": True, "steps": rollout.steps, "robot": self._tool_name}}],
        }

    def cleanup(self) -> None:
        """Close the serial port - or, on the twin, destroy an engine the driver built.

        Idempotent, and safe on a driver that never connected: the bus tracks
        whether it holds an open handle. Torque is deliberately left as it is -
        releasing it here would drop an arm holding a payload when a caller
        merely tore down a process; ``stop`` is the verb that de-energizes. An
        engine handed in through ``sim=`` is the caller's and is left alone.

        A rollout is halted first: this hook goes on to release the port that
        loop writes through, so a thread left in it would command an arm nothing
        in the process can still reach. The halt carries no verdict here either
        (``-> None``), so a refused one is logged.
        """
        if detail := halt_failure_detail(self.stop_task()):
            logger.error("%s: cleanup released the bus under a live rollout: %s", self._tool_name, detail)
        with bus_lock(self):
            self._bus.disconnect()

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                               #
    # ------------------------------------------------------------------ #

    @property
    def bus(self) -> FeetechBus:
        """The motor bus, for :func:`strands_robots.bus_access.joint_read_source`.

        Exposing a ``bus`` with ``sync_read`` alongside :attr:`is_connected` is
        the documented way a native driver publishes ``joints`` on the mesh
        state topic - see :func:`strands_robots.bus_access.read_joints`, which
        prefers this over a full observation so a dead camera cannot hide the
        joint positions.

        Every path in this class that touches the bus holds
        :func:`~strands_robots.bus_access.bus_lock` on ``self`` - the same lock
        :func:`~strands_robots.bus_access.read_joints` takes for the read it
        does through this property. A driver-side write outside that lock would
        interleave with a mesh-side read on a half-duplex bus and corrupt both
        frames, which is the collision that module exists to prevent.
        """
        return self._bus

    @property
    def is_connected(self) -> bool:
        """Whether the serial port is open, so a consumer can tell live from stale."""
        return self._bus.is_connected

    @property
    def transport(self) -> str:
        """Which seam the driver is on - one of :data:`TRANSPORTS`."""
        return self._transport

    @property
    def endpoint(self) -> str | None:
        """Where the driver reaches its servos: the serial path, or ``sim://<robot>`` on the twin."""
        return self._bus.port if self._transport == "twin" else self._port

    @property
    def sim(self) -> Any | None:
        """The engine behind the ``twin`` transport - for ``render`` and the like - else ``None``."""
        return getattr(self._bus, "sim", None) if self._transport == "twin" else None

    def _connect_if_needed(self) -> None:
        """Open the bus on first use, recording the reason when it fails."""
        if self._bus.is_connected:
            return
        try:
            self._bus.connect()
        except (ValueError, OSError) as e:
            self._connect_error = str(e)
            raise
        self._connect_error = None

    def _read_joints_envelope(self) -> dict[str, Any]:
        """Read joint positions into a tool envelope."""
        try:
            with bus_lock(self):
                self._connect_if_needed()
                joints = self._bus.sync_read()
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return _refuse(f"sensors: {e}")
        return {
            "status": "success",
            "content": [{"json": {"joint_state": joints, "unit": "degrees (gripper: percent open)"}}],
        }

    def _set_torque_envelope(self, enabled: bool) -> dict[str, Any]:
        """Energize or release the arm, reporting any motor that stayed driven."""
        try:
            with bus_lock(self):
                self._connect_if_needed()
                failed = self._bus.set_torque(enabled)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return _refuse(f"set_torque: {e}")
        if failed:
            # A partial release is a safety fact, not a success: say which
            # joints are still driven rather than reporting the arm released.
            return _refuse(f"set_torque({enabled}): these motors did not answer and may still be driven: {failed}")
        return {"status": "success", "content": [{"json": {"torque_enabled": enabled}}]}

    def connect_eagerly(self) -> str | None:
        """Open the serial bus now, or name why it could not open.

        Returns:
            ``None`` once the port is open, otherwise the failure reason. A
            named string rather than a raise, because a caller cannot tell a
            raise here from a real hardware fault mid-session.
        """
        try:
            # Under the lock like every other bus path: opening the port writes
            # to it (a torque-enable sweep on first contact), and this is the
            # one `_connect_if_needed` caller outside an already-locked block.
            # A lock only guarantees anything where EVERY caller takes it.
            with bus_lock(self):
                self._connect_if_needed()
        except (ValueError, OSError) as e:
            return str(e)
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's construction and configuration.

        Shape matches :meth:`DynamixelDriver.get_status` so both peers publish
        identically; fields absent on a Feetech bus (an FSM, a battery
        percentage) are simply not in the payload.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "tool_type": self.tool_type,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "transport": self._transport,
                        "endpoint": self.endpoint,
                        "port": self._port,
                        "baud_rate": self._baud_rate,
                        "motors": {name: spec.motor_id for name, spec in self._bus.motors.items()},
                        # Which travel the degrees on this bus are measured
                        # against. ``None`` is the servo's full rotation, and a
                        # caller who calibrated the arm reads that as the
                        # keyword they forgot rather than as a wrong number.
                        "calibration_source": self._calibration_source,
                        "motor_ids": list(self._motor_ids),
                        "supported_robots": list(SUPPORTED_ROBOTS),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Halt any rollout, then de-energize every motor.

        Logs the motors that did not answer - and a rollout thread that did not
        leave its loop - rather than raising: ``stop`` is called from teardown
        paths that cannot handle an exception and is annotated ``-> None``, so
        the log is the only place a halt this hook could not complete survives.
        A silent partial release would report an arm safe while joints are still
        driven, or while a policy is still commanding them.
        """
        if detail := halt_failure_detail(self.stop_task()):
            logger.error("%s: stop_task did not halt the rollout: %s", self._tool_name, detail)
        envelope = self._set_torque_envelope(False)
        if envelope["status"] == "error":
            logger.error("%s: %s", self._tool_name, envelope["content"][0]["text"])


# ---------------------------------------------------------------------------
# Envelope helpers. Kept private and one-liner-ish rather than reaching for a
# shared library, because the shape is small and the tests grade against the
# literal envelope. Duplicated with :mod:`strands_robots.drivers.dynamixel.driver`
# on purpose: two drivers with two two-line helpers is smaller than one driver
# and one shared module that binds their evolution together.
# ---------------------------------------------------------------------------
def _twin_robot(tool_name: str, sim: Any | None, context: str) -> str:
    """Name the registry robot the twin models, or refuse.

    The factory builds a driver as ``driver_cls(tool_name=<canonical>, ...)``
    and forwards nothing else that names the robot, so the honest source is
    ``tool_name`` resolved through the registry - unless ``sim`` already
    carries a robot, in which case that robot is the model and its name is the
    entry. A ``tool_name`` a caller chose (``"left_arm"``) resolves to nothing
    and is refused with the fix.

    Args:
        tool_name: The driver's tool name.
        sim: The caller's engine, or ``None``.
        context: The constructor's context for the refusal.

    Returns:
        A canonical name in :data:`SUPPORTED_ROBOTS` that has a simulation asset.

    Raises:
        ValueError: The name is not one this driver serves, or it has no asset.
    """
    from strands_robots.registry import has_sim, resolve_name  # noqa: PLC0415 - the registry is not a driver import

    candidate = tool_name
    if sim is not None:
        names = [str(name) for name in (sim.list_robots() or [])] if hasattr(sim, "list_robots") else []
        if names:
            candidate = names[0]
    canonical = resolve_name(candidate)
    if canonical not in SUPPORTED_ROBOTS:
        raise ValueError(
            f"{context}: transport='twin' models the robot tool_name names, and {candidate!r} resolves to "
            f"{canonical!r}, which is not one of {list(SUPPORTED_ROBOTS)}; pass tool_name as the arm's registry "
            "name (so101, so100) or sim= an engine carrying it"
        )
    if not has_sim(canonical):
        raise ValueError(
            f"{context}: transport='twin' needs a simulation asset, and the registry entry for {canonical!r} "
            "declares none; the SO arms (so100, so101) do"
        )
    return canonical


def _refuse(message: str) -> dict[str, Any]:
    """Return an error envelope with ``message``, matching the "not wired" contract."""
    return {"status": "error", "content": [{"text": message}]}


def _motor_targets(action: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Reduce every action key to the motor it names, once.

    ``"<motor>"`` and ``"<motor>.pos"`` are two spellings of one motor, and
    :func:`strands_robots.bus_access.read_joints` returns the suffixed one for
    this driver while the tool schema and the success envelope both speak the
    bare one - so a dict built from a read and then overridden by name carries
    both. Reducing them silently would make one of the two targets win by
    insertion order, write a single-motor frame, and report success naming only
    the survivor: the caller's other command would be gone with nothing saying
    so. One motor takes one target, so a doubled motor is refused instead.

    Args:
        action: The caller's mapping of joint name -> target, either spelling.

    Returns:
        ``(targets, None)`` keyed by motor name, or ``({}, message)`` naming
        every motor that was spelled more than once and the keys that spell it.
    """
    targets: dict[str, Any] = {}
    spellings: dict[str, list[str]] = {}
    for key, value in action.items():
        motor = str(key).removesuffix(".pos")
        spellings.setdefault(motor, []).append(str(key))
        targets[motor] = value
    doubled = {motor: keys for motor, keys in spellings.items() if len(keys) > 1}
    if doubled:
        named = "; ".join(f"{sorted(keys)} all name {motor!r}" for motor, keys in sorted(doubled.items()))
        return {}, (
            f"send_action: one motor takes one target, but {named}. "
            "A '.pos' suffix names the same motor as the bare joint, so spell each motor once."
        )
    return targets, None
