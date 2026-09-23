# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Feetech bus, answered by the arm's MuJoCo model.

:class:`~strands_robots.drivers.feetech.driver.FeetechDriver` reaches its arm
through one seam - :class:`~strands_robots.drivers.feetech.bus.FeetechBus`,
whose ``sync_read`` answers degrees (percent for the gripper) and whose
``write_goal_positions`` takes them. Everything the driver knows about the
arm - the six motor names, the calibration records, the refusals, the tool
schema - sits above that seam. :class:`FeetechTwinBus` is a second
implementation of it against a sim engine carrying the same arm, so
``Robot("so101", mode="real", driver="strands", transport="twin")`` builds the
**hardware driver** with the model at the far end: the same tool, the same
verbs, the same units, the same refusals as over the serial port. An agent
that rehearses ``move_to`` and ``set_torque`` here says the same words to the
arm.

What the twin does with each member the driver reads off the bus:

* ``connect`` - resolves every motor on the bus to a joint of the model
  through the registry's ``joint_labels`` (the SO-101 asset names its joints
  by servo id ``1``..``6``, the SO-100's by CAD term ``Rotation``..``Jaw``,
  the bus speaks ``shoulder_pan``..``gripper``) and to the actuator driving
  that joint, refusing by name when a label or an actuator is missing. Builds
  the engine when none was handed in.
* ``write_goal_positions`` - degrees go through the bus's **own**
  :meth:`~strands_robots.drivers.feetech.bus.FeetechBus.to_counts` against
  this arm's calibration records, exactly as on the wire; the twin then maps
  the calibrated ``[range_min, range_max]`` counts linearly onto the joint's
  travel in the model and writes the position actuator. A calibration file
  written for a real arm therefore places the twin's joints where the real
  arm's would be. The world then steps for one bus read period (the bus's
  ``timeout``), so the servo has arrived by the next ``sync_read`` - as a
  Feetech servo has at its default speed.
* ``sync_read`` - ``Present_Position`` is the model's joint angle back through
  the inverse map to counts and through
  :meth:`~strands_robots.drivers.feetech.bus.FeetechBus.to_value`, so the
  number the driver reads is the number a real bus would report for a servo
  standing there; ``Present_Velocity`` is the model's joint velocity in counts
  per second on the same scale; ``Torque_Enable`` is the twin's own state;
  ``Present_Load`` answers ``0`` - the model carries no load sensor. Any other
  register is refused by name, as the bus refuses one it cannot read.
* ``set_torque(False)`` - zeroes the position actuators' gains, so the arm goes
  limp under gravity in the model as it does on the bench;
  ``set_torque(True)`` restores them and re-targets the present position, so
  an energized arm holds where it is rather than snapping to a stale goal. A
  ``write_goal_positions`` with torque off is refused with the sentence the
  bus uses for a bus that cannot take the write.

The travel the counts map onto is the actuator's ``ctrlrange`` when it
declares one (the SO-100 asset does) and the joint's ``range`` otherwise: the
SO-101 asset's actuators declare ``ctrlrange="0 0"``, which MuJoCo reads as
*unlimited* and not as a zero-width travel, and the twin reads it the same way.
A joint with neither is refused at ``connect``, by name. The gripper's percent
spans its joint's travel end to end, ``0`` at the end the registry's
``gripper.closed`` names.

A target the calibration admits but the model cannot reach - a count past the
travel the joint's stops span - is clamped to that travel and **reported** on
the reply and in the log, never applied silently: a twin that drove to a
different angle than the one it was asked for and said nothing would teach the
agent the wrong arm.

The operator gate is not consulted here. The driver does not consult it on the
serial bus either - an SO arm is not on the blocklist - so nothing changes; it
is stated because the design every twin follows states it.

Wall clock: the twin steps physics as fast as it can, so a move returns in
milliseconds. ``realtime=True`` sleeps each read period out instead, for a
viewer that should see the motion at the speed the arm would make it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from strands_robots.drivers.feetech.bus import (
    DEFAULT_TIMEOUT_S,
    FeetechBus,
    MotorCalibration,
    MotorSpec,
)
from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)

#: The registers the twin can answer, and what each answers from. Stated so an
#: unknown register is refused by name rather than answered ``0``.
TWIN_REGISTERS: dict[str, str] = {
    "Present_Position": "the model's joint angle, through the inverse map and to_value",
    "Present_Velocity": "the model's joint velocity, in counts per second on the same scale",
    "Present_Load": "0 - the model carries no load sensor",
    "Torque_Enable": "the twin's own torque state, 1 or 0",
}

#: The scheme the twin's ``port`` reports, so a reader of ``status`` sees a model.
TWIN_SCHEME = "sim://"


def _scalar(value: Any) -> int:
    """A model field as an ``int``.

    MuJoCo's named views hand back one-element *arrays* rather than scalars, and
    ``int()`` on an array with ``ndim > 0`` is deprecated in NumPy and becomes an
    error - so the element is read before the conversion rather than after the
    warning. A test double may hand back a plain list, which has no ``shape``
    and is read the same way; a 0-d array has a falsy shape and converts
    directly.

    Args:
        value: A model field, array or scalar.

    Returns:
        The field as an ``int``.
    """
    if getattr(value, "shape", None) or isinstance(value, list | tuple):
        return _scalar(value[0])
    return int(value)


def _flag(value: Any) -> bool:
    """A model boolean flag (``limited``, ``ctrllimited``), whatever shape the view gives it."""
    return bool(_scalar(value))


def twin_endpoint(robot: str) -> str:
    """Where the twin for ``robot`` says it is - ``sim://<robot>``."""
    return f"{TWIN_SCHEME}{robot}"


@dataclass(frozen=True, slots=True)
class _Binding:
    """One motor's place in the model: the joint it reads, the actuator it drives, the travel between them.

    Attributes:
        joint: The model's joint name, as the engine's observation reports it
            (without the robot namespace).
        actuator: The actuator's name as the engine's ``send_action`` takes it.
        actuator_id: The actuator's index in the model, for the gain arrays.
        low: The low end of the travel the calibrated counts map onto, radians.
        high: The high end, radians.
        percent_low_is_closed: For a percent motor, whether ``0`` percent sits at
            ``low`` (the registry's ``gripper.closed`` is ``"low"``) or at ``high``.
        kp: The actuator's position gain (``gainprm[0]``), kept so
            ``set_torque(True)`` can restore what ``set_torque(False)`` zeroed.
        kv: The actuator's velocity bias (``biasprm[2]``), kept for the same reason.
    """

    joint: str
    actuator: str
    actuator_id: int
    low: float
    high: float
    percent_low_is_closed: bool
    kp: float
    kv: float


class FeetechTwinBus(FeetechBus):
    """A :class:`~strands_robots.drivers.feetech.bus.FeetechBus` whose servos are a model's joints.

    Inherits the units - ``to_value`` / ``to_counts`` / ``value_bounds`` and the
    calibration resolution - unchanged, so degrees on this bus mean exactly
    what they mean on the wire, and reimplements the lifecycle, the reads and
    the writes against a sim engine.

    Args:
        robot: The canonical registry name of the arm the model carries
            (``"so101"``, ``"so100"``). Names the engine to build when ``sim``
            is ``None`` and the registry entry whose ``joint_labels`` bridge
            the bus's motor names to the asset's joints.
        motors: The servos on this bus, defaulting to
            :data:`~strands_robots.drivers.feetech.bus.SO_ARM_MOTORS`.
        timeout: The bus's read period, seconds - how long the world steps
            after a write so the servo has arrived by the next read. Held to
            the domain :class:`~strands_robots.drivers.feetech.bus.FeetechBus`
            holds it to.
        calibration: This arm's records, from
            :func:`~strands_robots.drivers.feetech.bus.load_calibration`;
            ``None`` spans the servo's full travel, as the bus does.
        sim: A built sim engine carrying ``robot`` (what ``Robot(robot,
            mode="sim")`` returns). ``None`` builds one on :meth:`connect`.
        realtime: Sleep each read period out so a move takes the wall time it
            would on the arm. Default ``False``: as fast as the physics allows.

    Raises:
        ValueError: ``timeout`` or ``calibration`` is outside the bus's domain,
            or ``realtime`` is not a boolean.
    """

    def __init__(
        self,
        robot: str,
        motors: dict[str, MotorSpec] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        calibration: dict[str, MotorCalibration] | None = None,
        *,
        sim: Any | None = None,
        realtime: bool = False,
    ) -> None:
        if reason := boolean_flag_error(realtime, "realtime", type(self).__name__):
            raise ValueError(reason)
        super().__init__(port=twin_endpoint(robot), motors=motors, timeout=timeout, calibration=calibration)
        self.robot = robot
        self._sim = sim
        self._owns_sim = sim is None
        self._realtime = bool(realtime)
        self._robot_name: str | None = None
        self._bindings: dict[str, _Binding] = {}
        self._torque_enabled = True
        #: The clamp the last :meth:`write_goal_positions` reported, or ``""``.
        #: The driver's success envelope carries it, so a target the model
        #: could not reach is on the reply and not only in the log.
        self.last_clamp_note = ""

    # ------------------------------------------------------------------ #
    # The engine.                                                        #
    # ------------------------------------------------------------------ #

    @property
    def sim(self) -> Any | None:
        """The engine behind the twin, or ``None`` before :meth:`connect` built it."""
        return self._sim

    @property
    def robot_name(self) -> str:
        """The robot's name inside the engine - the first it lists, else ``robot``."""
        if self._robot_name is None:
            names = list(self._sim.list_robots()) if self._sim is not None else []
            self._robot_name = str(names[0]) if names else self.robot
        return self._robot_name

    @property
    def torque_enabled(self) -> bool:
        """Whether the actuators hold their gains - what ``Torque_Enable`` answers."""
        return self._torque_enabled

    @property
    def is_connected(self) -> bool:
        """Whether the engine is built and every motor is bound to a joint."""
        return self._sim is not None and bool(self._bindings)

    def connect(self) -> None:
        """Build the engine when none was given, then bind every motor to the model.

        Raises:
            OSError: The engine could not be built - MuJoCo or the asset is
                missing, or the registry does not know ``robot``. The same
                class the serial bus raises when its port will not open, so
                the driver's ``connect_eagerly`` reports it the same way.
            ValueError: A motor has no ``joint_labels`` entry for this robot,
                its joint is not in the model, no actuator drives that joint,
                or the joint declares neither a ``ctrlrange`` nor a ``range``
                to map the counts onto. Named per motor.
        """
        if self.is_connected:
            return
        if self._sim is None:
            # Built through the simulation package - ``create_simulation`` and
            # ``add_robot``, the two calls ``Robot(robot, mode="sim")`` makes -
            # rather than through the factory. The factory imports the driver
            # registry, and the registry imports this bus's driver, so a twin
            # that imported the factory would close a cycle around one arm.
            try:
                from strands_robots.simulation import (
                    create_simulation,  # noqa: PLC0415 - MuJoCo is optional; imported on use
                )

                sim = create_simulation("mujoco", tool_name=f"{self.robot}_twin")
                for step in (sim.create_world(), sim.add_robot(name=self.robot)):
                    if step.get("status") == "error":
                        sim.destroy()
                        detail = (step.get("content") or [{}])[0].get("text", str(step))
                        raise OSError(f"FeetechTwinBus: could not build the {self.robot} twin: {detail}")
            except OSError:
                self._sim = None
                raise
            except Exception as exc:  # noqa: BLE001 - the reason is reported, not raised, like every connect here
                self._sim = None
                raise OSError(
                    f"FeetechTwinBus: could not build the {self.robot} twin ({type(exc).__name__}: {exc})"
                ) from exc
            self._sim = sim
        try:
            self._bindings = self._bind_motors()
        except ValueError:
            self.disconnect()
            raise
        self._torque_enabled = True

    def disconnect(self) -> None:
        """Unbind the motors; destroy an engine this bus built and keep a caller-supplied one. Safe to repeat."""
        self._bindings = {}
        self._robot_name = None
        if not self._owns_sim:
            return
        sim, self._sim = self._sim, None
        if sim is not None:
            try:
                sim.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("twin: destroy() raised during disconnect", exc_info=True)

    # ------------------------------------------------------------------ #
    # The map: motors <-> the model.                                      #
    # ------------------------------------------------------------------ #

    def _bind_motors(self) -> dict[str, _Binding]:
        """Resolve every motor through the registry's labels to a joint, an actuator and a travel."""
        from strands_robots.registry import get_robot, joint_labels  # noqa: PLC0415 - registry imports are lazy here

        by_label = {label: joint for joint, label in joint_labels(self.robot).items()}
        if not by_label:
            raise ValueError(
                f"FeetechTwinBus: the registry entry for {self.robot!r} declares no joint_labels, so the bus's "
                f"motors {sorted(self.motors)} cannot be placed on the model's joints; add a joint_labels block "
                "to its registry entry (as so100 and so101 have)"
            )
        gripper = (get_robot(self.robot) or {}).get("gripper") or {}
        closed_low = str(gripper.get("closed", "low")).lower() != "high"
        model = self._model()
        if model is None:
            raise ValueError(f"FeetechTwinBus: the engine carrying {self.robot!r} exposes no mj_model to bind to")
        bindings: dict[str, _Binding] = {}
        for name, spec in self.motors.items():
            joint = by_label.get(name)
            if joint is None:
                raise ValueError(
                    f"FeetechTwinBus: no joint_labels entry names motor {name!r} on {self.robot!r}; "
                    f"the labels are {sorted(by_label)}"
                )
            joint_view = self._named(model, "joint", joint)
            if joint_view is None:
                raise ValueError(
                    f"FeetechTwinBus: motor {name!r} is labelled joint {joint!r}, which the {self.robot} model "
                    "does not carry"
                )
            actuator_view = self._actuator_for(model, int(joint_view.id))
            if actuator_view is None:
                raise ValueError(
                    f"FeetechTwinBus: no actuator drives joint {joint!r} (motor {name!r}) in the {self.robot} model"
                )
            low, high = self._travel(joint_view, actuator_view)
            if low is None or high is None:
                raise ValueError(
                    f"FeetechTwinBus: joint {joint!r} (motor {name!r}) declares neither an actuator ctrlrange nor "
                    f"a joint range in the {self.robot} model, so there is no travel to map its counts onto"
                )
            bindings[name] = _Binding(
                joint=joint,
                actuator=self._short(str(actuator_view.name)),
                actuator_id=int(actuator_view.id),
                low=low,
                high=high,
                percent_low_is_closed=closed_low,
                kp=float(actuator_view.gainprm[0]),
                kv=float(actuator_view.biasprm[2]),
            )
        return bindings

    def _model(self) -> Any | None:
        return getattr(self._sim, "mj_model", None)

    def _short(self, name: str) -> str:
        """Strip the robot namespace the model prefixes, to the name the engine's action takes."""
        prefix = f"{self.robot_name}/"
        return name[len(prefix) :] if name.startswith(prefix) else name

    def _named(self, model: Any, kind: str, name: str) -> Any | None:
        """``model.joint(...)`` / ``model.actuator(...)`` by namespaced name, then bare name; ``None`` when absent."""
        accessor = getattr(model, kind)
        for candidate in (f"{self.robot_name}/{name}", name):
            try:
                return accessor(candidate)
            except (KeyError, ValueError, IndexError):
                continue
        return None

    @staticmethod
    def _actuator_for(model: Any, joint_id: int) -> Any | None:
        """The first actuator whose transmission is joint ``joint_id``, else ``None``."""
        for index in range(int(model.nu)):
            view = model.actuator(index)
            if _scalar(view.trntype) == 0 and _scalar(view.trnid[0]) == joint_id:
                return view
        return None

    @staticmethod
    def _travel(joint_view: Any, actuator_view: Any) -> tuple[float | None, float | None]:
        """The travel the counts span: a declared ``ctrlrange``, else the joint ``range``, else nothing.

        ``ctrlrange="0 0"`` is MuJoCo's *unlimited* (``ctrllimited`` reads
        false), not a zero-width travel; it is read as such.
        """
        if _flag(actuator_view.ctrllimited):
            low, high = (float(value) for value in actuator_view.ctrlrange[:2])
            if high > low:
                return low, high
        if _flag(joint_view.limited):
            low, high = (float(value) for value in joint_view.range[:2])
            if high > low:
                return low, high
        return None, None

    def _binding(self, name: str) -> _Binding:
        binding = self._bindings.get(name)
        if binding is None:
            raise RuntimeError(
                f"FeetechBus: reading {name} needs an open bus; call connect() first (port={self.port!r})"
            )
        return binding

    def counts_to_model(self, name: str, counts: int) -> float:
        """One motor's encoder counts as the model's joint angle, radians.

        The calibrated ``[range_min, range_max]`` maps linearly onto the joint's
        travel; a percent motor maps through its percent so the registry's
        ``closed`` end lands on ``0`` whatever ``drive_mode`` the record carries.

        Args:
            name: A motor on this bus.
            counts: A count in the servo's own scale.

        Returns:
            Radians, on the model's zero. Not clamped: the caller decides what
            a target past the travel means (see :meth:`write_goal_positions`).
        """
        spec, record = self._calibrated(name)
        binding = self._binding(name)
        if spec.norm_mode == "range_0_100":
            fraction = self.to_value(name, counts) / 100.0
            closed, opened = (
                (binding.low, binding.high) if binding.percent_low_is_closed else (binding.high, binding.low)
            )
            return closed + fraction * (opened - closed)
        span = record.range_max - record.range_min
        return binding.low + (counts - record.range_min) / span * (binding.high - binding.low)

    def model_to_counts(self, name: str, radians: float) -> int:
        """The inverse of :meth:`counts_to_model`: a model joint angle as the count a servo there would report.

        Args:
            name: A motor on this bus.
            radians: The model's joint angle.

        Returns:
            The nearest count. Percent motors are bounded to the record's
            travel, as ``to_value`` bounds percent; degree motors are not -
            a joint pushed past its calibrated stop reads as the count it is at.
        """
        spec, record = self._calibrated(name)
        binding = self._binding(name)
        if spec.norm_mode == "range_0_100":
            closed, opened = (
                (binding.low, binding.high) if binding.percent_low_is_closed else (binding.high, binding.low)
            )
            fraction = (radians - closed) / (opened - closed) if opened != closed else 0.0
            percent = min(100.0, max(0.0, fraction * 100.0))
            return self.to_counts(name, percent)
        fraction = (radians - binding.low) / (binding.high - binding.low)
        return int(round(record.range_min + fraction * (record.range_max - record.range_min)))

    # ------------------------------------------------------------------ #
    # Stepping.                                                           #
    # ------------------------------------------------------------------ #

    def _engine(self) -> Any:
        if self._sim is None or not self._bindings:
            raise RuntimeError(f"FeetechBus: the twin needs an open bus; call connect() first (port={self.port!r})")
        return self._sim

    def _substeps(self, seconds: float) -> int:
        """Physics steps that cover ``seconds``, at least one."""
        dt = self._sim.physics_timestep() if self._sim is not None else None
        if not dt or dt <= 0:
            return 1
        return max(1, int(round(seconds / dt)))

    def _advance(self, action: dict[str, float], seconds: float) -> None:
        """Write ``action`` to the actuators and step ``seconds``; raise the engine's refusal as ``RuntimeError``."""
        result = self._engine().send_action(action, robot_name=self.robot_name, n_substeps=self._substeps(seconds))
        if result.get("status") != "success":
            text = str((result.get("content") or [{}])[0].get("text", "send_action failed"))
            raise RuntimeError(f"FeetechTwinBus: the model refused the write: {text}")
        if self._realtime and seconds > 0:
            time.sleep(seconds)

    def _observation(self) -> dict[str, float]:
        obs = self._engine().get_observation(robot_name=self.robot_name, skip_images=True)
        return {key: float(value) for key, value in obs.items() if isinstance(value, int | float)}

    # ------------------------------------------------------------------ #
    # Reads.                                                              #
    # ------------------------------------------------------------------ #

    def sync_read(self, register: str = "Present_Position", num_retry: int = 0) -> dict[str, float]:
        """Read ``register`` for every motor from the model.

        Args:
            register: A key of :data:`TWIN_REGISTERS`.
            num_retry: Accepted for the seam; the model always answers.

        Returns:
            Motor name -> value. ``Present_Position`` is in the joint's own
            unit through :meth:`to_value`, exactly as the bus decodes a
            servo's reply; the other registers are as :data:`TWIN_REGISTERS`
            states.

        Raises:
            ValueError: ``register`` is not one the twin answers.
            RuntimeError: The bus is not connected.
        """
        del num_retry
        if register not in TWIN_REGISTERS:
            raise ValueError(f"FeetechBus: cannot read {register!r}; the twin answers {sorted(TWIN_REGISTERS)}")
        self._engine()
        obs = self._observation()
        out: dict[str, float] = {}
        for name in self.motors:
            binding = self._binding(name)
            if register == "Torque_Enable":
                out[name] = 1.0 if self._torque_enabled else 0.0
            elif register == "Present_Load":
                out[name] = 0.0
            elif register == "Present_Velocity":
                _spec, record = self._calibrated(name)
                per_rad = (record.range_max - record.range_min) / abs(binding.high - binding.low)
                out[name] = float(obs.get(f"{binding.joint}.vel", 0.0)) * per_rad
            else:
                out[name] = self.to_value(name, self.model_to_counts(name, obs.get(binding.joint, binding.low)))
        return out

    # ------------------------------------------------------------------ #
    # Writes.                                                             #
    # ------------------------------------------------------------------ #

    def write_goal_positions(self, targets: dict[str, float]) -> None:
        """Command joint positions on the model and step one read period.

        Each target goes through :meth:`to_counts` first - the same domain
        check and the same calibration the wire applies - so a target the
        encoder cannot hold is refused with the bus's sentence, and a valid one
        lands where the same count would put the real arm.

        Args:
            targets: Motor name -> target, degrees (percent for the gripper).

        Raises:
            ValueError: A name is not on this bus, a value is not a finite
                number, or a value is outside the encoder's travel - the bus's
                own refusals, unchanged.
            RuntimeError: The bus is not connected, torque is off, or the model
                refused the write.
        """
        if not targets:
            raise ValueError("FeetechBus: no targets to write")
        self._engine()
        if not self._torque_enabled:
            raise RuntimeError(
                f"FeetechBus: writing goal positions needs torque on; call set_torque(True) first (port={self.port!r})"
            )
        action: dict[str, float] = {}
        clamped: list[str] = []
        for name, value in targets.items():
            self._calibrated(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"FeetechBus: {name} target must be a finite number, got {type(value).__name__}: {value!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"FeetechBus: {name} target must be finite, got {value!r}")
            counts = self.to_counts(name, number)
            radians = self.counts_to_model(name, counts)
            binding = self._binding(name)
            bounded = min(binding.high, max(binding.low, radians))
            if bounded != radians:
                clamped.append(f"{name} {radians:.3f} -> {bounded:.3f} rad")
            action[binding.actuator] = bounded
        self._advance(action, float(self.timeout))
        if clamped:
            self.last_clamp_note = f"model travel clamped: {'; '.join(clamped)}"
            logger.warning("twin: %s", self.last_clamp_note)
        else:
            self.last_clamp_note = ""

    def set_torque(self, enabled: bool) -> list[str]:
        """Energize or release every motor on the model.

        Releasing zeroes each actuator's position gain and velocity bias, so
        the joints go limp under gravity as the bench arm does; energizing
        restores them and re-targets the present position, so the arm holds
        where it is.

        Args:
            enabled: ``True`` to energize, ``False`` to release.

        Returns:
            The motors that did not take the write - always empty here, the
            model always answers; the list is the seam's shape.

        Raises:
            RuntimeError: The bus is not connected.
        """
        self._engine()
        model = self._model()
        obs = self._observation()
        hold: dict[str, float] = {}
        for name in self.motors:
            binding = self._binding(name)
            if model is not None:
                view = model.actuator(binding.actuator_id)
                view.gainprm[0] = binding.kp if enabled else 0.0
                view.biasprm[1] = -binding.kp if enabled else 0.0
                view.biasprm[2] = binding.kv if enabled else 0.0
            hold[binding.actuator] = float(obs.get(binding.joint, binding.low))
        self._torque_enabled = bool(enabled)
        if enabled:
            self._advance(hold, 0.0)
        return []


__all__ = ["TWIN_REGISTERS", "TWIN_SCHEME", "FeetechTwinBus", "twin_endpoint"]
