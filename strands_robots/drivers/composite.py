"""One robot from several drivers: a body, its grippers, its head camera.

A humanoid with add-on hardware is several transports at once: the Unitree G1
speaks DDS, OpenArm grippers on its wrists speak CAN-FD, the head camera is a
video device. An agent, the mesh, the recorder and the operator gate all want
ONE robot: one state dict, one action dict, one ``stop()``, one approval, one
peer. :class:`CompositeDriver` is that one object. It satisfies
:class:`~strands_robots.drivers.base.HardwareDriver` structurally and holds its
members as :class:`Part` objects that never become tools or peers themselves.

The rules it enforces are the ones a partial robot gets wrong silently:

* **Key space.** The ``primary`` part's joints keep their names; every other
  part's joints are prefixed ``<part>_``; a one-joint part whose joint is
  ``gripper`` collapses to the part name, so ``left_gripper`` falls out of a
  registry entry with no table.
* **Units.** Each part declares a unit per joint; the composite stores and
  reports values as the part gave them and never rescales, so a consumer that
  needs one unit reads :meth:`CompositeDriver.units` rather than guessing.
* **Writes.** A target for a key no part owns refuses the whole write. Parts
  are written sequentially, primary first, so a gripper never closes on a pose
  the body did not reach; a part's failure stops every part and refuses.
* **Stop.** ``stop()`` runs the declared ``stop_order`` (body first: it is the
  part that can hurt someone), each part with its own timeout, and reports how
  many stopped. A partial stop latches: motion is refused until
  :meth:`CompositeDriver.reset_estop`. Reads never latch; they carry per-part
  status so a dashboard shows the dead part.

What ships here is the proof the design needs: the protocol, a part over a
simulation engine, a mock gripper, and the composite. Wrapping a real driver and
a camera, and the registry ``parts:`` entry that builds all of it, are described
in ``docs/project/design-driver-composition.md`` and land in their own PRs.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Protocol

from strands.types.tools import ToolSpec, ToolUse

from strands_robots.drivers.base import refuse, undeclared_verb_error

logger = logging.getLogger(__name__)

#: Verbs :meth:`CompositeDriver.stream` dispatches; the tool spec's enum.
VERBS: tuple[str, ...] = ("status", "state", "features", "send_action", "stop", "reset_estop")

#: How long one part may take to halt before the composite reports it, seconds.
DEFAULT_STOP_TIMEOUT_S = 2.0


class Part(Protocol):
    """What the composite needs from each member."""

    name: str

    @property
    def joint_names(self) -> tuple[str, ...]:
        """This part's joints, its own names, its own order."""

    @property
    def units(self) -> Mapping[str, str]:
        """Joint name -> unit: ``"rad"``, ``"m"``, ``"deg"`` or ``"norm01"``."""

    @property
    def is_connected(self) -> bool:
        """Whether reads and writes can reach the hardware."""

    def read(self) -> dict[str, float]:
        """Latest joint values in :attr:`units`; never blocks on the bus."""

    def write(self, targets: Mapping[str, float]) -> dict[str, Any]:
        """Command joint targets; returns a status envelope, never raises."""

    def stop(self, timeout: float) -> dict[str, Any]:
        """Halt; idempotent; returns a status envelope within ``timeout`` seconds."""


class SimPart:
    """A robot inside a simulation engine, as one part.

    Duck-typed over the engine so this module imports nothing from the
    simulation layer: the engine needs ``robot_action_keys(name)``,
    ``get_observation(name, skip_images=True)``, ``send_action(action,
    robot_name=name)`` and ``stop_policy(name)``, which every
    :class:`~strands_robots.simulation.base.SimEngine` has.
    """

    def __init__(self, engine: Any, robot_name: str, part_name: str = "body"):
        """Wrap ``robot_name`` inside ``engine`` as part ``part_name``."""
        self.name = part_name
        self._engine = engine
        self._robot_name = robot_name
        self._joints = tuple(engine.robot_action_keys(robot_name))

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The engine's actuator names for this robot."""
        return self._joints

    @property
    def units(self) -> Mapping[str, str]:
        """Radians for every joint; the engine's own convention."""
        return dict.fromkeys(self._joints, "rad")

    @property
    def is_connected(self) -> bool:
        """A simulated robot is reachable while the engine still answers for its name."""
        try:
            return bool(self._engine.robot_action_keys(self._robot_name))
        except Exception:  # noqa: BLE001 - a removed robot is "not connected", not a crash
            return False

    def read(self) -> dict[str, float]:
        """Joint positions from the engine's observation, velocities dropped."""
        obs = self._engine.get_observation(self._robot_name, skip_images=True)
        return {name: float(obs[name]) for name in self._joints if name in obs}

    def write(self, targets: Mapping[str, float]) -> dict[str, Any]:
        """Forward to the engine's ``send_action`` for this robot."""
        envelope: dict[str, Any] = self._engine.send_action(dict(targets), robot_name=self._robot_name)
        return envelope

    def stop(self, timeout: float) -> dict[str, Any]:
        """Stop any policy driving this robot; the engine answers at once."""
        del timeout
        envelope: dict[str, Any] = self._engine.stop_policy(self._robot_name)
        return envelope


class MockGripperPart:
    """A one-joint gripper that settles toward its target; for tests and sim fallback.

    Its single joint is ``gripper`` in ``norm01`` (0 closed, 1 open), so under
    the composite's key rules it appears as the part name alone.
    """

    def __init__(
        self,
        name: str,
        settle_s: float = 0.02,
        connected: bool = True,
        fail_stop: bool = False,
        fail_write: bool = False,
    ):
        """Configure the mock.

        Args:
            name: Part name, e.g. ``"left_gripper"``.
            settle_s: Time for a commanded target to be fully reported.
            connected: Initial connection state.
            fail_stop: When True, :meth:`stop` reports an error (a wedged CAN node).
            fail_write: When True, :meth:`write` reports an error.
        """
        if not math.isfinite(settle_s) or settle_s < 0:
            raise ValueError(f"settle_s must be a non-negative finite number, got {settle_s!r}")
        self.name = name
        self._settle = float(settle_s)
        self._connected = bool(connected)
        self._fail_stop = bool(fail_stop)
        self._fail_write = bool(fail_write)
        self._value = 0.0
        self._target = 0.0
        self._t_cmd = time.monotonic()
        self.writes = 0
        self.stops = 0

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The one joint."""
        return ("gripper",)

    @property
    def units(self) -> Mapping[str, str]:
        """Open fraction."""
        return {"gripper": "norm01"}

    @property
    def is_connected(self) -> bool:
        """Set by the constructor and :meth:`disconnect`."""
        return self._connected

    def disconnect(self) -> None:
        """Simulate the bus going away."""
        self._connected = False

    def read(self) -> dict[str, float]:
        """The value, linearly settled toward the target."""
        if self._settle == 0:
            self._value = self._target
        else:
            frac = min(1.0, (time.monotonic() - self._t_cmd) / self._settle)
            self._value = self._value + (self._target - self._value) * frac
        return {"gripper": self._value}

    def write(self, targets: Mapping[str, float]) -> dict[str, Any]:
        """Accept a ``gripper`` target in ``[0, 1]``."""
        self.writes += 1
        if self._fail_write:
            return refuse(f"{self.name}: bus write failed")
        value = float(targets["gripper"])
        if not 0.0 <= value <= 1.0:
            return refuse(f"{self.name}: gripper target {value} outside [0, 1]")
        self.read()
        self._target = value
        self._t_cmd = time.monotonic()
        return {"status": "success", "content": [{"text": f"{self.name}: gripper -> {value:.2f}"}]}

    def stop(self, timeout: float) -> dict[str, Any]:
        """Hold the current value; fails when built with ``fail_stop``."""
        self.stops += 1
        if self._fail_stop:
            return refuse(f"{self.name}: no halt acknowledgement after {timeout:.1f} s")
        self._target = self.read()["gripper"]
        return {"status": "success", "content": [{"text": f"{self.name}: holding"}]}


def part_key(part_name: str, joint: str, *, primary: bool) -> str:
    """The composite key for ``joint`` of ``part_name`` under the naming rules."""
    if primary:
        return joint
    if joint == "gripper":
        return part_name
    return f"{part_name}_{joint}"


class CompositeDriver:
    """Several :class:`Part` objects behind one driver surface."""

    def __init__(
        self,
        tool_name: str = "composite",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        parts: Sequence[Part],
        primary: str,
        stop_order: Sequence[str] | None = None,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
    ):
        """Compose the parts.

        Args:
            tool_name: Name the agent invokes the robot by; also its mesh peer id.
            cameras: Accepted for the driver constructor contract; camera parts
                carry their own devices, so this is unused here.
            data_config: Accepted for parity; unused.
            parts: The members, in the order their keys appear after the primary's.
            primary: Name of the part whose joints stay unprefixed.
            stop_order: Part names in halt order; missing parts are appended in
                declaration order, never dropped.
            stop_timeout_s: Per-part halt budget.

        Raises:
            ValueError: Duplicate part names, an unknown ``primary`` or
                ``stop_order`` entry, or two parts producing one composite key.
        """
        del cameras, data_config
        self._tool_name = tool_name
        names = [p.name for p in parts]
        if len(set(names)) != len(names):
            raise ValueError(f"CompositeDriver: duplicate part names in {names}")
        if primary not in names:
            raise ValueError(f"CompositeDriver: primary {primary!r} is not one of the parts {names}")
        if not math.isfinite(stop_timeout_s) or stop_timeout_s <= 0:
            raise ValueError(f"stop_timeout_s must be a positive finite number, got {stop_timeout_s!r}")
        order = list(stop_order or ())
        unknown = [n for n in order if n not in names]
        if unknown:
            raise ValueError(f"CompositeDriver: stop_order names parts that do not exist: {unknown}")
        order += [n for n in names if n not in order]
        self._parts: dict[str, Part] = {p.name: p for p in parts}
        self._primary = primary
        self._stop_order = tuple(order)
        self._stop_timeout = float(stop_timeout_s)
        self._estop_latched_by: tuple[str, ...] = ()
        # key -> (part name, joint name); built once, refuses collisions.
        self._owners: dict[str, tuple[str, str]] = {}
        ordered = [self._parts[primary]] + [p for p in parts if p.name != primary]
        for part in ordered:
            for joint in part.joint_names:
                key = part_key(part.name, joint, primary=part.name == primary)
                if key in self._owners:
                    other = self._owners[key][0]
                    raise ValueError(f"CompositeDriver: key {key!r} is produced by both {other!r} and {part.name!r}")
                self._owners[key] = (part.name, joint)

    # ------------------------------------------------------------------ #
    # Tool surface.                                                       #
    # ------------------------------------------------------------------ #
    @property
    def tool_name(self) -> str:
        """The name the agent invokes this robot by."""
        return self._tool_name

    @property
    def tool_name_str(self) -> str:
        """Same as :attr:`tool_name`; the spelling the mesh presence reads."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"``, like every driver."""
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """The verbs an agent may send and the one payload they take."""
        return {
            "name": self._tool_name,
            "description": (
                f"Composite robot: parts {list(self._parts)} behind one tool. "
                "status: parts and latch; state: one joint dict; features: key -> unit; "
                "send_action: targets keyed by composite joint name; stop: halt every part; "
                "reset_estop: clear a latched partial stop after the operator checked the robot."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(VERBS)},
                        "targets": {
                            "type": "object",
                            "description": "send_action: composite joint name -> value in that key's unit.",
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result."""
        del invocation_state, kwargs
        tool_use_id = tool_use.get("toolUseId", "")
        request = tool_use.get("input") or {}
        action = request.get("action", "status")
        envelope: dict[str, Any]
        if action == "status":
            envelope = self.get_status()
        elif action == "state":
            envelope = {"status": "success", "content": [{"json": self.get_observation()}]}
        elif action == "features":
            envelope = {"status": "success", "content": [{"json": dict(self.units())}]}
        elif action == "send_action":
            envelope = self.send_action(request.get("targets") or {})
        elif action == "stop":
            envelope = self.stop(reason="agent stop")
        elif action == "reset_estop":
            envelope = self.reset_estop()
        else:
            envelope = undeclared_verb_error(self, action)
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # One robot.                                                          #
    # ------------------------------------------------------------------ #
    @property
    def parts(self) -> tuple[str, ...]:
        """Part names, primary first."""
        return (self._primary,) + tuple(n for n in self._parts if n != self._primary)

    @property
    def primary(self) -> str:
        """The part whose joints are unprefixed."""
        return self._primary

    @property
    def stop_order(self) -> tuple[str, ...]:
        """The halt order :meth:`stop` follows."""
        return self._stop_order

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Every composite key, primary joints first then parts in declaration order."""
        return tuple(self._owners)

    @property
    def observation_features(self) -> dict[str, type]:
        """The one state dict's columns; what a recorder declares."""
        return dict.fromkeys(self._owners, float)

    @property
    def action_features(self) -> dict[str, type]:
        """Same key space as the state for joint-target control."""
        return dict.fromkeys(self._owners, float)

    def units(self) -> dict[str, str]:
        """Composite key -> unit, as each part declared it."""
        return {key: self._parts[part].units[joint] for key, (part, joint) in self._owners.items()}

    @property
    def is_connected(self) -> bool:
        """True only when every part is; the mesh publishes joints on this."""
        return all(p.is_connected for p in self._parts.values())

    @property
    def estop_latched(self) -> bool:
        """Whether a partial stop is holding motion refused."""
        return bool(self._estop_latched_by)

    def parts_status(self) -> dict[str, dict[str, Any]]:
        """Per-part connection state, for ``status`` and the mesh state topic."""
        return {
            name: {"connected": bool(p.is_connected), "joints": len(p.joint_names)} for name, p in self._parts.items()
        }

    def get_observation(self) -> dict[str, Any]:
        """One dict: every composite key that could be read, plus ``parts`` status.

        A part that is disconnected or fails to read contributes no keys and is
        reported under ``parts``; nothing is invented for it.
        """
        out: dict[str, Any] = {}
        status = self.parts_status()
        for name, part in self._parts.items():
            if not part.is_connected:
                continue
            try:
                values = part.read()
            except Exception as exc:  # noqa: BLE001 - a read failure is a status, not a crash
                status[name]["error"] = f"{type(exc).__name__}: {exc}"
                continue
            for joint, value in values.items():
                out[part_key(name, joint, primary=name == self._primary)] = value
        out["parts"] = status
        return out

    def get_status(self) -> dict[str, Any]:
        """Parts, latch and key count as an envelope."""
        latched = ", ".join(self._estop_latched_by) if self._estop_latched_by else "no"
        connected = sum(1 for p in self._parts.values() if p.is_connected)
        text = (
            f"{self._tool_name}: {len(self._parts)} parts ({connected} connected), "
            f"{len(self._owners)} joints, e-stop latched: {latched}"
        )
        return {
            "status": "success",
            "content": [
                {"text": text},
                {"json": {"parts": self.parts_status(), "estop_latched_by": list(self._estop_latched_by)}},
            ],
        }

    def _motion_refusal(self) -> str | None:
        """Why no part may move right now, or ``None``."""
        if self._estop_latched_by:
            who = ", ".join(self._estop_latched_by)
            return f"{self._tool_name}: e-stop latched by {who}; call reset_estop() after checking the robot."
        dead = [name for name, p in self._parts.items() if not p.is_connected]
        if dead:
            return f"{self._tool_name}: part(s) {dead} not connected; the whole robot refuses motion."
        return None

    def route(self, action: Mapping[str, Any]) -> tuple[dict[str, dict[str, float]], list[str]]:
        """Split a composite action into per-part targets; unknown keys come back separately."""
        split: dict[str, dict[str, float]] = {}
        unknown: list[str] = []
        for key, value in action.items():
            owner = self._owners.get(key)
            if owner is None:
                unknown.append(key)
                continue
            part, joint = owner
            split.setdefault(part, {})[joint] = float(value)
        return split, unknown

    def send_action(self, action: Mapping[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Write targets to their parts, primary first; any failure stops everything.

        Args:
            action: Composite key -> value in that key's unit.
            robot_name: Accepted for the driver contract; a composite is one robot,
                so anything but ``None`` or its own name is refused.
        """
        if robot_name not in (None, self._tool_name):
            return refuse(f"{self._tool_name} is one robot; robot_name={robot_name!r} names none of it.")
        if not action:
            return refuse(f"{self._tool_name}: send_action needs at least one target.")
        if reason := self._motion_refusal():
            return refuse(reason)
        split, unknown = self.route(action)
        if unknown:
            return refuse(
                f"{self._tool_name}: keys {sorted(unknown)} belong to no part {list(self._parts)}; "
                f"nothing was written. Valid keys: {list(self._owners)}"
            )
        results: dict[str, Any] = {}
        for name in self.parts:
            targets = split.get(name)
            if not targets:
                continue
            envelope = self._parts[name].write(targets)
            results[name] = envelope
            if envelope.get("status") != "success":
                halt = self.stop(reason=f"write to {name} failed")
                detail = envelope.get("content", [{}])[0].get("text", "")
                return refuse(
                    f"{self._tool_name}: part {name!r} refused the write ({detail}); "
                    f"every part was stopped ({halt['content'][0]['text']})."
                )
        return {
            "status": "success",
            "content": [
                {"text": f"{self._tool_name}: wrote {len(action)} targets to {list(results)}"},
                {"json": results},
            ],
        }

    def stop(self, reason: str = "stop") -> dict[str, Any]:
        """Halt every part in :attr:`stop_order`; latch when any part did not."""
        results: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        for name in self._stop_order:
            part = self._parts[name]
            try:
                envelope = part.stop(self._stop_timeout)
            except Exception as exc:  # noqa: BLE001 - a halt must report, never raise past the next part
                envelope = refuse(f"{name}: {type(exc).__name__}: {exc}")
            results[name] = envelope
            if envelope.get("status") != "success":
                failed.append(name)
        if failed:
            self._estop_latched_by = tuple(failed)
            details = "; ".join(results[n]["content"][0].get("text", n) for n in failed)
            logger.error(
                "[%s] stop (%s): %d/%d parts halted; %s",
                self._tool_name,
                reason,
                len(results) - len(failed),
                len(results),
                details,
            )
            return {
                "status": "error",
                "content": [
                    {"text": f"stopped {len(results) - len(failed)}/{len(results)} parts: {details}"},
                    {"json": results},
                ],
            }
        return {
            "status": "success",
            "content": [{"text": f"stopped {len(results)}/{len(results)} parts"}, {"json": results}],
        }

    def stop_task(self) -> dict[str, Any]:
        """The driver contract's halt verb; same as :meth:`stop`."""
        return self.stop(reason="stop_task")

    def reset_estop(self) -> dict[str, Any]:
        """Clear the latch; refused while a part is still disconnected."""
        if not self._estop_latched_by:
            return {"status": "success", "content": [{"text": f"{self._tool_name}: no e-stop latched"}]}
        dead = [name for name, p in self._parts.items() if not p.is_connected]
        if dead:
            return refuse(f"{self._tool_name}: cannot clear the e-stop while {dead} are not connected.")
        was = self._estop_latched_by
        self._estop_latched_by = ()
        return {
            "status": "success",
            "content": [{"text": f"{self._tool_name}: e-stop cleared (was latched by {list(was)})"}],
        }

    # ------------------------------------------------------------------ #
    # Driver-contract members a composite answers by refusing, for now.   #
    # ------------------------------------------------------------------ #
    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "mock",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Refused: policy rollouts over a composite land with the token decoder slot."""
        del policy_port, policy_host, policy_provider, duration, policy_kwargs
        return refuse(f"{self._tool_name}: start_task({instruction!r}) is not implemented for a composite yet.")

    def run_policy(
        self,
        policy_object: Any,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refused for the same reason as :meth:`start_task`."""
        del policy_object, duration, n_steps
        return refuse(f"{self._tool_name}: run_policy({instruction!r}) is not implemented for a composite yet.")

    def get_task_status(self) -> dict[str, Any]:
        """No task can be running; says so."""
        return {"status": "success", "content": [{"text": f"{self._tool_name}: no task running"}]}

    def cleanup(self) -> None:
        """Stop every part; a cleanup never raises."""
        self.stop(reason="cleanup")


__all__ = [
    "DEFAULT_STOP_TIMEOUT_S",
    "VERBS",
    "CompositeDriver",
    "MockGripperPart",
    "Part",
    "SimPart",
    "part_key",
]
