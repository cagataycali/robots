"""Native Dynamixel Protocol 2.0 driver satisfying :class:`HardwareDriver`.

What this driver actually does today: it constructs, it satisfies the
:class:`~strands_robots.drivers.base.HardwareDriver` surface, and it names its
motion, task, and policy paths as deferred. It does **not** open a serial
port. The bus / serial I/O work is scope item 1 in :issue:`359` and is
deliberately its own PR - the same slice :issue:`354`'s triage recommends and
the same slice :issue:`360`'s triage explicitly names as landable-without-
hardware.

Why land it as a stub anyway: because the driver seam (:issue:`353` /
:pr:`2734`) is on ``main`` with no native driver registered against it for any
Dynamixel robot. ``Robot("koch", mode="real", driver="strands")`` today raises
``ValueError`` from :func:`~strands_robots.robot._build_native_driver` because
:func:`~strands_robots.drivers.registry.get_native_driver_class` returns
``None``. That error message says "add a driver package"; this file is the
smallest driver package that removes that failure mode without lying about
what works.

The surface a caller sees at each stub method is the shape it will hold when
the bus lands:

* ``send_action`` - refused with ``"not wired yet (issue #359 bus)"``.
* ``start_task`` / ``run_policy`` - refused with the same envelope. There is
  no FSM to gate against on a servo bus (unlike the G1's ``mode_machine``),
  but a caller who plumbed error handling for the deferred G1 path gets to
  reuse it here.
* ``get_task_status`` / ``stop_task`` - return a live-but-empty envelope; the
  driver has no in-flight work to report while writes are deferred.
* ``cleanup`` - a no-op; there is nothing to release.

None of this pretends. Every stub returns an envelope of the same shape a
successful path would return, so the mesh and the agent do not need a code
change on the day the bus lands.

The class is registered for every Dynamixel robot the package registry knows
about - see :func:`~strands_robots.drivers._register_shipped_drivers` for the
list. Registering after import (``from strands_robots.drivers.dynamixel import
DynamixelDriver`` then :func:`register_native_driver`) is also supported and
is how an out-of-tree driver package would extend the table.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The robots this driver serves. Every entry corresponds to a canonical name
# in ``strands_robots/registry/robots.json``. The list is deliberately narrow:
# a Dynamixel driver could in principle serve every arm on that bus, but the
# ones :issue:`359` names are the ones the acceptance criteria measure, and
# registering for a robot we cannot verify is a promise this driver does not
# yet keep.
# ---------------------------------------------------------------------------
SUPPORTED_ROBOTS: tuple[str, ...] = (
    "koch",
    "aloha",
    "vx300s",
    "wx250s",
    "trossen_wxai",
    "dynamixel_2r",
)

_TOOL_TYPE = "robot"

# Refusal reason shared between the four deferred verbs. The literal string is
# checked in tests, so a change here is a change to the driver contract.
_NOT_WIRED = "not wired yet (issue #359 bus)"


class DynamixelDriver:
    """Native Protocol-2.0 driver for the arms in :data:`SUPPORTED_ROBOTS`.

    Constructor contract matches :class:`~strands_robots.drivers.base.HardwareDriver`
    - the factory builds every native driver as ``driver_cls(tool_name=...,
    cameras=..., data_config=..., **kwargs)`` and forwards the caller's extras
    in ``kwargs``. Dynamixel-specific keywords land in ``kwargs``:

    * ``port`` - a serial device path (``/dev/tty.usbserial-*``) for a single
      bus, or a sequence of them for a bimanual rig. Kept polymorphic:
      Aloha's ``ports=[...]`` and the single-bus ``port=`` both land here.
    * ``baud_rate`` - integer, defaults to ``1_000_000``. The Robotis default.
    * ``motor_ids`` - the servo IDs on the bus, in wire order. Optional at
      construction; the bus discovers them on connect.
    """

    tool_type = _TOOL_TYPE

    def __init__(
        self,
        tool_name: str,
        cameras: Any | None = None,
        data_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_name = tool_name
        self._cameras = cameras
        self._data_config = data_config
        # port and ports are two spellings of the same field; the mesh's
        # keyboard-teleop passes ``port=`` and Aloha's example passes
        # ``ports=[...]``. Store both, normalise to a tuple.
        port = kwargs.pop("port", None)
        ports = kwargs.pop("ports", None)
        if port is not None and ports is not None:
            raise ValueError(
                f"DynamixelDriver({tool_name!r}): pass port= for a single bus or ports= for multiple, not both",
            )
        if ports is None and port is None:
            self._ports: tuple[str, ...] = ()
        elif ports is not None:
            self._ports = tuple(ports)
        else:
            self._ports = (port,)
        self._baud_rate: int = int(kwargs.pop("baud_rate", 1_000_000))
        self._motor_ids: tuple[int, ...] = tuple(kwargs.pop("motor_ids", ()))
        self._connected: bool = False
        self._connect_error: str | None = None
        # extras from the caller are kept for a downstream driver package
        # to consume; refusing them here would refuse a valid future
        # extension.
        self._extras = kwargs

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

        Three read-only verbs land now (``status``, ``sensors``, ``stop``);
        the write verbs (``move_to``, ``set_torque``, ``home``) land with the
        bus. Refusing a verb the schema declares is worse than not declaring
        it - an agent that plans against the schema will pick a verb it sees.
        """
        return {
            "name": self._tool_name,
            "description": f"Dynamixel-native driver for {self._tool_name} (Protocol 2.0). Read-only until the bus lands (issue #359).",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "sensors", "stop"],
                            "description": "status: connection + motor list; sensors: last-read joint state; stop: refuse further writes.",
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

        Follows the shape :class:`G1Driver` uses for its own deferred motion
        path so a caller writes the same error-checking code either way.
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
            envelope = {
                "status": "success",
                "content": [{"json": {"joint_state": None, "reason": _NOT_WIRED}}],
            }
        else:  # "stop"
            await self.stop()
            envelope = {
                "status": "success",
                "content": [{"text": f"stop: {_NOT_WIRED}"}],
            }
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Motion, task and policy paths. All refuse in the same envelope.     #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Refuse: the bus that would carry the write is not yet wired.

        The envelope shape mirrors a successful ``send_action`` so the mesh's
        error path handles both cases uniformly. Once the bus lands the
        signature and shape do not change; only the refusal is lifted.
        """
        del action, robot_name
        return _refuse(f"send_action: {_NOT_WIRED}")

    def start_task(
        self,
        task: str,
        robot_name: str | None = None,
        policy: Policy | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: no policy execution path exists on the servo bus yet."""
        del task, robot_name, policy, kwargs
        return _refuse(f"start_task: {_NOT_WIRED}")

    def run_policy(
        self,
        policy: Policy,
        robot_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: no policy execution path exists on the servo bus yet."""
        del policy, robot_name, kwargs
        return _refuse(f"run_policy: {_NOT_WIRED}")

    def get_task_status(self) -> dict[str, Any]:
        """Return an empty-but-well-formed envelope.

        A caller polling task status sees nothing running rather than an
        error, because "nothing to run" is the honest answer during the stub
        phase.
        """
        return {
            "status": "success",
            "content": [{"json": {"in_flight": False, "reason": _NOT_WIRED}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """No-op success: there is nothing to stop."""
        return {"status": "success", "content": [{"text": f"stop_task: {_NOT_WIRED}"}]}

    def cleanup(self) -> None:
        """No-op: the driver holds no OS resources until the bus lands.

        The method exists because the driver contract requires it (see
        :data:`~strands_robots.drivers.base.DRIVER_SURFACE`) and is a
        genuine no-op today rather than a placeholder that opens something
        it needs to release.
        """
        return None

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                               #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Report the connection state; do not open a port.

        Kept as a method rather than a lie: a driver whose bus is not wired
        yet cannot connect. Returning a named reason ("bus not wired") is
        clearer than either returning ``None`` (which the caller would read
        as success) or raising (which cannot be distinguished from a real
        hardware failure).
        """
        if self._connected:
            return None
        reason = _NOT_WIRED
        self._connect_error = reason
        return reason

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's construction and configuration.

        Shape matches :class:`G1Driver.get_status` so both peers publish
        identically; fields absent on a Dynamixel bus (an FSM, a battery
        percentage) are simply not in the payload.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "tool_type": self.tool_type,
                        "connected": self._connected,
                        "connect_error": self._connect_error,
                        "ports": list(self._ports),
                        "baud_rate": self._baud_rate,
                        "motor_ids": list(self._motor_ids),
                        "supported_robots": list(SUPPORTED_ROBOTS),
                        "reason": _NOT_WIRED,
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Refuse further writes. A no-op today; the shape lands with the bus."""
        return None


# ---------------------------------------------------------------------------
# Envelope helpers. Kept private and one-liner-ish rather than reaching for a
# shared library, because the shape is small and the tests grade against the
# literal envelope.
# ---------------------------------------------------------------------------
def _refuse(message: str) -> dict[str, Any]:
    """Return an error envelope with ``message``, matching the "not wired" contract."""
    return {"status": "error", "content": [{"text": message}]}
