"""The opt-in ``Call Service`` surface of a :class:`~strands_robots.foxglove.FoxgloveBridge`.

Off by default. With ``foxglove_services=True`` the server advertises one
service, ``strands/set_joint_positions``, whose request is JSON::

    {"robot": "so101", "positions": {"1": 0.2, "2": -0.4}}

``robot`` may be omitted when the engine holds one robot. Every call goes
through the same operator gate a ROS 2 or serial command does
(:func:`strands_robots._command_gate.gate_motion`): a Foxglove client carries
no agent ``tool_context``, so there is nobody to ask, and the call is refused
until the operator pre-approves the surface with
``STRANDS_FOXGLOVE_COMMAND_ALLOW=strands/set_joint_positions`` (or ``*``) in
the robot's environment, or sets ``BYPASS_TOOL_CONSENT=true``. A refusal is
raised, so the Call Service panel shows the gate's own sentence, and both the
refusal and an executed command are written to ``/strands/events`` and
``/strands/log``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from strands_robots._command_gate import gate_motion

if TYPE_CHECKING:
    from strands_robots.foxglove.bridge import FoxgloveBridge

logger = logging.getLogger(__name__)

#: Environment variable naming the pre-approved Foxglove services, comma-separated.
FOXGLOVE_COMMAND_ALLOW_ENV = "STRANDS_FOXGLOVE_COMMAND_ALLOW"

#: The one service this module advertises.
SET_JOINT_POSITIONS = "strands/set_joint_positions"

_TOOL = "foxglove"

_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "robot": {"type": "string"},
        "positions": {"type": "object", "additionalProperties": {"type": "number"}},
    },
    "required": ["positions"],
}
_RESPONSE_SCHEMA = {"type": "object"}


def parse_request(payload: bytes | None) -> tuple[str | None, dict[str, float]]:
    """Decode a service request into ``(robot, positions)``.

    Raises:
        ValueError: The payload is not a JSON object with a ``positions`` object of numbers.
    """
    try:
        body = json.loads(payload or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{SET_JOINT_POSITIONS}: request is not JSON ({exc})") from exc
    if not isinstance(body, dict) or not isinstance(body.get("positions"), dict) or not body["positions"]:
        raise ValueError(
            f'{SET_JOINT_POSITIONS}: request must be {{"robot": "<name>", "positions": {{"<joint>": <radians>}}}} '
            "with at least one joint."
        )
    positions: dict[str, float] = {}
    for joint, value in body["positions"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{SET_JOINT_POSITIONS}: positions[{joint}] must be a number.")
        positions[str(joint)] = float(value)
    robot = body.get("robot")
    if robot is not None and not isinstance(robot, str):
        raise ValueError(f"{SET_JOINT_POSITIONS}: robot must be a string when given.")
    return robot, positions


def gate_service(name: str) -> str | None:
    """The operator gate for one Foxglove service call; the refusal sentence, or ``None`` to proceed."""
    return gate_motion(
        _TOOL,
        "call_service",
        name,
        f"{name!r} moves the robot from a Foxglove panel, which carries no operator approval.",
        None,
        allow_env=FOXGLOVE_COMMAND_ALLOW_ENV,
    )


def build_services(bridge: FoxgloveBridge, command_sink: Any) -> list[Any]:
    """The ``foxglove.Service`` list a bridge registers when services are on.

    Args:
        bridge: The owning bridge, for its log and event channels.
        command_sink: ``callable(robot, positions) -> result dict`` that
            applies the command once the gate has let it through.

    Returns:
        A one-element list holding :data:`SET_JOINT_POSITIONS`.
    """
    from foxglove import MessageSchema, Schema, Service, ServiceSchema

    def _json_schema(schema: dict[str, Any]) -> MessageSchema:
        return MessageSchema(
            encoding="json",
            schema=Schema(name="json", encoding="jsonschema", data=json.dumps(schema).encode("utf-8")),
        )

    def _set_joint_positions(request: Any) -> bytes:
        robot, positions = parse_request(getattr(request, "payload", None))
        refusal = gate_service(SET_JOINT_POSITIONS)
        if refusal is not None:
            bridge.log("warning", refusal, name="gate")
            bridge.event({"event": "gate", "service": SET_JOINT_POSITIONS, "robot": robot, "decision": "refused"})
            raise PermissionError(refusal)
        bridge.event({"event": "gate", "service": SET_JOINT_POSITIONS, "robot": robot, "decision": "allowed"})
        result = command_sink(robot, positions)
        text = ""
        if isinstance(result, dict):
            content = result.get("content") or []
            text = str(content[0].get("text", "")) if content and isinstance(content[0], dict) else ""
        status = result.get("status", "success") if isinstance(result, dict) else "success"
        bridge.log(
            "info" if status == "success" else "error", text or f"{SET_JOINT_POSITIONS}: {status}", name="service"
        )
        bridge.event(
            {"event": "service", "service": SET_JOINT_POSITIONS, "robot": robot, "status": status, "text": text}
        )
        return json.dumps({"status": status, "text": text}).encode("utf-8")

    return [
        Service(
            SET_JOINT_POSITIONS,
            schema=ServiceSchema(
                SET_JOINT_POSITIONS,
                request=_json_schema(_REQUEST_SCHEMA),
                response=_json_schema(_RESPONSE_SCHEMA),
            ),
            handler=_set_joint_positions,
        )
    ]
