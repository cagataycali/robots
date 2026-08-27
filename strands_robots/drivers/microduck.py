"""Native daemon driver for the Pollen Robotics Microduck.

``Robot("microduck", mode="real", port="/run/robotd.sock")`` builds one of
these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver` structurally, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like any other driver.

Why a native driver: the Microduck is driven by ``robotd``, a daemon that owns
the 50 Hz control loop *and runs the walking/skill policy on-device*. Its IPC
surface (``duck-ipc-proto``) is JSON-RPC 2.0, one object per line (NDJSON), over
a unix socket. There is no lerobot robot type for it and no serial servo bus to
address; the honest client speaks ``robotd``'s protocol.

THE DECISIVE FACT - robotd exposes **no per-joint write**. The whole ``robot.*``
surface is intent-level: ``robot.move`` (twist), ``robot.head``, ``robot.pose``,
``robot.do`` (skills), ``robot.enable``/``robot.relax`` (torque), ``robot.init``,
``robot.stop``, and ``robot.state`` (read). So ``mode="real"`` is *delegate-only*
by the robot's own design: this driver sends INTENTS and robotd's on-device
policy produces the joint targets. ``run_policy``/``start_task`` therefore do not
pretend to stream a MicroduckPolicy's 14 joint targets to hardware - the wire has
no such method - they refuse and name the intent path instead. Sim-to-real parity
is preserved regardless: the on-robot policy is the *same* ``alpha_walking.onnx``
run in sim (byte-compat 0.0), so a sim rollout predicts the hardware for equal obs.

Continuous intents (``robot.move``/``robot.head``/``robot.pose``/``robot.mouth``)
are sent as JSON-RPC *notifications* (no ``id``, no reply); discrete ones
(``robot.do``/``robot.enable``/``robot.init``/``robot.stop``/``robot.relax``) as
*requests* whose id-correlated reply is awaited.

The 15-vs-14 papercut: robotd's ``JOINT_NAMES`` is 15 wide with ``"mouth"``
spliced at index 9; the policy/sim contract is the 14 locomotion joints (no
mouth). :meth:`MicroduckDriver.read_state` drops index 9 so the joints it
publishes are the 14 the policy speaks, and mouth travels via ``robot.mouth``.

Nothing here imports a transport at module load: every socket touch is inside a
method body, so the module imports on CI and in every unit test.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands_robots.policies.microduck import MICRODUCK_JOINT_NAMES
from strands_robots.utils import finite_number_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The socket robotd serves (``duck-ipc-proto`` ``socket::ROBOT``). ``port=``
#: overrides it; a ``host:path`` form names a remote whose socket has been
#: forwarded (how ``duckctl`` reaches a robot over SSH).
DEFAULT_SOCKET: str = "/run/robotd.sock"

#: The API version this driver speaks, pinned to ``duck-ipc-proto`` ``API_VERSION``.
#: The Hello handshake refuses a robotd whose version differs rather than
#: mis-parsing its frames later.
MICRODUCK_API_VERSION: int = 16

#: JSON-RPC version string every frame carries.
JSONRPC_VERSION: str = "2.0"

#: robotd's HARDWARE joint order - 15 wide, ``"mouth"`` spliced at index 9.
#: ``robot.state`` ``joints``/``targets`` are indexed by this. Kept as the wire
#: truth; :data:`MOUTH_INDEX` is dropped to reach the 14 locomotion joints.
HARDWARE_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

#: Index of ``"mouth"`` in :data:`HARDWARE_JOINT_NAMES`; dropped to map 15->14.
MOUTH_INDEX: int = 9

#: The 14 locomotion joints the policy/sim contract speaks, in contract order.
#: Equal to :data:`HARDWARE_JOINT_NAMES` with index 9 removed - asserted in the
#: tests so a divergence between the wire map and the policy contract is caught.
LOCOMOTION_JOINT_NAMES: tuple[str, ...] = MICRODUCK_JOINT_NAMES

# robotd method names (duck-ipc-proto ``method`` module).
_M_HELLO = "hello"
_M_MOVE = "robot.move"
_M_HEAD = "robot.head"
_M_POSE = "robot.pose"
_M_MOUTH = "robot.mouth"
_M_DO = "robot.do"
_M_ENABLE = "robot.enable"
_M_INIT = "robot.init"
_M_RELAX = "robot.relax"
_M_STOP = "robot.stop"
_M_STATE = "robot.state"
_M_HEALTH = "robot.health"
_M_SUBSCRIBE = "robot.subscribe"

#: Skills robotd's ``robot.do`` accepts. The wire enum is ``snake_case``
#: (``Skill`` ``#[serde(rename_all="snake_case")]``), so a ``skill`` action value
#: is normalised to these before it is sent - a typo is refused here, not
#: silently no-op'd on the robot.
SKILLS: tuple[str, ...] = ("ground_pick", "kick_left", "kick_right", "sit_toggle", "roulade")

#: Action keys this driver knows how to turn into an intent, for the refusal
#: message when an action names none of them.
_ACTION_KEYS: tuple[str, ...] = (
    "vx",
    "vy",
    "vyaw",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "z",
    "roll",
    "pitch",
    "active",
    "open",
    "skill",
)


# --------------------------------------------------------------------------- #
# Pure wire encoding - no socket, so a test asserts the exact bytes.          #
# --------------------------------------------------------------------------- #


def _encode(obj: dict[str, Any]) -> bytes:
    """Serialise one JSON-RPC frame to a single NDJSON line.

    Compact separators (no spaces) and a trailing ``\\n`` match what
    ``serde_json`` emits on the robotd side, so the bytes this driver writes are
    the bytes a real robotd would accept.
    """
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def _request(request_id: int, method: str, params: dict[str, Any]) -> bytes:
    """A JSON-RPC request (carries ``id``) as NDJSON bytes.

    Field order - ``jsonrpc``, ``id``, ``method``, ``params`` - matches the Rust
    ``Request`` struct so the serialised line is byte-identical. Methods that
    take no parameters send ``params: {}`` (an empty object), the same as
    robotd's ``Call::params`` for its unit variants.
    """
    return _encode({"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method, "params": params})


def _notification(method: str, params: dict[str, Any]) -> bytes:
    """A JSON-RPC notification (no ``id``, no reply) as NDJSON bytes."""
    return _encode({"jsonrpc": JSONRPC_VERSION, "method": method, "params": params})


def action_to_wire(action: dict[str, Any]) -> list[tuple[str, dict[str, Any], bool]] | str:
    """Translate a validated action dict into robotd intents.

    Returns a list of ``(method, params, is_notification)`` in a fixed order -
    twist, head, pose, mouth, skill - so two identical actions always produce
    the same wire sequence. Continuous intents are notifications
    (``is_notification=True``); ``robot.do`` is a request (``False``). Returns a
    reason string when a ``skill`` value is not a known skill, so the driver
    refuses at the door rather than sending a frame robotd will reject.

    The param structs mirror the Rust field order exactly:
    ``MoveParams{vx,vy,vyaw}``, ``HeadParams{neck_pitch,head_pitch,head_yaw,
    head_roll}``, ``PoseParams{z,roll,pitch,active}``, ``MouthParams{open}``,
    ``DoParams{skill}``.
    """
    commands: list[tuple[str, dict[str, Any], bool]] = []

    if any(k in action for k in ("vx", "vy", "vyaw")):
        commands.append(
            (
                _M_MOVE,
                {
                    "vx": float(action.get("vx", 0.0)),
                    "vy": float(action.get("vy", 0.0)),
                    "vyaw": float(action.get("vyaw", 0.0)),
                },
                True,
            )
        )

    if any(k in action for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")):
        commands.append(
            (
                _M_HEAD,
                {
                    "neck_pitch": float(action.get("neck_pitch", 0.0)),
                    "head_pitch": float(action.get("head_pitch", 0.0)),
                    "head_yaw": float(action.get("head_yaw", 0.0)),
                    "head_roll": float(action.get("head_roll", 0.0)),
                },
                True,
            )
        )

    if any(k in action for k in ("z", "roll", "pitch", "active")):
        commands.append(
            (
                _M_POSE,
                {
                    "z": float(action.get("z", 0.0)),
                    "roll": float(action.get("roll", 0.0)),
                    "pitch": float(action.get("pitch", 0.0)),
                    "active": bool(action.get("active", True)),
                },
                True,
            )
        )

    if "open" in action:
        commands.append((_M_MOUTH, {"open": float(action["open"])}, True))

    if "skill" in action:
        skill = str(action["skill"]).strip().lower()
        if skill not in SKILLS:
            return f"unknown skill {action['skill']!r}; expected one of {list(SKILLS)}"
        commands.append((_M_DO, {"skill": skill}, False))

    return commands


def map_hardware_joints(values: list[float]) -> dict[str, float]:
    """Map robotd's 15-wide ``joints``/``targets`` to the 14 locomotion joints.

    Drops index 9 (``mouth``) and names the rest by
    :data:`LOCOMOTION_JOINT_NAMES`. A vector that is not 15 wide is mapped by
    position for whatever it does carry rather than raising - a robotd that grew
    or shrank the vector should degrade to a partial read, not a crash.
    """
    if len(values) == len(HARDWARE_JOINT_NAMES):
        locomotion = [v for i, v in enumerate(values) if i != MOUTH_INDEX]
        return dict(zip(LOCOMOTION_JOINT_NAMES, locomotion, strict=True))
    return {name: float(v) for name, v in zip(LOCOMOTION_JOINT_NAMES, values, strict=False)}


def _refuse(reason: str) -> dict[str, Any]:
    """The driver's error envelope, one shape for every refusal path."""
    return {"status": "error", "content": [{"text": reason}]}
