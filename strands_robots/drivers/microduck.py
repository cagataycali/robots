"""Native daemon driver for the Pollen Robotics Microduck.

``Robot("microduck", mode="real")`` builds one of these. With no ``port`` the
socket is discovered: ``$MICRODUCK_SOCKET`` (a local unix path, e.g. one you
forwarded yourself), then ``$MICRODUCK_HOST`` (``[user@]host`` - the driver runs
``ssh -N -L`` and forwards robotd's socket the way an operator does; the user
defaults to ``radxa``, ``$DUCK_BOARD_USER`` is honoured as ``duckctl`` does),
then ``/run/robotd.sock`` itself - the answer when this code runs on the duck.
A ``port`` of the form ``ssh://[user@]host`` names the forward explicitly. The instance satisfies
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
(``robot.do``/``robot.enable``/``robot.stop``/``robot.relax``) as
*requests* whose id-correlated reply is awaited.

The 15-vs-14 papercut: robotd's ``JOINT_NAMES`` is 15 wide with ``"mouth"``
spliced at index 9; the policy/sim contract is the 14 locomotion joints (no
mouth). :meth:`MicroduckDriver.read_state` drops index 9 so the joints it
publishes are the 14 the policy speaks, and mouth travels via ``robot.mouth``.

Nothing here imports a transport at module load: every socket touch is inside a
method body, so the module imports on CI and in every unit test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strands_robots._path_validation import resolve_output_path, validate_save_path
from strands_robots.drivers.base import undeclared_verb_error
from strands_robots.utils import (
    boolean_flag_error,
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
    refusal_repr,
    refusal_str,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The socket robotd serves (``duck-ipc-proto`` ``socket::ROBOT``). ``port=``
#: overrides it (a local unix path, or ``ssh://[user@]host`` for a forward).
DEFAULT_SOCKET: str = "/run/robotd.sock"

#: ``mediad``'s on-demand raw-frame socket (``socket::MEDIA``) - ``media.frame``.
DEFAULT_MEDIA_SOCKET: str = "/run/mediad/media.sock"

#: ``tofd``'s depth stream socket (``socket::TOF``) - ``tof.stream``.
DEFAULT_TOF_SOCKET: str = "/run/tofd/tof.sock"

#: Environment knobs the zero-argument discovery reads, in this order.
ENV_SOCKET: str = "MICRODUCK_SOCKET"
ENV_HOST: str = "MICRODUCK_HOST"
#: Local paths for mediad's and tofd's sockets when you forwarded them yourself.
ENV_MEDIA_SOCKET: str = "MICRODUCK_MEDIA_SOCKET"
ENV_TOF_SOCKET: str = "MICRODUCK_TOF_SOCKET"
#: The ssh login on the board; ``duckctl`` reads the same variable.
ENV_SSH_USER: str = "DUCK_BOARD_USER"
DEFAULT_SSH_USER: str = "radxa"

#: The ``duck-ipc-proto`` ``API_VERSION`` this driver's frames were checked
#: against (microduck release 0.14.1). Sent in Hello and compared with the
#: daemon's answer; a difference is logged, not refused. duck-ipc-proto's own
#: rule: "No daemon refuses a call because this number differs" - every frame is
#: ``#[serde(default)]``-tolerant, a route that moved refuses itself by name
#: (``METHOD_NOT_FOUND``/``INVALID_PARAMS``), and a gate on the handshake would
#: refuse every serveable call along with it (which is exactly how the pin at 16
#: refused every 0.14 duck).
MICRODUCK_API_VERSION: int = 31

#: ``[safety] deadman_ms`` default (robotd-params): a twist older than this is
#: zeroed by the control loop and ``move.limited_by`` says ``deadman``. Head,
#: pose and mouth intents are NOT deadman'd - a stale head pose is harmless, a
#: stale velocity walks the robot into a wall (duck-control ``safety.rs``).
DEADMAN_S: float = 0.5

#: How often :meth:`MicroduckDriver.move` re-sends the twist while it runs -
#: the pad sends at 50 Hz; a fifth of the deadman window is plenty of margin.
MOVE_REFRESH_HZ: float = 20.0
MOVE_DURATION_DEFAULT: float = 1.0
MOVE_DURATION_MIN: float = 0.1
MOVE_DURATION_MAX: float = 10.0

#: robotd clamps nothing on a twist. The only velocity envelope the robot has is
#: the pad's stick shaping (``padd`` ``--max-linear 0.3``, ``--max-angular 1.5``;
#: roller ``ROLLER_PUSH 0.6`` / ``ROLLER_BRAKE 0.5`` / ``ROLLER_YAW 0.3``, no
#: strafe), which is what the policies were driven with. Beyond it is a policy
#: leaning on inputs it never saw, so this driver refuses there, by mode.
WALK_MAX_LINEAR: float = 0.3
WALK_MAX_ANGULAR: float = 1.5
ROLLER_MAX_FORWARD: float = 0.6
ROLLER_MAX_BACKWARD: float = 0.5
ROLLER_MAX_ANGULAR: float = 0.3

#: ``PoseParams`` doc: trained ranges z -0.025..+0.010 m, roll/pitch +/-0.26 rad;
#: "the robot clamps nothing here".
POSE_Z_MIN: float = -0.025
POSE_Z_MAX: float = 0.010
POSE_TILT_MAX: float = 0.26

#: The voice-bank tags ``robot.sound`` accepts (``SoundTag``, snake_case).
#: ``wheee`` is a held ride driven per tick by the pad's trigger - a hold this
#: driver cannot honestly keep, so :meth:`MicroduckDriver.play_sound` refuses it.
SOUND_TAGS: tuple[str, ...] = ("alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee")
PLAYABLE_SOUND_TAGS: tuple[str, ...] = tuple(tag for tag in SOUND_TAGS if tag != "wheee")

#: ``robot.setMode`` accepts exactly these (``SetModeParams`` doc).
DRIVE_MODES: tuple[str, ...] = ("walk", "roller")

#: ``LoadPolicyParams.slot`` names (0.14.1); the robot refuses one it lacks.
POLICY_SLOTS: tuple[str, ...] = ("walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade")

#: ``ImuHealth::FROZEN_RUN`` - this many consecutive stale IMU blocks means the
#: board has stopped fusing while still answering the bus.
IMU_FROZEN_RUN: int = 25

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

#: The 14 locomotion joints the policy/sim contract speaks, in contract order:
#: :data:`HARDWARE_JOINT_NAMES` with :data:`MOUTH_INDEX` removed. Derived from
#: this driver's own wire map rather than read back from
#: :data:`~strands_robots.policies.microduck.MICRODUCK_JOINT_NAMES`, so the
#: driver does not take its joint roster from a policy's tensor ordering; that
#: the two still agree is asserted in the tests.
LOCOMOTION_JOINT_NAMES: tuple[str, ...] = tuple(
    name for index, name in enumerate(HARDWARE_JOINT_NAMES) if index != MOUTH_INDEX
)

# robotd method names (duck-ipc-proto ``method`` module).
_M_HELLO = "hello"
_M_MOVE = "robot.move"
_M_HEAD = "robot.head"
_M_POSE = "robot.pose"
_M_MOUTH = "robot.mouth"
_M_DO = "robot.do"
_M_ENABLE = "robot.enable"
_M_RELAX = "robot.relax"
_M_STOP = "robot.stop"
_M_STATE = "robot.state"
_M_HEALTH = "robot.health"
_M_SUBSCRIBE = "robot.subscribe"
_M_LOOK = "robot.look"
_M_INIT = "robot.init"
_M_REBOOT_MOTORS = "robot.rebootMotors"
_M_SOUND = "robot.sound"
_M_THEREMIN = "robot.theremin"
_M_MODE = "robot.mode"
_M_SET_MODE = "robot.setMode"
_M_POLICIES = "robot.policies"
_M_MODEL = "robot.model"
_M_MODEL_API = "robot.modelApi"
_M_LOAD_POLICY = "robot.loadPolicy"
_M_RELOAD_POLICIES = "robot.reloadPolicies"
_M_MEDIA_FRAME = "media.frame"
_M_TOF_STREAM = "tof.stream"
_M_TOF_FRAME = "tof.frame"

#: The skills a STOCK robot's ``robot.do`` answers to. Since API v28 a skill is
#: a name from the robot's config, learned from ``robot.subscribe``/
#: ``robot.policies`` (``skills``); these five are the fallback vocabulary when
#: the robot has not said, and what :func:`action_to_wire` checks against when
#: no live list is given - a typo is refused here, not silently dropped on the
#: robot.
SKILLS: tuple[str, ...] = ("ground_pick", "kick_left", "kick_right", "sit_toggle", "roulade")

#: Action keys this driver knows how to turn into an intent. An action key
#: outside this tuple is refused, so it is also the vocabulary the two refusal
#: messages name - the one for an action naming none of them, and the one for an
#: action naming a key beside them.
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


def action_to_wire(
    action: dict[str, Any], known_skills: tuple[str, ...] | list[str] = SKILLS
) -> list[tuple[str, dict[str, Any], bool]] | str:
    """Translate a validated action dict into robotd intents.

    Returns a list of ``(method, params, is_notification)`` in a fixed order -
    twist, head, pose, mouth, skill - so two identical actions always produce
    the same wire sequence. Continuous intents are notifications
    (``is_notification=True``); ``robot.do`` is a request (``False``). Returns a
    reason string when a ``skill`` value is not a known skill, so the driver
    refuses at the door rather than sending a frame robotd will reject.

    A key this driver has no intent for is refused for the same reason, even
    with a known key beside it. Absent means resting - an action naming only ``vx``
    walks straight - but unknown means the caller named a component robotd never
    receives, and every intent frame carries its whole group, so the two are not
    distinguishable downstream: ``{"vx": 0.15, "yaw": 0.6}`` - ``yaw`` being the
    spelling ``get_status``' pose block uses for the heading - sends
    ``robot.move{vx:0.15,vy:0,vyaw:0}``, walking straight past the turn that was
    asked for, and reports success. The same holds for a 14-joint
    :data:`~strands_robots.policies.microduck.MICRODUCK_JOINT_NAMES` action:
    four of its keys are head axes, so it would send a ``robot.head`` frame
    built from those four and drop the other ten - the per-joint stream
    :meth:`MicroduckDriver.run_policy` refuses by name, arriving through the
    intent path instead.

    The param structs mirror the Rust field order exactly:
    ``MoveParams{vx,vy,vyaw}``, ``HeadParams{neck_pitch,head_pitch,head_yaw,
    head_roll}``, ``PoseParams{z,roll,pitch,active}``, ``MouthParams{open}``,
    ``DoParams{skill}``.
    """
    unknown = sorted(set(action) - set(_ACTION_KEYS))
    if unknown and not set(action).isdisjoint(_ACTION_KEYS):
        # Only when something else in the action did parse, so each refusal
        # diagnoses one fault: an action naming no intent at all is told what to
        # send (send_action's "nothing to send"), and one that mostly parsed is
        # told which component was dropped.
        return f"{unknown} name no Microduck intent; expected any of {sorted(_ACTION_KEYS)}"

    if "active" in action and (reason := boolean_flag_error(action["active"], "active", "send_action")) is not None:
        # A posture flag selects a standing pose on hardware; reading it by
        # truthiness would send active=true for the string "false". Refuse.
        return reason

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
        if skill not in known_skills:
            return f"unknown skill {action['skill']!r}; expected one of {list(known_skills)}"
        commands.append((_M_DO, {"skill": skill}, False))

    return commands


def map_hardware_joints(values: list[float]) -> dict[str, float]:
    """Map robotd's 15-wide ``joints``/``targets`` to the 14 locomotion joints.

    Drops index 9 (``mouth``) and names the rest by
    :data:`LOCOMOTION_JOINT_NAMES`.

    A vector that is not 15 wide is mapped positionally for whatever it does
    carry rather than raising, so a robotd whose vector changed width degrades
    instead of crashing the reader thread. What "degrades" means differs by
    direction, and the two are not symmetric: a *shorter* vector yields a
    partial read (only the joints it reaches are named), while a *longer* one
    yields a full 14-joint read in which index 9 is no longer dropped, so every
    joint after the mouth is named one position early. Which of those a 14-wide
    vector is - robotd having dropped the mouth itself, or having dropped some
    other joint - is not knowable from the width, so this maps by position and
    leaves the reading to the caller rather than guessing.
    """
    if len(values) == len(HARDWARE_JOINT_NAMES):
        locomotion = [v for i, v in enumerate(values) if i != MOUTH_INDEX]
        return dict(zip(LOCOMOTION_JOINT_NAMES, locomotion, strict=True))
    return {name: float(v) for name, v in zip(LOCOMOTION_JOINT_NAMES, values, strict=False)}


def parse_robot_state(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise a ``robot.state`` payload into the driver's cached shape.

    Pure, so a test parses a real RobotState fixture without a socket. The
    ``move``/``loop`` wire keys (renamed from the Rust ``movement``/
    ``control_loop`` fields) are read as they arrive; ``joints`` and ``targets``
    are mapped 15->14 via :func:`map_hardware_joints`.

    Args:
        params: The ``params`` object of a ``robot.state`` notification.

    Returns:
        A dict with ``t``, ``policy``, ``move``, ``head``, ``safety``, ``loop``,
        ``odom`` verbatim, plus ``joints``/``targets`` as 14-joint name->angle
        maps.
    """
    state: dict[str, Any] = {
        "t": params.get("t"),
        "policy": params.get("policy"),
        "move": params.get("move", {}),
        "head": params.get("head", []),
        "safety": params.get("safety", {}),
        "loop": params.get("loop", {}),
        "odom": params.get("odom", {}),
        "joints": map_hardware_joints(list(params.get("joints", []))),
        "targets": map_hardware_joints(list(params.get("targets", []))),
    }
    # Blocks a 0.14 robotd adds (``#[serde(default, skip_serializing_if)]``):
    # carried verbatim when present, absent when the robot did not send them,
    # so a reader can tell "no theremin" from "theremin silent".
    for key in ("imu", "frames", "theremin", "chorale", "t_ns"):
        if key in params:
            state[key] = params[key]
    return state


def _refuse(reason: str) -> dict[str, Any]:
    """The driver's error envelope, one shape for every refusal path."""
    return {"status": "error", "content": [{"text": reason}]}


# --------------------------------------------------------------------------- #
# Transport - one NDJSON JSON-RPC conversation over a unix socket.            #
# --------------------------------------------------------------------------- #


class _RobotdClient:
    """A single robotd connection: one reader thread, id-correlated replies.

    All inbound bytes are read by one thread. A discrete request registers an
    event keyed by its id which the reader fulfils; a ``robot.state``
    notification is handed to ``on_state``. Writes are serialised by a lock so a
    notification cannot interleave a request mid-line. The Hello handshake runs
    synchronously before the reader starts, since nothing else reads yet.
    """

    def __init__(self, socket_path: str, *, timeout: float = 5.0) -> None:
        self._path = socket_path
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._wlock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._on_state: Any = None
        self._stop = threading.Event()
        self.alive = False

    def _alloc_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _write(self, data: bytes) -> None:
        sock = self._sock
        if sock is None:
            raise ConnectionError("robotd socket is not connected")
        with self._wlock:
            sock.sendall(data)

    def connect(self) -> None:
        """Open the unix socket. Raises on failure so the caller names it."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._path)
        self._sock = sock
        self._rfile = sock.makefile("rb")
        self.alive = True

    def hello(self, api_version: int) -> dict[str, Any]:
        """Send the Hello handshake and return its ``result`` (synchronous)."""
        self._write(_request(self._alloc_id(), _M_HELLO, {"api_version": api_version}))
        line = self._rfile.readline()
        if not line:
            raise ConnectionError("robotd closed the connection during the Hello handshake")
        obj = json.loads(line)
        if obj.get("error"):
            raise ConnectionError(f"robotd refused Hello: {obj['error']}")
        return obj.get("result") or {}

    def start_reader(self, on_state: Any) -> None:
        """Begin the background reader that dispatches replies and state."""
        self._on_state = on_state
        self._reader = threading.Thread(target=self._read_loop, name="robotd-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            for line in self._rfile:
                if self._stop.is_set():
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("robotd sent an unparseable line: %r", line[:200])
                    continue
                if "method" not in obj and obj.get("id") is not None:
                    self._resolve(obj)
                elif obj.get("method") == _M_STATE and self._on_state is not None:
                    self._on_state(obj.get("params") or {})
        except OSError as exc:
            logger.debug("robotd reader stopped: %s", exc)
        finally:
            self.alive = False

    def _resolve(self, obj: dict[str, Any]) -> None:
        with self._pending_lock:
            slot = self._pending.pop(obj["id"], None)
        if slot is not None:
            event, box = slot
            box["result"] = obj.get("result")
            box["error"] = obj.get("error")
            event.set()

    def call(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Send a request and block for its id-correlated reply.

        Returns:
            The reply's ``result`` object (``{}`` when absent).

        Raises:
            TimeoutError: If no reply with the matching id arrives in time.
            ConnectionError: If robotd returned a JSON-RPC error.
        """
        rid = self._alloc_id()
        event = threading.Event()
        box: dict[str, Any] = {}
        with self._pending_lock:
            self._pending[rid] = (event, box)
        self._write(_request(rid, method, params))
        if not event.wait(timeout or self._timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"robotd did not answer {method!r} within {timeout or self._timeout}s")
        if box.get("error"):
            raise ConnectionError(f"robotd error on {method!r}: {box['error']}")
        return box.get("result") or {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification (no id, no reply)."""
        self._write(_notification(method, params))

    def close(self) -> None:
        """Stop the reader and close the socket. Idempotent."""
        self._stop.set()
        self.alive = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already half-closed; a shutdown race here is not actionable
            try:
                self._sock.close()
            except OSError:
                pass  # closing a socket that is already gone is fine
        self._sock = None
        self._rfile = None


# --------------------------------------------------------------------------- #
# Discovery - which socket, and how to reach it.                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Endpoint:
    """Where robotd is, as the driver resolved it.

    ``kind`` is ``"socket"`` (a local unix path) or ``"ssh"`` (the driver
    forwards the robot's sockets over ssh and connects to the local ends).
    ``source`` names what decided it - ``port``, an environment variable, or
    ``default`` - so a refusal can say which knob to change.
    """

    kind: str
    socket_path: str
    source: str
    host: str | None = None
    user: str | None = None

    @property
    def label(self) -> str:
        """The endpoint as a refusal names it."""
        if self.kind == "ssh":
            return f"ssh://{self.user}@{self.host}{self.socket_path}"
        return repr(self.socket_path)


def _split_login(spec: str, env: Mapping[str, str]) -> tuple[str, str]:
    """``[user@]host`` -> ``(user, host)``, user from ``$DUCK_BOARD_USER`` else ``radxa``.

    An emptied variable reads as unset, the rule ``duckctl`` follows: a
    variable blanked to switch it off must not become a login of ``@host``.
    """
    if "@" in spec:
        user, host = spec.rsplit("@", 1)
        return user, host
    return (env.get(ENV_SSH_USER) or DEFAULT_SSH_USER), spec


def resolve_endpoint(port: Any = None, env: Mapping[str, str] | None = None) -> Endpoint:
    """Decide where robotd is from ``port`` and the environment.

    Order: ``port`` (a unix path, or ``ssh://[user@]host``) → ``$MICRODUCK_SOCKET``
    → ``$MICRODUCK_HOST`` (``[user@]host``) → :data:`DEFAULT_SOCKET`. Pure: a
    test hands it an ``env`` mapping.
    """
    environment: Mapping[str, str] = os.environ if env is None else env
    if port:
        text = str(port).strip()
        if text.startswith("ssh://"):
            user, host = _split_login(text[len("ssh://") :].strip("/"), environment)
            return Endpoint("ssh", DEFAULT_SOCKET, "port", host=host, user=user)
        return Endpoint("socket", text, "port")
    if local := environment.get(ENV_SOCKET, "").strip():
        return Endpoint("socket", local, ENV_SOCKET)
    if remote := environment.get(ENV_HOST, "").strip():
        user, host = _split_login(remote, environment)
        return Endpoint("ssh", DEFAULT_SOCKET, ENV_HOST, host=host, user=user)
    return Endpoint("socket", DEFAULT_SOCKET, "default")


def _discovery_hint(endpoint: Endpoint) -> str:
    """What to set, appended to every "did not answer" refusal."""
    return (
        f"(resolved from {endpoint.source}). robotd listens on a unix socket ON the duck "
        f"(root:robot 0660). From another machine set {ENV_HOST}=[user@]<duck ip> so the driver "
        f"forwards it over ssh (user defaults to {DEFAULT_SSH_USER!r} or ${ENV_SSH_USER}), or forward "
        f"it yourself and set {ENV_SOCKET}=<local socket path>, or pass port="
    )


def ssh_forward_argv(
    user: str,
    host: str,
    forwards: list[tuple[str, str]],
    *,
    connect_timeout: float = 5.0,
) -> list[str]:
    """The ``ssh`` command that forwards robotd's sockets to local paths.

    Pure so a test asserts the exact arguments. ``-N`` runs no remote command,
    ``BatchMode=yes`` fails fast instead of prompting for a password (the board
    is key-authenticated, like ``duckctl ssh``), ``ExitOnForwardFailure`` turns a
    bind failure into an exit rather than a silent stall, and
    ``StreamLocalBindUnlink`` lets a stale local socket from a crashed forward
    be reused. Each ``(local, remote)`` pair is one ``-L local:remote`` - OpenSSH
    forwards unix sockets with the same flag it forwards ports with.

    Args:
        user: Login on the board.
        host: The duck's address.
        forwards: ``(local, remote)`` socket pairs to forward.
        connect_timeout: Seconds ssh may spend reaching the board, rounded to the
            whole second ssh reads. The shared positive-finite domain, because
            ``ssh`` cannot spend ``0``, a negative or a non-number, and the
            rounding would otherwise turn each of those into a silent
            ``ConnectTimeout=1`` or an exception naming no parameter.

    Raises:
        ValueError: If ``connect_timeout`` is not a positive finite number.
    """
    if reason := positive_finite_number_error(connect_timeout, "connect_timeout", "ssh_forward_argv"):
        raise ValueError(reason)
    argv = [
        "ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "StreamLocalBindUnlink=yes",
        "-o",
        f"ConnectTimeout={max(1, int(round(connect_timeout)))}",
    ]
    for local, remote in forwards:
        argv += ["-L", f"{local}:{remote}"]
    argv.append(f"{user}@{host}")
    return argv


class _SshForward:
    """One ``ssh -N -L`` process forwarding the duck's sockets to a temp dir.

    ``spawn`` is the process factory (``subprocess.Popen`` by default); a test
    injects one that stands a mock robotd up at the local path instead. The
    forward is up when the local robotd socket exists; ``ssh`` exiting first is
    the failure, reported with its stderr.
    """

    def __init__(
        self,
        user: str,
        host: str,
        *,
        remote_sockets: Mapping[str, str],
        timeout: float = 5.0,
        spawn: Callable[..., Any] | None = None,
    ) -> None:
        self._user = user
        self._host = host
        self._timeout = timeout
        self._spawn = spawn or subprocess.Popen
        self._dir = tempfile.mkdtemp(prefix="microduck-ssh-")
        self.local: dict[str, str] = {name: os.path.join(self._dir, f"{name}.sock") for name in remote_sockets}
        self._forwards = [(self.local[name], remote) for name, remote in remote_sockets.items()]
        self._proc: Any = None

    @property
    def argv(self) -> list[str]:
        return ssh_forward_argv(self._user, self._host, self._forwards, connect_timeout=self._timeout)

    def start(self) -> str | None:
        """Spawn ssh and wait for the robotd socket to appear. Returns a reason on failure."""
        if shutil.which("ssh") is None and self._spawn is subprocess.Popen:
            return "no `ssh` binary on this machine to forward the duck's socket with"
        try:
            self._proc = self._spawn(
                self.argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except OSError as exc:
            return f"could not start ssh: {refusal_str(exc)}"
        robot_socket = self.local["robot"]
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if os.path.exists(robot_socket):
                return None
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)
        stderr = b""
        if self._proc.poll() is not None and self._proc.stderr is not None:
            try:
                stderr = self._proc.stderr.read() or b""
            except OSError:
                stderr = b""
        self.close()
        detail = stderr.decode("utf-8", "replace").strip() or "ssh gave no reason"
        return f"ssh forward to {self._user}@{self._host} did not come up within {self._timeout}s: {detail}"

    def close(self) -> None:
        """Terminate ssh and remove the local sockets. Idempotent."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass  # already gone
        shutil.rmtree(self._dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Envelopes - refuse at the door, before float().                             #
# --------------------------------------------------------------------------- #


def _number(params: Mapping[str, Any], name: str, context: str, default: float | None = None) -> float | str:
    """Read one finite number from an agent's params, or the refusal text.

    Validated BEFORE ``float()``: a string, ``None``, a bool or a ``nan`` is a
    refusal naming the parameter and the value, never a ``TypeError`` out of
    the coercion or a NaN on the wire.
    """
    if name not in params or params[name] is None:
        if default is None:
            return f"{context}: {name} is required"
        return default
    value = params[name]
    if (reason := finite_number_error(value, name, context)) is not None:
        return reason
    return float(value)


def _within(value: float, low: float, high: float, name: str, unit: str, context: str) -> str | None:
    if not low <= value <= high:
        return f"{context}: {name}={value!r} {unit} is outside the trained range [{low}, {high}]"
    return None


def twist_error(vx: float, vy: float, vyaw: float, mode: str | None) -> str | None:
    """Refuse a twist outside the pad's envelope for ``mode`` (walk when unknown).

    Walk: ``|vx|, |vy| <= 0.3 m/s``, ``|vyaw| <= 1.5 rad/s``. Roller: ``vx`` in
    ``[-0.5, 0.6]`` (brake / push), no strafe, ``|vyaw| <= 0.3``. The robot does
    not clamp these - it hands them to the policy - so the envelope is here.
    """
    if mode == "roller":
        if not -ROLLER_MAX_BACKWARD <= vx <= ROLLER_MAX_FORWARD:
            return f"move: vx={vx!r} m/s is outside the roller envelope [-{ROLLER_MAX_BACKWARD}, {ROLLER_MAX_FORWARD}]"
        if vy != 0.0:
            return f"move: vy={vy!r} - the roller mode has no strafe; vy must be 0"
        if abs(vyaw) > ROLLER_MAX_ANGULAR:
            return f"move: vyaw={vyaw!r} rad/s exceeds the roller envelope +/-{ROLLER_MAX_ANGULAR}"
        return None
    if abs(vx) > WALK_MAX_LINEAR or abs(vy) > WALK_MAX_LINEAR:
        return f"move: vx={vx!r}, vy={vy!r} m/s exceed the walking envelope +/-{WALK_MAX_LINEAR} (the pad's full deflection)"
    if abs(vyaw) > WALK_MAX_ANGULAR:
        return f"move: vyaw={vyaw!r} rad/s exceeds the walking envelope +/-{WALK_MAX_ANGULAR}"
    return None


def _accepted(result: Mapping[str, Any]) -> bool:
    """robotd's ``IntentResult.accepted``; a result with no such field counts as accepted (``{}``)."""
    return bool(result.get("accepted", True))


# --------------------------------------------------------------------------- #
# The move stream - what keeps a one-call `move` alive past the deadman.      #
# --------------------------------------------------------------------------- #


class _MoveStream:
    """Re-send one twist at :data:`MOVE_REFRESH_HZ` for a bounded duration.

    The deadman zeroes a twist older than :data:`DEADMAN_S`, so a single
    ``robot.move`` walks the duck for half a second. An agent's ``move`` is one
    intent for ``duration`` seconds, so this thread keeps it fresh and then
    sends a zero twist once, so the robot stops by intent rather than by the
    deadman. :meth:`cancel` is the stop path; it sends nothing itself - the
    caller's ``robot.stop`` does.
    """

    def __init__(self, client: _RobotdClient, twist: dict[str, float], duration: float) -> None:
        self._client = client
        self.twist = twist
        self.duration = duration
        self.started = time.monotonic()
        self._cancel = threading.Event()
        self.sent = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="microduck-move", daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def active(self) -> bool:
        return self._thread.is_alive() and not self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self) -> None:
        period = 1.0 / MOVE_REFRESH_HZ
        deadline = self.started + self.duration
        try:
            while not self._cancel.is_set() and time.monotonic() < deadline:
                self._client.notify(_M_MOVE, self.twist)
                self.sent += 1
                self._cancel.wait(period)
            if not self._cancel.is_set():
                self._client.notify(_M_MOVE, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
        except OSError as exc:
            self.error = refusal_str(exc)


# --------------------------------------------------------------------------- #
# Other daemons' sockets - one request, one connection, as their clients do. #
# --------------------------------------------------------------------------- #


def _one_shot_call(path: str, method: str, params: dict[str, Any], timeout: float) -> tuple[Any, dict[str, Any]]:
    """Connect, Hello, send one request, return ``(reader, response)``.

    The shape ``robotctl frame`` uses for ``media.frame`` and ``tofd`` expects
    for ``tof.stream``: a fresh connection per ask. The caller reads what
    follows the response (a binary tail, or notifications) and closes.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(path)
    reader = sock.makefile("rb")
    try:
        sock.sendall(_request(1, _M_HELLO, {"api_version": MICRODUCK_API_VERSION}))
        hello = json.loads(reader.readline() or b"{}")
        if hello.get("error"):
            raise ConnectionError(f"hello refused: {hello['error']}")
        sock.sendall(_request(2, method, params))
        line = reader.readline()
        if not line:
            raise ConnectionError(f"{method}: the daemon closed the connection without answering")
        response = json.loads(line)
        if response.get("error"):
            raise ConnectionError(f"{method}: {response['error']}")
        return (sock, reader), response
    except Exception:
        reader.close()
        sock.close()
        raise


#: Where a captured frame lands. ``save_path`` is a tool parameter an agent
#: fills in, and a JPEG written wherever it points is a write to any file this
#: process can touch - a shell profile, a cron fragment - so the value names a
#: file *inside this root* and nothing else, the same confinement
#: ``Simulation.render(output_path=...)`` applies to a still. Point the variable
#: elsewhere to collect frames elsewhere; the refusal quotes the root in force.
FRAME_ROOT_ENV = "STRANDS_ROBOTS_MICRODUCK_FRAME_ROOT"


def frame_root() -> Path:
    """The directory a captured frame may be written into, read at call time."""
    raw = os.getenv(FRAME_ROOT_ENV) or str(Path.home() / ".strands_robots" / "microduck" / "frames")
    return Path(raw).expanduser()


def uyvy_to_jpeg(width: int, height: int, rotate: int, raw: bytes, path: str) -> None:
    """Decode a ``media.frame`` UYVY tail to a JPEG on disk, the mount turn applied.

    ``rotate`` is the header's mount angle (0/90/180/270): the raw path names
    it and leaves the picture as captured, so the file is turned here the way
    ``robotctl frame`` turns it. OpenCV is a core dependency of the package.
    """
    import cv2  # noqa: PLC0415 - heavy import kept off the module path
    import numpy as np  # noqa: PLC0415

    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 2)
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
    turns = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotate in turns:
        bgr = cv2.rotate(bgr, turns[rotate])
    if not cv2.imwrite(path, bgr):
        raise OSError(f"could not write {path}")


def summarise_tof(frame: Mapping[str, Any]) -> dict[str, Any]:
    """One ``tof.frame`` as a reader wants it: ranged cells, nearest, median.

    ``distance_mm`` is ``i16`` per cell; ``status`` is the sensor's per-cell
    code, where 5 and 9 are ST's "valid" verdicts (what ``robotctl monitor``
    draws as ranged). A cell that could not measure says nothing about what is
    out there, so it is counted separately rather than read as "far".
    """
    rows, cols = int(frame.get("rows", 0)), int(frame.get("cols", 0))
    distances = [int(d) for d in frame.get("distance_mm", [])]
    statuses = [int(s) for s in frame.get("status", [])]
    valid = [d for d, s in zip(distances, statuses, strict=False) if s in (5, 9) and d > 0]
    ordered = sorted(valid)
    return {
        "seq": frame.get("seq"),
        "rows": rows,
        "cols": cols,
        "cells": len(distances),
        "ranged": len(valid),
        "unmeasurable": len(distances) - len(valid),
        "nearest_m": (ordered[0] / 1000.0) if ordered else None,
        "median_m": (ordered[len(ordered) // 2] / 1000.0) if ordered else None,
        "farthest_m": (ordered[-1] / 1000.0) if ordered else None,
        "distance_mm": distances,
        "status": statuses,
    }


# --------------------------------------------------------------------------- #
# The driver.                                                                 #
# --------------------------------------------------------------------------- #


class MicroduckDriver:
    """Native robotd delegate driver for the Pollen Microduck.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally,
    so no import from that module is needed; the surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "microduck",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        api_version: int = MICRODUCK_API_VERSION,
        timeout: float = 5.0,
        subscribe_hz: int | None = None,
        ssh_spawn: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` talks to robotd.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer id.
            cameras: Accepted for parity with other drivers; unused here (the
                Microduck's cameras are not addressed by this driver).
            data_config: Accepted for parity; unused.
            port: Where robotd is: a unix socket path, or ``ssh://[user@]host``
                to have the driver forward the duck's sockets itself. ``None``
                discovers it - see :func:`resolve_endpoint` (``$MICRODUCK_SOCKET``,
                ``$MICRODUCK_HOST``, then :data:`DEFAULT_SOCKET`).
            api_version: API version to send in the Hello handshake, defaults to
                :data:`MICRODUCK_API_VERSION`. A robotd answering a different
                version is connected and the skew logged; each call it cannot
                serve refuses itself by name.
            timeout: Socket and request timeout in seconds. A positive, finite
                number: it is handed to ``socket.settimeout`` and to the reply
                wait, neither of which can report what it was given.
            subscribe_hz: State-stream decimation. ``None`` = every control tick;
                otherwise a positive integer, because it is sent to robotd as one.
            ssh_spawn: Process factory for the ssh forward (tests inject one);
                ``None`` means ``subprocess.Popen``.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If ``timeout`` is not a positive finite number, or
                ``subscribe_hz`` is neither ``None`` nor a positive integer.
                Raised here rather than returned from
                :meth:`connect_eagerly`, which is declared ``-> str | None``:
                a value the transport cannot use is not a connection this
                driver can degrade to reporting.
        """
        del cameras, data_config
        if kwargs:
            logger.debug("MicroduckDriver ignoring extra kwargs: %s", sorted(kwargs))

        # The two transport knobs are held to the shared numeric domains for
        # the same reason the actuation flags are held to ``boolean_flag_error``:
        # each reaches a consumer that cannot report what it was handed.
        # ``timeout`` goes to ``socket.settimeout`` and to the reply wait, so a
        # ``nan``, an ``inf``, a negative or a numeric string raised out of
        # :meth:`connect_eagerly` from inside the socket call - naming neither
        # this driver nor the parameter, out of a method declared
        # ``-> str | None`` - while ``True`` acted as a silent one second and
        # ``None`` left both the socket and the reply wait unbounded.
        # ``subscribe_hz`` is interpolated straight into the ``robot.subscribe``
        # params: ``nan``/``inf`` serialise to the bare ``NaN``/``Infinity``
        # tokens, which are not JSON (RFC 8259), so a strict daemon parser
        # refuses the frame while ``connect_eagerly`` reports success; a
        # ``numpy`` integer is not JSON-serialisable at all.  The strict-int
        # domain is the right member because robotd is sent an integer.
        if reason := positive_finite_number_error(timeout, "timeout", "MicroduckDriver"):
            raise ValueError(reason)
        if subscribe_hz is not None and (
            reason := positive_count_error(subscribe_hz, "subscribe_hz", "MicroduckDriver")
        ):
            raise ValueError(reason)
        # ``api_version`` is the same shape on the same wire: ``HelloParams`` is
        # a ``u32`` that robotd decodes with ``deny_unknown_fields``.
        if reason := positive_count_error(api_version, "api_version", "MicroduckDriver"):
            raise ValueError(reason)

        self._tool_name = tool_name
        self._endpoint = resolve_endpoint(port)
        self._socket_path = self._endpoint.socket_path
        self._media_socket = os.environ.get(ENV_MEDIA_SOCKET, "").strip() or DEFAULT_MEDIA_SOCKET
        self._tof_socket = os.environ.get(ENV_TOF_SOCKET, "").strip() or DEFAULT_TOF_SOCKET
        self._forward: _SshForward | None = None
        self._ssh_spawn = ssh_spawn
        self._api_version = api_version
        self._timeout = timeout
        self._subscribe_hz = subscribe_hz

        self._cache_lock = threading.Lock()
        self._joints: dict[str, float] = {}
        self._pose: dict[str, Any] | None = None
        self._imu: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._last_state: dict[str, Any] | None = None

        self._client: _RobotdClient | None = None
        self._connected: bool = False
        self._connect_error: str | None = None
        self._hello: dict[str, Any] | None = None
        self._stopped: bool = False
        self._move: _MoveStream | None = None
        self._skills: list[str] | None = None
        self._policies: dict[str, Any] | None = None
        self._health: dict[str, Any] | None = None
        self._health_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``\"robot\"`` - mirrors the other drivers."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether the robotd connection is live, for the mesh joint read."""
        return self._connected and self._client is not None and self._client.alive

    @property
    def tool_spec(self) -> ToolSpec:
        """The duck's whole vocabulary as one JSON-callable tool.

        The verbs an operator has at the pad, ``duckctl`` and ``robotctl`` -
        drive, head, pose, mouth, skills, sit/stand, look-at, sounds, the
        theremin, walk/roller mode, policy slots, camera, depth, health,
        odometry, a monitor snapshot - each mapping onto exactly one robotd,
        mediad or tofd path. The description says the one thing every write
        shares: robotd accepting an intent is not the robot having done it.
        """
        verbs = "; ".join(f"{name}: {text}" for name, text in _ACTION_HELP.items())
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Pollen Microduck native driver (bipedal duck: 14 locomotion joints + mouth, "
                    "head camera, ToF depth, speaker, IMU) over the on-robot robotd JSON-RPC unix "
                    "socket. Discovers the robot itself (MICRODUCK_SOCKET / MICRODUCK_HOST=[user@]ip "
                    "via ssh forward / /run/robotd.sock). Velocities m/s and rad/s, angles radians, "
                    "z metres. `move` runs for `duration` seconds (robotd zeroes a twist older than "
                    f"{DEADMAN_S}s, so the driver re-sends it, then sends a zero twist). Every write "
                    "returns robotd's acceptance (`accepted`, `reason`): NOT proof the robot moved - "
                    "read `sensors` or `monitor` afterwards. A running policy walks a duck whose torque "
                    "is on: `disable`/`relax` are the hands-off verbs, `stop` halts and holds."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": verbs,
                                "enum": list(_ACTIONS),
                                "default": "sensors",
                            },
                            "vx": {
                                "type": "number",
                                "description": "move: forward m/s (walk +/-0.3; roller -0.5..0.6)",
                            },
                            "vy": {"type": "number", "description": "move: left m/s (walk +/-0.3; roller must be 0)"},
                            "vyaw": {
                                "type": "number",
                                "description": "move: yaw rad/s, +left (walk +/-1.5; roller +/-0.3)",
                            },
                            "duration": {
                                "type": "number",
                                "description": f"move: seconds to keep the twist alive, {MOVE_DURATION_MIN}..{MOVE_DURATION_MAX} (default {MOVE_DURATION_DEFAULT})",
                            },
                            "neck_pitch": {
                                "type": "number",
                                "description": "head: neck pitch rad; look_at: neck posture to aim around (default 0)",
                            },
                            "head_pitch": {"type": "number", "description": "head: head pitch rad"},
                            "head_yaw": {"type": "number", "description": "head: yaw rad (left +)"},
                            "head_roll": {"type": "number", "description": "head: roll rad"},
                            "x": {
                                "type": "number",
                                "description": "look_at: target x in the robot frame, metres (forward +)",
                            },
                            "y": {"type": "number", "description": "look_at: target y, metres (left +)"},
                            "z": {
                                "type": "number",
                                "description": f"pose: body height offset m ({POSE_Z_MIN}..{POSE_Z_MAX}); look_at: target z, metres (up +)",
                            },
                            "roll": {"type": "number", "description": f"pose: body roll rad (+/-{POSE_TILT_MAX})"},
                            "pitch": {"type": "number", "description": f"pose: body pitch rad (+/-{POSE_TILT_MAX})"},
                            "active": {
                                "type": "boolean",
                                "description": "pose: hold the offsets (true) or release them (false); theremin: on/off",
                            },
                            "open": {"type": "number", "description": "mouth: 0 closed .. 1 open"},
                            "skill": {
                                "type": "string",
                                "description": "do: a skill name the robot lists (see `skills`)",
                            },
                            "on": {
                                "type": "boolean",
                                "description": "enable: true energises and starts the policy, false disables",
                            },
                            "mode": {
                                "type": "string",
                                "enum": list(DRIVE_MODES),
                                "description": "set_mode: walk or roller",
                            },
                            "slot": {
                                "type": "string",
                                "description": "load_policy: which policy slot (see `policies`)",
                            },
                            "path": {"type": "string", "description": "load_policy: ONNX path ON THE ROBOT"},
                            "tag": {
                                "type": "string",
                                "enum": list(PLAYABLE_SOUND_TAGS),
                                "description": "play_sound: voice-bank clip",
                            },
                            "ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "reboot_motors: servo ids (empty = all)",
                            },
                            "confirm": {
                                "type": "boolean",
                                "description": "relax / reboot_motors / init: must be true - the robot goes limp",
                            },
                            "save_path": {
                                "type": "string",
                                "description": (
                                    "camera: file name inside the frame root "
                                    "(default: a temp name there); an absolute path or `..` is refused"
                                ),
                            },
                        },
                        "required": ["action"],
                    }
                },
            },
        )

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result.

        A driver the factory built but nobody connected is connected here on
        the first verb that needs robotd, so ``Agent(tools=[Robot("microduck")])``
        works without a separate ``connect_eagerly`` line; a connect that fails
        is the verb's refusal, naming the socket and what to set. ``status``
        never connects - it is the question "are we connected", and answering
        it by connecting would make it unable to say no.
        """
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        params: dict[str, Any] = dict(tool_use.get("input") or {})
        action = params.pop("action", "sensors")
        # One dispatch decision. The halt has a branch of its own so the verb
        # returns ``stop_task``'s verdict (a robotd that declines the stop is
        # reported, never restated as success) and never waits on a connect.
        # Every other declared verb runs its handler from the table the schema
        # enum is generated from; anything else is refused by name and never
        # reaches a write. The terminal ``else`` refuses.
        if action == "stop":
            envelope = self.stop_task()
        elif isinstance(action, str) and action in _ACTIONS:
            if self._needs_connect(action) and (reason := self.connect_eagerly()) is not None:
                envelope = _refuse(f"{action}: {reason}")
            else:
                envelope = await asyncio.to_thread(_ACTIONS[action], self, params)
        else:
            envelope = undeclared_verb_error(self, action)
        yield {"toolUseId": tool_use_id, **envelope}

    def _needs_connect(self, action: str) -> bool:
        """Does this verb need robotd brought up first?

        ``status``/``sensors``/``sounds``/``odometry`` answer from the cache.
        ``camera``/``tof`` talk to mediad/tofd, not robotd - they only need the
        ssh forward, which :meth:`connect_eagerly` is what stands up.
        """
        if action in _OFFLINE_ACTIONS:
            return False
        if action in _MEDIA_ACTIONS:
            return self._endpoint.kind == "ssh" and self._forward is None
        return not self.is_connected

    @classmethod
    def probe_hardware(cls, timeout: float = 1.0) -> bool:
        """Is a robotd reachable with the environment as it stands? For ``mode="auto"``.

        Resolves the endpoint exactly as :meth:`connect_eagerly` would (so an
        ``ssh`` forward is attempted when ``$MICRODUCK_HOST`` is set) and asks
        Hello once. Nothing is subscribed and nothing is left open.
        """
        driver = cls(timeout=timeout)
        try:
            return driver.connect_eagerly() is None
        finally:
            driver.cleanup()

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                         #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Connect to robotd, Hello-handshake, then subscribe to state.

        Returns ``None`` on success. Off hardware - no socket at
        :data:`_socket_path` - returns a reason naming the socket that did not
        answer and leaves the driver usable (every read returns its empty cache,
        every write refuses "not connected"). Idempotent: a second call on a live
        connection is a no-op success.

        A version mismatch is *reported*, not refused: robotd bumps
        ``API_VERSION`` for additive methods too, and answers a call it cannot
        serve with an error naming the method or parameter, so a gate here would
        refuse a daemon every frame of which this driver reads.
        """
        if self.is_connected:
            return None

        if self._endpoint.kind == "ssh" and self._forward is None:
            forward = _SshForward(
                self._endpoint.user or DEFAULT_SSH_USER,
                self._endpoint.host or "",
                remote_sockets={"robot": DEFAULT_SOCKET, "media": DEFAULT_MEDIA_SOCKET, "tof": DEFAULT_TOF_SOCKET},
                timeout=self._timeout,
                spawn=self._ssh_spawn,
            )
            if (reason := forward.start()) is not None:
                self._connect_error = f"{reason} {_discovery_hint(self._endpoint)}"
                return self._connect_error
            self._forward = forward
            self._socket_path = forward.local["robot"]
            self._media_socket = forward.local["media"]
            self._tof_socket = forward.local["tof"]

        client = _RobotdClient(self._socket_path, timeout=self._timeout)
        try:
            client.connect()
        except OSError as exc:
            self._connect_error = (
                f"robotd socket {self._socket_path!r} did not answer: {exc} {_discovery_hint(self._endpoint)}"
            )
            return self._connect_error

        try:
            hello = client.hello(self._api_version)
        except (OSError, ValueError) as exc:
            client.close()
            self._connect_error = f"robotd Hello failed: {exc}"
            return self._connect_error

        their_version = hello.get("api_version")
        if their_version != self._api_version:
            logger.warning(
                "robotd speaks api_version %s, this driver was checked against %s; "
                "a call whose shape moved will refuse itself by name",
                their_version,
                self._api_version,
            )

        client.start_reader(self._on_state)
        try:
            subscribed = client.call(_M_SUBSCRIBE, {} if self._subscribe_hz is None else {"hz": self._subscribe_hz})
        except OSError as exc:
            client.close()
            self._connect_error = f"robot.subscribe failed: {exc}"
            return self._connect_error

        # v28+: the subscribe result carries the robot's skill names, mode,
        # homed/sitting. Older shapes (``{}``) leave the stock vocabulary.
        if isinstance(subscribed, dict) and subscribed:
            self._absorb_policies(subscribed)

        # Battery is not in robot.state - it rides on robot.health. Best-effort:
        # a robotd that cannot read the bus omits it, and the driver stays up.
        try:
            self._absorb_health(client.call(_M_HEALTH, {}))
        except OSError as exc:
            logger.debug("%s: robot.health unavailable at connect: %s", self._tool_name, exc)

        self._client = client
        self._hello = hello
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability, robotd version and the latest battery read."""
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "socket": self._socket_path,
                        "api_version": (self._hello or {}).get("api_version"),
                        "daemon_version": (self._hello or {}).get("daemon_version"),
                        "motion_stopped": self._stopped,
                        "battery_pct": (self._battery or {}).get("pct"),
                        "endpoint": {
                            "kind": self._endpoint.kind,
                            "source": self._endpoint.source,
                            "host": self._endpoint.host,
                            "user": self._endpoint.user,
                            "forwarded": self._forward is not None,
                        },
                        "driver_api_version": self._api_version,
                        "mode": (self._policies or {}).get("mode"),
                        "skills": list(self.known_skills),
                        "moving": self._move is not None and self._move.active,
                        "policy": (self._last_state or {}).get("policy"),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Ask robotd to stop the robot (``robot.stop``)."""
        self._cancel_move()
        if self._client is None or not self._client.alive:
            return
        try:
            self._client.call(_M_STOP, {})
            self._stopped = True
        except OSError as exc:
            logger.warning("%s.stop(): robotd refused the stop: %s", self._tool_name, exc)

    def cleanup(self) -> None:
        """Close the robotd connection and any ssh forward. Idempotent."""
        self._cancel_move()
        if self._client is not None:
            self._client.close()
        self._client = None
        self._connected = False
        if self._forward is not None:
            self._forward.close()
            self._forward = None

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Map an action dict onto robotd intents and send them.

        Gates, in order: this driver fronts this robot; it is connected; every
        numeric value is finite; every key names an intent this driver can send;
        the action names at least one of them. Continuous intents (twist/head/pose/mouth) are notifications;
        a ``skill`` is a ``robot.do`` request whose IntentResult is returned.

        Accepted keys: ``vx``/``vy``/``vyaw`` (twist, m/s and rad/s),
        ``neck_pitch``/``head_pitch``/``head_yaw``/``head_roll`` (rad),
        ``z``/``roll``/``pitch``/``active`` (standing pose), ``open`` (mouth
        0..1), ``skill`` (one of :data:`SKILLS`).
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        if not self.is_connected or self._client is None:
            return _refuse("not connected - call connect_eagerly() first")

        for name, value in action.items():
            if name in ("skill", "active"):
                continue
            if (reason := finite_number_error(value, name, "send_action")) is not None:
                return _refuse(reason)

        commands = action_to_wire(action, self.known_skills)
        if isinstance(commands, str):
            return _refuse(f"send_action: {commands}")
        if not commands:
            return _refuse(
                f"send_action: nothing to send - none of {sorted(action)} names a Microduck intent; "
                f"expected any of {sorted(_ACTION_KEYS)}"
            )

        sent: list[dict[str, Any]] = []
        for method, params, is_notification in commands:
            try:
                if is_notification:
                    self._client.notify(method, params)
                    sent.append({"method": method, "params": params})
                else:
                    result = self._client.call(method, params)
                    sent.append({"method": method, "params": params, "result": result})
            except OSError as exc:
                return _refuse(f"send_action: {method} failed: {exc}")
        self._stopped = False
        return {"status": "success", "content": [{"json": {"sent": sent, "robot": self._tool_name}}]}

    # ------------------------------------------------------------------ #
    # Task and policy paths - delegate-only, so these refuse.            #
    # ------------------------------------------------------------------ #

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: robotd runs the policy on-device, there is no joint-stream path."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: robotd runs the walking/skill policy on-device and exposes no per-joint write; "
            "send an intent through send_action (twist vx/vy/vyaw, or skill=...) instead"
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a host-driven rollout, for the same reason as :meth:`start_task`.

        The sim path (``Robot("microduck", mode="sim").run_policy``) runs a
        MicroduckPolicy against MuJoCo; on hardware the *same* ONNX runs inside
        robotd, so there is nothing for a host rollout to stream.
        """
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: on hardware the policy runs inside robotd (the same ONNX proven in sim); "
            "robotd exposes no per-joint write to stream targets to. Use send_action intents, or "
            'mode="sim" for a host-driven MicroduckPolicy rollout'
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report the policy robotd is running, from the last state frame."""
        with self._cache_lock:
            policy = (self._last_state or {}).get("policy")
        return {
            "status": "success",
            "content": [{"json": {"running": policy not in (None, "held"), "policy": policy}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop the robot, the closest thing to halting an on-device policy."""
        self._cancel_move()
        if self._client is None or not self._client.alive:
            return _refuse("stop_task: not connected")
        try:
            self._client.call(_M_STOP, {})
        except OSError as exc:
            return _refuse(f"stop_task: robotd refused the stop: {exc}")
        self._stopped = True
        return {"status": "success", "content": [{"text": "asked robotd to stop the robot (robot.stop)"}]}

    # ------------------------------------------------------------------ #
    # Torque + emergency stop - discrete intents.                        #
    # ------------------------------------------------------------------ #

    def enable_torque(self, on: bool = True) -> dict[str, Any]:
        """Enable or disable the policy/torque via ``robot.enable``."""
        if (reason := boolean_flag_error(on, "on", "enable_torque")) is not None:
            # on=false energising the servos is the failure mode this refuses.
            return _refuse(reason)
        return self._discrete(_M_ENABLE, {"on": on, "toggle": False}, "enable_torque")

    def relax(self) -> dict[str, Any]:
        """Cut joint power via ``robot.relax`` (the robot collapses if unheld)."""
        return self._discrete(_M_RELAX, {}, "relax")

    def emergency_stop(self) -> dict[str, Any]:
        """Stop the robot immediately via ``robot.stop``, and record the halt.

        The same request :meth:`stop` and :meth:`stop_task` send, so which of
        the three a caller reached for cannot change what :meth:`get_status`
        publishes under ``motion_stopped`` - the field an operator reads to
        decide whether the robot is safe to approach. Only an accepted stop
        records it: a refusal returns before the latch, because reporting a
        halt robotd declined is the affirmative lie the flag exists to avoid.

        Torque state is a different fact: :meth:`relax` and
        :meth:`enable_torque` de-energise rather than halt a commanded motion,
        and deliberately leave the flag alone.

        Returns:
            A success envelope naming the method and robotd's result, or a
            refusal under this verb's own label.
        """
        self._cancel_move()
        envelope = self._discrete(_M_STOP, {}, "emergency_stop")
        if envelope["status"] == "success":
            self._stopped = True
        return envelope

    def _discrete(self, method: str, params: dict[str, Any], label: str) -> dict[str, Any]:
        if self._client is None or not self._client.alive:
            return _refuse(f"{label}: not connected")
        try:
            result = self._client.call(method, params)
        except OSError as exc:
            return _refuse(f"{label}: {method} failed: {exc}")
        return {"status": "success", "content": [{"json": {"method": method, "result": result}}]}

    # ------------------------------------------------------------------ #
    # Mesh telemetry.                                                    #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, float]:
        """The 14 locomotion joints (name -> radians) for the mesh joint read."""
        with self._cache_lock:
            return dict(self._joints)

    def read_state(self) -> dict[str, Any]:
        """The last robot.state frame the subscribe stream delivered, or a refusal."""
        with self._cache_lock:
            state = self._last_state
        if state is None:
            return _refuse("read_state: no robot.state received yet (not connected, or no frames)")
        return {"status": "success", "content": [{"json": state}]}

    # ------------------------------------------------------------------ #
    # Reader callback + snapshots. Run on the reader thread; keep fast.  #
    # ------------------------------------------------------------------ #

    def _on_state(self, params: dict[str, Any]) -> None:
        state = parse_robot_state(params)
        safety = state.get("safety") or {}
        odom = state.get("odom") or {}
        with self._cache_lock:
            self._last_state = state
            self._joints = dict(state.get("joints") or {})
            self._imu = {"projected_gravity": safety.get("gravity")}
            self._pose = {"position": odom.get("position"), "yaw": odom.get("yaw")}

    def _absorb_health(self, health: dict[str, Any]) -> None:
        battery = (health or {}).get("battery")
        with self._cache_lock:
            self._health = dict(health) if isinstance(health, dict) else None
            self._health_at = time.monotonic()
            if isinstance(battery, dict):
                self._battery = {"pct": battery.get("percent"), "volts": battery.get("volts")}

    def _absorb_policies(self, policies: dict[str, Any]) -> None:
        """Cache a ``robot.policies``/``robot.subscribe`` result: skills, mode, homed, sitting."""
        with self._cache_lock:
            self._policies = dict(policies)
            skills = policies.get("skills")
            if isinstance(skills, list) and skills:
                self._skills = [str(name) for name in skills]

    @property
    def known_skills(self) -> tuple[str, ...]:
        """The skills this robot answers to: what it told us, else the stock five."""
        with self._cache_lock:
            return tuple(self._skills) if self._skills else SKILLS

    def _cancel_move(self) -> None:
        move, self._move = self._move, None
        if move is not None:
            move.cancel()

    def _snapshot(self, attr: str) -> Any:
        with self._cache_lock:
            value = getattr(self, attr)
            if isinstance(value, dict):
                return dict(value)
            return value


# --------------------------------------------------------------------------- #
# Verb handlers. One per agent-facing action; each maps onto exactly one      #
# robotd/mediad/tofd path and returns an envelope. They live at module level   #
# so the dispatch table below is the single source of the schema enum.         #
# --------------------------------------------------------------------------- #


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "content": [{"json": payload}]}


def _intent(driver: MicroduckDriver, method: str, params: dict[str, Any], label: str) -> dict[str, Any]:
    """A request whose result is an ``IntentResult``: report robotd's verdict, not ours.

    ``accepted: false`` is a refusal carrying robotd's ``reason`` (``"limp"``,
    ``"fallen"``, ``"not homed"``, ``"unknown skill"``) - the robot did not do
    it, so the envelope must not say success.
    """
    envelope = driver._discrete(method, params, label)
    if envelope["status"] != "success":
        return envelope
    result = envelope["content"][0]["json"].get("result")
    if isinstance(result, dict) and not _accepted(result):
        return _refuse(f"{label}: robotd declined ({method}): {result.get('reason') or 'no reason given'}")
    driver._stopped = False
    return _ok(
        {"method": method, "params": params, "result": result, "note": "accepted by robotd; not proof of motion"}
    )


def _act_sensors(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    """The four caches every driver reports; the fuller frame is ``monitor``."""
    del params
    return _ok(
        {
            "joints": driver._snapshot("_joints"),
            "pose": driver._snapshot("_pose"),
            "imu": driver._snapshot("_imu"),
            "battery": driver._snapshot("_battery"),
        }
    )


def _act_status(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return asyncio.run(driver.get_status())


def _act_stop(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver.stop_task()


def _act_move(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, float] = {}
    for name in ("vx", "vy", "vyaw"):
        value = _number(params, name, "move", default=0.0)
        if isinstance(value, str):
            return _refuse(value)
        values[name] = value
    if not any(values.values()):
        return _refuse("move: vx, vy and vyaw are all zero - to halt, use `stop`")
    duration = _number(params, "duration", "move", default=MOVE_DURATION_DEFAULT)
    if isinstance(duration, str):
        return _refuse(duration)
    if not MOVE_DURATION_MIN <= duration <= MOVE_DURATION_MAX:
        return _refuse(f"move: duration={duration!r}s is outside [{MOVE_DURATION_MIN}, {MOVE_DURATION_MAX}]")
    mode = (driver._policies or {}).get("mode")
    if (reason := twist_error(values["vx"], values["vy"], values["vyaw"], mode)) is not None:
        return _refuse(reason)
    if driver._client is None or not driver._client.alive:
        return _refuse("move: not connected")
    with driver._cache_lock:
        state = dict(driver._last_state or {})
    if state.get("policy") not in (None, "walk", "roller") and state.get("policy") != mode:
        # A skill is playing (kick, roulade...): robotd's blend ignores the twist.
        return _refuse(f"move: the robot is running {state.get('policy')!r}; a twist is ignored until it finishes")
    driver._cancel_move()
    stream = _MoveStream(driver._client, values, duration)
    stream.start()
    driver._move = stream
    driver._stopped = False
    return _ok(
        {
            "method": _M_MOVE,
            "twist": values,
            "duration_s": duration,
            "refresh_hz": MOVE_REFRESH_HZ,
            "deadman_s": DEADMAN_S,
            "mode": mode,
            "policy_at_send": state.get("policy"),
            "note": "streaming the twist; a zero twist is sent at the end. `stop` halts early. Read `monitor` for applied/limited_by.",
        }
    )


def _act_head(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    wire: dict[str, float] = {}
    for name in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
        if name in params:
            value = _number(params, name, "head")
            if isinstance(value, str):
                return _refuse(value)
            wire[name] = value
    if not wire:
        return _refuse("head: give at least one of neck_pitch, head_pitch, head_yaw, head_roll (radians)")
    if driver._client is None or not driver._client.alive:
        return _refuse("head: not connected")
    try:
        driver._client.notify(_M_HEAD, wire)
    except OSError as exc:
        return _refuse(f"head: {_M_HEAD} failed: {refusal_str(exc)}")
    return _ok(
        {
            "method": _M_HEAD,
            "params": wire,
            "note": "notification - robotd holds this head pose (no deadman); no acknowledgement exists",
        }
    )


def _act_pose(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    active = params.get("active", True)
    if (reason := boolean_flag_error(active, "active", "pose")) is not None:
        return _refuse(reason)
    wire: dict[str, Any] = {"active": active}
    z = _number(params, "z", "pose", default=0.0)
    roll = _number(params, "roll", "pose", default=0.0)
    pitch = _number(params, "pitch", "pose", default=0.0)
    for value in (z, roll, pitch):
        if isinstance(value, str):
            return _refuse(value)
    assert not isinstance(z, str) and not isinstance(roll, str) and not isinstance(pitch, str)
    for reason in (
        _within(z, POSE_Z_MIN, POSE_Z_MAX, "z", "m", "pose"),
        _within(roll, -POSE_TILT_MAX, POSE_TILT_MAX, "roll", "rad", "pose"),
        _within(pitch, -POSE_TILT_MAX, POSE_TILT_MAX, "pitch", "rad", "pose"),
    ):
        if reason is not None:
            return _refuse(reason)
    wire.update({"z": z, "roll": roll, "pitch": pitch})
    if driver._client is None or not driver._client.alive:
        return _refuse("pose: not connected")
    try:
        driver._client.notify(_M_POSE, wire)
    except OSError as exc:
        return _refuse(f"pose: {_M_POSE} failed: {refusal_str(exc)}")
    return _ok(
        {
            "method": _M_POSE,
            "params": wire,
            "note": "notification; the robot clamps nothing here, so the trained range was enforced by the driver",
        }
    )


def _act_mouth(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    value = _number(params, "open", "mouth")
    if isinstance(value, str):
        return _refuse(value)
    if not 0.0 <= value <= 1.0:
        return _refuse(f"mouth: open={value!r} must be within 0..1")
    if driver._client is None or not driver._client.alive:
        return _refuse("mouth: not connected")
    try:
        driver._client.notify(_M_MOUTH, {"open": value})
    except OSError as exc:
        return _refuse(f"mouth: {_M_MOUTH} failed: {refusal_str(exc)}")
    return _ok({"method": _M_MOUTH, "params": {"open": value}})


def _act_do(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    raw = params.get("skill")
    if not isinstance(raw, str) or not raw.strip():
        return _refuse(f"do: skill (one of {list(driver.known_skills)}) is required")
    skill = raw.strip().lower()
    if skill not in driver.known_skills:
        # The robot's list may have changed since connect; ask once before refusing.
        _refresh_policies(driver)
        if skill not in driver.known_skills:
            return _refuse(f"do: unknown skill {raw!r}; this robot lists {list(driver.known_skills)}")
    driver._cancel_move()
    return _intent(driver, _M_DO, {"skill": skill}, "do")


def _refresh_policies(driver: MicroduckDriver) -> dict[str, Any] | str:
    if driver._client is None or not driver._client.alive:
        return "not connected"
    try:
        result = driver._client.call(_M_POLICIES, {})
    except OSError as exc:
        return refusal_str(exc)
    if isinstance(result, dict):
        driver._absorb_policies(result)
        return result
    return f"robot.policies answered {refusal_repr(result)}"


def _act_skills(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    result = _refresh_policies(driver)
    if isinstance(result, str):
        return _refuse(f"skills: {result}")
    return _ok(
        {
            "skills": list(driver.known_skills),
            "mode": result.get("mode"),
            "homed": result.get("homed"),
            "sitting": result.get("sitting"),
        }
    )


def _act_policies(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    result = _refresh_policies(driver)
    if isinstance(result, str):
        return _refuse(f"policies: {result}")
    return _ok(result)


def _sit_or_stand(driver: MicroduckDriver, want_sitting: bool, label: str) -> dict[str, Any]:
    result = _refresh_policies(driver)
    if isinstance(result, str):
        return _refuse(f"{label}: {result}")
    sitting = result.get("sitting")
    if sitting is want_sitting:
        return _ok(
            {
                "skill": "sit_toggle",
                "sent": False,
                "sitting": sitting,
                "note": f"already {label[:-1] if label.endswith('e') else label}ing - nothing sent",
            }
        )
    if "sit_toggle" not in driver.known_skills:
        return _refuse(f"{label}: this robot lists no sit_toggle skill ({list(driver.known_skills)})")
    driver._cancel_move()
    envelope = _intent(driver, _M_DO, {"skill": "sit_toggle"}, label)
    if envelope["status"] == "success":
        payload = envelope["content"][0]["json"]
        payload["sitting_before"] = sitting
        payload["note"] = (
            "sit_toggle accepted"
            + (
                ""
                if sitting is not None
                else " (this robotd does not report `sitting`, so the direction is unverified)"
            )
            + "; the transition takes a few seconds - read `policies` for `sitting`"
        )
    return envelope


def _act_sit(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return _sit_or_stand(driver, True, "sit")


def _act_stand(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return _sit_or_stand(driver, False, "stand")


def _act_look_at(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    wire: dict[str, float] = {}
    for name in ("x", "y", "z"):
        value = _number(params, name, "look_at")
        if isinstance(value, str):
            return _refuse(value)
        wire[name] = value
    if wire["x"] == 0.0 and wire["y"] == 0.0:
        return _refuse("look_at: a target straight above or below the head (x=y=0) has no yaw; give an x or y")
    # ``LookParams.neck_pitch`` is a plain f64, not optional: the IK holds the
    # neck at this posture and aims the head around it (robotctl's default 0).
    neck = _number(params, "neck_pitch", "look_at", default=0.0)
    if isinstance(neck, str):
        return _refuse(neck)
    wire["neck_pitch"] = neck
    envelope = driver._discrete(_M_LOOK, wire, "look_at")
    if envelope["status"] != "success":
        return envelope
    result = envelope["content"][0]["json"].get("result") or {}
    driver._stopped = False
    return _ok(
        {
            "method": _M_LOOK,
            "target": wire,
            "head": result.get("head"),
            "clamped": result.get("clamped"),
            "note": (
                "robotd solved and holds the head pose (trunk frame: the floor is ~0.12 m below z=0); "
                "`clamped` true means the target is outside the head's reach"
            ),
        }
    )


def _act_enable(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    on = params.get("on", True)
    if (reason := boolean_flag_error(on, "on", "enable")) is not None:
        return _refuse(reason)
    if not on:
        driver._cancel_move()
    return _intent(driver, _M_ENABLE, {"on": on, "toggle": False}, "enable")


def _act_disable(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    driver._cancel_move()
    return _intent(driver, _M_ENABLE, {"on": False, "toggle": False}, "disable")


def _confirmed(params: dict[str, Any], label: str, consequence: str) -> str | None:
    confirm = params.get("confirm", False)
    if (reason := boolean_flag_error(confirm, "confirm", label)) is not None:
        return reason
    if not confirm:
        return f"{label}: {consequence} - pass confirm=true (and hold the robot) to do it"
    return None


def _act_relax(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    if (reason := _confirmed(params, "relax", "cuts joint power; an unheld duck collapses")) is not None:
        return _refuse(reason)
    driver._cancel_move()
    return _intent(driver, _M_RELAX, {}, "relax")


def _act_init(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    if (
        reason := _confirmed(params, "init", "re-homes every servo to the init pose (limbs move, no obstacle check)")
    ) is not None:
        return _refuse(reason)
    driver._cancel_move()
    return _intent(driver, _M_INIT, {}, "init")


def _act_reboot_motors(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    if (reason := _confirmed(params, "reboot_motors", "power-cycles servos; the robot goes limp")) is not None:
        return _refuse(reason)
    ids = params.get("ids", [])
    if not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        return _refuse(
            f"reboot_motors: ids must be a list of servo ids (non-negative integers), got {refusal_repr(ids)}"
        )
    driver._cancel_move()
    return _intent(driver, _M_REBOOT_MOTORS, {"ids": ids}, "reboot_motors")


def _act_sounds(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del driver, params
    return _ok(
        {
            "tags": list(PLAYABLE_SOUND_TAGS),
            "held": ["wheee"],
            "note": "wheee is a pad-trigger ride the driver cannot hold; the rest play once",
        }
    )


def _act_play_sound(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    tag = params.get("tag")
    if not isinstance(tag, str) or tag.strip().lower() not in SOUND_TAGS:
        return _refuse(f"play_sound: tag must be one of {list(PLAYABLE_SOUND_TAGS)}, got {refusal_repr(tag)}")
    tag = tag.strip().lower()
    if tag == "wheee":
        return _refuse(
            "play_sound: 'wheee' is a held ride driven per tick by the pad trigger; this driver cannot keep that hold honestly"
        )
    return _intent(driver, _M_SOUND, {"tag": tag}, "play_sound")


def _act_theremin(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    active = params.get("active", True)
    if (reason := boolean_flag_error(active, "active", "theremin")) is not None:
        return _refuse(reason)
    return _intent(driver, _M_THEREMIN, {"active": active}, "theremin")


def _act_mode(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    envelope = driver._discrete(_M_MODE, {}, "mode")
    if envelope["status"] == "success":
        result = envelope["content"][0]["json"].get("result")
        if isinstance(result, dict):
            with driver._cache_lock:
                driver._policies = {**(driver._policies or {}), **result}
    return envelope


def _act_set_mode(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    mode = params.get("mode")
    if not isinstance(mode, str) or mode.strip().lower() not in DRIVE_MODES:
        return _refuse(f"set_mode: mode must be one of {list(DRIVE_MODES)}, got {refusal_repr(mode)}")
    driver._cancel_move()
    envelope = _intent(driver, _M_SET_MODE, {"mode": mode.strip().lower()}, "set_mode")
    if envelope["status"] == "success":
        _refresh_policies(driver)
        envelope["content"][0]["json"]["mode_after"] = (driver._policies or {}).get("mode")
    return envelope


def _act_load_policy(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    slot, path = params.get("slot"), params.get("path")
    if not isinstance(slot, str) or not slot.strip():
        return _refuse(f"load_policy: slot is required (known: {list(POLICY_SLOTS)})")
    if not isinstance(path, str) or not path.strip():
        return _refuse("load_policy: path (an ONNX file ON THE ROBOT) is required")
    driver._cancel_move()
    return _intent(driver, _M_LOAD_POLICY, {"slot": slot.strip(), "path": path.strip()}, "load_policy")


def _act_reload_policies(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    driver._cancel_move()
    envelope = _intent(driver, _M_RELOAD_POLICIES, {}, "reload_policies")
    if envelope["status"] == "success":
        _refresh_policies(driver)
    return envelope


def _act_health(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    envelope = driver._discrete(_M_HEALTH, {}, "health")
    if envelope["status"] != "success":
        return envelope
    health = envelope["content"][0]["json"].get("result")
    if not isinstance(health, dict):
        return _refuse(f"health: robot.health answered {refusal_repr(health)}")
    driver._absorb_health(health)
    imu = health.get("imu") or {}
    run = imu.get("consecutive_stale_blocks")
    return _ok({**health, "imu_frozen": isinstance(run, int) and run >= IMU_FROZEN_RUN})


def _act_version(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    hello = dict(driver._hello or {})
    envelope = driver._discrete(_M_MODEL_API, {}, "version")
    model_api = envelope["content"][0]["json"].get("result") if envelope["status"] == "success" else None
    return _ok(
        {
            "daemon_version": hello.get("daemon_version"),
            "api_version": hello.get("api_version"),
            "driver_api_version": driver._api_version,
            "model_api": model_api,
            "pinned_release": "microduck 0.14.1",
        }
    )


def _act_model(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver._discrete(_M_MODEL, {}, "model")


def _act_odometry(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    with driver._cache_lock:
        state = dict(driver._last_state or {})
    if not state:
        return _refuse("odometry: no robot.state received yet")
    odom = state.get("odom") or {}
    return _ok(
        {
            "position": odom.get("position"),
            "yaw": odom.get("yaw"),
            "odom": odom,
            "t": state.get("t"),
            "note": "dead-reckoned from the gait, not a fix",
        }
    )


def _act_monitor(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    """One `robotctl monitor` row: requested vs applied twist, limiter, fall, loop, thermals."""
    del params
    with driver._cache_lock:
        state = dict(driver._last_state or {})
        health = dict(driver._health or {})
        health_age = time.monotonic() - driver._health_at if driver._health else None
    if health_age is None or health_age > 2.0:
        fresh = driver._discrete(_M_HEALTH, {}, "monitor")
        if fresh["status"] == "success" and isinstance(fresh["content"][0]["json"].get("result"), dict):
            health = fresh["content"][0]["json"]["result"]
            driver._absorb_health(health)
    move = state.get("move") or {}
    safety = state.get("safety") or {}
    motors = health.get("motors") or {}
    return _ok(
        {
            "policy": state.get("policy"),
            "mode": (driver._policies or {}).get("mode"),
            "requested": move.get("requested"),
            "applied": move.get("applied"),
            "limited_by": move.get("limited_by"),
            "fallen": safety.get("fallen"),
            "limp": safety.get("limp"),
            "gain": safety.get("gain"),
            "projected_gravity": safety.get("gravity"),
            "loop": state.get("loop"),
            "battery": health.get("battery"),
            "hottest_motor": motors.get("hottest"),
            "imu": health.get("imu"),
            "moving": driver._move is not None and driver._move.active,
            "motion_stopped": driver._stopped,
            "head": state.get("head"),
            "imu_raw": state.get("imu"),
            "theremin": state.get("theremin"),
            "frames": state.get("frames"),
            "t": state.get("t"),
        }
    )


def _act_camera(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    """One raw frame from mediad (``media.frame``), decoded to a JPEG on disk.

    ``save_path`` names a file inside :func:`frame_root`; a value that leaves it
    - absolute, or through ``..`` - is refused by name before mediad is dialed,
    because the frame is written with this process's privileges. The guards are
    the package's own path sandbox
    (:mod:`strands_robots._path_validation`), so this write is confined on the
    same terms as every other caller-named artifact.
    """
    save_path = params.get("save_path")
    if save_path is not None and (not isinstance(save_path, str) or not save_path.strip()):
        return _refuse(f"camera: save_path must be a file name, got {refusal_repr(save_path)}")
    root = frame_root()
    target: str | None = None
    if isinstance(save_path, str):
        try:
            target = resolve_output_path(validate_save_path(str(root), label="frame root"), save_path.strip())
        except ValueError as exc:
            return _refuse(
                f"camera: save_path must name a file inside {root} "
                f"({refusal_str(exc)}); set {FRAME_ROOT_ENV} to collect frames elsewhere"
            )
    try:
        (sock, reader), response = _one_shot_call(driver._media_socket, _M_MEDIA_FRAME, {}, driver._timeout)
    except (OSError, ValueError) as exc:
        return _refuse(f"camera: mediad at {driver._media_socket!r} did not answer: {refusal_str(exc)}")
    try:
        header = response.get("result") or {}
        try:
            width, height, size = int(header["width"]), int(header["height"]), int(header["bytes"])
        except (KeyError, TypeError, ValueError):
            return _refuse(f"camera: media.frame header lacks width/height/bytes: {refusal_repr(header)}")
        if size != width * height * 2 or size <= 0:
            return _refuse(
                f"camera: header says {width}x{height} UYVY but {size} bytes (expected {width * height * 2})"
            )
        raw = reader.read(size)
        if len(raw) != size:
            return _refuse(f"camera: mediad sent {len(raw)} of {size} frame bytes")
    finally:
        reader.close()
        sock.close()
    try:
        if target is None:
            root.mkdir(parents=True, exist_ok=True)
            handle, path = tempfile.mkstemp(prefix="microduck-", suffix=".jpg", dir=root)
            os.close(handle)
        else:
            path = target
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        uyvy_to_jpeg(width, height, int(header.get("rotate", 0) or 0), raw, path)
    except (OSError, ValueError) as exc:
        return _refuse(f"camera: could not decode/write the frame: {refusal_str(exc)}")
    return _ok(
        {
            "path": path,
            "width": width,
            "height": height,
            "rotate": header.get("rotate", 0),
            "format": header.get("format", "UYVY"),
            "bytes_raw": size,
            "captured_at_unix_us": header.get("captured_at_unix_us"),
            "note": "one frame; mediad's live stream stays with the phone app (WebRTC)",
        }
    )


def _act_tof(driver: MicroduckDriver, params: dict[str, Any]) -> dict[str, Any]:
    """One depth frame from tofd (subscribe with ``tof.stream``, read one ``tof.frame``)."""
    del params
    try:
        (sock, reader), response = _one_shot_call(driver._tof_socket, _M_TOF_STREAM, {"on": True}, driver._timeout)
    except (OSError, ValueError) as exc:
        return _refuse(f"tof: tofd at {driver._tof_socket!r} did not answer: {refusal_str(exc)}")
    try:
        deadline = time.monotonic() + driver._timeout
        while time.monotonic() < deadline:
            line = reader.readline()
            if not line:
                return _refuse("tof: tofd closed the stream before a frame arrived")
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("method") == _M_TOF_FRAME:
                frame = message.get("params") or {}
                return _ok({**summarise_tof(frame), "subscribed": response.get("result")})
        return _refuse(f"tof: no tof.frame within {driver._timeout}s - is the sensor mounted and tofd running?")
    except OSError as exc:
        return _refuse(f"tof: reading the stream failed: {refusal_str(exc)}")
    finally:
        reader.close()
        sock.close()


#: Every agent-facing verb, in the order the schema lists them. The dispatch
#: in :meth:`MicroduckDriver.stream` reads this table; the enum is its keys.
_ACTIONS: dict[str, Callable[[MicroduckDriver, dict[str, Any]], dict[str, Any]]] = {
    "sensors": _act_sensors,
    "status": _act_status,
    "stop": _act_stop,
    "move": _act_move,
    "head": _act_head,
    "look_at": _act_look_at,
    "pose": _act_pose,
    "mouth": _act_mouth,
    "do": _act_do,
    "skills": _act_skills,
    "sit": _act_sit,
    "stand": _act_stand,
    "enable": _act_enable,
    "disable": _act_disable,
    "relax": _act_relax,
    "init": _act_init,
    "reboot_motors": _act_reboot_motors,
    "sounds": _act_sounds,
    "play_sound": _act_play_sound,
    "theremin": _act_theremin,
    "mode": _act_mode,
    "set_mode": _act_set_mode,
    "policies": _act_policies,
    "load_policy": _act_load_policy,
    "reload_policies": _act_reload_policies,
    "health": _act_health,
    "version": _act_version,
    "model": _act_model,
    "odometry": _act_odometry,
    "monitor": _act_monitor,
    "camera": _act_camera,
    "tof": _act_tof,
}

#: Verbs that answer without a robotd connection (so ``stream`` does not connect for them).
_OFFLINE_ACTIONS: frozenset[str] = frozenset({"status", "sensors", "sounds", "odometry"})

#: Verbs served by mediad/tofd rather than robotd.
_MEDIA_ACTIONS: frozenset[str] = frozenset({"camera", "tof"})

_ACTION_HELP: dict[str, str] = {
    "sensors": "cached joints/pose/imu/battery/policy/move from the state stream",
    "status": "reachability, endpoint, versions, mode, skills (never connects)",
    "stop": "robot.stop - halt and hold the current policy",
    "move": "walk/roll with vx,vy,vyaw for duration s (bounded by the pad envelope)",
    "head": "hold a head pose: neck_pitch/head_pitch/head_yaw/head_roll rad",
    "look_at": "turn the head toward a point x,y,z m in the robot frame (robotd solves it)",
    "pose": "standing offsets z/roll/pitch with active=true|false",
    "mouth": "open 0..1",
    "do": "run a named skill (see skills)",
    "skills": "the skills this robot lists, plus mode/homed/sitting",
    "sit": "sit_toggle only if standing",
    "stand": "sit_toggle only if sitting",
    "enable": "on=true energise + start policy, on=false disable",
    "disable": "disable the policy (robot holds, no torque cut)",
    "relax": "cut joint power (confirm=true)",
    "init": "re-home servos to the init pose (confirm=true)",
    "reboot_motors": "power-cycle servos by ids (confirm=true)",
    "sounds": "list voice-bank tags",
    "play_sound": "play one tag",
    "theremin": "active=true|false head-driven synth",
    "mode": "current drive mode (walk/roller)",
    "set_mode": "switch drive mode",
    "policies": "loaded policy slots, skills, mode, homed, sitting",
    "load_policy": "load an ONNX (on the robot) into a slot",
    "reload_policies": "reload every slot from the robot's config",
    "health": "battery, motor temperatures, IMU freshness (robot.health)",
    "version": "daemon + API + model API versions",
    "model": "the robot's model/config as robotd reports it",
    "odometry": "dead-reckoned position/yaw from the state stream",
    "monitor": "one robotctl-monitor row: requested vs applied twist, limiter, fallen, loop, thermals",
    "camera": "one head-camera frame as a JPEG file (mediad)",
    "tof": "one depth frame summary (tofd)",
}
