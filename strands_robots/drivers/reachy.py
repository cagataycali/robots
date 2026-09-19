"""Native daemon driver for the Pollen Robotics Reachy Mini.

``Robot("reachy_mini", mode="real", port="reachy-a.local:8000")`` builds one of
these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver`, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like the lerobot driver they replace for
this robot. There is nothing to replace in practice: the Reachy Mini has no
lerobot robot type, so before this driver ``mode="real"`` raised
``ValueError: Unsupported robot type: 'reachy_mini'``.

Why a native driver rather than a lerobot entry: the Mini is an expressive desk
robot - a 6-DOF Stewart head on a rotating body, two antennas, a speaker and a
recorded-emotion library - with no arms and no gait. lerobot's robot classes
model a serial servo bus or a network arm; the Mini is addressed through the
Reachy daemon's REST API plus a real-time link, and its state of interest is
head orientation rather than a joint-space arm pose.

What the driver actually does:

* Probes ``GET /api/daemon/status`` in :meth:`~ReachyDriver.connect_eagerly` -
  the reachability check, and the same call that reports which hardware variant
  answered. The daemon WebSocket serves both **Lite** and **Wireless** hardware
  (verified on daemon 1.10.0). An explicitly supplied Wireless bridge transport
  retains the Zenoh path. Both links come from
  :mod:`strands_robots.device_connect.reachy_transport`, which the Device
  Connect driver already ships - this module reuses them rather than growing a
  second daemon client.
* Runs that link on one background asyncio loop and caches what it delivers.
  ``_imu`` is the head IMU verbatim; ``_pose`` is the head orientation, taken
  from the IMU quaternion because the Mini's IMU is *in the head*; ``_battery``
  is read from the daemon status payload when it carries one. The mesh reads all
  three with ``getattr(robot, name, None)``, so a Mini that has not connected
  publishes no sensor topic and is otherwise complete.
* Refuses a motion write outside the envelope, naming the limit, via the shared
  :func:`~strands_robots.drivers.reachy_envelope.envelope_error`.

Deliberately absent, so a reader is not left guessing:

* **No** ``_lidar_*``. The Mini has no lidar.
* No forward kinematics of the Stewart platform. The link reports six leg
  positions; turning those into a head pose needs a model of the platform this
  repo does not have, so the legs are cached as legs (``_joints``) and the head
  *orientation* comes from the IMU rather than being derived from them.
  Inventing the kinematics would put a number on the mesh that nothing
  measured.
* ``start_task`` and ``run_policy`` refuse outright rather than standing in for
  work in progress. A recorded emotion is not a policy rollout: with no arms and
  no gait there is no action space for a policy to be trained against, so the
  refusal names the recorded-move path instead of implying a rollout is coming.

Nothing here imports a transport at module load: every daemon touch is inside a
method body, so the module imports on Thor, on CI and in every unit test with a
mocked daemon.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import re
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands.tools.tools import AgentTool

from strands_robots.drivers.base import undeclared_verb_error
from strands_robots.drivers.reachy_doa import DoaLoop, DoaTurner
from strands_robots.drivers.reachy_envelope import envelope_error
from strands_robots.drivers.reachy_vocabulary import (
    DEFAULT_TTS_PORT,
    ENV_HOST,
    ENV_PORT,
    ENV_TTS_URL,
    daemon_answers,
    discovery_candidates,
    goto_body,
    goto_body_error,
    resolve_move_name,
    resolve_volume_level,
)
from strands_robots.utils import finite_number_error, tcp_port_error

if TYPE_CHECKING:
    from concurrent.futures import Future

    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The daemon's own default port. ``port="host"`` with no ``:port`` suffix means
#: this one, which is what every Reachy install uses unless it was reconfigured.
DEFAULT_API_PORT: int = 8000

#: REST paths this driver calls. Grouped so a reader sees the whole daemon
#: surface the driver depends on in one place, and so a test can assert against
#: the same constants the driver sends.
_PATH_STATUS = "/api/daemon/status"
_PATH_STOP = "/api/move/stop"
_PATH_MOVES_RUNNING = "/api/move/running"
_PATH_SET_TARGET = "/api/move/set_target"
_PATH_WAKE = "/api/move/play/wake_up"
_PATH_SLEEP = "/api/move/play/goto_sleep"
_PATH_MOVE_PLAY = "/api/move/play/recorded-move-dataset/{dataset}/{move}"
_PATH_MOVE_LIST = "/api/move/recorded-move-datasets/list/{dataset}"
_PATH_GOTO = "/api/move/goto"
_PATH_MOTORS_MODE = "/api/motors/set_mode/{mode}"
_PATH_VOLUME_GET = "/api/volume/current"
_PATH_VOLUME_SET = "/api/volume/set"
_PATH_PLAY_SOUND = "/api/media/play_sound"
_PATH_WOBBLE_ENABLE = "/api/media/wobbling/enable"
_PATH_WOBBLE_DISABLE = "/api/media/wobbling/disable"
_PATH_TRACKING_ENABLE = "/api/media/tracking/enable"
_PATH_TRACKING_DISABLE = "/api/media/tracking/disable"
_PATH_TRACKING_FACE = "/api/media/tracking/face"
_PATH_STATE = "/api/state/full"
_PATH_STATE_DOA = "/api/state/full?with_doa=true&with_head_pose=true&with_body_yaw=true"
#: A tracked face older than this no longer vetoes a DoA turn.
_FACE_FRESH_S = 1.5

#: The three torque modes the daemon's ``/api/motors/set_mode/{mode}`` accepts.
#: ``enabled`` and ``disabled`` also have a real-time link command (``torque``),
#: which :meth:`ReachyDriver.set_motors` keeps using for them; the third has
#: only this REST path.
_MOTOR_MODES: tuple[str, ...] = ("enabled", "disabled", "gravity_compensation")

#: A recorded move's name goes into a URL path, so the admitted alphabet is the
#: same one :mod:`strands_robots.device_connect.reachy_mini_driver` enforces -
#: anything else is refused before a request is built from it.
#:
#: The leading character is alphanumeric, which is what makes this a *bare path
#: segment* rather than only a charset: ``.`` and ``..`` are spelled entirely
#: from the admitted alphabet, so a charset alone admits the two tokens a URL
#: path resolves relative to its parent. ``move_name=".."`` builds
#: ``/api/move/play/recorded-move-dataset/<owner>/<library>/..``, which resolves
#: to ``/api/move/play/recorded-move-dataset/<owner>`` - a daemon endpoint the
#: caller did not name. Same shape and same reason as the bare-path-segment
#: gate :mod:`strands_robots.drivers.feetech.bus` applies to the two names it
#: interpolates into a calibration file path.
_MOVE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

#: The two recorded-move libraries the daemon serves, mapped to their
#: HuggingFace dataset ids. A dict rather than string surgery so a refusal can
#: name the admitted set.
_MOVE_LIBRARIES: dict[str, str] = {
    "emotions": "pollen-robotics/reachy-mini-emotions-library",
    "dances": "pollen-robotics/reachy-mini-dances-library",
}

#: The module every daemon touch here goes through. It is a leaf that imports
#: nothing but the standard library, and its parent package
#: :mod:`strands_robots.device_connect` resolves ``device_connect_edge`` and the
#: three Device Connect drivers lazily, so importing the leaf executes no
#: third-party import. Nothing an extra installs can therefore decide whether
#: this import succeeds: on a stock ``pip install strands-robots`` it does, and a
#: failure that still reaches :func:`_resolve_transport` is a broken install of a
#: module the core distribution ships rather than a missing optional dependency.
_TRANSPORT_MODULE = "strands_robots.device_connect.reachy_transport"

#: How long :meth:`ReachyDriver._start_link` waits for a link's handshake before
#: giving up on it. Read back off the module rather than inlined so a caller that
#: needs a different budget, and the tests that exercise the give-up path, can set
#: it - the same shape as ``device_connect``'s ``_INIT_TIMEOUT_S``.
_LINK_START_TIMEOUT_S: float = 10.0

#: How long :meth:`ReachyDriver._stop_loop` waits for the thread running the
#: link's loop to return from ``run_forever`` before giving up on closing that
#: loop. A budget rather than an unbounded wait because the thread is only as
#: free to return as the callbacks on it: a link callback wedged on a socket read
#: would otherwise hold teardown open for as long as the read takes. Matches
#: ``_CAMS_REC_JOIN_TIMEOUT_S`` and ``_TELEOP_JOIN_TIMEOUT_S`` in purpose, and is
#: read off the module for the same reason as the budget above.
_LOOP_JOIN_TIMEOUT_S: float = 5.0


def _resolve_transport() -> Any:
    """Return the Reachy transport module, or a reason naming what failed.

    The same shape as
    :func:`~strands_robots.drivers.g1._resolve_message_class`: the seam's other
    driver resolves its lazy SDK import through a helper that hands back a
    reason string, and every refusal boundary turns that string into a named
    failure. Doing the same here keeps the driver's no-raise contract intact
    when the transport module cannot be imported - ``connect_eagerly`` reports a
    reason and leaves the driver disconnected but usable, rather than raising
    ``ModuleNotFoundError`` through the agent tool surface.

    The reason reports the module and the underlying ``ImportError`` and stops
    there. It prescribes no install remedy because there is none this branch
    could establish: the transport leaf imports nothing outside the standard
    library, so no ``pip install`` supplies a module whose absence would reach
    here. That is the position the shared optional-dependency helper
    :func:`~strands_robots.utils.require_optional` already refuses to print a
    pip line in, because such a line "would hand the caller an instruction that
    reports success without supplying the module". Every cause that can still
    reach this branch - a shadowing module, a partial wheel, a corrupt install -
    is described by the ``ImportError`` itself.

    Returns:
        The imported module, or a reason string naming the module and the cause.
    """
    try:
        import importlib

        return importlib.import_module(_TRANSPORT_MODULE)
    except ImportError as exc:
        return f"cannot import {_TRANSPORT_MODULE}: {exc}"


#: Keys the daemon status payload might carry a battery percentage under. The
#: daemon documents status as "daemon status, motor state, and control
#: frequency" and this repo cannot confirm a battery field without hardware, so
#: the read is defensive: a payload that carries one populates ``_battery``, a
#: payload that does not leaves it ``None`` and the mesh publishes no battery
#: topic. Guessing a dedicated ``/api/battery`` endpoint would be a request no
#: measurement supports.
_BATTERY_KEYS: tuple[str, ...] = ("battery_level", "battery_pct", "battery", "soc")

#: How much of an unreadable daemon body a refusal quotes. Enough to recognise a
#: proxy's error page or a bare scalar, short enough that a large array does not
#: fill the log line the refusal ends up on.
_BODY_PREVIEW_CHARS = 60


class ReachyDriver(AgentTool):
    """Native driver for the Pollen Reachy Mini.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally
    - a Protocol - so no import from :mod:`strands_robots.drivers.base` is
    needed. The surface check that
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    time is what pins the contract for this class.
    """

    def __init__(
        self,
        tool_name: str = "reachy_mini",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        api_port: int = DEFAULT_API_PORT,
        media_port: int = 8443,
        zenoh_prefix: str | None = None,
        transport: Any = None,
        tts_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` talks to the daemon.

        The three leading arguments are the ones every native driver takes - see
        :mod:`strands_robots.drivers.base`'s constructor contract - so the
        factory can build any driver the same way.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer
                id when the driver is wrapped by
                :class:`~strands_robots.mesh.Mesh`. Two Minis on one mesh differ
                by this and nothing else, which is what makes a two-instance
                bring-up work without either knowing about the other.
            cameras: Accepted for parity with the lerobot driver. The Mini's
                camera is reached through the daemon rather than v4l2, so this
                driver does not open it.
            data_config: Accepted for parity; unused.
            port: The daemon host, optionally with a port -
                ``"reachy-a.local"`` or ``"reachy-a.local:8000"``. ``port`` is
                polymorphic across drivers by contract; here it names a host,
                because that is what addresses a Mini. ``None`` reads
                ``REACHY_HOST``/``REACHY_PORT`` from the environment, and when
                those are unset too :meth:`connect_eagerly` discovers the daemon:
                ``localhost`` (a Lite, or a process on the robot) and then
                ``reachy-mini.local`` (a Wireless's factory mDNS name), taking
                the first that answers ``/api/daemon/status`` without an error.
                Until then ``_host`` reads ``localhost``.
            api_port: Daemon port to use when ``port`` carries no ``:port``
                suffix. An explicit suffix in ``port`` wins.
            media_port: Native GStreamer LAN signaling port used for camera capture.
            zenoh_prefix: Zenoh key prefix for a Wireless Mini. Defaults to
                ``tool_name``, so two Minis do not share a key space.
            transport: Zenoh transport for a Wireless Mini, passed through to
                :class:`~strands_robots.device_connect.reachy_transport.ZenohLink`.
                ``None`` selects the daemon WebSocket on either hardware variant;
                supplying a transport keeps the Wireless Zenoh bridge path.
            tts_url: Base URL of a Piper/``tiny-tts`` speech service for
                :meth:`say`. ``None`` reads ``REACHY_TTS_URL`` and otherwise
                combines the daemon host with port 5002 at call time (where
                ``tiny-the-reachy`` installs its ``tiny-tts.service``). Speech
                synthesis is not part of the Reachy daemon.
            **kwargs: Ignored; accepted so the factory can forward extras
                without the driver knowing what they are.

        Raises:
            ValueError: If ``api_port`` or a ``:port`` suffix in ``port`` is not
                a usable TCP port. A bad port cannot address a daemon, and
                refusing here means the mistake surfaces at construction rather
                than as an unreachable host minutes later.
        """
        super().__init__()
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("ReachyDriver ignoring extra kwargs: %s", sorted(kwargs))

        self._tool_name = tool_name
        # ``port=None`` with ``REACHY_HOST`` set reads as if the caller had passed
        # that host, so an operator who exported it for the Pollen SDK or for
        # tiny-the-reachy has configured this driver too. ``REACHY_PORT`` is the
        # matching port when the host carries no ``:port`` suffix; a value that
        # is not a usable port is refused here by the same domain as ``api_port``.
        if port is None and (env_host := (os.getenv(ENV_HOST) or "").strip()):
            port = env_host
            if env_port := (os.getenv(ENV_PORT) or "").strip():
                if reason := tcp_port_error(
                    int(env_port) if env_port.isdigit() else env_port, ENV_PORT, "ReachyDriver"
                ):
                    raise ValueError(reason)
                api_port = int(env_port)
        #: Whether :meth:`connect_eagerly` may look for the daemon rather than
        #: dial one address. Only a bring-up that named no host at all (no
        #: ``port=``, no ``REACHY_HOST``) discovers; ``_host`` holds the first
        #: candidate meanwhile so a status read before connecting names it.
        self._discover_host: bool = port is None
        self._host, self._api_port = _split_host_port(port, api_port)
        self._tts_url: str | None = tts_url
        self._zenoh_prefix = zenoh_prefix or tool_name
        self._transport = transport
        if reason := tcp_port_error(media_port, "media_port", "ReachyDriver"):
            raise ValueError(reason)
        self._media_port = int(media_port)

        # Sensor caches. Every one is optional per the mesh contract, so a
        # driver that has not connected is not broken. Written by link
        # callbacks on the loop thread; read by mesh loops on theirs.
        self._cache_lock = threading.Lock()
        self._imu: dict[str, Any] | None = None
        self._pose: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        #: ``(daemon face stamp, this host's monotonic time when first seen)``.
        #: The daemon stamps on its own clock, so only the local age is usable.
        self._face_stamp: tuple[float, float] | None = None
        self._joints: dict[str, Any] | None = None
        self._joints_received_at: float | None = None

        # The head yaw, in degrees, this driver last put on the wire, and so the
        # one the daemon is still targeting. Not a sensor reading: no telemetry
        # carries it, because the head IMU measures the head's orientation and
        # nothing reports body yaw back. ``None`` means unknown - never
        # commanded, or a path that re-pins the daemon's target has run since.
        # Guarded by ``_cache_lock`` with the sensor caches: same kind of
        # cross-thread read, same lock rather than a second one.
        self._head_yaw_target: float | None = None

        # Connection state. ``None`` link on a machine that never connected is
        # a valid state for tests and for a peer built ahead of a bring-up.
        self._link: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._connected: bool = False
        self._connect_error: str | None = None
        self._variant: str | None = None

        # The halt an operator reads through ``motion_stopped``. True only once
        # the daemon has accepted a stop, and False again as soon as this driver
        # commits motion to the wire - a latch that outlived the halt it named
        # would report a stopped robot while the head was moving, which is the
        # same affirmative lie as claiming a halt that never happened, told one
        # step later. Every clearing site is a *successful* write path, so a
        # refused command leaves the halt standing.
        self._stopped: bool = False

        #: The turn-toward-a-voice loop, built by :meth:`turn_to_sound` and
        #: stopped by it, by :meth:`stop` and by :meth:`cleanup`. ``None`` until
        #: first enabled; kept afterwards so its status survives a disable.
        #: Every read-then-write of this slot holds ``_doa_lock``: the tool
        #: surface dispatches handlers on worker threads, so two concurrent
        #: ``turn_to_sound(True)`` calls would otherwise each see no running
        #: loop, each start one, and the second assignment would drop the only
        #: reference to the first - a 10 Hz thread issuing head turns that
        #: ``stop``, ``turn_to_sound(False)`` and ``cleanup`` could no longer
        #: reach. The lock is never held across a daemon call, and the loop
        #: thread never takes it, so holding it across ``DoaLoop.stop`` (which
        #: joins that thread) cannot deadlock.
        self._doa_lock = threading.Lock()
        self._doa: DoaLoop | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface (matches AgentTool's abstract members).         #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"`` - mirrors the lerobot driver."""
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """The Mini's whole vocabulary as one JSON-callable tool.

        Universal verbs (``status``, ``sensors``, ``stop``), native capture
        (``camera``, ``record_audio``), a pixel-to-head turn (``look_at``),
        turning toward a voice (``turn_to_sound``) and
        the expressive verbs a desk robot is actually asked for - ``look``,
        ``antennas``, ``body_turn``, ``home``, ``wake``/``sleep``, ``express``,
        ``say``, ``play_sound``, ``volume``, ``track_face`` - each mapping onto
        exactly one daemon path. The description tells the agent the one thing
        every write shares: the daemon accepting a command is not the head
        having moved or a sound having been heard.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Pollen Reachy Mini native driver (6-DOF head on a rotating body, two antennas, "
                    "speaker, camera, microphone, a recorded-emotion library). Reads IMU/joints/pose, "
                    "captures camera/microphone, and drives smooth interpolated head, body and antenna "
                    "moves, recorded emotions, wake/sleep, speech (needs a TTS sidecar), sound playback, "
                    "speaker volume, daemon face tracking and turning toward a voice (turn_to_sound, "
                    "microphone-array direction of arrival). Angles are degrees, translations millimetres, "
                    "durations seconds. Every write returns the daemon's acceptance: it is NOT proof the "
                    "head reached the pose or that audio was heard - read `sensors` afterwards."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors/get_state: latest cached IMU, head pose, battery and joints (6 head legs, "
                                    "body_yaw, antennas, degrees); "
                                    "status: daemon reachability, hardware variant, connection error; "
                                    "stop: stop every running move (by uuid) and the turn_to_sound loop; "
                                    "look: smooth head pose (pitch/roll/yaw deg, x/y/z mm, optional body_yaw and "
                                    "antenna_right/antenna_left) over duration; "
                                    "antennas: move just the ears (antenna_right/antenna_left deg); "
                                    "body_turn: rotate the body (body_yaw deg, +/-160); "
                                    "home: neutral pose, antennas level, body centred; "
                                    "wake / sleep: the daemon's built-in wake-up / go-to-sleep choreography; "
                                    "express: play a recorded emotion or dance by name (emotion, library) - plain words "
                                    "like happy/curious/yes/no resolve to library moves; "
                                    "list_moves: the library's move names (library=emotions|dances); "
                                    "motors: torque mode (mode=enabled|disabled|gravity_compensation), or omit mode to read it - a move is accepted with torque off and holds nothing; "
                                    "say: speak text through the robot's TTS sidecar + speaker (text, wobble); "
                                    "play_sound: play a WAV the daemon can read (sound_file, wobble); "
                                    "volume: read the speaker level; "
                                    "set_volume: set it (level 0-100 or a word) - the daemon ALSO plays a short test "
                                    "sound, so allow_test_sound=true is required; "
                                    "track_face: daemon face tracking on/off (enabled, weight) - while it is on at "
                                    "weight 1 it overrides look/express; "
                                    "tracking_status: whether a face is detected and where; "
                                    "camera: save a fresh native camera JPEG locally; "
                                    "record_audio: record a bounded microphone WAV locally (both require GStreamer); "
                                    "look_at: turn the head toward a camera pixel (u, v, frame_width, frame_height, "
                                    "duration); "
                                    "turn_to_sound: face whoever is talking using the microphone array's direction of "
                                    "arrival (enabled, sign) - one smooth turn per utterance, face tracking outranks it; "
                                    "turn_to_sound_status: bearing, speech flag, turns and why the last frame did not turn"
                                ),
                                "enum": list(_ACTIONS),
                                "default": "sensors",
                            },
                            "pitch": {"type": "number", "description": "Head pitch, degrees, +/-40 (look)."},
                            "roll": {"type": "number", "description": "Head roll, degrees, +/-40 (look)."},
                            "yaw": {"type": "number", "description": "Head yaw, degrees, +/-180 (look)."},
                            "x": {
                                "type": "number",
                                "description": "Head forward translation, millimetres, +/-25 (look).",
                            },
                            "y": {"type": "number", "description": "Head left translation, millimetres, +/-25 (look)."},
                            "z": {"type": "number", "description": "Head up translation, millimetres, +/-25 (look)."},
                            "body_yaw": {
                                "type": "number",
                                "description": (
                                    "Body yaw, degrees, +/-160 (body_turn; optional in look, within 65 deg of the head yaw)."
                                ),
                            },
                            "antenna_right": {
                                "type": "number",
                                "description": "Right antenna angle, degrees, +/-150 (antennas; optional in look).",
                            },
                            "antenna_left": {
                                "type": "number",
                                "description": "Left antenna angle, degrees, +/-150 (antennas; optional in look).",
                            },
                            "duration": {
                                "type": "number",
                                "description": (
                                    "Seconds: the interpolation time for look, antennas, body_turn, home and look_at "
                                    "(0.1-10, default 0.8), or the microphone recording length for record_audio (0.1-5, default 1)."
                                ),
                                "minimum": 0.1,
                                "maximum": 10,
                            },
                            "interpolation": {
                                "type": "string",
                                "description": "Interpolation for a move: minjerk (default), linear, ease_in_out or cartoon.",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "Move name or plain emotion word for express (happy, curious, yes, no, sad, dance1 ...).",
                            },
                            "library": {
                                "type": "string",
                                "description": "Recorded-move library for express/list_moves: emotions (default) or dances.",
                            },
                            "mode": {
                                "type": "string",
                                "description": "Torque mode for motors: enabled, disabled or gravity_compensation; omit to read the current mode.",
                            },
                            "text": {"type": "string", "description": "What to say (say), up to 500 characters."},
                            "sound_file": {
                                "type": "string",
                                "description": "WAV for play_sound: an absolute path on the robot, a built-in asset name or an uploaded name.",
                            },
                            "wobble": {
                                "type": "boolean",
                                "description": "say/play_sound: bob the head in sync with the audio (moves the head). Default false.",
                            },
                            "level": {
                                "type": "string",
                                "description": "set_volume: 0-100, or silent/low/normal/loud/max/quieter/louder.",
                            },
                            "allow_test_sound": {
                                "type": "boolean",
                                "description": "set_volume: acknowledge that the daemon plays a short test sound when the level changes. Required true.",
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "track_face / turn_to_sound: true to start, false to stop.",
                            },
                            "weight": {
                                "type": "number",
                                "description": "track_face: 0-1 blend; 1 lets tracking own the head (default), 0 pauses it without stopping the detector.",
                            },
                            "u": {
                                "type": "integer",
                                "description": "Pixel column for look_at.",
                            },
                            "v": {"type": "integer", "description": "Pixel row for look_at."},
                            "frame_width": {
                                "type": "integer",
                                "description": "Unmodified source frame width for look_at.",
                            },
                            "frame_height": {
                                "type": "integer",
                                "description": "Unmodified source frame height for look_at.",
                            },
                            "save_path": {
                                "type": "string",
                                "description": "Camera JPEG or audio WAV output path; empty creates a private temporary file. Never overwrites.",
                                "default": "",
                            },
                            "sign": {
                                "type": "number",
                                "description": "turn_to_sound: +1 (default) when +yaw is the array's 0-rad side, -1 for a mirrored mount.",
                                "default": 1.0,
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
        the first verb that needs the daemon, so
        ``Agent(tools=[Robot("reachy_mini", mode="real")])`` works without a
        separate :meth:`connect_eagerly` line; a connect that fails is reported
        as the verb's refusal, naming the reason. ``status`` never connects -
        it is the question "are we connected", and answering it by connecting
        would make it unable to say no.

        Args:
            tool_use: The agent's request, carrying the tool id and parameters.
            invocation_state: Caller-provided state; unused here.
            **kwargs: Forward compatibility only.

        Yields:
            One tool result envelope, carrying the requested read-out.
        """
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        params: dict[str, Any] = dict(tool_use.get("input") or {})
        action = params.pop("action", "sensors")
        # One dispatch decision. The halt has a branch of its own so the verb
        # returns ``stop_task``'s verdict (a daemon that declines the stop is
        # reported, never restated as success) and so it never waits on a
        # connect. Every other declared verb runs its handler from the table
        # the schema enum is generated from; anything else - a typo, a
        # non-string, a verb borrowed from a sibling driver - is refused by
        # name and never reaches a write. The terminal ``else`` refuses.
        if action == "stop":
            envelope = self.stop_task()
        elif isinstance(action, str) and action in _ACTIONS:
            if action != "status" and not self._connected and (reason := self.connect_eagerly()) is not None:
                envelope = _refuse(f"{action}: {reason}")
            elif inspect.iscoroutinefunction(_ACTIONS[action]):
                envelope = await _ACTIONS[action](self, params)
            else:
                envelope = await asyncio.to_thread(_ACTIONS[action], self, params)
        else:
            envelope = undeclared_verb_error(self, action)
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                              #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Probe the daemon, then start the real-time link. Idempotent.

        The factory only constructs the driver; whoever performs the bring-up
        calls this, so a real bring-up fails here rather than on the first mesh
        poll. Off hardware - the Thor case - the REST probe cannot reach a
        daemon and the returned reason names the address that did not answer.

        A second call on a connected driver is a no-op success rather than a
        second link: rebuilding would drop the only reference to the running
        link and leave its reader task subscribed.

        Returns:
            ``None`` on success and on a call against an already-connected
            driver. A named reason on failure - the driver is left disconnected
            but usable, so a mesh peer for a Mini that is switched off can still
            be constructed for later use.
        """
        if self._connected:
            logger.debug("%s already connected; connect_eagerly() is a no-op", self._tool_name)
            return None

        # Resolved before the probe so a missing extra is reported as a missing
        # extra. Routed through the probe instead, it would come back wrapped as
        # "daemon unreachable (host:port)" and send an operator to check a
        # network that was never touched.
        transport = _resolve_transport()
        if isinstance(transport, str):
            self._connect_error = transport
            return transport

        if self._discover_host:
            found = self._discover_daemon()
            if isinstance(found, str):
                self._connect_error = found
                return found
            self._host, self._api_port, status = found
        else:
            status = self._daemon_get(_PATH_STATUS)
            if (error := status.get("error")) is not None:
                reason = f"daemon unreachable ({self._host}:{self._api_port}): {error}"
                self._connect_error = reason
                return reason

        # The daemon reports the variant; a payload without the flag is treated
        # as a Wireless because that is the shipped default, matching the
        # Device Connect driver's reading of the same field.
        is_lite = not status.get("wireless_version", True)
        self._variant = "lite" if is_lite else "wireless"
        self._absorb_status(status)

        link = self._build_link(is_lite=is_lite)
        if isinstance(link, str):
            self._connect_error = link
            return link

        error_text = self._start_link(link)
        if error_text is not None:
            self._connect_error = error_text
            return error_text

        self._link = link
        self._connected = True
        self._connect_error = None
        self._stopped = False
        return None

    def _discover_daemon(self) -> tuple[str, int, dict[str, Any]] | str:
        """Find the daemon a zero-argument bring-up should talk to.

        Probes :func:`~strands_robots.drivers.reachy_vocabulary.discovery_candidates`
        in order - ``REACHY_HOST`` when set, else ``localhost`` then
        ``reachy-mini.local`` - and takes the first whose
        ``/api/daemon/status`` :func:`~strands_robots.drivers.reachy_vocabulary.daemon_answers`.
        A daemon that answers with its own start-up error (a desktop Lite daemon
        that found no robot) is skipped, not connected to: it serves no robot,
        and taking it would hide the Wireless one further down the list.

        Returns:
            ``(host, port, status)`` for the daemon found, or a reason listing
            every candidate and what each one said.
        """
        tried: list[str] = []
        for host, port in discovery_candidates(self._api_port):
            status = self._daemon_get_at(host, port, _PATH_STATUS)
            if daemon_answers(status):
                return host, port, status
            said = status.get("error") if isinstance(status, dict) else status
            tried.append(f"{host}:{port} -> {said if said is not None else 'daemon reports state=error'}")
        return (
            "daemon unreachable ("
            + ", ".join(t.split(" -> ")[0] for t in tried)
            + "): "
            + "; ".join(tried)
            + f'. Pass port="host[:port]" to Robot(...) or export {ENV_HOST}'
        )

    @classmethod
    def probe_hardware(cls) -> bool:
        """Whether a Reachy daemon answers at any discovery address right now.

        The hook :func:`strands_robots.robot.Robot` ``mode="auto"`` consults for
        a robot whose hardware is reached over the network rather than a serial
        bus: the USB scan that decides ``auto`` for a servo arm cannot see a
        daemon. Read-only - one ``GET /api/daemon/status`` per candidate - and
        it builds no driver.

        Returns:
            ``True`` when a usable daemon answered.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return False
        for host, port in discovery_candidates():
            if daemon_answers(transport.api(host, port, _PATH_STATUS)):
                return True
        return False

    def _build_link(self, *, is_lite: bool) -> Any:
        """Return the link for this hardware variant, or a reason string.

        Kept as a method, like
        :meth:`~strands_robots.drivers.g1.G1Driver._subscription_plan`, so a
        test can substitute a link without a daemon and without patching an
        import.

        Args:
            is_lite: Whether the daemon reported a Lite (no onboard computer).

        Returns:
            A ``HardwareLink``, or a reason naming what the variant needs and
            did not get.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return transport

        if is_lite or self._transport is None:
            return transport.WebSocketLink(self._host, self._api_port)
        return transport.ZenohLink(self._transport, self._zenoh_prefix)

    def _start_link(self, link: Any) -> str | None:
        """Start ``link`` on a dedicated background asyncio loop.

        The links are async and the mesh, the agent tool path and
        :meth:`connect_eagerly` are all synchronous callers, so the driver owns
        one loop on one thread for the link's lifetime. A daemon thread so a
        process that forgets :meth:`cleanup` still exits.

        Args:
            link: The link to start.

        Returns:
            ``None`` on success, or a reason naming the failure.
        """
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name=f"{self._tool_name}-reachy-link",
            daemon=True,
        )
        thread.start()
        future = asyncio.run_coroutine_threadsafe(
            link.start(on_joints=self._on_joints, on_imu=self._on_imu),
            loop,
        )
        try:
            future.result(timeout=_LINK_START_TIMEOUT_S)
        except TimeoutError:
            # Named before the general handler because this one carries no
            # message: ``str(TimeoutError())`` is the empty string, so reporting
            # it as a cause produced "failed to start: " and told an operator
            # nothing. The budget is the cause, so the budget is what is named.
            self._release_link(link, future, loop, thread)
            return (
                f"link to {self._host}:{self._api_port} did not finish its handshake within {_LINK_START_TIMEOUT_S:g}s"
            )
        except Exception as exc:  # noqa: BLE001 - any link failure is a connect failure
            self._release_link(link, future, loop, thread)
            return f"link to {self._host}:{self._api_port} failed to start: {exc}"
        self._loop = loop
        self._loop_thread = thread
        return None

    def _stop_loop(self, loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
        """Stop the loop running *thread*, wait for it, and close the loop.

        ``asyncio.new_event_loop`` is what :meth:`_start_link` calls to get this
        loop, and ``loop.close()`` is that call's documented
        counterpart: it releases the selector and the self-pipe the loop opened.
        ``loop.stop()`` is not that counterpart - it only asks ``run_forever`` to
        return. So a teardown that stops without closing abandons an open loop,
        and Python says so: every connect/teardown cycle raised one
        ``ResourceWarning: unclosed event loop``, reported not here but wherever
        the collector happened to reclaim it, which is code that has nothing to
        do with this driver.

        The wait is not politeness, it is what makes the close legal: closing a
        loop that is still running raises ``RuntimeError``, and ``stop()`` is
        asynchronous - it schedules the stop and returns, so the thread is still
        inside ``run_forever`` when it does. Waiting for the thread is therefore
        the only way to know the loop has stopped, and it is why the thread
        handle is kept at all.

        Bounded by :data:`_LOOP_JOIN_TIMEOUT_S`, and a thread that outlasts it
        keeps its loop: the loop is by definition still running, so closing it
        would raise, and reporting a teardown that did not happen is worse than
        one open loop. That outcome is logged rather than raised, because
        teardown is a caller's last action and has no error to return to.

        Args:
            loop: The loop to stop and close.
            thread: The thread running that loop, waited for here.
        """
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=_LOOP_JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.warning(
                "%s: the link loop did not stop within %.1fs, so its loop is left open; "
                "a link callback is still running on it.",
                self._tool_name,
                _LOOP_JOIN_TIMEOUT_S,
            )
            return
        loop.close()

    def _release_link(
        self,
        link: Any,
        future: Future[None],
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
    ) -> None:
        """Close whatever a bring-up that will not be adopted already opened.

        A ``start`` that raised, or that outran
        :data:`_LINK_START_TIMEOUT_S`, can still have put the link on the wire:
        :meth:`~strands_robots.device_connect.reachy_transport.WebSocketLink.start`
        assigns the connected socket before it spawns its read task, and the
        Zenoh link subscribes to its first topic before its second. The link is
        not adopted after such a failure - ``_link`` stays ``None`` so the driver
        reports itself disconnected - which means :meth:`cleanup` has nothing to
        stop and no verb can ever reach that socket again. Every later
        :meth:`connect_eagerly` builds a fresh link, so the stranded reader stays
        subscribed for the life of the process, writing sensor frames nobody
        reads: exactly the outcome :meth:`connect_eagerly` refuses a second link
        in order to avoid.

        So the handshake is cancelled and the link is asked to stop, on the loop
        it was started on, before that loop is stopped. Both concrete links
        tolerate a ``stop`` after a partial ``start``: the WebSocket link guards
        each handle it clears, and the Zenoh link's stop is a no-op.

        Args:
            link: The link whose bring-up failed.
            future: The pending handshake, cancelled here.
            loop: The loop the handshake was submitted to; stopped last.
            thread: The thread running that loop, so the loop this bring-up
                opened is closed rather than left for the collector.
        """
        future.cancel()
        try:
            asyncio.run_coroutine_threadsafe(link.stop(), loop).result(timeout=_LINK_START_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - teardown of a failed bring-up must not raise
            logger.debug("%s: stopping the failed link raised: %s", self._tool_name, exc)
        self._stop_loop(loop, thread)

    async def get_status(self) -> dict[str, Any]:
        """Report reachability, hardware variant and the latest battery read.

        The shape matches the lerobot driver's ``get_status`` envelope so the
        mesh publishes both peers identically.

        Returns:
            A success envelope carrying the driver's connection state.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self._connected,
                        "connect_error": self._connect_error,
                        "host": self._host,
                        "api_port": self._api_port,
                        "variant": self._variant,
                        # The halt an operator reads; cleared by the next
                        # motion this driver commits.
                        "motion_stopped": self._stopped,
                        "battery_pct": (self._battery or {}).get("pct"),
                        "discovery": self._discover_host,
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Ask the daemon to stop every move in progress.

        Unlike a robot with no motion path, the Mini has a real stop - but the
        daemon's ``POST /api/move/stop`` takes the ``uuid`` of ONE running move
        (a bare post is a 422), so a halt is ``GET /api/move/running`` followed
        by one stop per uuid. The link stays up, so sensors keep arriving - a
        stopped Mini is still observable, which is what an operator wants after
        halting it. A DoA turner, if one is running, is stopped first so it
        cannot queue a fresh turn behind the halt.

        This is the protocol's verdict-free shutdown hook: the halt itself, and
        the verdict, live in :meth:`stop_task` - one owner - and a daemon that
        declines is logged here rather than swallowed silently.
        """
        outcome = self.stop_task()
        if outcome.get("status") != "success":
            logger.warning(
                "%s.stop(): %s",
                self._tool_name,
                " ".join(str(block.get("text", "")) for block in outcome.get("content", []) if isinstance(block, dict)),
            )

    def cleanup(self) -> None:
        """Stop the link, then stop and close the loop it ran on. Idempotent.

        The loop is closed and not merely stopped - see :meth:`_stop_loop` for
        why the two are different and why closing it means waiting for the
        thread first. ``_loop`` and ``_loop_thread`` are adopted together by
        :meth:`_start_link` and cleared together here, so one being set is the
        same condition as both.
        """
        self._stop_doa()
        if self._link is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._link.stop(), self._loop).result(timeout=5)
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("%s: link stop failed during cleanup: %s", self._tool_name, exc)
        if self._loop is not None and self._loop_thread is not None:
            self._stop_loop(self._loop, self._loop_thread)
        self._link = None
        self._loop = None
        self._loop_thread = None
        self._connected = False
        with self._cache_lock:
            self._joints_received_at = None
        self._remember_head_yaw_target(None)

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
        *,
        require_ack: bool = False,
    ) -> dict[str, Any]:
        """Command head pose, body yaw and antennas, refusing what cannot be met.

        Three gates, in this order:

        1. The driver is connected. A write to a link that was never started has
           nowhere to go.
        2. Every numeric value is finite, and every bounded axis is inside the
           envelope - both from the shared
           :func:`~strands_robots.drivers.reachy_envelope.envelope_error`, so this driver
           and the ``reachy_*`` tools cannot disagree about the same robot. An
           action carrying ``body_yaw`` and no head pose is checked against the
           head yaw this driver last commanded, so the head-body coupling limit
           applies to a body-only turn as well as to a pair: the daemon holds
           the head pose and turns the body no further than the limit, so a
           lone body yaw beyond it would report success and stop short. The
           limit is skipped, not guessed, while that target is unknown.
        3. Every key names something this driver can send. An action naming no
           axis at all is refused rather than reported as a successful no-op,
           and so is one that names a real axis alongside a key this driver has
           no actuator for: a dropped head axis is not left alone but commanded
           to zero, because the daemon's head command is a whole pose.

        Degrees in, radians and pose matrices out. The caller-facing unit is
        degrees because the envelope is expressed in degrees and because the
        daemon's own RPC surface takes degrees; the wire wants radians for a
        joint and a 4x4 matrix for the head, which
        :func:`~strands_robots.device_connect.reachy_transport.rpy_to_pose`
        builds.

        Args:
            action: Any of ``head_pitch``, ``head_roll``, ``head_yaw`` (degrees),
                ``head_x``, ``head_y``, ``head_z`` (millimetres), ``body_yaw``
                (degrees), ``antenna_left``, ``antenna_right`` (degrees). Absent
                head axes default to zero, so ``{"head_yaw": 20}`` means "look
                20 degrees left, level" rather than "leave pitch as it was" -
                the daemon's head command is a whole pose, not a delta.
            robot_name: Accepted for contract parity. This driver fronts exactly
                one Mini, so a name that is neither ``None`` nor this driver's
                own is refused rather than silently applied to the wrong robot.
            require_ack: Opt into a native REST acknowledgement for an explicit
                antenna_right/antenna_left pair only. Requires telemetry received
                within 0.5 monotonic seconds. Only complete, finite joint frames
                refresh this receipt. All existing numeric/envelope gates still run.
                Busy/unknown/failed replies refuse without a WebSocket fallback
                or retry; a timeout leaves delivery uncertain. Default false
                preserves the existing fire-and-forget link path. Neither path
                verifies physical motion or grants exclusive controller ownership.

        Returns:
            A success envelope naming what was sent, or an error envelope naming
            the first thing that refused.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        if not self._connected:
            return _refuse("not connected - call connect_eagerly() first")

        if not isinstance(require_ack, bool):
            return _refuse("send_action: require_ack must be a boolean")
        for name, value in action.items():
            if (reason := finite_number_error(value, name, "send_action")) is not None:
                return _refuse(reason)
        commanded_head_yaw = _head_yaw_of(action)
        held_head_yaw = commanded_head_yaw if commanded_head_yaw is not None else self._read_head_yaw_target()
        if (reason := envelope_error(action, "send_action", head_yaw_target=held_head_yaw)) is not None:
            return _refuse(reason)

        commands = _wire_commands(action)
        if isinstance(commands, str):
            return _refuse(f"send_action: {commands}")
        if not commands:
            return _refuse(
                f"send_action: nothing to send - none of {sorted(action)} names a Reachy Mini axis; "
                f"expected any of {sorted(_ACTION_KEYS)}"
            )
        # Checked after the gate above rather than before it, so each refusal
        # diagnoses one fault: an action naming no axis at all is told what to
        # send, and an action that mostly parsed is told which key was dropped.
        if unknown := sorted(set(action) - _ACTION_KEYS):
            return _refuse(
                f"send_action: {unknown} names no Reachy Mini axis; expected any of {sorted(_ACTION_KEYS)}. "
                "A dropped head axis is commanded to zero rather than left alone, because the daemon's "
                "head command is a whole pose"
            )

        if require_ack:
            if set(action) != {"antenna_right", "antenna_left"}:
                return _refuse("send_action: require_ack supports only an explicit antenna_right/antenna_left pair")
            with self._cache_lock:
                stamp = self._joints_received_at
            if (
                stamp is None
                or finite_number_error(stamp, "telemetry timestamp", "send_action")
                or not 0 <= time.monotonic() - stamp <= 0.5
            ):
                return _refuse("send_action: require_ack needs joint telemetry received within 0.5 seconds")
            result = self._daemon_post(_PATH_SET_TARGET, {"target_antennas": commands[0]["antennas_joint_positions"]})
            if "error" in result or result.get("status") != "ok":
                return _refuse(
                    f"send_action: target not acknowledged: {result!r}; delivery may be uncertain, no automatic retry"
                )
            self._stopped = False
            return {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "sent": [sorted(c) for c in commands],
                            "robot": self._tool_name,
                            "acknowledgement": "daemon_target_handler_ok",
                            "motion_verified": False,
                        }
                    }
                ],
            }

        for command in commands:
            if (error := self._send_cmd(command)) is not None:
                return _refuse(f"send_action: {error}")
        if commanded_head_yaw is not None:
            self._remember_head_yaw_target(commanded_head_yaw)
        self._stopped = False
        return {
            "status": "success",
            "content": [{"json": {"sent": [sorted(c) for c in commands], "robot": self._tool_name}}],
        }

    # ------------------------------------------------------------------ #
    # Task and policy paths.                                             #
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
        """Refuse a policy-driven task: the Mini has no policy path.

        Not a stub awaiting wiring like a manipulator's would be. The Mini has
        no arms and no gait, so there is no action space a ``groot`` or lerobot
        policy is trained against; its expressive behaviour is a *recorded move*
        played by the daemon, which the ``reachy_*`` tool bundle plays by name.
        The refusal says so, rather than implying a rollout is coming.

        Args:
            instruction: Ignored; named for contract parity.
            policy_port: Ignored.
            policy_host: Ignored.
            policy_provider: Ignored.
            duration: Ignored.
            **policy_kwargs: Ignored.

        Returns:
            An error envelope naming the recorded-move path instead.
        """
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: the Reachy Mini has no policy action space (no arms, no gait); "
            "play a recorded move through the reachy_* tools instead"
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a rollout, for the same reason as :meth:`start_task`.

        Args:
            policy_object: Ignored; named for contract parity.
            instruction: Ignored.
            duration: Ignored.
            n_steps: Ignored.

        Returns:
            An error envelope naming the recorded-move path instead.
        """
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: the Reachy Mini has no policy action space (no arms, no gait); "
            "play a recorded move through the reachy_* tools instead"
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report that no task is running, because none can be started.

        Returns:
            A success envelope: the question is answerable even though
            :meth:`start_task` refuses.
        """
        return {
            "status": "success",
            "content": [{"json": {"running": False, "reason": "the Reachy Mini has no policy task path"}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop motion, since a Mini's closest thing to a task is a recorded move.

        The one place the halt is issued and recorded: the DoA turner is
        stopped first so it cannot queue a turn behind the halt, then every
        running move is listed and stopped by uuid (the daemon's stop takes ONE
        uuid; a bare post is a 422, which is why an earlier bare post never
        halted anything). ``motion_stopped`` is recorded only once every stop
        was accepted; a daemon that declines one is reported, not restated.

        Returns:
            A success envelope naming the uuids stopped (an empty list when
            nothing was running - a no-op halt), or a refusal naming what the
            daemon declined.
        """
        self._stop_doa()
        running = self._daemon_get_list(_PATH_MOVES_RUNNING)
        if isinstance(running, dict):
            return _refuse(f"stop_task: could not list running moves: {running.get('error')}")
        uuids = [entry.get("uuid") for entry in running if isinstance(entry, dict) and entry.get("uuid")]
        stopped: list[str] = []
        for uuid in uuids:
            result = self._daemon_post(_PATH_STOP, {"uuid": uuid})
            if (error := result.get("error")) is not None:
                return _refuse(
                    f"stop_task: daemon refused the stop of move {uuid}: {error} "
                    f"(stopped before the refusal: {stopped or 'none'})"
                )
            stopped.append(str(uuid))
        self._stopped = True
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "stopped": stopped,
                        "running_before": len(uuids),
                        "note": "each running move was stopped by uuid; nothing running is a no-op halt",
                    }
                }
            ],
        }

    # ------------------------------------------------------------------ #
    # Recorded-move and motor paths the reachy_* tools call.             #
    # ------------------------------------------------------------------ #

    def play_move(self, move_name: str, library: str = "emotions") -> dict[str, Any]:
        """Play one recorded move (emotion or dance) by name through the daemon.

        The Mini's expressive behaviour is a recorded head+antenna+body
        choreography served by the daemon from a HuggingFace library - the same
        rail :meth:`stop_task` halts. Three gates: connected, ``library`` in the
        admitted set, ``move_name`` a bare URL path segment - the alphabet plus
        an alphanumeric first character, so a dot segment cannot re-point the
        request at the daemon's parent path.

        Args:
            move_name: The move's library name (``'cheerful1'``) or a plain
                emotion word (``'happy'``, ``'curious'``, ``'no'``) resolved
                through :func:`~strands_robots.drivers.reachy_vocabulary.resolve_move_name`.
            library: Which library, one of ``'emotions'`` or ``'dances'``.

        Returns:
            A success envelope naming the move actually sent (and the word it
            was resolved from), or an error envelope naming the first gate that
            refused. Success is the daemon accepting the move: while daemon face
            tracking holds the head at weight 1 the choreography is overridden.
        """
        if not self._connected:
            return _refuse("play_move: not connected - call connect_eagerly() first")
        dataset = _MOVE_LIBRARIES.get(library)
        if dataset is None:
            return _refuse(f"play_move: unknown library {library!r}; expected one of {sorted(_MOVE_LIBRARIES)}")
        # A plain word ("happy", "go away") becomes the library's own name
        # ("cheerful1", "go_away1") against the live catalogue; a catalogue
        # that cannot be read leaves the alias table to answer alone. The
        # resolved name still has to pass the path-segment gate below, so the
        # translation cannot manufacture a name the gate would have refused.
        # The path-segment gate runs BEFORE the catalogue read, on the
        # normalised request: a dot segment must build no request at all, not
        # one GET and then a refusal. ``resolve_move_name`` only ever returns a
        # catalogue entry, an alias-table value or ``key + "1"``, all of which
        # pass the same gate when ``key`` does, so the chosen name needs no
        # second check.
        key = (move_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not _MOVE_NAME_RE.fullmatch(key):
            return _refuse(
                f"play_move: invalid move_name {move_name!r}; expected 1-128 chars of [A-Za-z0-9._-] "
                "starting with a letter or digit (one bare path segment, so no '.' or '..') - "
                "list_moves() names the library's catalogue"
            )
        catalogue = self._daemon_get_list(_PATH_MOVE_LIST.format(dataset=dataset))
        names = catalogue if isinstance(catalogue, list) else []
        resolved = resolve_move_name(key, names)
        if resolved is None and names:
            sample = ", ".join(sorted(str(n) for n in names)[:12])
            return _refuse(
                f"play_move: {move_name!r} names no move in the {library} library and matches no alias; "
                f"list_moves() has the catalogue (starts: {sample} ...)"
            )
        chosen = resolved if resolved is not None else key
        result = self._daemon_post(_PATH_MOVE_PLAY.format(dataset=dataset, move=chosen))
        if (error := result.get("error")) is not None:
            return _refuse(f"play_move: daemon refused {chosen!r}: {error}")
        self._remember_head_yaw_target(None)
        self._stopped = False
        payload: dict[str, Any] = {"played": chosen, "library": library, "motion_verified": False}
        if chosen != move_name:
            payload["requested"] = move_name
        return {"status": "success", "content": [{"json": payload}]}

    def list_moves(self, library: str = "emotions") -> dict[str, Any]:
        """List the recorded moves one library serves.

        Args:
            library: Which library, one of ``'emotions'`` or ``'dances'``.

        Returns:
            A success envelope whose ``moves`` is the daemon's array of names,
            or an error envelope naming what refused.
        """
        if not self._connected:
            return _refuse("list_moves: not connected - call connect_eagerly() first")
        dataset = _MOVE_LIBRARIES.get(library)
        if dataset is None:
            return _refuse(f"list_moves: unknown library {library!r}; expected one of {sorted(_MOVE_LIBRARIES)}")
        result = self._daemon_get_list(_PATH_MOVE_LIST.format(dataset=dataset))
        # A catalogue read succeeds as a JSON array. The only dict this endpoint
        # can produce is the transport's {"error": ...} envelope, so a dict here
        # is a failure whatever key it carries.
        if isinstance(result, dict):
            return _refuse(f"list_moves: daemon refused: {result.get('error', result)}")
        return {"status": "success", "content": [{"json": {"library": library, "moves": result}}]}

    def wake_up(self) -> dict[str, Any]:
        """Play the daemon's built-in wake-up move (init pose, ears up).

        Returns:
            A success envelope, or an error envelope naming what refused.
        """
        if not self._connected:
            return _refuse("wake_up: not connected - call connect_eagerly() first")
        result = self._daemon_post(_PATH_WAKE)
        if (error := result.get("error")) is not None:
            return _refuse(f"wake_up: daemon refused: {error}")
        self._remember_head_yaw_target(None)
        self._stopped = False
        return {"status": "success", "content": [{"text": "asked the daemon to play the wake-up move"}]}

    def goto_sleep(self) -> dict[str, Any]:
        """Play the daemon's built-in go-to-sleep move (head down, ears rest).

        Returns:
            A success envelope, or an error envelope naming what refused.
        """
        if not self._connected:
            return _refuse("goto_sleep: not connected - call connect_eagerly() first")
        result = self._daemon_post(_PATH_SLEEP)
        if (error := result.get("error")) is not None:
            return _refuse(f"goto_sleep: daemon refused: {error}")
        self._remember_head_yaw_target(None)
        self._stopped = False
        return {"status": "success", "content": [{"text": "asked the daemon to play the go-to-sleep move"}]}

    def set_motors(self, mode: str) -> dict[str, Any]:
        """Set motor torque: ``'enabled'`` holds, ``'disabled'`` goes limp, ``'gravity_compensation'`` floats.

        The first two are sent on the real-time command link, the same rail
        :meth:`send_action` writes. The third has no link command, so it goes
        through the daemon's own ``POST /api/motors/set_mode/{mode}`` - the path
        the Pollen SDK's ``enable_gravity_compensation()`` takes. Disabling
        torque on a head that is not resting lets it drop; the caller is the one
        holding it.

        Args:
            mode: ``'enabled'`` (torque on), ``'disabled'`` (safe to move by
                hand) or ``'gravity_compensation'`` (marionette mode).

        Returns:
            A success envelope naming the mode, or an error envelope naming
            what refused.
        """
        if not self._connected:
            return _refuse("set_motors: not connected - call connect_eagerly() first")
        if mode not in _MOTOR_MODES:
            return _refuse(f"set_motors: unknown mode {mode!r}; expected one of {list(_MOTOR_MODES)}")
        if mode == "gravity_compensation":
            result = self._daemon_post(_PATH_MOTORS_MODE.format(mode=mode))
            if (error := result.get("error")) is not None:
                return _refuse(f"set_motors: daemon refused {mode!r}: {error}")
        else:
            torque = mode == "enabled"
            if (error := self._send_cmd({"torque": torque, "ids": None})) is not None:
                return _refuse(f"set_motors: {error}")
        self._remember_head_yaw_target(None)
        return {"status": "success", "content": [{"json": {"motors": mode}}]}

    def read_motors(self) -> dict[str, Any]:
        """Report the torque mode the robot is in, as the daemon sees it.

        The read for :meth:`set_motors`, the same way :meth:`get_volume` answers
        for :meth:`set_volume`. It matters because the daemon accepts a ``goto``
        in every mode: a move commanded while torque is off is acknowledged with
        a move uuid and moves nothing, so without this an agent whose ``look``
        succeeded and changed no pose had no way to learn why.

        Returns:
            A success envelope carrying the daemon's ``control_mode`` and
            whether a commanded pose will be held - only ``'enabled'`` holds
            one; ``'disabled'`` is limp and ``'gravity_compensation'`` floats.
            A refusal names what refused: no link, a transport failure, or a
            daemon body carrying no mode.
        """
        if not self._connected:
            return _refuse("read_motors: not connected - call connect_eagerly() first")
        result = self._daemon_get(_PATH_STATE)
        if (error := result.get("error")) is not None:
            return _refuse(f"read_motors: {error}")
        mode = result.get("control_mode")
        if not isinstance(mode, str):
            return _refuse(f"read_motors: daemon reported no control_mode (got {mode!r})")
        return {"status": "success", "content": [{"json": {"motors": mode, "holds_a_pose": mode == "enabled"}}]}

    # ------------------------------------------------------------------ #
    # Expressive verbs: smooth moves, speech, sound, volume, tracking.   #
    # ------------------------------------------------------------------ #

    def goto(
        self,
        *,
        head: dict[str, float] | None = None,
        body_yaw: float | None = None,
        antennas: tuple[float, float] | None = None,
        duration: float = 0.8,
        interpolation: str = "minjerk",
    ) -> dict[str, Any]:
        """Ask the daemon for one smooth interpolated move - the gesture rail.

        ``POST /api/move/goto`` is what the Pollen SDK's ``goto_target`` and
        every desk app use for a gesture: the daemon interpolates from where the
        head is to the target over ``duration``. :meth:`send_action` is the
        other rail - the real-time ``set_target`` stream with no interpolation,
        meant for a 10 Hz+ control loop - and a single agent call on it is a
        step, not a gesture.

        Gates, in order: connected; the rotational envelope
        (:func:`~strands_robots.drivers.reachy_envelope.envelope_error`, with the
        head-body coupling checked against this driver's last commanded head
        yaw when only ``body_yaw`` is given); translation, antenna, duration and
        interpolation domains
        (:func:`~strands_robots.drivers.reachy_vocabulary.goto_body_error`).

        Args:
            head: Head pose with any of ``pitch``, ``roll``, ``yaw`` (degrees)
                and ``x``, ``y``, ``z`` (millimetres). Absent axes are zero -
                the daemon's head command is a whole pose, so ``{"pitch": 15}``
                is "look up, otherwise level". ``None`` leaves the head alone.
            body_yaw: Body yaw in degrees, or ``None`` to leave the body alone.
            antennas: ``(right, left)`` in degrees, or ``None`` to leave them.
            duration: Seconds, 0.1-10.
            interpolation: ``minjerk`` (default), ``linear``, ``ease_in_out`` or
                ``cartoon``.

        Returns:
            A success envelope carrying the daemon's move uuid and the body
            sent, or a refusal naming the first gate. Success means the daemon
            accepted and started the move; ``motion_verified`` is ``False``
            because nothing here reads the head back - and while daemon face
            tracking holds the head (weight 1) the move is overridden.
        """
        if not self._connected:
            return _refuse("goto: not connected - call connect_eagerly() first")
        values: dict[str, Any] = {}
        if head is not None:
            values.update(
                {
                    "head_pitch": head.get("pitch", 0.0),
                    "head_roll": head.get("roll", 0.0),
                    "head_yaw": head.get("yaw", 0.0),
                }
            )
        if body_yaw is not None:
            values["body_yaw"] = body_yaw
        # Validate BEFORE any float(): ``envelope_error`` runs
        # ``finite_number_error`` over every value, so a string ``yaw`` from
        # the model ("left") is refused here instead of raising through
        # ``stream``. ``head_yaw_target`` is only consulted when the action
        # names no head yaw, so it is only read in that case.
        if (
            reason := envelope_error(
                values,
                "goto",
                head_yaw_target=None if "head_yaw" in values else self._read_head_yaw_target(),
            )
        ) is not None:
            return _refuse(reason)
        if (
            reason := goto_body_error(
                head=head,
                body_yaw=body_yaw,
                antennas=antennas,
                duration=duration,
                interpolation=interpolation,
                context="goto",
            )
        ) is not None:
            return _refuse(reason)
        body = goto_body(
            head=head, body_yaw=body_yaw, antennas=antennas, duration=duration, interpolation=interpolation
        )
        result = self._daemon_post(_PATH_GOTO, body)
        if (error := result.get("error")) is not None:
            return _refuse(f"goto: daemon refused the move: {error}")
        if head is not None:
            self._remember_head_yaw_target(float(values["head_yaw"]))
        self._stopped = False
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "move": result,
                        "sent": body,
                        "duration_s": float(duration),
                        "robot": self._tool_name,
                        "motion_verified": False,
                    }
                }
            ],
        }

    def home(self, duration: float = 1.0) -> dict[str, Any]:
        """Return to the neutral pose: head level and centred, antennas level, body centred.

        Args:
            duration: Seconds for the interpolation.

        Returns:
            :meth:`goto`'s envelope.
        """
        return self.goto(
            head={"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
            body_yaw=0.0,
            antennas=(0.0, 0.0),
            duration=duration,
        )

    def get_volume(self) -> dict[str, Any]:
        """Read the speaker volume - ``GET /api/volume/current``. Read-only.

        Returns:
            A success envelope with ``volume`` (0-100) and the daemon's payload,
            or a refusal.
        """
        if not self._connected:
            return _refuse("get_volume: not connected - call connect_eagerly() first")
        result = self._daemon_get(_PATH_VOLUME_GET)
        if (error := result.get("error")) is not None:
            return _refuse(f"get_volume: daemon refused: {error}")
        return {"status": "success", "content": [{"json": {"volume": result.get("volume"), "daemon": result}}]}

    def set_volume(self, level: Any, *, allow_test_sound: bool = False) -> dict[str, Any]:
        """Set the speaker volume - and know that the daemon plays a sound doing it.

        Daemon 1.10.0's ``POST /api/volume/set`` handler also calls
        ``backend.play_sound("impatient1.wav")`` after writing the level
        (``daemon/app/routers/volume.py``). That is audible, and when head
        wobbling is enabled it moves the head. A volume change is therefore not
        a silent write, so this method refuses unless the caller says
        ``allow_test_sound=True`` - the same posture as every other
        side-effecting verb here: the caller names the consequence, the driver
        does not hide it.

        Args:
            level: 0-100, a numeric string (``"40%"``), one of the words in
                :data:`~strands_robots.drivers.reachy_vocabulary.VOLUME_WORDS`
                (``silent``, ``low``, ``normal``, ``loud``, ``max`` ...), or
                ``quieter``/``louder`` relative to the current level.
            allow_test_sound: Must be ``True``; acknowledges the test sound.

        Returns:
            A success envelope with ``previous`` and ``volume``, or a refusal.
            ``volume`` is what the daemon reported after the write, which is
            the mixer level - not proof of anything heard.
        """
        if not self._connected:
            return _refuse("set_volume: not connected - call connect_eagerly() first")
        if allow_test_sound is not True:
            return _refuse(
                "set_volume: the daemon plays a short test sound (impatient1.wav) whenever the level is set, and "
                "with wobbling enabled that moves the head; pass allow_test_sound=True to accept that, or use "
                "get_volume() to read without touching it"
            )
        current_result = self._daemon_get(_PATH_VOLUME_GET)
        current = current_result.get("volume") if current_result.get("error") is None else None
        target = resolve_volume_level(level, current if isinstance(current, int) else None)
        if isinstance(target, str):
            return _refuse(f"set_volume: {target}")
        result = self._daemon_post(_PATH_VOLUME_SET, {"volume": target})
        if (error := result.get("error")) is not None:
            return _refuse(f"set_volume: daemon refused {target}: {error}")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "previous": current,
                        "volume": result.get("volume", target),
                        "test_sound_played_by_daemon": True,
                        "audible_verified": False,
                    }
                }
            ],
        }

    def play_sound(self, sound_file: str, *, wobble: bool = False) -> dict[str, Any]:
        """Play a WAV on the robot's speaker - ``POST /api/media/play_sound``.

        The daemon resolves ``sound_file`` as an absolute path on ITS
        filesystem, a built-in asset name (``wake_up.wav``) or a name uploaded
        through ``/api/media/sounds/upload``; a path on the calling machine is
        not one of those. The handler answers ``{"status": "ok"}`` as soon as
        the backend accepted the file - including on a backend with no media
        server, where playback is a no-op - so acceptance is not audibility.

        Args:
            sound_file: The daemon-side file, as above.
            wobble: Enable audio-reactive head wobbling first
                (``/api/media/wobbling/enable``). This MOVES THE HEAD for the
                length of the audio, so it is off by default; nothing here
                disables it again, because the driver does not know when the
                sound ends - call ``set_wobbling(False)`` afterwards, or let
                :meth:`say` do it, which knows the WAV's length.

        Returns:
            A success envelope carrying the daemon's reply, or a refusal.
        """
        if not self._connected:
            return _refuse("play_sound: not connected - call connect_eagerly() first")
        if not isinstance(sound_file, str) or not sound_file.strip():
            return _refuse(
                f"play_sound: sound_file must be a non-empty daemon-side path or asset name, got {sound_file!r}"
            )
        if not isinstance(wobble, bool):
            return _refuse(f"play_sound: wobble must be a boolean, got {wobble!r}")
        if wobble and (reason := self.set_wobbling(True).get("status")) != "success":
            return _refuse(f"play_sound: could not enable wobbling before playback ({reason})")
        result = self._daemon_post(_PATH_PLAY_SOUND, {"file": sound_file})
        if (error := result.get("error")) is not None:
            return _refuse(f"play_sound: daemon refused {sound_file!r}: {error}")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "played": sound_file,
                        "wobble": wobble,
                        "daemon": result,
                        "audible_verified": False,
                    }
                }
            ],
        }

    def set_wobbling(self, enabled: bool) -> dict[str, Any]:
        """Enable or disable audio-reactive head wobbling on the daemon.

        Args:
            enabled: ``True`` for ``/api/media/wobbling/enable``, ``False`` for
                ``/disable`` (which also resets the speech offsets to zero).

        Returns:
            A success envelope, or a refusal.
        """
        if not self._connected:
            return _refuse("set_wobbling: not connected - call connect_eagerly() first")
        if not isinstance(enabled, bool):
            return _refuse(f"set_wobbling: enabled must be a boolean, got {enabled!r}")
        result = self._daemon_post(_PATH_WOBBLE_ENABLE if enabled else _PATH_WOBBLE_DISABLE)
        if (error := result.get("error")) is not None:
            return _refuse(f"set_wobbling: daemon refused: {error}")
        return {"status": "success", "content": [{"json": {"wobbling": enabled}}]}

    def tts_url(self) -> str:
        """The speech service URL :meth:`say` posts to.

        Returns:
            The constructor's ``tts_url``, else ``REACHY_TTS_URL``, else
            ``http://<daemon host>:5002`` - the daemon's host because the
            service has to write a WAV the DAEMON can read, so it lives on the
            robot.
        """
        if self._tts_url:
            return self._tts_url.rstrip("/")
        if env := (os.getenv(ENV_TTS_URL) or "").strip():
            return env.rstrip("/")
        return f"http://{self._host}:{DEFAULT_TTS_PORT}"

    def say(self, text: str, *, wobble: bool = False, timeout_s: float = 30.0) -> dict[str, Any]:
        """Speak ``text`` through the robot's speaker.

        Two hops, both on the robot: ``POST <tts_url>/tts {"text", "as": "path"}``
        asks a Piper/``tiny-tts`` service for a WAV and gets back the path it
        wrote, then :meth:`play_sound` hands that path to the daemon. The
        speech service is a sidecar ``tiny-the-reachy`` installs as
        ``tiny-tts.service``; it is NOT part of the Reachy daemon, so a robot
        without it refuses here by name - nothing is spoken through a cloud
        fallback the operator did not configure.

        With ``wobble`` the head bobs for the WAV's duration (read from the
        service's ``seconds`` field when present) and wobbling is disabled
        afterwards on a timer, so the head is handed back.

        Args:
            text: What to say, 1-500 characters.
            wobble: Bob the head while speaking. Moves the head; off by default.
            timeout_s: How long to wait for synthesis.

        Returns:
            A success envelope with the WAV path, its length when known and the
            daemon's playback reply, or a refusal. ``audible_verified`` is
            ``False``: the daemon accepting a file is not sound in the room.
        """
        if not self._connected:
            return _refuse("say: not connected - call connect_eagerly() first")
        if not isinstance(text, str) or not text.strip():
            return _refuse(f"say: text must be a non-empty string, got {text!r}")
        text = text.strip()
        if len(text) > 500:
            return _refuse(f"say: text is {len(text)} characters; the limit is 500 per call")
        if not isinstance(wobble, bool):
            return _refuse(f"say: wobble must be a boolean, got {wobble!r}")
        url = self.tts_url()
        synth = _post_json(f"{url}/tts", {"text": text, "as": "path"}, timeout_s)
        if (error := synth.get("error")) is not None:
            return _refuse(
                f"say: no speech service at {url} ({error}). Speech is a sidecar (tiny-tts / Piper) on the robot, "
                f"not part of the Reachy daemon; install it, or pass tts_url= / export {ENV_TTS_URL}"
            )
        path = synth.get("path")
        if not isinstance(path, str) or not path:
            return _refuse(f"say: speech service answered without a wav path: {synth!r}")
        seconds = synth.get("seconds")
        seconds_f = float(seconds) if isinstance(seconds, int | float) and math.isfinite(seconds) else None
        played = self.play_sound(path, wobble=wobble)
        if played.get("status") != "success":
            return played
        if wobble:
            tail = min((seconds_f if seconds_f is not None else max(1.5, len(text) * 0.06)) + 0.3, 30.0)
            timer = threading.Timer(tail, self.set_wobbling, args=(False,))
            timer.daemon = True
            timer.start()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "said": text,
                        "wav": path,
                        "seconds": seconds_f,
                        "tts_url": url,
                        "wobble": wobble,
                        "audible_verified": False,
                    }
                }
            ],
        }

    def set_tracking(self, enabled: bool, weight: float = 1.0) -> dict[str, Any]:
        """Turn the daemon's own face tracking on or off.

        Daemon 1.10 runs a YuNet face detector on its camera pipeline and blends
        the head toward the nearest face every control tick
        (``POST /api/media/tracking/enable {"weight"}`` / ``/disable``). Two
        facts a caller has to know: the DAEMON must own the camera (a client
        that released media stops tracking), and while tracking holds the head
        at weight 1 the daemon IGNORES ``goto``/``set_target`` - a ``look`` or
        an ``express`` sent meanwhile is overridden. Pollen's own conversation
        app pauses tracking (weight 0) while it speaks or gestures and restores
        it after; that choreography is a long-lived controller's job and is not
        done here - set ``weight=0.0`` before a gesture and ``1.0`` after it.

        Args:
            enabled: ``True`` to start following, ``False`` to stop the detector.
            weight: 0-1 blend when enabling; 1 lets tracking own the head.

        Returns:
            A success envelope with the daemon's ``enabled`` verdict (``False``
            with ``status: unavailable`` when the daemon has no camera), or a
            refusal.
        """
        if not self._connected:
            return _refuse("set_tracking: not connected - call connect_eagerly() first")
        if not isinstance(enabled, bool):
            return _refuse(f"set_tracking: enabled must be a boolean, got {enabled!r}")
        if (reason := finite_number_error(weight, "weight", "set_tracking")) is not None:
            return _refuse(reason)
        if not 0.0 <= float(weight) <= 1.0:
            return _refuse(f"set_tracking: weight {weight:g} is outside [0, 1]")
        if enabled:
            result = self._daemon_post(_PATH_TRACKING_ENABLE, {"weight": float(weight)})
        else:
            result = self._daemon_post(_PATH_TRACKING_DISABLE)
        if (error := result.get("error")) is not None:
            return _refuse(f"set_tracking: daemon refused: {error}")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "requested": enabled,
                        "weight": float(weight) if enabled else None,
                        "enabled": bool(result.get("enabled", False)),
                        "daemon": result,
                    }
                }
            ],
        }

    def tracked_face(self) -> dict[str, Any]:
        """The latest face the daemon's tracker saw - ``GET /api/media/tracking/face``. Read-only.

        Returns:
            A success envelope with ``face_target`` (``detected``, ``x``, ``y``
            in [-1, 1], ``roll``, ``ts``), or a refusal.
        """
        if not self._connected:
            return _refuse("tracked_face: not connected - call connect_eagerly() first")
        result = self._daemon_get(_PATH_TRACKING_FACE)
        if (error := result.get("error")) is not None:
            return _refuse(f"tracked_face: daemon refused: {error}")
        return {
            "status": "success",
            "content": [{"json": {"face_target": result.get("face_target"), "daemon": result}}],
        }

    def turn_to_sound(self, enabled: bool = True, *, sign: float = 1.0) -> dict[str, Any]:
        """Turn toward whoever is talking, or stop doing so.

        The Wireless Mini's microphone array reports a Direction of Arrival
        (``GET /api/state/full?with_doa=true``: ``angle`` in radians, ``0`` =
        the robot's left, ``pi/2`` = ahead, plus ``speech_detected``). Enabling
        starts a driver-owned thread that polls it at 10 Hz and, per utterance,
        commits ONE smooth :meth:`goto` toward the speaker - head first, the
        body carrying what the head cannot. The judgement lives in
        :class:`~strands_robots.drivers.reachy_doa.DoaTurner`: six agreeing
        speech frames, no rail readings, at least a 10 degree change, one turn
        per three seconds, and a windup guard against chasing the robot's own
        speaker. A face the tracker saw within the last 1.5 s, or a recorded
        move in flight, vetoes a turn - both outrank a bearing.

        The loop is stopped by ``turn_to_sound(False)``, by :meth:`stop` and by
        :meth:`cleanup`; it never outlives the driver. A Lite has no array: the
        daemon's frames then carry no ``doa`` and the loop simply never turns,
        which :meth:`turn_to_sound_status` shows as ``angle_deg: None``.

        Args:
            enabled: ``True`` to start (a no-op when already running), ``False``
                to stop.
            sign: ``+1`` (default) when ``+yaw`` is the side the array reports
                as ``0`` rad - the SDK convention - or ``-1`` for a mirrored
                mount. Empirical: confirm by speaking from the robot's left.

        Returns:
            A success envelope carrying the loop's status, or a refusal when the
            driver is disconnected or an argument is out of domain.
        """
        if not self._connected:
            return _refuse("turn_to_sound: not connected - call connect_eagerly() first")
        if not isinstance(enabled, bool):
            return _refuse(f"turn_to_sound: enabled must be a boolean, not {enabled!r}")
        if (reason := finite_number_error(sign, "sign", "turn_to_sound")) is not None:
            return _refuse(reason)
        if enabled:
            with self._doa_lock:
                if self._doa is None or not self._doa.running:
                    self._doa = DoaLoop(
                        read_frame=self._doa_frame,
                        look=self._doa_look,
                        blocked=self._doa_blocked,
                        turner=DoaTurner(sign=sign),
                        name=f"{self._tool_name}-doa",
                    )
                    self._doa.start()
        else:
            self._stop_doa()
        return self.turn_to_sound_status()

    def turn_to_sound_status(self) -> dict[str, Any]:
        """What the DoA turner hears and last did. Read-only.

        Returns:
            A success envelope: ``running``, ``enabled``, ``speech``,
            ``angle_deg``/``delta_deg`` of the last bearing, ``turns``,
            ``windups``, ``why`` the last frame did not turn, ``last_sent``
            (plan + the daemon's goto answer), and the rule constants.
        """
        if self._doa is None:
            payload: dict[str, Any] = {"running": False, "enabled": False, "why": "never enabled"}
        else:
            payload = self._doa.status()
        return {"status": "success", "content": [{"json": payload}]}

    def _stop_doa(self) -> None:
        """Stop the DoA loop if one is running. Idempotent; never raises.

        Holds ``_doa_lock`` across the stop so a concurrent
        :meth:`turn_to_sound` cannot install a fresh loop between this read of
        the slot and the stop of what it found.
        """
        with self._doa_lock:
            if self._doa is not None:
                self._doa.stop()

    def _doa_frame(self) -> dict[str, Any] | None:
        """One daemon state frame with DoA, head pose and body yaw, or ``None`` on a failed read."""
        frame = self._daemon_get(_PATH_STATE_DOA)
        if frame.get("error") is not None:
            return None
        return frame

    def _doa_look(
        self, *, yaw: float, body_yaw: float | None, pitch: float, roll: float, duration: float
    ) -> dict[str, Any]:
        """The turner's motion sink: one bounded :meth:`goto` in degrees."""
        return self.goto(head={"pitch": pitch, "roll": roll, "yaw": yaw}, body_yaw=body_yaw, duration=duration)

    def _face_lock_is_fresh(self, stamp: Any) -> bool:
        """Whether the face tracker's lock is recent enough to veto a DoA turn.

        The daemon's ``ts`` is stamped on the robot's OWN clock: a Wireless Mini
        reports an uptime (71708.0 while this host's epoch read 1.79e9), so
        neither ``time.time()`` nor this host's ``time.monotonic()`` is
        subtractable from it - doing so made every difference vastly larger than
        the window, and a live lock never vetoed anything.

        What is measurable here is how long ago THIS driver first saw that value,
        on one monotonic clock: a tracker still detecting hands back a new stamp
        each read, and a lock nobody refreshed keeps the stamp it had.

        Args:
            stamp: The daemon's ``face_target.ts``, whatever it sent.

        Returns:
            ``True`` while the lock counts as current - including when the daemon
            sends no usable stamp, because a reported detection is not made
            safer by being unstamped.
        """
        if not isinstance(stamp, int | float) or isinstance(stamp, bool):
            return True
        now = time.monotonic()
        with self._cache_lock:
            seen = self._face_stamp
            if seen is None or seen[0] != float(stamp):
                self._face_stamp = (float(stamp), now)
                return True
            first_seen = seen[1]
        return now - first_seen <= _FACE_FRESH_S

    def _doa_blocked(self) -> str | None:
        """Why a DoA turn must not happen right now, or ``None``. Two GETs, only when a turn is armed."""
        face = self._daemon_get(_PATH_TRACKING_FACE)
        target = face.get("face_target") if isinstance(face, dict) else None
        if isinstance(target, dict) and target.get("detected"):
            if self._face_lock_is_fresh(target.get("ts")):
                return "face tracker has a lock"
        running = self._daemon_get_list(_PATH_MOVES_RUNNING)
        if isinstance(running, list) and running:
            return "move in flight"
        return None

    def look_at(self, u: int, v: int, frame_width: int, frame_height: int, duration: float = 1.0) -> dict[str, Any]:
        """Turn the head toward a camera pixel.

        One verb, one meaning: the pixel is resolved through the daemon's own
        camera calibration and a fresh head pose (:meth:`_look_at_pose`, GETs
        only) into a 4x4 target with translation recentred at the origin, and
        that target is sent as a :meth:`goto`. The head pose is sampled
        separately from the frame the pixel came from. The move is bounded
        here the way every head move is: the
        target's yaw/pitch/roll are extracted and put through the shared
        envelope, so a pixel that asks for more pitch than the platform has is
        refused, not clamped. The move goes out as ``head_pose`` in
        ``roll/pitch/yaw`` form, so the daemon's IK solves it like any ``look``.

        Args:
            u: Pixel column in the unmodified camera frame.
            v: Pixel row.
            frame_width: Width of that frame.
            frame_height: Height of that frame.
            duration: Seconds for the interpolation.

        Returns:
            :meth:`goto`'s envelope, with the resolved geometry attached under
            ``plan`` and the bounded target in degrees under ``target_deg``, or
            a refusal from either step. ``motion_verified`` stays ``False``:
            the daemon accepted the move; nothing here proves the head arrived.
        """
        plan = self._look_at_pose(u, v, frame_width, frame_height)
        if plan.get("status") != "success":
            return plan
        geometry = plan["content"][0]["json"]
        matrix = geometry.get("head_pose")
        try:
            roll, pitch, yaw = _rpy_from_matrix(matrix)
        except (TypeError, ValueError, IndexError) as exc:
            return _refuse(f"look_at: the plan's head pose is not a usable 4x4 matrix: {exc}")
        sent = self.goto(head={"pitch": pitch, "roll": roll, "yaw": yaw}, duration=duration)
        if sent.get("status") != "success":
            return sent
        payload = dict(sent["content"][0]["json"])
        payload["plan"] = geometry
        payload["target_deg"] = {"roll": roll, "pitch": pitch, "yaw": yaw}
        return {"status": "success", "content": [{"json": payload}]}

    def state_snapshot(self) -> dict[str, Any]:
        """Return the cached sensor state: joints, pose, IMU, battery.

        A synchronous read off the caches the link thread fills - no daemon
        round-trip, so it is cheap enough to call before and after every
        motion. A ``None`` field means that stream has not delivered yet (IMU
        and battery stay ``None`` forever on a Lite, which has neither).

        Returns:
            A success envelope carrying the four snapshots, or a refusal when
            the driver was never connected (caches from a link that never ran
            would be indistinguishable from a robot reporting nothing).
        """
        if not self._connected:
            return _refuse("state_snapshot: not connected - call connect_eagerly() first")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "joints": self._snapshot("_joints"),
                        "pose": self._snapshot("_pose"),
                        "imu": self._snapshot("_imu"),
                        "battery": self._snapshot("_battery"),
                    }
                }
            ],
        }

    # ------------------------------------------------------------------ #
    # Link callbacks. Each runs on the loop thread; keep fast and pure.  #
    # ------------------------------------------------------------------ #

    def _on_joints(self, payload: dict[str, Any]) -> None:
        """Cache joint positions, converting the link's radians to degrees.

        Args:
            payload: The link's joints message, carrying
                ``head_joint_positions`` and ``antennas_joint_positions`` in
                radians. Seven head values mean body yaw followed by six
                Stewart legs. Antennas are ordered [right, left].
        """
        try:
            head = [math.degrees(float(j)) for j in payload.get("head_joint_positions", [])]
            antennas = [math.degrees(float(j)) for j in payload.get("antennas_joint_positions", [])]
            with self._cache_lock:
                complete = len(head) in (6, 7) and len(antennas) == 2 and all(math.isfinite(v) for v in head + antennas)
                self._joints_received_at = time.monotonic() if complete else None
                self._joints = {
                    # The daemon's seven head motor values start with body yaw;
                    # only the remaining six are Stewart-platform legs. Older
                    # bridges carrying six legs have no body measurement.
                    "head_leg_deg": head[1:] if len(head) == 7 else head,
                    "body_yaw_deg": head[0] if len(head) == 7 else None,
                    "antennas_deg": antennas,
                    "t": time.time(),
                }
        except (TypeError, ValueError) as exc:
            with self._cache_lock:
                self._joints_received_at = None
            logger.debug("%s: joints decode failed: %s", self._tool_name, exc)

    def _on_imu(self, payload: dict[str, Any]) -> None:
        """Cache the head IMU, and derive :attr:`_pose` from its quaternion.

        The Mini's IMU is mounted in the head, so its quaternion *is* the head's
        orientation - a measurement, not a model. That is why ``_pose`` is
        derived here rather than from the six leg positions, which would need
        Stewart-platform forward kinematics this repo does not have.

        Args:
            payload: The link's IMU message, carrying ``accelerometer``,
                ``gyroscope``, ``quaternion`` and ``temperature``.
        """
        try:
            imu = {
                "accelerometer": payload.get("accelerometer"),
                "gyroscope": payload.get("gyroscope"),
                "quaternion": payload.get("quaternion"),
                "temperature": payload.get("temperature"),
                "t": time.time(),
            }
            pose: dict[str, Any] | None = None
            quaternion = payload.get("quaternion")
            if quaternion is not None:
                pose = {
                    "quat": list(quaternion),
                    "frame": "head",
                    "source": "imu",
                    "t": imu["t"],
                }
            with self._cache_lock:
                self._imu = imu
                if pose is not None:
                    self._pose = pose
        except (TypeError, ValueError) as exc:
            logger.debug("%s: imu decode failed: %s", self._tool_name, exc)

    def _absorb_status(self, status: dict[str, Any]) -> None:
        """Populate :attr:`_battery` from a daemon status payload, if it has one.

        Args:
            status: The decoded ``/api/daemon/status`` body.
        """
        for key in _BATTERY_KEYS:
            value = status.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                pct = float(value)
            except (TypeError, ValueError):
                continue
            with self._cache_lock:
                self._battery = {"pct": pct, "source": key, "t": time.time()}
            return

    # ------------------------------------------------------------------ #
    # Internal helpers.                                                  #
    # ------------------------------------------------------------------ #

    def _daemon_get(self, path: str) -> dict[str, Any]:
        """Call the daemon's REST API with GET.

        Args:
            path: Request path, one of this module's ``_PATH_*`` constants.

        Returns:
            The decoded body, or ``{"error": ...}`` - the shape
            :func:`~strands_robots.device_connect.reachy_transport.api` returns
            for every failure, which is why no call here needs a ``try``.

            A body that decodes to something other than a JSON object is
            reported as that same shape. ``api`` hands the decoded body back
            unreshaped, so a daemon - or an interposed proxy - answering with an
            array, a string or ``null`` reached every caller here typed as a
            mapping, and the ``result.get("error")`` each one opens with raised
            ``AttributeError`` out of a driver whose contract is to report a
            reason and stay usable. Judging the shape here is the rule
            :meth:`~strands_robots.device_connect.reachy_mini_driver.ReachyMiniDriver._transport_failure`
            states for this transport: the callers that require an object are
            the ones that judge it.
        """
        return self._daemon_get_at(self._host, self._api_port, path)

    def _daemon_get_at(self, host: str, port: int, path: str) -> dict[str, Any]:
        """GET ``path`` from an explicit daemon address - :meth:`_daemon_get` for a candidate.

        Args:
            host: Daemon host to dial.
            port: Daemon REST port.
            path: Request path, one of this module's ``_PATH_*`` constants.

        Returns:
            The decoded object, or ``{"error": ...}`` on the reasoning
            :meth:`_daemon_get` gives.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return {"error": transport}

        result = transport.api(host, port, path)
        if not isinstance(result, dict):
            return _body_shape_error("GET", path, "an object", result)
        return result

    def _daemon_get_list(self, path: str) -> list[Any] | dict[str, Any]:
        """Call a daemon REST endpoint whose success body is a JSON array.

        ``/api/move/recorded-move-datasets/list/{dataset}`` is declared
        ``-> list[str]`` by the daemon, and
        :func:`~strands_robots.device_connect.reachy_transport.api` hands the
        decoded body back unreshaped, so a successful catalogue read is a
        ``list`` while every failure is still the ``{"error": ...}`` dict.

        Kept separate from :meth:`_daemon_get` rather than widening that
        return type, so the union stays confined to the one endpoint that can
        answer with an array and the dict-shaped callers keep narrowing for
        free. Reading an error key off a catalogue is then a type error at
        check time rather than an ``AttributeError`` on the happy path.

        Args:
            path: Request path, one of this module's ``_PATH_*`` constants.

        Returns:
            The decoded array on success, or ``{"error": ...}``. A body that is
            neither - a scalar, or an object that is not the transport's failure
            envelope - is reported as the shape it arrived as, which is what
            makes ":meth:`list_moves`' only dict is the transport's error
            envelope" true by construction rather than by assumption. Left
            unjudged, a scalar body was returned to the caller as the
            catalogue.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return {"error": transport}

        result = transport.api(self._host, self._api_port, path)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "error" in result:
            return result
        return _body_shape_error("GET", path, "an array", result)

    def _daemon_post(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call the daemon's REST API with POST.

        Args:
            path: Request path, one of this module's ``_PATH_*`` constants.
            data: JSON body, or ``None``.

        Returns:
            The decoded body, or ``{"error": ...}`` - including for a body that
            decodes to something other than a JSON object, on the reasoning
            :meth:`_daemon_get` gives.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return {"error": transport}

        result = transport.api(self._host, self._api_port, path, method="POST", data=data)
        if not isinstance(result, dict):
            return _body_shape_error("POST", path, "an object", result)
        return result

    def capture_frame(self, save_path: str = "") -> dict[str, Any]:
        """Save one fresh camera JPEG through native LAN WebRTC, without taking media ownership.

        The Python capture polling budget is ten seconds, followed by a
        three-second teardown verification wait. Native calls are not preempted;
        unconfirmed cleanup refuses data but may leave the receiver active.
        Negotiated audio is discarded; nothing is played or commanded.
        Requires PyGObject/GStreamer rswebrtc.
        Authenticated/TLS daemon configurations are refused because this media
        signaller cannot forward that authentication, never downgraded silently.

        Args:
            save_path: New JPEG path; empty creates a private temporary file.
                Existing files and symlinks are never overwritten.

        Returns:
            A success envelope containing path, dimensions and source, or a refusal.
        """
        if not self._connected:
            return _refuse("capture_frame: not connected - call connect_eagerly() first")
        if not isinstance(save_path, str):
            return _refuse("capture_frame: save_path must be a string")
        transport = _resolve_transport()
        if isinstance(transport, str):
            return _refuse(f"capture_frame: {transport}")
        if transport._daemon_auth_token() or transport._daemon_use_tls():
            return _refuse(
                "capture_frame: authenticated/TLS media signaling is not supported; daemon credentials were not forwarded"
            )
        transport._warn_unauthenticated_once("media signaling")
        try:
            from strands_robots.drivers.reachy_media import _capture_jpeg, _save_jpeg

            result = _save_jpeg(_capture_jpeg(self._host, self._media_port), save_path)
        except Exception as exc:  # noqa: BLE001 - GI/plugins and image decoders expose vendor exception types
            return _refuse(f"capture_frame: {exc}")
        return {"status": "success", "content": [{"json": result}]}

    def record_audio(self, duration: float = 1.0, save_path: str = "") -> dict[str, Any]:
        """Record a bounded microphone WAV through native LAN WebRTC, without playback.

        Receives mono 16 kHz signed 16-bit PCM after GStreamer conversion.
        Decoder-reported damage, malformed buffers, stream errors and timeouts
        refuse without saving. Receiver-clock adjustments are reported separately
        from sample-derived duration; lossless transport is not verified.
        Negotiated camera frames are discarded. The Python startup/capture
        polling budget is ten seconds plus duration, followed by teardown
        verification as in :meth:`capture_frame`; native calls are not preempted.
        Requires the same trusted-LAN media setup as :meth:`capture_frame`.

        Args:
            duration: Seconds to record, finite and between 0.1 and 5 inclusive.
            save_path: New WAV path; empty creates a private temporary file.

        Returns:
            Path and sample-derived recording metadata, or a refusal envelope.
        """
        if not self._connected:
            return _refuse("record_audio: not connected - call connect_eagerly() first")
        if reason := finite_number_error(duration, "duration", "record_audio"):
            return _refuse(reason)
        if not 0.1 <= duration <= 5:
            return _refuse("record_audio: duration must be between 0.1 and 5 seconds")
        if not isinstance(save_path, str):
            return _refuse("record_audio: save_path must be a string")
        transport = _resolve_transport()
        if isinstance(transport, str):
            return _refuse(f"record_audio: {transport}")
        if transport._daemon_auth_token() or transport._daemon_use_tls():
            return _refuse(
                "record_audio: authenticated/TLS media signaling is not supported; daemon credentials were not forwarded"
            )
        transport._warn_unauthenticated_once("media signaling")
        try:
            from strands_robots.drivers.reachy_media import _capture_pcm, _save_wav

            pcm, quality = _capture_pcm(self._host, self._media_port, duration)
            if len(pcm) != round(duration * 16000) * 2:
                return _refuse("record_audio: received sample count does not match the requested duration")
            result = _save_wav(pcm, save_path)
            result["quality"] = quality
        except Exception as exc:  # noqa: BLE001 - GI/plugins expose vendor exception types
            return _refuse(f"record_audio: {exc}")
        return {"status": "success", "content": [{"json": result}]}

    def _look_at_pose(self, u: int, v: int, frame_width: int, frame_height: int) -> dict[str, Any]:
        """Resolve a camera pixel to a head-pose target using GETs only; never command the head.

        The private half of :meth:`look_at`. Uses daemon calibration/crop
        metadata and a fresh head pose, but the frame and pose are not
        time-synchronized. The returned target recenters translation, like the
        vendor geometry, and is NOT safety-validated here - :meth:`look_at`
        puts it through the shared envelope before sending. Unknown camera
        models/resolutions or invalid geometry refuse rather than guessing.

        Args:
            u: Pixel column, starting at zero on the left.
            v: Pixel row, starting at zero at the top.
            frame_width: Width of the unmodified source camera frame.
            frame_height: Height of the unmodified source camera frame.

        Returns:
            The geometry-only plan (``head_pose`` 4x4, calibration facts,
            ``safety_validated: false``), or a refusal. Never sends a motor,
            stop or media-ownership command.
        """
        if not self._connected:
            return _refuse("look_at: not connected - call connect_eagerly() first")
        from strands_robots.drivers.reachy_look_at import _coordinates_error, _pixel_plan

        if reason := _coordinates_error(u, v, frame_width, frame_height):
            return _refuse(reason)
        specs = self._daemon_get("/api/camera/specs")
        if "error" in specs:
            return _refuse(f"look_at: {specs['error']}")
        pose = self._daemon_get("/api/state/present_head_pose?use_pose_matrix=true")
        if "error" in pose:
            return _refuse(f"look_at: {pose['error']}")
        try:
            plan = _pixel_plan(specs, pose, u, v, frame_width, frame_height)
        except Exception as exc:  # noqa: BLE001 - geometry and OpenCV failures become explicit refusals
            return _refuse(f"look_at: {exc}")
        return {"status": "success", "content": [{"json": plan}]}

    def _send_cmd(self, command: dict[str, Any]) -> str | None:
        """Put one real-time command on the link.

        Args:
            command: The link-level command dict.

        Returns:
            ``None`` on success, or a reason naming the failure.
        """
        if self._link is None or self._loop is None:
            return "link is not running"
        try:
            asyncio.run_coroutine_threadsafe(self._link.send_cmd(command), self._loop).result(timeout=5)
        except Exception as exc:  # noqa: BLE001 - a link can fail in any way
            return f"link refused the command: {exc}"
        return None

    def _remember_head_yaw_target(self, yaw: float | None) -> None:
        """Record the head yaw the daemon is targeting, or ``None`` for unknown.

        Called with a value after :meth:`send_action` puts a head pose on the
        wire, and with ``None`` by every path that moves the head without one:
        :meth:`play_move`, :meth:`wake_up` and :meth:`goto_sleep` hand the
        daemon a whole choreography, :meth:`set_motors` lets it re-pin its own
        target to wherever the head physically is when torque comes back, and
        :meth:`cleanup` ends the session that knew. Forgetting is the safe
        direction: an unknown target skips the coupling check, where a stale one
        would refuse a turn the robot could make.

        Args:
            yaw: Degrees, or ``None`` to mark the target unknown.
        """
        with self._cache_lock:
            self._head_yaw_target = yaw

    def _read_head_yaw_target(self) -> float | None:
        """The head yaw the daemon is targeting, or ``None`` when unknown."""
        with self._cache_lock:
            return self._head_yaw_target

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        """Return a copy of one cached sensor dict, or ``None``.

        Args:
            attr: Cache attribute name, e.g. ``"_imu"``.

        Returns:
            A shallow copy, so a caller who mutates the result does not corrupt
            the cache the link thread writes into.
        """
        with self._cache_lock:
            value: dict[str, Any] | None = getattr(self, attr, None)
            return None if value is None else dict(value)


#: Every key :func:`_wire_commands` understands. Surfaced so a refusal can name
#: the accepted set rather than leaving a caller to read the source.
def _post_json(url: str, data: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """POST ``data`` as JSON to ``url`` and decode the object it answers with.

    The one HTTP call in this module that does not go through the daemon
    transport, because the speech service is not the daemon: it has no token,
    no TLS switch and no variant, so the transport's headers would be wrong for
    it. Same no-raise contract as the transport's ``api`` - every failure comes
    back as ``{"error": ...}``.

    Args:
        url: Full URL.
        data: JSON body.
        timeout_s: Socket timeout.

    Returns:
        The decoded object, or ``{"error": ...}``.
    """
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - caller-configured http(s) URL
            body = json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}"}
    except Exception as exc:  # noqa: BLE001 - every transport failure is one reason
        return {"error": str(exc)}
    if not isinstance(body, dict):
        return _body_shape_error("POST", url, "an object", body)
    return body


def _rpy_from_matrix(matrix: Any) -> tuple[float, float, float]:
    """Roll, pitch, yaw in degrees from a 4x4 (or 3x3) rotation, ZYX convention.

    The convention the daemon's ``XYZRPYPose.from_pose_array`` uses, so a
    matrix the planner produced round-trips to the ``roll/pitch/yaw`` form the
    ``goto`` endpoint accepts.

    Args:
        matrix: Nested lists, at least 3x3.

    Returns:
        ``(roll, pitch, yaw)`` in degrees.

    Raises:
        TypeError, ValueError, IndexError: For anything that is not a numeric
            3x3-or-larger nested list.
    """
    r00, r01, r02 = (float(matrix[0][i]) for i in range(3))
    r10, r11, r12 = (float(matrix[1][i]) for i in range(3))
    r20, r21, r22 = (float(matrix[2][i]) for i in range(3))
    del r01, r02, r12
    sy = math.hypot(r00, r10)
    if sy < 1e-9:
        # Gimbal lock: pitch is +/-90 deg; roll is taken as zero and yaw
        # absorbs the remaining rotation, the standard convention.
        pitch = math.degrees(math.atan2(-r20, sy))
        roll = 0.0
        yaw = math.degrees(math.atan2(-r11, r21)) if r20 < 0 else math.degrees(math.atan2(r11, -r21))
    else:
        roll = math.degrees(math.atan2(r21, r22))
        pitch = math.degrees(math.atan2(-r20, sy))
        yaw = math.degrees(math.atan2(r10, r00))
    for value in (roll, pitch, yaw):
        if not math.isfinite(value):
            raise ValueError("rotation produced a non-finite angle")
    return roll, pitch, yaw


def _optional_pair(params: dict[str, Any], right: str, left: str, context: str) -> tuple[float, float] | None | str:
    """Read an antenna pair: both keys, neither, or a reason.

    Args:
        params: The agent's parameters.
        right: Key of the right antenna.
        left: Key of the left antenna.
        context: Verb to quote.

    Returns:
        ``(right, left)``, ``None`` when neither is present, or a reason when
        only one is - the daemon moves both antennas as a pair, so one value
        alone would command the other to zero without the caller asking.
    """
    has_right, has_left = right in params, left in params
    if has_right != has_left:
        return f"{context}: {right} and {left} go together - pass both or neither"
    if not has_right:
        return None
    return params[right], params[left]


def _act_sensors(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return {
        "status": "success",
        "content": [
            {
                "json": {
                    "imu": driver._snapshot("_imu"),
                    "pose": driver._snapshot("_pose"),
                    "battery": driver._snapshot("_battery"),
                    "joints": driver._snapshot("_joints"),
                }
            }
        ],
    }


async def _act_status(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    # The ``status`` verb carries the whole ``get_status`` envelope - the
    # shape the mesh publishes - inside one json block, as every native
    # driver's status verb does.
    del params
    return {"status": "success", "content": [{"json": await driver.get_status()}]}


def _act_stop(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    # Kept in the table so ``enum == list(_ACTIONS)`` holds; ``stream``
    # dispatches the halt on its own branch (before any connect) and returns
    # ``stop_task``'s verdict directly.
    del params
    return driver.stop_task()


def _act_look(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    pair = _optional_pair(params, "antenna_right", "antenna_left", "look")
    if isinstance(pair, str):
        return _refuse(pair)
    head = {axis: params.get(axis, 0.0) for axis in ("pitch", "roll", "yaw", "x", "y", "z")}
    return driver.goto(
        head=head,
        body_yaw=params.get("body_yaw"),
        antennas=pair,
        duration=params.get("duration", 0.8),
        interpolation=params.get("interpolation", "minjerk"),
    )


def _act_antennas(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    pair = _optional_pair(params, "antenna_right", "antenna_left", "antennas")
    if isinstance(pair, str):
        return _refuse(pair)
    if pair is None:
        return _refuse("antennas: antenna_right and antenna_left (degrees) are required")
    return driver.goto(
        antennas=pair, duration=params.get("duration", 0.5), interpolation=params.get("interpolation", "minjerk")
    )


def _act_body_turn(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    if "body_yaw" not in params:
        return _refuse("body_turn: body_yaw (degrees, +/-160) is required")
    return driver.goto(
        body_yaw=params["body_yaw"],
        duration=params.get("duration", 1.0),
        interpolation=params.get("interpolation", "minjerk"),
    )


def _act_home(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.home(duration=params.get("duration", 1.0))


def _act_wake(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver.wake_up()


def _act_sleep(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver.goto_sleep()


def _act_express(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    emotion = params.get("emotion")
    if not isinstance(emotion, str) or not emotion.strip():
        return _refuse("express: emotion (a move name or a plain word such as happy, curious, yes, no) is required")
    library = params.get("library", "emotions")
    if not isinstance(library, str):
        return _refuse(f"express: library must be a string, got {library!r}")
    return driver.play_move(emotion, library)


def _act_list_moves(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    library = params.get("library", "emotions")
    if not isinstance(library, str):
        return _refuse(f"list_moves: library must be a string, got {library!r}")
    return driver.list_moves(library)


def _act_motors(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    # No mode is the question, not a malformed command: one verb reads the
    # torque state and writes it, as volume/set_volume and
    # tracking_status/track_face pair a read with its write.
    mode = params.get("mode")
    if mode is None:
        return driver.read_motors()
    if not isinstance(mode, str):
        return _refuse(f"motors: mode must be one of {list(_MOTOR_MODES)}, got {mode!r}")
    return driver.set_motors(mode)


def _act_say(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.say(params.get("text", ""), wobble=params.get("wobble", False))


def _act_play_sound(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.play_sound(params.get("sound_file", ""), wobble=params.get("wobble", False))


def _act_volume(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver.get_volume()


def _act_set_volume(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    if "level" not in params:
        return _refuse("set_volume: level (0-100, or silent/low/normal/loud/max/quieter/louder) is required")
    return driver.set_volume(params["level"], allow_test_sound=params.get("allow_test_sound", False))


def _act_track_face(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.set_tracking(params.get("enabled", True), weight=params.get("weight", 1.0))


def _act_tracking_status(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver.tracked_face()


def _act_camera(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.capture_frame(params.get("save_path", ""))


def _act_record_audio(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.record_audio(params.get("duration", 1.0), params.get("save_path", ""))


def _act_look_at(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.look_at(
        params.get("u", -1),
        params.get("v", -1),
        params.get("frame_width", 0),
        params.get("frame_height", 0),
        duration=params.get("duration", 1.0),
    )


def _act_turn_to_sound(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    return driver.turn_to_sound(params.get("enabled", True), sign=params.get("sign", 1.0))


def _act_turn_to_sound_status(driver: ReachyDriver, params: dict[str, Any]) -> dict[str, Any]:
    del params
    return driver.turn_to_sound_status()


#: Agent verb -> handler. Order is the order the ``enum`` lists them; every
#: handler takes the driver and the agent's remaining parameters and returns
#: one envelope, so :meth:`ReachyDriver.stream` is a table lookup and the
#: table is what a test grades against the spec.
_ACTIONS: dict[str, Any] = {
    "sensors": _act_sensors,
    "get_state": _act_sensors,
    "status": _act_status,
    "stop": _act_stop,
    "look": _act_look,
    "antennas": _act_antennas,
    "body_turn": _act_body_turn,
    "home": _act_home,
    "wake": _act_wake,
    "sleep": _act_sleep,
    "express": _act_express,
    "list_moves": _act_list_moves,
    "motors": _act_motors,
    "say": _act_say,
    "play_sound": _act_play_sound,
    "volume": _act_volume,
    "set_volume": _act_set_volume,
    "track_face": _act_track_face,
    "tracking_status": _act_tracking_status,
    "camera": _act_camera,
    "record_audio": _act_record_audio,
    "look_at": _act_look_at,
    "turn_to_sound": _act_turn_to_sound,
    "turn_to_sound_status": _act_turn_to_sound_status,
}


_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "head_pitch",
        "head_roll",
        "head_yaw",
        "head_x",
        "head_y",
        "head_z",
        "body_yaw",
        "antenna_left",
        "antenna_right",
    }
)

#: The head pose is one command built from up to six keys, so the presence of
#: any of them means a head command is wanted.
_HEAD_KEYS: tuple[str, ...] = ("head_pitch", "head_roll", "head_yaw", "head_x", "head_y", "head_z")


def _head_yaw_of(action: dict[str, Any]) -> float | None:
    """The head yaw, in degrees, ``action`` puts on the wire - ``None`` for no head pose.

    The daemon's head command is a whole pose, so naming any one head key
    commands all six: an action that moves the pitch and says nothing about the
    yaw commands the yaw to zero. That makes the head yaw of a head-bearing
    action knowable exactly, which is what the coupling limit needs, and it is
    why this returns ``0.0`` rather than ``None`` for ``{"head_pitch": 10}``.
    The single owner of that rule - :func:`_wire_commands` builds the pose from
    it, and :meth:`ReachyDriver.send_action` checks the coupling against it.

    Args:
        action: An action dict; see :meth:`ReachyDriver.send_action`.

    Returns:
        The commanded head yaw in degrees, or ``None`` when the action names no
        head axis and so leaves the head pose alone.
    """
    if not any(key in action for key in _HEAD_KEYS):
        return None
    return float(action.get("head_yaw", 0.0))


def _wire_commands(action: dict[str, Any]) -> list[dict[str, Any]] | str:
    """Translate a degrees-and-millimetres action into link commands.

    Three commands at most, because the daemon addresses the Mini's three
    movable groups separately: a 4x4 ``head_pose``, a scalar ``body_yaw`` in
    radians, and a two-element ``antennas_joint_positions`` in radians. Only the
    groups the action mentions are built, so commanding the antennas does not
    also re-send a head pose.

    Args:
        action: A validated action dict; see
            :meth:`ReachyDriver.send_action` for the accepted keys.

    Returns:
        The commands to send, in a fixed order - head, body, antennas - so two
        identical actions always produce the same sequence. Or a reason string
        if the transport module cannot be imported: the head pose is built by
        the transport's own ``rpy_to_pose``, so this is a refusal boundary like
        the daemon calls, and returning the reason keeps it out of the raising
        path even though ``send_action``'s connected gate should already have
        refused.
    """
    transport = _resolve_transport()
    if isinstance(transport, str):
        return transport

    commands: list[dict[str, Any]] = []
    head_yaw = _head_yaw_of(action)
    if head_yaw is not None:
        commands.append(
            {
                "head_pose": transport.rpy_to_pose(
                    float(action.get("head_pitch", 0.0)),
                    float(action.get("head_roll", 0.0)),
                    head_yaw,
                    float(action.get("head_x", 0.0)),
                    float(action.get("head_y", 0.0)),
                    float(action.get("head_z", 0.0)),
                )
            }
        )
    if "body_yaw" in action:
        commands.append({"body_yaw": math.radians(float(action["body_yaw"]))})
    if "antenna_left" in action or "antenna_right" in action:
        commands.append(
            {
                "antennas_joint_positions": [
                    math.radians(float(action.get("antenna_right", 0.0))),
                    math.radians(float(action.get("antenna_left", 0.0))),
                ]
            }
        )
    return commands


def _split_host_port(port: str | None, api_port: int) -> tuple[str, int]:
    """Split a ``host[:port]`` string into a host and a validated TCP port.

    ``port=`` is polymorphic across drivers by contract, and for a daemon-backed
    robot the natural spelling is the one an operator already types into a
    browser. Accepting both ``"host"`` and ``"host:8000"`` means a caller does
    not have to know which one this driver wanted.

    Args:
        port: The caller's ``port=``, or ``None`` for ``localhost``.
        api_port: Fallback port when ``port`` carries no suffix.

    Returns:
        ``(host, port)``.

    Raises:
        ValueError: If either the suffix or ``api_port`` is not a usable TCP
            port. Raised rather than returned because a driver with no
            addressable daemon has nothing to degrade to.
    """
    host = "localhost"
    resolved = api_port
    if port:
        text = str(port)
        head, separator, tail = text.rpartition(":")
        if separator and head and tail.isdigit():
            host, resolved = head, int(tail)
        elif separator and head:
            raise ValueError(
                f"ReachyDriver: port {text!r} names a host and a port, but {tail!r} is not a number - "
                'expected "host" or "host:8000"'
            )
        else:
            host = text
    if (reason := tcp_port_error(resolved, "api_port", "ReachyDriver")) is not None:
        raise ValueError(reason)
    return host, resolved


def _json_kind(body: Any) -> str:
    """Name the JSON type a decoded daemon body arrived as.

    Args:
        body: A value :func:`~strands_robots.device_connect.reachy_transport.api`
            handed back.

    Returns:
        The JSON type name - ``"object"``, ``"array"``, ``"string"``,
        ``"number"``, ``"boolean"`` or ``"null"`` - so a refusal names the shape
        in the daemon's own vocabulary rather than Python's. ``bool`` is tested
        before ``int`` because it is a subclass of it, and a stray non-JSON value
        falls back to its Python type name rather than being mislabelled.
    """
    if body is None:
        return "null"
    if isinstance(body, bool):
        return "boolean"
    if isinstance(body, int | float):
        return "number"
    if isinstance(body, str):
        return "string"
    if isinstance(body, list):
        return "array"
    if isinstance(body, dict):
        return "object"
    return type(body).__name__


def _body_shape_error(method: str, path: str, expected: str, body: Any) -> dict[str, Any]:
    """Report a daemon body that decoded to the wrong JSON shape.

    Args:
        method: HTTP method, so a reason names the call.
        path: Request path, one of this module's ``_PATH_*`` constants.
        expected: The shape the caller needs, worded for the message - e.g.
            ``"an object"``.
        body: What the daemon answered with instead.

    Returns:
        The ``{"error": ...}`` envelope every caller here already branches on,
        so a body of the wrong shape refuses by the same path as an unreachable
        daemon. The value is previewed rather than named by type alone: an
        interposed proxy's JSON error page and a daemon answering ``null`` are
        both "not an object", and only the preview tells them apart.
    """
    preview = repr(body)
    if len(preview) > _BODY_PREVIEW_CHARS:
        preview = preview[:_BODY_PREVIEW_CHARS] + "..."
    return {"error": f"{method} {path}: daemon answered a JSON {_json_kind(body)}, not {expected}: {preview}"}


def _refuse(reason: str) -> dict[str, Any]:
    """Return the driver's error envelope with ``reason`` inside.

    Args:
        reason: Text naming what refused.

    Returns:
        The error envelope, in the one shape every refusal path here renders.
    """
    return {"status": "error", "content": [{"text": reason}]}
