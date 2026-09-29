"""A live Foxglove view of any robot, on the telemetry seam the ROS 2 bridges already use.

:class:`FoxgloveBridge` implements the three-method duck interface both
engines publish through - ``publish_joint_states(robot, names, positions)``,
``publish_image(robot, key, frame)`` and ``shutdown()`` - so it slots into
``SimEngine._init_ros_bridge`` and the hardware ``Robot._init_ros_bridge``
without a change to the publish path. Every message is logged to the WebSocket
server (subprotocol ``foxglove.sdk.v1``, what Foxglove 3.x speaks) and, when
asked, to an MCAP file, so what an operator sees live is what lands on disk.
The two sinks sit in two ``foxglove.Context`` objects with one channel per
topic each: that is what lets the static scene go to the file exactly once
while every new live subscriber still receives it (a sink cannot be asked to
skip one message of a channel it accepted).

Topics::

    /tf                       FrameTransforms  every MuJoCo body, when an engine is attached
    /<robot>/scene            SceneUpdate      the robot's visual meshes, once, frame_locked
    /scene                    SceneUpdate      floor and task objects, once per recompile
    /<robot>/joint_states     JointStates      positions (+ velocities with an engine)
    /<robot>/camera/<name>    CompressedImage  jpeg q80, the mesh's own encoding
    /strands/log              Log              what the bridge and its gate said
    /strands/events           JSON             session, gate and service events

Read only by default. The server advertises no capability unless
``services=True``, and even then every call goes through the operator gate in
:mod:`strands_robots.foxglove.services`; ``clientPublish`` and ``parameters``
are never advertised.

Costs an operator should know: the static scene is a few MB (so101: 3.8 MB)
and is re-sent to each new subscriber, never on a timer; steady state is about
60 KB/s at 50 Hz transforms plus 4 Hz 320x240 images; two 640x480 cameras at
30 Hz write about 75 MiB per minute of MCAP.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

import numpy as np

from strands_robots.foxglove.options import PORT_SEARCH_WIDTH, FoxgloveOptions
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

#: JPEG quality used for camera frames, the mesh's own setting.
JPEG_QUALITY = 80

#: Defaults for the two rate limits, in Hz.
DEFAULT_STATE_HZ = 50.0
DEFAULT_CAMERA_HZ = 10.0

_EVENT_SCHEMA = {"type": "object", "title": "strands.Event"}


def _foxglove() -> Any:
    return require_optional(
        "foxglove",
        pip_install="foxglove-sdk",
        extra="foxglove",
        purpose="the live Foxglove view and MCAP recording (Robot(foxglove=True))",
    )


def encode_jpeg(frame: Any) -> bytes | None:
    """An ``(H, W, 3)`` RGB uint8 array as JPEG bytes, or ``None`` when it is not an image.

    Args:
        frame: The camera frame as the observation carries it.

    Returns:
        JPEG bytes at :data:`JPEG_QUALITY`, or ``None`` for a shape or dtype
        that is not an RGB image.
    """
    import cv2

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        return None
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def image_format(data: bytes) -> str | None:
    """``"jpeg"`` or ``"png"`` from the magic bytes, or ``None``."""
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


class _RateLimit:
    """``due()`` answers True at most ``hz`` times per second, keyed by name."""

    def __init__(self, hz: float) -> None:
        self.period = 1.0 / hz if hz > 0 else 0.0
        self._next: dict[str, float] = {}

    def due(self, key: str, now: float) -> bool:
        if now < self._next.get(key, 0.0):
            return False
        self._next[key] = now + self.period
        return True

    def any_due(self, now: float) -> bool:
        """True when at least one key is due, or none has been seen yet (checked, not consumed)."""
        return now >= min(self._next.values(), default=0.0)


class FoxgloveBridge:
    """Publish one robot's telemetry to Foxglove (live WebSocket and/or MCAP).

    Args:
        options: The resolved ``foxglove=`` keywords.
        name: Server name shown in Foxglove's connection list.
        engine: Optional MuJoCo engine exposing ``mj_model`` / ``mj_data`` and
            ``robot_joint_names``; with one, ``/tf`` and the mesh scene are
            published too. Real arms pass ``None`` and get joints and cameras.
        command_sink: ``callable(robot, {joint: position}) -> result dict`` used
            by the gated ``set_joint_positions`` service; ``None`` disables
            services regardless of ``options.services``.
        state_hz: Ceiling for joint state and ``/tf`` messages.
        camera_hz: Ceiling for camera frames; :meth:`wants_images` tells the
            engine when a render is worth paying for.

    Raises:
        ImportError: The ``[foxglove]`` extra is not installed.
        RuntimeError: No free port within :data:`PORT_SEARCH_WIDTH` of the requested one.
    """

    def __init__(
        self,
        options: FoxgloveOptions,
        *,
        name: str = "strands-robots",
        engine: Any = None,
        command_sink: Any = None,
        state_hz: float = DEFAULT_STATE_HZ,
        camera_hz: float = DEFAULT_CAMERA_HZ,
    ) -> None:
        _foxglove()
        import foxglove as fox

        self.options = options
        self.name = name
        self._engine = engine
        self._lock = threading.RLock()
        self._state_rate = _RateLimit(state_hz)
        self._camera_rate = _RateLimit(camera_hz)
        # Two sinks, two contexts: ``_live`` carries the server, ``_file`` the
        # MCAP writer. A topic has one channel per context (see the module
        # docstring for why the scene needs that split).
        self._live = fox.Context()
        self._file = fox.Context() if options.mcap is not None else None
        self._channels: dict[str, list[Any]] = {}
        self._scene_model_id: int | None = None
        self._scene_messages: dict[str, Any] = {}
        self._scene_digests: dict[str, str] = {}
        self._resend_scene = threading.Event()
        self._writer: Any = None
        self._server: Any = None
        self.closed = False
        self.frames_sent = 0
        self.mcap_path = str(options.mcap) if options.mcap is not None else None

        from foxglove import Capability

        services: list[Any] = []
        capabilities: list[Any] = []
        if options.services and command_sink is not None:
            from strands_robots.foxglove.services import build_services

            services = build_services(self, command_sink)
            capabilities = [Capability.Services]
        self.services_enabled = bool(services)

        if self._file is not None:
            self._writer = fox.open_mcap(str(options.mcap), context=self._file)
        self._server = self._start_server(fox, capabilities, services)
        self.host = options.host
        self.port = int(self._server.port)
        logger.info("Foxglove: %s is live at %s (mcap=%s)", name, self.url, self.mcap_path or "off")
        self.log("info", f"{name}: Foxglove server up at {self.url}")
        self.event(
            {
                "event": "session",
                "name": name,
                "url": self.url,
                "mcap": self.mcap_path,
                "services": self.services_enabled,
            }
        )

    # -- construction helpers -------------------------------------------------

    def _start_server(self, fox: Any, capabilities: list[Any], services: list[Any]) -> Any:
        listener = _Listener(self)
        last_error: Exception | None = None
        ports = (
            [self.options.port]
            if self.options.port == 0
            else range(self.options.port, self.options.port + PORT_SEARCH_WIDTH)
        )
        for port in ports:
            try:
                return fox.start_server(
                    name=self.name,
                    host=self.options.host,
                    port=port,
                    capabilities=capabilities or None,
                    supported_encodings=["json"] if services else None,
                    services=services or None,
                    context=self._live,
                    server_listener=listener,
                )
            except RuntimeError as exc:  # the SDK reports a busy port as RuntimeError
                if "in use" not in str(exc).lower():
                    self._close_writer()
                    raise
                last_error = exc
        self._close_writer()
        raise RuntimeError(
            f"Foxglove: no free port in {self.options.port}-{self.options.port + PORT_SEARCH_WIDTH - 1} on "
            f"{self.options.host} ({last_error}); pass foxglove='host:port' or ':0' for an ephemeral port."
        )

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # -- public surface --------------------------------------------------------

    @property
    def url(self) -> str:
        """The WebSocket URL Foxglove connects to."""
        return f"ws://{self.host}:{self.port}"

    @property
    def link(self) -> str:
        """A ``foxglove://`` deep link that opens the desktop app on this server."""
        return f"foxglove://open?ds=foxglove-websocket&ds.url={self.url}"

    @property
    def capabilities(self) -> list[str]:
        """What the server advertises: ``[]`` unless services are on."""
        return ["services"] if self.services_enabled else []

    def wants_images(self) -> bool:
        """True when a camera frame is due under the camera rate limit (checked, not consumed)."""
        return self._camera_rate.any_due(time.monotonic())

    def log(self, level: str, message: str, *, name: str = "strands") -> None:
        """Write one line to ``/strands/log``."""
        from foxglove.channels import LogChannel
        from foxglove.messages import Log, LogLevel

        levels = {"debug": LogLevel.Debug, "info": LogLevel.Info, "warning": LogLevel.Warning, "error": LogLevel.Error}
        with self._lock:
            if self.closed:
                return
            self._log(
                "/strands/log",
                LogChannel,
                Log(timestamp=self._stamp(), level=levels.get(level, LogLevel.Info), message=message, name=name),
            )

    def event(self, payload: dict[str, Any]) -> None:
        """Write one JSON object to ``/strands/events`` (``time`` is added)."""
        import foxglove

        def _json_channel(topic: str, *, context: Any) -> Any:
            return foxglove.Channel(topic, schema=_EVENT_SCHEMA, message_encoding="json", context=context)

        with self._lock:
            if self.closed:
                return
            self._log("/strands/events", _json_channel, {"time": time.time(), **payload})

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """One ``JointStates`` for ``robot``; with an engine, ``/tf`` and the scenes too."""
        now = time.monotonic()
        with self._lock:
            if self.closed or not self._state_rate.due(f"state:{robot}", now):
                return
            stamp_ns = time.time_ns()
            model = getattr(self._engine, "mj_model", None) if self._engine is not None else None
            data = getattr(self._engine, "mj_data", None) if self._engine is not None else None
            if model is not None and data is not None:
                self._publish_3d(robot, names, model, data, stamp_ns)
                return
            from foxglove.channels import JointStatesChannel
            from foxglove.messages import JointState, JointStates, Timestamp

            ts = Timestamp(sec=stamp_ns // 1_000_000_000, nsec=stamp_ns % 1_000_000_000)
            joints = [JointState(name=n, position=float(p)) for n, p in zip(names, positions, strict=False)]
            self._log(f"/{robot}/joint_states", JointStatesChannel, JointStates(timestamp=ts, joints=joints))

    def publish_image(self, robot: str, key: str, frame: Any) -> None:
        """One ``CompressedImage`` on ``/<robot>/camera/<key>``; arrays are JPEG-encoded, bytes pass through."""
        now = time.monotonic()
        with self._lock:
            if self.closed or not self._camera_rate.due(f"{robot}/{key}", now):
                return
            data: bytes | None
            fmt: str | None
            if isinstance(frame, (bytes, bytearray, memoryview)):
                data = bytes(frame)
                fmt = image_format(data)
            else:
                data = encode_jpeg(frame)
                fmt = "jpeg" if data is not None else None
            if data is None or fmt is None:
                return
            from foxglove.channels import CompressedImageChannel
            from foxglove.messages import CompressedImage

            self._log(
                f"/{robot}/camera/{key}",
                CompressedImageChannel,
                CompressedImage(timestamp=self._stamp(), frame_id=f"{robot}/camera/{key}", data=data, format=fmt),
            )
            self.frames_sent += 1

    def shutdown(self) -> None:
        """Stop the server and close the MCAP file. Safe to call twice."""
        with self._lock:
            if self.closed:
                return
            self.log("info", f"{self.name}: Foxglove server stopping")
            self.closed = True
            server, self._server = self._server, None
        if server is not None:
            server.stop()
        self._close_writer()

    # -- internals ----------------------------------------------------------------

    def _stamp(self) -> Any:
        from foxglove.messages import Timestamp

        return Timestamp.from_epoch_secs(time.time())

    def _channels_for(self, topic: str, factory: Any) -> list[Any]:
        """The channel pair for ``topic``: ``[live, file]`` (``[live]`` without an MCAP), built on first use."""
        channels = self._channels.get(topic)
        if channels is None:
            contexts = [self._live] + ([self._file] if self._file is not None else [])
            channels = self._channels[topic] = [factory(topic, context=context) for context in contexts]
        return channels

    def _log(self, topic: str, factory: Any, message: Any) -> None:
        """Log ``message`` on ``topic`` to every sink."""
        for channel in self._channels_for(topic, factory):
            channel.log(message)

    def _log_live(self, topic: str, factory: Any, message: Any) -> None:
        """Log ``message`` on ``topic`` to the live server only (a re-send the file already holds)."""
        self._channels_for(topic, factory)[0].log(message)

    def _publish_3d(self, robot: str, names: list[str], model: Any, data: Any, stamp_ns: int) -> None:
        from foxglove.channels import FrameTransformsChannel, JointStatesChannel, SceneUpdateChannel

        from strands_robots.foxglove import scene as scene_mod

        if self._scene_model_id != id(model):
            # A recompile (add_object, add_robot, ...) is a new model: rebuild
            # every scene once and send it; the file filter keeps one copy.
            self._scene_model_id = id(model)
            self._scene_messages = {}
            robots = sorted(
                {r for b in range(1, model.nbody) if (r := scene_mod.robot_of_body(scene_mod.body_name(model, b)))}
            )
            for r in robots:
                if (message := scene_mod.scene_update(model, stamp_ns, robot=r)) is not None:
                    self._scene_messages[f"/{r}/scene"] = message
            if (world := scene_mod.scene_update(model, stamp_ns, robot=None)) is not None:
                self._scene_messages["/scene"] = world
            self._resend_scene.clear()
            for topic, message in self._scene_messages.items():
                # A recompile that left a robot's meshes untouched (an added
                # cube) produces the same bytes: the file keeps one copy.
                digest = hashlib.sha1(message.encode(), usedforsecurity=False).hexdigest()
                if self._scene_digests.get(topic) == digest:
                    self._log_live(topic, SceneUpdateChannel, message)
                else:
                    self._scene_digests[topic] = digest
                    self._log(topic, SceneUpdateChannel, message)
        elif self._resend_scene.is_set():
            # A new subscriber: the file already holds this scene, so only the
            # live sink gets the copy.
            self._resend_scene.clear()
            for topic, message in self._scene_messages.items():
                self._log_live(topic, SceneUpdateChannel, message)
        self._log("/tf", FrameTransformsChannel, scene_mod.frame_transforms(model, data, stamp_ns))
        self._log(
            f"/{robot}/joint_states",
            JointStatesChannel,
            scene_mod.joint_states(model, data, names, stamp_ns, robot=robot),
        )

    def _on_subscribe(self, topic: str) -> None:
        """A new subscriber to a scene topic gets the static scene again (from the publish thread)."""
        if topic.endswith("/scene"):
            self._resend_scene.set()

    def __repr__(self) -> str:
        try:
            return f"FoxgloveBridge(url={self.url!r}, mcap={self.mcap_path!r}, services={self.services_enabled})"
        except AttributeError:
            from strands_robots.utils import partial_construction_repr

            return partial_construction_repr(self)


class _Listener:
    """The ``ServerListener`` half: only ``on_subscribe`` does anything."""

    def __init__(self, bridge: FoxgloveBridge) -> None:
        self._bridge = bridge

    def on_subscribe(self, client: Any, channel: Any) -> None:
        self._bridge._on_subscribe(channel.topic)

    def on_unsubscribe(self, client: Any, channel: Any) -> None:
        return None

    def on_client_advertise(self, client: Any, channel: Any) -> None:
        return None

    def on_client_unadvertise(self, client: Any, client_channel_id: int) -> None:
        return None

    def on_message_data(self, client: Any, client_channel_id: int, data: bytes) -> None:
        return None

    def on_get_parameters(self, client: Any, param_names: list[str], request_id: str | None = None) -> list[Any]:
        return []

    def on_set_parameters(self, client: Any, parameters: list[Any], request_id: str | None = None) -> list[Any]:
        return []

    def on_parameters_subscribe(self, param_names: list[str]) -> None:
        return None

    def on_parameters_unsubscribe(self, param_names: list[str]) -> None:
        return None

    def on_connection_graph_subscribe(self) -> None:
        return None

    def on_connection_graph_unsubscribe(self) -> None:
        return None
