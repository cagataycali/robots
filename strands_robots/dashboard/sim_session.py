"""A simulated robot stepping in this process, owned by one thread.

``SimSession`` wraps the engine ``Robot(name, mode="sim")`` returns. One worker
thread creates the engine, steps it in real time, renders frames and applies
queued commands; every other thread reads an immutable snapshot. This is not a
style choice: the renderer's GL context is bound to the thread that made it, and
MuJoCo's ``MjData`` is not safe to step and read from two threads at once.

Freezing is the e-stop. A frozen session stops stepping but keeps rendering
and keeps answering telemetry, so an operator sees the robot exactly where it
stopped. A queued command that would move the robot is refused when the worker
reaches it rather than applied - the refusal is that command's own result, so
the caller is told, never silently dropped. A command the worker was already
inside when the freeze landed cannot be recalled: that write completes, and the
route answers the request 423 with the lockout still latched.

A session can also be a MIRROR: given a ``source`` (see
:mod:`strands_robots.dashboard.mirror`), the worker never steps physics. Each
tick it takes the source's joint angles, writes them into ``qpos`` and runs
the kinematics, so the render and the twin show the real arm where it is. A
mirror refuses ``set_joints`` and ``reset`` - the arm decides the pose - and a
source that stops answering, or a pose the model refuses, shows as ``stale`` or
``error`` with the reason rather than as a frozen pose reported as healthy.

Nothing here knows about the mesh, hardware, or HTTP.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_SESSIONS = 4

#: Commands that move the robot, and so are refused while a session is frozen.
#: A read (``state``) is still answered, because answering it moves nothing.
MOTION_COMMANDS = frozenset({"reset", "set_joints", "step"})
_RENDER_FPS = 15.0
_RENDER_SIZE = (512, 384)  # width, height: a live view, not a dataset frame
_TELEMETRY_HZ = 30.0


@dataclass(frozen=True)
class Snapshot:
    """What every reader gets: the sim as of one instant, on this dashboard's clock."""

    id: str
    robot: str
    state: str  # starting | running | mirroring | stale | refused | frozen | error | stopped
    sim_time: float
    steps: int
    joint_names: tuple[str, ...]
    qpos: tuple[float, ...]
    fps: float
    cameras: tuple[str, ...]
    error: str | None = None
    model_path: str | None = None
    created: float = 0.0
    #: Every geom's world pose as little-endian float32 ``[x y z | 3x3 row-major]``
    #: rows, ``ngeom`` of them - what the browser twin needs, in the bytes it
    #: needs, so no route re-encodes it per client. Empty until the engine exists.
    poses: bytes = b""
    #: ``"sim"`` for physics, or ``"real:<port>"`` when a source drives the pose.
    source: str = "sim"
    #: The source's health (port, hz, age, raw ticks, error), or None for physics.
    bus: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The snapshot as JSON-ready fields, floats rounded for the wire."""
        return {
            "id": self.id,
            "robot": self.robot,
            "state": self.state,
            "sim_time": round(self.sim_time, 4),
            "steps": self.steps,
            "joint_names": list(self.joint_names),
            "qpos": [round(q, 5) for q in self.qpos],
            "fps": round(self.fps, 1),
            "cameras": list(self.cameras),
            "error": self.error,
            "model_path": self.model_path,
            "created": self.created,
            "source": self.source,
            "bus": self.bus,
        }


@dataclass
class _Command:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


class SimSession:
    """One simulated robot in one worker thread. See the module docstring."""

    def __init__(
        self,
        robot: str,
        *,
        engine_factory: Callable[[str], Any] | None = None,
        realtime: bool = True,
        source: Any | None = None,
    ):
        self.id = uuid.uuid4().hex[:8]
        self.robot = robot
        self._factory = engine_factory or _default_factory
        self._realtime = realtime
        self._source = source
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._frozen = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._engine: Any | None = None
        self._snapshot = Snapshot(
            self.id,
            robot,
            "starting",
            0.0,
            0,
            (),
            (),
            0.0,
            (),
            created=time.time(),
            source="sim" if source is None else f"real:{source.port}",
        )
        self._thread = threading.Thread(target=self._run, name=f"sim-{robot}-{self.id}", daemon=True)
        self._thread.start()

    # -- readers (any thread) ------------------------------------------------

    @property
    def snapshot(self) -> Snapshot:
        """The latest immutable snapshot."""
        with self._lock:
            return self._snapshot

    @property
    def model(self) -> Any | None:
        """The compiled ``MjModel`` (immutable after load; safe to read from any thread), or None."""
        engine = self._engine
        return getattr(engine, "mj_model", None) if engine is not None else None

    def latest_frame(self) -> np.ndarray | None:
        """The most recent rendered RGB frame, for :func:`mjpeg_frames`."""
        with self._lock:
            return self._frame

    @property
    def frozen(self) -> bool:
        """Whether the e-stop has this session stopped."""
        return self._frozen.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the engine exists and rendered once (or failed). Tests and the create route use it."""
        return self._ready.wait(timeout)

    # -- controls (any thread) -----------------------------------------------

    def freeze(self) -> None:
        """Stop stepping; keep rendering and answering. The e-stop."""
        self._frozen.set()
        self._publish(state="frozen")

    def thaw(self) -> None:
        """Resume stepping after :meth:`freeze`."""
        self._frozen.clear()
        self._publish(state="running" if self._source is None else "mirroring")

    def stop(self, timeout: float = 5.0) -> None:
        """End the worker and close the engine."""
        self._stop.set()
        self._thread.join(timeout)
        self._publish(state="stopped")

    def command(self, kind: str, timeout: float = 5.0, **payload: Any) -> dict[str, Any]:
        """Run one engine call on the worker thread and return its result.

        Raises:
            RuntimeError: when the session is not running, or the call timed out.
        """
        if self._stop.is_set() or self.snapshot.state == "error":
            raise RuntimeError(f"session {self.id} is not running")
        cmd = _Command(kind, payload)
        self._commands.put(cmd)
        if not cmd.done.wait(timeout):
            raise RuntimeError(f"{kind} timed out after {timeout:.0f}s")
        assert cmd.result is not None
        return cmd.result

    # -- worker --------------------------------------------------------------

    def _publish(self, **changes: Any) -> None:
        with self._lock:
            current = self._snapshot.__dict__ | changes
            self._snapshot = Snapshot(**current)

    def _run(self) -> None:
        # The source is opened before the worker starts and outlives every engine
        # attempt, so it is released here and not beside the engine: a factory
        # that refuses, a renderer that fails, a loop that dies and a stop all
        # leave through this frame, and a mirror's port is held exclusively
        # until they do.
        try:
            self._serve()
        finally:
            if self._source is not None:
                self._source.close()

    def _serve(self) -> None:
        try:
            engine = self._factory(self.robot)
        except Exception as exc:
            logger.warning("sim %s (%s) failed to start: %s", self.id, self.robot, exc)
            self._publish(state="error", error=f"{type(exc).__name__}: {exc}")
            self._ready.set()
            return

        self._engine = engine
        names = tuple(str(n) for n in engine.robot_joint_names(self.robot))
        cameras = tuple(engine.list_cameras())
        dt = float(engine.mj_model.opt.timestep)
        self._publish(
            # An e-stop that arrived while the engine was building already set
            # the flag; this first publish reports that, not "running".
            state="frozen" if self._frozen.is_set() else ("running" if self._source is None else "mirroring"),
            joint_names=names,
            cameras=cameras,
            model_path=_model_path(self.robot),
        )
        # The first render builds the GL context, which takes longer than the
        # engine itself (0.7 s here, more under software GL). Ready means the
        # session steps AND renders, so that cost is paid before the create
        # route answers and a machine with no renderer reports ``error`` there,
        # not as a session that never shows a frame.
        try:
            rgb, _ = engine.get_frame(width=_RENDER_SIZE[0], height=_RENDER_SIZE[1])
        except Exception as exc:
            logger.warning("sim %s (%s) has no renderer: %s", self.id, self.robot, exc)
            self._publish(state="error", error=f"{type(exc).__name__}: {exc}")
            self._ready.set()
            self._close(engine)
            return
        with self._lock:
            self._frame = rgb
        self._ready.set()

        last_render = time.monotonic()
        last_wall = last_render
        frames = 0
        fps = 0.0
        fps_window = time.monotonic()
        steps = 0
        try:
            while not self._stop.is_set():
                self._drain(engine)
                now = time.monotonic()
                mirrored: str | None = None
                mirror_error: str | None = None
                if self._source is not None:
                    # A mirror never steps: the arm decides, the kinematics follow.
                    mirrored, mirror_error = self._follow(engine, names)
                    last_wall = now
                elif not self._frozen.is_set():
                    # Real time: step as many physics ticks as wall time asks for, capped
                    # so a stall never turns into a burst.
                    n = min(int((now - last_wall) / dt), 200) if self._realtime else 1
                    if n > 0:
                        engine.step(n)
                        steps += n
                        last_wall = now
                else:
                    last_wall = now
                if now - last_render >= 1.0 / _RENDER_FPS:
                    rgb, _ = engine.get_frame(width=_RENDER_SIZE[0], height=_RENDER_SIZE[1])
                    with self._lock:
                        self._frame = rgb
                    last_render = now
                    frames += 1
                    if now - fps_window >= 1.0:
                        fps = frames / (now - fps_window)
                        frames = 0
                        fps_window = now
                self._publish(
                    sim_time=float(engine.mj_data.time),
                    steps=steps,
                    qpos=tuple(float(q) for q in engine.mj_data.qpos),
                    fps=fps,
                    state="frozen" if self._frozen.is_set() else (mirrored or "running"),
                    poses=_pack_poses(engine.mj_data),
                    bus=None if self._source is None else self._source.health(),
                    error=mirror_error,
                )
                time.sleep(1.0 / _TELEMETRY_HZ)
        except Exception as exc:
            logger.exception("sim %s (%s) died", self.id, self.robot)
            self._publish(state="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            self._close(engine)

    def _close(self, engine: Any) -> None:
        close = getattr(engine, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("engine close failed", exc_info=True)

    def _follow(self, engine: Any, names: tuple[str, ...]) -> tuple[str, str | None]:
        """Write the source's angles into the model: this tick's state, and why it is not ``mirroring``.

        ``("stale", None)`` while no fresh reading arrives, ``("error", reason)``
        when the bus stopped answering, and ``("refused", refusal)`` when the
        model refuses the pose. That write is all-or-nothing: one joint past its
        ``jnt_range`` writes nothing, so the twin stays on the last accepted
        pose, and answering ``mirroring`` for that tick would show a frozen twin
        beside a healthy 20 Hz bus with the reason nowhere on the page. An arm a
        few degrees of calibration offset outside the model's range trips this on
        every sweep, and it clears again by itself when the arm comes back inside.
        """
        source = self._source
        assert source is not None
        if source.error:
            return "error", source.error
        if self._frozen.is_set():
            return "frozen", None
        q = source.qpos()
        if q is None:
            return "stale", None
        wrote = engine.set_joint_positions(dict(zip(names, q, strict=False)), robot_name=self.robot, hold=True)
        if dict(wrote).get("status") == "error":
            texts = [b["text"] for b in dict(wrote).get("content") or [] if isinstance(b, dict) and b.get("text")]
            return "refused", " ".join(str(t) for t in texts) or "the model refused the pose"
        return "mirroring", None

    def _drain(self, engine: Any) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if self._frozen.is_set() and cmd.kind in MOTION_COMMANDS:
                    cmd.result = {
                        "status": "error",
                        "content": [{"text": f"{cmd.kind}: refused, this session is frozen by an e-stop"}],
                    }
                else:
                    cmd.result = self._apply(engine, cmd)
            except Exception as exc:
                cmd.result = {"status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
            finally:
                cmd.done.set()

    def _apply(self, engine: Any, cmd: _Command) -> dict[str, Any]:
        if self._source is not None and cmd.kind in ("reset", "set_joints"):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"this session mirrors the real arm on {self._source.port}: it reads the pose, it cannot set one"
                    }
                ],
            }
        if cmd.kind == "reset":
            return dict(engine.reset())
        if cmd.kind == "set_joints":
            # A target, not a nudge: ``set_joint_positions`` is a kinematic qpos
            # write, and this worker steps again the moment ``_drain`` returns.
            # Without ``hold`` the position servos are still commanded to their
            # previous setpoint and pull the pose back inside a few hundred
            # steps (measured on ``so101``: 0.5 rad written, 0.03 rad half a
            # second later) while the route has already answered 200.
            # ``hold=True`` moves the setpoints with the pose, which is the
            # "set joint targets" the route and the changelog promise.
            return dict(engine.set_joint_positions(cmd.payload["positions"], robot_name=self.robot, hold=True))
        if cmd.kind == "state":
            return dict(engine.get_robot_state(self.robot))
        if cmd.kind == "step":
            return dict(engine.step(int(cmd.payload.get("n", 1))))
        return {"status": "error", "content": [{"text": f"unknown command {cmd.kind}"}]}


def _pack_poses(data: Any) -> bytes:
    """``[geom_xpos | geom_xmat]`` per geom as little-endian float32 rows.

    The row is :data:`strands_robots.dashboard.scene.POSE_ROW_FLOATS` wide, which
    is the width ``describe`` publishes and ``static/twin.js`` strides by.
    """
    xpos = np.asarray(data.geom_xpos, dtype=np.float32).reshape(-1, 3)
    xmat = np.asarray(data.geom_xmat, dtype=np.float32).reshape(-1, 9)
    return np.ascontiguousarray(np.hstack([xpos, xmat]), dtype="<f4").tobytes()


def _default_factory(robot: str) -> Any:
    from strands_robots import Robot

    return Robot(robot, mode="sim")


def _model_path(robot: str) -> str | None:
    try:
        from strands_robots.simulation.model_registry import resolve_model

        return resolve_model(robot)
    except Exception:
        return None


class SessionStore:
    """The sessions this process is running, capped so a page cannot fork the machine."""

    def __init__(self, limit: int = MAX_SESSIONS):
        self.limit = limit
        self._sessions: dict[str, SimSession] = {}
        self._lock = threading.Lock()

    def create(self, robot: str, **kwargs: Any) -> SimSession:
        """Start a session, or raise ``RuntimeError`` when the cap is reached."""
        with self._lock:
            live = [s for s in self._sessions.values() if s.snapshot.state not in ("stopped", "error")]
            if len(live) >= self.limit:
                raise RuntimeError(f"{self.limit} sessions already running; stop one first")
            session = SimSession(robot, **kwargs)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> SimSession | None:
        """The session with this id, or None."""
        with self._lock:
            return self._sessions.get(session_id)

    def all(self) -> list[SimSession]:
        """Every session, including stopped ones not yet removed."""
        with self._lock:
            return list(self._sessions.values())

    def remove(self, session_id: str) -> bool:
        """Stop and forget a session. False when there was none."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.stop()
        return True

    def freeze_all(self) -> list[str]:
        """Freeze every session that can still step; returns the ids frozen.

        A session in ``starting`` is included: its engine is still being built,
        so it has not stepped yet and will begin the moment the build returns.
        Selecting on ``running`` alone would leave that one stepping after an
        e-stop, which is the window the e-stop exists for.
        """
        ids = []
        for s in self.all():
            if s.snapshot.state not in ("stopped", "error"):
                s.freeze()
                ids.append(s.id)
        return ids

    def thaw_all(self) -> list[str]:
        """Thaw every frozen session; returns the ids thawed."""
        ids = []
        for s in self.all():
            if s.snapshot.state == "frozen":
                s.thaw()
                ids.append(s.id)
        return ids

    def shutdown(self) -> None:
        """Stop every session. Called at app shutdown."""
        for s in self.all():
            s.stop()
