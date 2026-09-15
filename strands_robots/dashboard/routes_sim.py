"""``/api/sim``, ``/ws/telemetry`` and ``/api/safety`` - the simulated robot and the e-stop.

The lockout is :class:`strands_robots.dashboard.safety_state.Lockout`, folded
with the functions that module already tests: an e-stop is ``apply_event
(kind="estop")``, a resume is ``apply_event(kind="resume")`` - which leaves
the state ``unknown`` on purpose, because a resume is a request, not proof -
and the first command a session then accepts is ``note_command_accepted``,
which is the proof. Every route that would move the sim checks
``proves_clear(action)`` and refuses with 423 while the lockout is ``locked``.

Sessions are local to this process. The mesh-wide signed e-stop rail is a
separate slice; this one stops what this dashboard is running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from strands_robots.dashboard import access, safety_state, scene
from strands_robots.dashboard.sim_session import SessionStore, SimSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sim"])

_STREAM_FPS = 12.0
_TELEMETRY_HZ = 15.0


class Safety:
    """The process-wide lockout and the sessions it governs."""

    def __init__(self, store: SessionStore):
        self.store = store
        self.lockout = safety_state.Lockout()
        self._lock = threading.Lock()

    def estop(self, by: str) -> dict[str, Any]:
        """Freeze every session and latch the lockout."""
        now = time.time()
        frozen = self.store.freeze_all()
        with self._lock:
            self.lockout = safety_state.apply_event(self.lockout, kind="estop", data={"source": by, "t": now}, now=now)
        logger.warning("e-stop by %s froze %d session(s)", by, len(frozen), extra=self.lockout.as_fields())
        return {"lockout": self.lockout.as_fields(), "frozen": frozen}

    def resume(self, by: str) -> dict[str, Any]:
        """Fold a resume (state becomes ``unknown``) and thaw sessions."""
        now = time.time()
        with self._lock:
            self.lockout = safety_state.apply_event(self.lockout, kind="resume", data={"source": by, "t": now}, now=now)
        thawed = self.store.thaw_all()
        return {"lockout": self.lockout.as_fields(), "thawed": thawed}

    def gate(self, action: str) -> None:
        """Refuse *action* while locked, unless the lockout exempts it."""
        if self.lockout.state == "locked" and safety_state.proves_clear(action):
            raise HTTPException(423, f"e-stop engaged: {self.lockout.reason}")

    def accepted(self) -> None:
        """A session took a command: proof the lockout is not engaged."""
        with self._lock:
            self.lockout = safety_state.note_command_accepted(self.lockout, now=time.time())


def _safety(request: Request) -> Safety:
    return request.app.state.safety  # type: ignore[no-any-return]


def _session(request: Request, session_id: str) -> SimSession:
    session = _safety(request).store.get(session_id)
    if session is None:
        raise HTTPException(404, f"no session {session_id}")
    return session


def _known_sim_robot(name: str) -> str:
    from strands_robots.registry.robots import get_robot, resolve_name

    entry = get_robot(name)
    if entry is None or not entry.get("asset"):
        raise HTTPException(400, f"{name!r} is not a robot with a simulation asset")
    return resolve_name(name)


# -- sessions ---------------------------------------------------------------


@router.get("/api/sim")
async def list_sessions(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Every session this process is running."""
    return {"sessions": [s.snapshot.as_dict() for s in _safety(request).store.all()]}


def _is_serial_device(port: str) -> bool:
    return port.startswith("/dev/") and "/../" not in port and os.path.exists(port)


def _mirror_source(spec: Any) -> Any:
    """Open the real arm's bus read-only for ``{"port": "/dev/..."}``, or refuse with why."""
    from strands_robots.dashboard.mirror import BusMirror

    if not isinstance(spec, dict) or not isinstance(spec.get("port"), str):
        raise HTTPException(400, 'mirror must be {"port": "/dev/<serial device>"}')
    port = spec["port"]
    if not _is_serial_device(port):
        raise HTTPException(400, f"{port!r} is not a serial device on this machine")
    source = BusMirror(port)
    source.wait_ready(15.0)
    if source.error:
        source.close()
        raise HTTPException(502, source.error)
    return source


@router.get("/api/sim/ports")
async def serial_ports(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Serial devices on this machine a mirror could read, servo buses first. Nothing is opened."""
    from strands_robots._serial_discovery import scan_serial_devices

    found = await asyncio.to_thread(scan_serial_devices)
    ports = [
        {"port": c.port, "stable_id": c.stable_id, "likely_servo_bus": bool(c.likely_servo_bus)}
        for c in sorted(found, key=lambda c: (not c.likely_servo_bus, c.port))
    ]
    return {"ports": ports}


@router.post("/api/sim", status_code=201)
async def create_session(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Start a simulated robot - or a twin that mirrors the real one. Refused while the e-stop is engaged.

    ``{"robot": "so101"}`` steps physics. ``{"robot": "so101", "mirror":
    {"port": "/dev/cu.usbmodem..."}}`` reads the servo bus at that port
    (never writes it) and poses the model from the readings; ``set_joints``
    and ``reset`` are refused on such a session.
    """
    safety = _safety(request)
    safety.gate("create")
    body = await request.json() if await request.body() else {}
    if not isinstance(body, dict) or not isinstance(body.get("robot"), str):
        raise HTTPException(400, 'body must be {"robot": "<name>"}')
    robot = _known_sim_robot(body["robot"])
    source = await asyncio.to_thread(_mirror_source, body["mirror"]) if "mirror" in body else None
    try:
        session = safety.store.create(robot, source=source)
    except RuntimeError as exc:
        if source is not None:
            source.close()
        raise HTTPException(429, str(exc))
    await asyncio.to_thread(session.wait_ready, 60.0)
    snap = session.snapshot
    if snap.state == "error":
        safety.store.remove(session.id)
        raise HTTPException(500, f"could not start {robot}: {snap.error}")
    safety.accepted()
    logger.info("sim %s started for %s (%s) by %s", session.id, robot, snap.source, who.get("via"))
    return snap.as_dict()


@router.get("/api/sim/{session_id}")
async def get_session(request: Request, session_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The session's latest snapshot."""
    return _session(request, session_id).snapshot.as_dict()


@router.delete("/api/sim/{session_id}")
async def delete_session(
    request: Request, session_id: str, _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Stop and forget a session. Allowed under lockout: stopping is never refused."""
    if not await asyncio.to_thread(_safety(request).store.remove, session_id):
        raise HTTPException(404, f"no session {session_id}")
    return {"ok": True}


@router.post("/api/sim/{session_id}/reset")
async def reset_session(request: Request, session_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Return the robot to its home pose."""
    safety = _safety(request)
    safety.gate("reset")
    session = _session(request, session_id)
    result = await asyncio.to_thread(session.command, "reset")
    safety.accepted()
    return result


@router.post("/api/sim/{session_id}/joints")
async def set_joints(request: Request, session_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Set joint targets: ``{"positions": {"<joint>": rad, ...}}`` or a list in joint order."""
    safety = _safety(request)
    safety.gate("set_joints")
    session = _session(request, session_id)
    body = await request.json() if await request.body() else {}
    positions = body.get("positions") if isinstance(body, dict) else None
    if not isinstance(positions, (dict, list)) or not positions:
        raise HTTPException(400, "positions must be a non-empty object or list")
    if any(not isinstance(v, (int, float)) for v in (positions.values() if isinstance(positions, dict) else positions)):
        raise HTTPException(400, "every position must be a number")
    result = await asyncio.to_thread(session.command, "set_joints", positions=positions)
    if result.get("status") == "error":
        raise HTTPException(400, str(result.get("content")))
    safety.accepted()
    return result


@router.get("/api/sim/{session_id}/stream.mjpg")
async def stream(
    request: Request, session_id: str, frames: int | None = None, _: dict = Depends(access.require_session)
) -> StreamingResponse:
    """The rendered camera as multipart MJPEG, via ``rendering.video.mjpeg_frames``.

    ``?frames=N`` bounds the burst (a probe, a test, a thumbnail); unbounded
    is the live view.
    """
    from strands_robots.rendering.video import mjpeg_frames

    if frames is not None and not 1 <= frames <= 600:
        raise HTTPException(400, "frames must be 1..600")
    session = _session(request, session_id)
    frames_iter = mjpeg_frames(session.latest_frame, fps=_STREAM_FPS, quality=80, max_frames=frames)
    return StreamingResponse(frames_iter, media_type="multipart/x-mixed-replace; boundary=frame")


# -- the twin's geometry ----------------------------------------------------


def _model_of(request: Request, session_id: str) -> Any:
    model = _session(request, session_id).model
    if model is None:
        raise HTTPException(409, "session has no model yet")
    return model


@router.get("/api/sim/{session_id}/scene")
async def scene_description(
    request: Request, session_id: str, _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Geoms, meshes and cameras of the compiled model - what the browser twin draws."""
    return scene.describe(_model_of(request, session_id))


@router.get("/api/sim/{session_id}/mesh/{index}")
async def mesh(request: Request, session_id: str, index: int, _: dict = Depends(access.require_session)) -> Response:
    """One compiled mesh as ``SRM1`` bytes. Read from ``MjModel``; no file is opened."""
    model = _model_of(request, session_id)
    try:
        body = await asyncio.to_thread(scene.mesh_bytes, model, index)
    except IndexError:
        raise HTTPException(404, f"no mesh {index}")
    return Response(body, media_type="application/octet-stream", headers={"Cache-Control": "private, max-age=3600"})


# -- telemetry --------------------------------------------------------------


@router.websocket("/ws/telemetry/{session_id}")
async def telemetry(ws: WebSocket, session_id: str, poses: bool = False) -> None:
    """Snapshots at ~15 Hz. Same admission as every other route; a stranger is closed with 4401.

    With ``?poses=1`` every JSON snapshot is followed by one binary frame: the
    geom world poses as ``ngeom`` rows of 12 little-endian float32 (see
    :mod:`strands_robots.dashboard.scene`). The joint strip never asks; the
    twin always does.
    """
    try:
        access.caller(ws)  # type: ignore[arg-type]  # WebSocket answers headers/cookies/client like a Request
    except HTTPException:
        await ws.close(code=4401)
        return
    store: SessionStore = ws.app.state.safety.store
    session = store.get(session_id)
    if session is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    try:
        while True:
            snap = session.snapshot.as_dict()
            snap["lockout"] = ws.app.state.safety.lockout.as_fields()
            await ws.send_json(snap)
            if poses and session.snapshot.poses:
                await ws.send_bytes(session.snapshot.poses)
            if snap["state"] in ("stopped", "error"):
                break
            await asyncio.sleep(1.0 / _TELEMETRY_HZ)
    except WebSocketDisconnect:
        pass


# -- safety -----------------------------------------------------------------


@router.get("/api/safety")
async def safety_status(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The lockout as this dashboard understands it."""
    return {"lockout": _safety(request).lockout.as_fields()}


@router.post("/api/safety/estop")
async def estop(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Freeze every session and latch the lockout. Never refused."""
    return _safety(request).estop(by=str(who.get("name") or who.get("via") or "dashboard"))


@router.post("/api/safety/resume")
async def resume(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Lift the lockout to ``unknown`` and thaw sessions; the next accepted command proves ``clear``."""
    if who.get("via") == "loopback" and access.came_through_a_proxy(request):
        raise HTTPException(401, "sign in required")
    return _safety(request).resume(by=str(who.get("name") or who.get("via") or "dashboard"))
