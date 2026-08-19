"""/api/record - the session controller behind the record screen (U8).

The RecordController owns exactly one teleop recording session at a time and
the fleet-side choreography around it:

* Serial ports are exclusive, so the arms' managed fleet peers are STOPPED
  before the session opens and RESPAWNED with their original configs after
  it closes. The USB auto-spawn watcher is suspended for the whole session -
  otherwise it would resurrect the peers within one poll and two processes
  would drive one servo bus.

* If opening fails halfway (a device refuses, lerobot missing), everything
  already torn down is put back. A failed open must leave the fleet exactly
  as it found it.

* HTTP speaks the FRONTEND_HANDOFF.md contract the record screen was built
  against; the state machine itself lives in
  :mod:`strands_robots.dashboard.record_worker`.

Factories for the backend and recorder are injectable for tests; the real
defaults are :func:`record_worker.hardware_backend` and
:class:`strands_robots.dataset_recorder.DatasetRecorder`.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from strands_robots.dashboard.record_worker import RecordWorker, hardware_backend

logger = logging.getLogger(__name__)

#: robot_name of the arm driven as leader -> lerobot teleoperator type
LEADER_TYPES = {
    "so100": "so100_leader",
    "so101": "so101_leader",
    "koch": "koch_leader",
}

EMPTY_SESSION: dict[str, Any] = {
    "dataset": None, "task": "", "leader": None, "follower": None,
    "target_episodes": 10, "fps": 30, "phase": "idle", "episodes": [],
    "error": None,
}


def _default_recorder_factory(backend: Any) -> Callable[..., Any]:
    def make(*, repo_id: str, fps: int, task: str) -> Any:
        from strands_robots.dataset_recorder import DatasetRecorder

        extra = backend.recorder_kwargs() if hasattr(backend, "recorder_kwargs") else {}
        return DatasetRecorder.create(repo_id=repo_id, fps=fps, task=task, **extra)

    return make


class RecordController:
    """One recording session at a time, with fleet peers parked around it."""

    def __init__(
        self,
        devices: Any,
        *,
        backend_factory: Callable[..., Any] | None = None,
        recorder_factory_factory: Callable[[Any], Callable[..., Any]] | None = None,
        thumb_root: str | None = None,
    ) -> None:
        self._devices = devices
        self._backend_factory = backend_factory or hardware_backend
        self._recorder_factory_factory = (
            recorder_factory_factory or _default_recorder_factory
        )
        self._thumb_root = Path(
            thumb_root or (Path(tempfile.gettempdir()) / "strands-record-thumbs")
        )
        self._lock = threading.Lock()
        self._worker: RecordWorker | None = None
        self._parked: list[dict[str, Any]] = []

    # ------------------------------------------------------------- session

    def session(self) -> dict[str, Any]:
        with self._lock:
            if self._worker is None:
                return dict(EMPTY_SESSION)
            return self._worker.session()

    @property
    def thumb_dir(self) -> Path:
        return self._thumb_root

    # ---------------------------------------------------------------- open

    def open(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._worker is not None:
                raise HTTPException(
                    409, "a recording session is already open - close it first"
                )
            dataset = str(body.get("dataset", "")).strip()
            task = str(body.get("task", "")).strip()
            leader_id = str(body.get("leader", "")).strip()
            follower_id = str(body.get("follower", "")).strip()

            leader = self._managed(leader_id, role="leader")
            follower = self._managed(follower_id, role="follower")

            leader_type = str(
                body.get("leader_type")
                or LEADER_TYPES.get(leader.robot_name, "")
            ).strip()
            if not leader_type:
                raise HTTPException(
                    422,
                    f"no teleoperator type known for leader robot "
                    f"'{leader.robot_name}' - pass leader_type explicitly "
                    f"(known: {sorted(LEADER_TYPES)})",
                )

            # Park the peers: remember their spawn configs, stop them, and
            # keep the watcher's hands off the freed ports.
            parked = [
                self._spawn_cfg(leader), self._spawn_cfg(follower),
            ]
            watcher = getattr(self._devices, "autospawn", None)
            if watcher is not None:
                watcher.suspended = True
            self._devices.despawn(leader_id)
            self._devices.despawn(follower_id)
            self._parked = parked

            try:
                backend = self._backend_factory(
                    follower_name=follower.robot_name,
                    follower_port=follower.port,
                    leader_type=leader_type,
                    leader_port=leader.port,
                    cameras=follower.cameras,
                )
                self._worker = RecordWorker(
                    dataset=dataset,
                    task=task,
                    leader=leader_id,
                    follower=follower_id,
                    target_episodes=int(body.get("target_episodes", 10) or 10),
                    fps=int(body.get("fps", 30) or 30),
                    backend=backend,
                    recorder_factory=self._recorder_factory_factory(backend),
                    thumb_dir=self._thumb_root,
                )
            except HTTPException:
                self._unpark_locked()
                raise
            except ValueError as exc:
                self._unpark_locked()
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                # a failed open must leave the fleet exactly as it found it
                self._unpark_locked()
                raise HTTPException(500, f"could not open the arms: {exc}") from exc
            return self._worker.session()

    def _managed(self, peer_id: str, *, role: str) -> Any:
        if not peer_id:
            raise HTTPException(422, f"{role} peer_id is required")
        m = getattr(self._devices, "robots", {}).get(peer_id)
        if m is None:
            raise HTTPException(
                404,
                f"{role} '{peer_id}' is not a robot this dashboard manages - "
                f"recording needs both arms spawned from this machine",
            )
        if m.mode != "real" or not m.port:
            raise HTTPException(
                422,
                f"{role} '{peer_id}' is {m.mode} - recording teleop episodes "
                f"needs a real arm with a serial port",
            )
        return m

    @staticmethod
    def _spawn_cfg(m: Any) -> dict[str, Any]:
        return {
            "robot_name": m.robot_name, "mode": m.mode, "peer_id": m.peer_id,
            "port": m.port, "cameras": m.cameras, "remember": False,
        }

    # --------------------------------------------------------------- steps

    def _require_worker(self) -> RecordWorker:
        if self._worker is None:
            raise HTTPException(409, "no recording session is open")
        return self._worker

    def start_episode(self) -> dict[str, Any]:
        with self._lock:
            return self._require_worker().start_episode()

    def stop_episode(self) -> dict[str, Any]:
        with self._lock:
            return self._require_worker().stop_episode()

    def redo_episode(self) -> dict[str, Any]:
        with self._lock:
            return self._require_worker().redo_episode()

    def discard(self, index: int) -> dict[str, Any]:
        with self._lock:
            try:
                return self._require_worker().discard(index)
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc

    # --------------------------------------------------------------- close

    def close(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        with self._lock:
            worker = self._require_worker()
            try:
                result = worker.close(
                    upload=bool(body.get("upload")),
                    repo_id=(str(body.get("repo_id") or "").strip() or None),
                )
            finally:
                # the fleet comes back whether or not finalize/upload worked -
                # a hub hiccup must not leave the desk armless
                self._worker = None
                self._unpark_locked()
            return result

    def _unpark_locked(self) -> None:
        for cfg in self._parked:
            try:
                res = self._devices.spawn(**cfg)
                if res.get("error"):
                    logger.warning(
                        "respawn of %s after record session failed: %s",
                        cfg.get("peer_id"), res["error"],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "respawn of %s after record session failed: %r",
                    cfg.get("peer_id"), exc,
                )
        self._parked = []
        watcher = getattr(self._devices, "autospawn", None)
        if watcher is not None:
            watcher.suspended = False


def build_router(controller: RecordController) -> APIRouter:
    """The HTTP surface, exactly as FRONTEND_HANDOFF.md specifies it."""
    r = APIRouter(prefix="/api/record")

    @r.get("/session")
    async def session() -> dict[str, Any]:
        return controller.session()

    @r.post("/open")
    async def open_session(body: dict[str, Any]) -> dict[str, Any]:
        return controller.open(body)

    @r.post("/episode/start")
    async def episode_start() -> dict[str, Any]:
        return controller.start_episode()

    @r.post("/episode/stop")
    async def episode_stop() -> dict[str, Any]:
        return controller.stop_episode()

    @r.post("/episode/redo")
    async def episode_redo() -> dict[str, Any]:
        return controller.redo_episode()

    @r.post("/episode/discard")
    async def episode_discard(body: dict[str, Any]) -> dict[str, Any]:
        if "index" not in body:
            raise HTTPException(422, "index required")
        return controller.discard(int(body["index"]))

    @r.post("/close")
    async def close_session(body: dict[str, Any] | None = None) -> dict[str, Any]:
        return controller.close(body)

    @r.get("/thumb/{episode}/{camera}")
    async def thumb(episode: int, camera: str) -> FileResponse:
        # camera comes from a URL path - keep it a bare name, no traversal
        safe = "".join(c for c in camera if c.isalnum() or c in "-_")
        path = controller.thumb_dir / f"{int(episode)}_{safe}.jpg"
        if not path.is_file():
            raise HTTPException(404, "no thumbnail for that episode/camera")
        return FileResponse(path, media_type="image/jpeg")

    return r
