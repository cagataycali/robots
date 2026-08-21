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

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from strands_robots.dashboard import camera_liveness
from strands_robots.dashboard import record_crash
from strands_robots.dashboard import disk_headroom
from strands_robots.dashboard.dataset_check import _as_int, record_target_verdict
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


def _target_facts(dataset: str) -> dict[str, Any]:
    """What is on disk where this dataset would be written.

    Read here rather than inside the verdict so the judgment stays pure and testable, and read
    DEFENSIVELY: an unreadable dataset home must not stop a recording from opening. Silence here
    means "no evidence of a collision", which is exactly what the old behaviour assumed - the
    check can only ever add refusals it can prove.
    """
    name = (dataset or "").strip()
    if not name:
        return {}
    try:
        from strands_robots.dataset_recorder import resolve_dataset_dir

        d = resolve_dataset_dir(name)
        if not d.exists():
            return {"exists": False}
        meta = (d / "meta" / "info.json").exists()
        episodes = None
        if meta:
            episodes = _as_int(
                json.loads((d / "meta" / "info.json").read_text()).get("total_episodes")
            )
        return {
            "exists": True,
            "has_meta": meta,
            "episodes": episodes,
            "non_empty": any(d.iterdir()),
        }
    except Exception:  # noqa: BLE001 - a blind check must never block a recording
        return {}


def _hub_facts(repo_id: str | None) -> dict[str, Any]:
    """Q78: what is already published at this destination, or {} when it cannot be established.

    Read DEFENSIVELY and never raise: no network, no token, a 5xx from the Hub, an unexpected
    payload - all mean "no evidence", which is exactly how the check behaved before it existed. A
    Hub lookup must not be able to stop a recording from being published.
    """
    name = (repo_id or "").strip().strip("/")
    if not name or "/" not in name:
        return {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if not api.repo_exists(repo_id=name, repo_type="dataset"):
            return {"exists": False}
        episodes = None
        try:
            info = api.dataset_info(repo_id=name)
            card = getattr(info, "cardData", None) or {}
            raw = card.get("total_episodes") if isinstance(card, dict) else None
            episodes = _as_int(raw)
        except Exception:  # noqa: BLE001 - the count is a nicety, existence is the fact
            episodes = None
        return {"exists": True, "episodes": episodes}
    except Exception:  # noqa: BLE001
        return {}


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
        bridge: Any | None = None,
    ) -> None:
        self._devices = devices
        # Where the live per-camera frame evidence comes from. Optional and
        # late-bound: an older caller (and every existing test) constructs this
        # controller with devices alone, and a controller with no bridge simply
        # has no evidence - which must mean "do not refuse", never "everything
        # is dead".
        self._bridge = bridge
        self._backend_factory = backend_factory or hardware_backend
        self._recorder_factory_factory = (
            recorder_factory_factory or _default_recorder_factory
        )
        self._thumb_root = Path(
            thumb_root or (Path(tempfile.gettempdir()) / "strands-record-thumbs")
        )
        self._lock = threading.Lock()
        self._worker: RecordWorker | None = None
        #: Q40: proof that a session was open, for the next process to read.
        self._crumb = record_crash.crumb_path()
        self._parked: list[dict[str, Any]] = []
        #: Q92: cached free-space verdict + when it was read.
        self._disk_cache: dict[str, Any] | None = None
        self._disk_seen: float | None = None

    #: How long a free-space reading stays good enough (Q92). The record screen polls session() about
    #: once a second; a statfs is cheap but not free, and disk pressure is a minutes-scale story.
    DISK_POLL_S = 20.0

    def _disk_notice(self) -> dict[str, Any] | None:
        """The free-space warning for the volume datasets land on, or None.

        Reported on EVERY session read, idle or recording, because Q91 was a RATE: the volume this
        rig records into lost ~2Gi/h to swap growth, so a session that was comfortable when it opened
        can be in trouble by episode 20. A check that only runs at open would have missed exactly
        that. Cached, blind to failure, and silent when there is nothing to say.
        """
        now = time.time()
        if self._disk_seen is not None and now - self._disk_seen < self.DISK_POLL_S:
            return self._disk_cache
        got = disk_headroom.free_space()
        self._disk_cache = disk_headroom.headroom_verdict(
            free_mb=got.get("free_mb"),
            total_mb=got.get("total_mb"),
            where="the dataset home",
        )
        self._disk_seen = now
        return self._disk_cache

    # ------------------------------------------------------------- session

    def session(self) -> dict[str, Any]:
        with self._lock:
            if self._worker is None:
                # Q40: "no session" is true but not the whole truth. A breadcrumb this dashboard
                # wrote and never removed proves a session was open and did not close - so the
                # screen can name the dataset and the arms instead of showing an empty form over a
                # half-written take.
                idle = dict(EMPTY_SESSION)
                crumb = record_crash.read_crumb(self._crumb)
                notice = record_crash.interrupted_notice(
                    crumb, same_process=bool(crumb) and crumb.get("pid") == os.getpid()
                )
                if notice:
                    idle["interrupted"] = notice
                disk = self._disk_notice()
                if disk:
                    idle["disk_notice"] = disk
                return idle
            live = self._worker.session()
            # Q92: the worker owns the recording, not the machine it runs on, so the disk reading is
            # attached here. Absent rather than null when there is nothing to say, matching every
            # other *_notice on this document.
            disk = self._disk_notice()
            if disk:
                live = {**live, "disk_notice": disk}
            return live

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

            # Q39: judge the dataset NAME first. Every failure after the parking step below
            # reports through "could not open the arms: {exc}", so a name that is already taken
            # sent the operator to check cables for what is a one-word rename - and by then both
            # arms have been despawned and respawned for nothing.
            bad = record_target_verdict(dataset, **_target_facts(dataset))
            if bad:
                raise HTTPException(409 if dataset.strip() else 422, bad)

            leader = self._managed(leader_id, role="leader")
            follower = self._managed(follower_id, role="follower")

            # A dataset is the expensive artifact here: an hour of hand-guiding an
            # arm, discovered to be useless at training time. A camera whose last
            # capture is hours old (measured: arm-1's wrist, 10.4h) would record a
            # frozen or missing image stream, so it is worth one check now. Only
            # POSITIVE evidence of death refuses - a camera with no frame history
            # may simply never have been subscribed to (camera_liveness).
            if not body.get("ignore_dead_cameras"):
                dead = camera_liveness.dead_cameras(
                    follower.cameras, self._camera_meta(follower_id), now=time.time()
                )
                if dead:
                    raise HTTPException(
                        409, camera_liveness.refusal(dead, peer_id=follower_id)
                    )

            # The rail above judges frame AGE, so it is blind to a camera that never
            # published at all - which is exactly a camera unplugged before the arm
            # subscribed. The machine's own enumeration answers that independently.
            if not body.get("ignore_missing_cameras"):
                missing = camera_liveness.missing_cameras(
                    follower.cameras, self._present_camera_indices()
                )
                if missing:
                    raise HTTPException(
                        409, camera_liveness.missing_refusal(missing, peer_id=follower_id)
                    )

            # The third shape, and the one both rails above call healthy: an index is a POSITION in
            # a list that closes up when a device is removed, so pulling the camera at index 1
            # slides index 2 into its place. The configured index still exists, still opens, still
            # streams - with the wrong view, and the episodes look perfect until training. Only the
            # name the index carried when it was configured (stamped at spawn) can say so.
            if not body.get("ignore_camera_identity"):
                drift = camera_liveness.identity_drift(
                    follower.cameras, self._present_camera_roster()
                )
                if drift:
                    raise HTTPException(
                        409, camera_liveness.drift_refusal(drift, peer_id=follower_id)
                    )

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
            opened = self._worker.session()
            record_crash.write_crumb(opened, path=self._crumb)
            return opened

    #: How old an already-taken camera roster may be and still count as evidence. A roster from
    #: this morning could omit a camera plugged in since, and refusing a session over that would be
    #: this gate inventing a fault.
    ROSTER_MAX_AGE_S = 300.0

    def _present_camera_roster(self) -> tuple[dict[str, Any], ...]:
        """The camera roster this machine ALREADY took, if it is still evidence.

        Deliberately does not trigger a scan. Two reasons, and the second is the important one:
        a fresh name scan shells out to ffmpeg with a 10s timeout, which would sit in front of the
        record button; and any probe that *opens* cameras to enumerate them could itself take the
        index the arm is about to use. So this reads the cache the devices screen has already
        filled - and an empty or stale cache reads as no evidence, which cannot refuse anything.

        Single source for both machine-side camera rails (absence and identity), so they can never
        disagree about what this Mac is showing right now.
        """
        devices = self._devices
        try:
            roster = getattr(devices, "_camera_names_cache", None)
            taken_at = float(getattr(devices, "_camera_names_cache_t", 0.0) or 0.0)
            if not roster or taken_at <= 0 or time.time() - taken_at > self.ROSTER_MAX_AGE_S:
                return ()
            return tuple(dict(r) for r in roster if isinstance(r, Mapping))
        except Exception:  # noqa: BLE001 - evidence gathering must never break a session
            logger.debug("[record] could not read the camera roster")
            return ()

    def _present_camera_indices(self) -> tuple[int, ...]:
        """Just the indices of :meth:`_present_camera_roster`, for the absence rail."""
        return tuple(
            int(r["listing_index"])
            for r in self._present_camera_roster()
            if isinstance(r.get("listing_index"), int) and not isinstance(r.get("listing_index"), bool)
        )

    def _camera_meta(self, peer_id: str) -> dict[str, Any]:
        """What the fleet snapshot last saw from this peer's cameras.

        Never raises and never guesses: a missing bridge, a peer the bridge has
        not heard of, or a snapshot shaped differently all read as "no evidence",
        and no evidence must not be able to refuse a recording.
        """
        bridge = self._bridge
        if bridge is None:
            return {}
        try:
            peers = bridge.snapshot().get("peers") or {}
            cams = (peers.get(peer_id) or {}).get("cameras") or {}
            return dict(cams) if isinstance(cams, dict) else {}
        except Exception:  # noqa: BLE001 - a probe must not break the session
            logger.debug("[record] could not read camera evidence for %s", peer_id)
            return {}

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
                # Q40: cleared even if finalize threw. The breadcrumb answers "was a session left
                # open when this process died", and a close that reached the worker answers it: no.
                record_crash.clear_crumb(self._crumb)
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


def build_router(
    controller: RecordController,
    on_activity: Callable[..., None] | None = None,
) -> APIRouter:
    """The HTTP surface, exactly as FRONTEND_HANDOFF.md specifies it."""
    r = APIRouter(prefix="/api/record")

    @r.get("/session")
    async def session() -> dict[str, Any]:
        return controller.session()

    @r.get("/upload-preflight")
    async def upload_preflight_route() -> dict[str, Any]:
        """Q72: can this machine publish THIS session's dataset -- asked before the work, not after.

        Read-only. Every failure `close(upload=True)` can report is knowable now, and knowing it
        later costs a finished session that cannot be re-pushed from here. hf_auth_state() is
        cached and local unless the token changed, so the record screen may poll this.
        """
        from strands_robots.dashboard.checkpoints import hf_auth_state
        from strands_robots.dashboard.upload_preflight import upload_preflight

        from strands_robots.dashboard.upload_preflight import destination

        current = controller.session() or {}
        dataset = current.get("dataset") or current.get("repo_id")

        def judge() -> dict[str, Any]:
            auth = hf_auth_state()
            user = auth.get("user") if isinstance(auth, dict) else None
            # Q78: only ask the Hub once the destination is knowable AND the credential is good -
            # an unauthenticated probe answers about public repos only, and the auth refusal is the
            # one that matters first anyway.
            existing = None
            if isinstance(auth, dict) and auth.get("authenticated") is True:
                existing = _hub_facts(destination(dataset or "", user if isinstance(user, str) else None))
            return upload_preflight(dataset=dataset, auth=auth, existing=existing)

        return await asyncio.to_thread(judge)

    @r.post("/open")
    async def open_session(body: dict[str, Any]) -> dict[str, Any]:
        result = controller.open(body)
        # Opening a session PARKS real arms into teleop - an audit-worthy
        # moment: it is the record screen taking the robots away from the
        # fleet. The hook is optional so the router stays testable alone.
        if on_activity is not None:
            on_activity(
                "record", "session_open",
                target=str(body.get("dataset") or body.get("repo_id") or "session"),
                detail=f"task={body.get('task', '')!r}",
                ok="error" not in result,
            )
        return result

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
        result = controller.close(body)
        if on_activity is not None:
            episodes = result.get("episodes_kept", result.get("episodes"))
            on_activity(
                "record", "session_close", target="session",
                detail=None if episodes is None else f"{episodes} episodes",
                ok="error" not in result,
            )
        return result

    @r.get("/thumb/{episode}/{camera}")
    async def thumb(episode: int, camera: str) -> FileResponse:
        # camera comes from a URL path - keep it a bare name, no traversal
        safe = "".join(c for c in camera if c.isalnum() or c in "-_")
        path = controller.thumb_dir / f"{int(episode)}_{safe}.jpg"
        if not path.is_file():
            raise HTTPException(404, "no thumbnail for that episode/camera")
        return FileResponse(path, media_type="image/jpeg")

    return r
