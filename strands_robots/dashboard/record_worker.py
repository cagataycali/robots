"""Teleop episode recording engine for the dashboard record screen (U8).

This module is the machinery behind ``/api/record``: a human drives the
leader arm, the follower mirrors it, and every control step is written to a
LeRobotDataset via :class:`strands_robots.dataset_recorder.DatasetRecorder`.

Design notes, so the next reader does not have to rediscover them:

* Recording happens IN ONE PROCESS that owns both arms for the whole
  session - the same exclusivity model as ``lerobot record``. Serial ports
  cannot be shared, so the dashboard stops the arms' fleet peers before a
  session opens and respawns them after close (that half lives in the
  session controller, not here).

* Everything hardware-shaped is injected. ``RecordWorker`` takes a
  ``backend`` (leader action + follower observation + apply) and a
  ``recorder_factory``; tests drive the state machine deterministically
  with fakes and never open a serial port. The real backend lives in
  :func:`hardware_backend` and imports strands_robots lazily.

* The session dict this produces IS the wire contract from
  FRONTEND_HANDOFF.md ("/api/record contract"): the frontend was built
  against it first, so field names here are load-bearing.

* The control loop is a plain thread stepping at ``fps``. Commands mutate
  state under one lock; the loop reads under the same lock. An episode is
  buffered by the recorder and only ``save_episode()`` makes it real -
  ``redo`` drops the buffer via ``clear_episode_buffer()``.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

THUMB_MAX_WIDTH = 160


class TeleopBackend(Protocol):
    """What the worker needs from hardware. One leader, one follower."""

    #: distinct camera names in the follower observation (schema order)
    camera_keys: list[str]

    def leader_action(self) -> dict[str, float]:
        """Read the leader arm - the action to mirror and record."""
        ...

    def follower_apply(self, action: dict[str, float]) -> dict[str, float]:
        """Send the action to the follower; returns the action as sent."""
        ...

    def follower_observation(self) -> dict[str, Any]:
        """Follower joints + camera frames (name -> float | ndarray)."""
        ...

    def close(self) -> None: ...


class EpisodeState:
    """Bookkeeping for one episode; ``to_dict`` speaks the wire contract."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.frames = 0
        self.started_at = 0.0
        self.duration_s = 0.0
        self.thumbnails: dict[str, str] = {}
        self.discarded = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frames": self.frames,
            "duration_s": round(self.duration_s, 1),
            "thumbnails": dict(self.thumbnails),
            "discarded": self.discarded,
        }


def _save_thumbnail(frame: Any, path: Path) -> bool:
    """Write a small JPEG for the episode strip; best-effort, never raises.

    A thumbnail is decoration - failing to encode one must not kill a
    recording session that is otherwise writing good frames.
    """
    try:
        import numpy as np

        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return False
        if arr.shape[1] > THUMB_MAX_WIDTH:
            step = max(1, arr.shape[1] // THUMB_MAX_WIDTH)
            arr = arr[::step, ::step]
        try:
            import cv2

            path.parent.mkdir(parents=True, exist_ok=True)
            # observations are RGB; cv2 writes BGR
            return bool(cv2.imwrite(str(path), arr[:, :, 2::-1]))
        except ImportError:
            from PIL import Image

            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr[:, :, :3].astype("uint8")).save(path)
            return True
    except Exception as exc:  # noqa: BLE001 - decoration only
        logger.debug("thumbnail skipped: %r", exc)
        return False


class RecordWorker:
    """The record session state machine + control loop.

    Phases: ``idle`` (between episodes) and ``recording``. All commands are
    idempotent the way the contract promises: ``start`` while recording and
    ``stop``/``redo`` while idle return the session unchanged.
    """

    def __init__(
        self,
        *,
        dataset: str,
        task: str,
        leader: str,
        follower: str,
        target_episodes: int,
        fps: int,
        backend: TeleopBackend,
        recorder_factory: Callable[..., Any],
        thumb_dir: Path | str,
        clock: Callable[[], float] = time.monotonic,
        autostart_loop: bool = True,
    ) -> None:
        if not dataset or not dataset.strip():
            raise ValueError("dataset name is required")
        if not task or not task.strip():
            raise ValueError("task is required - it is written on every frame")
        if leader == follower:
            raise ValueError("leader and follower must be different arms")
        fps = int(fps)
        if fps < 1 or fps > 120:
            raise ValueError("fps must be between 1 and 120")
        target = int(target_episodes)
        if target < 1:
            raise ValueError("target_episodes must be at least 1")

        self.dataset = dataset.strip()
        self.task = task.strip()
        self.leader = leader
        self.follower = follower
        self.target_episodes = target
        self.fps = fps
        self._backend = backend
        self._recorder = recorder_factory(
            repo_id=self.dataset, fps=fps, task=self.task
        )
        self._thumb_dir = Path(thumb_dir)
        self._clock = clock

        self._lock = threading.RLock()
        self._phase = "idle"
        self._episodes: list[EpisodeState] = []
        self._current: EpisodeState | None = None
        self._closed = False
        self._last_error: str | None = None

        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        if autostart_loop:
            self._thread = threading.Thread(
                target=self._loop, name="record-loop", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------- session

    def session(self) -> dict[str, Any]:
        with self._lock:
            return {
                "dataset": None if self._closed else self.dataset,
                "task": self.task,
                "leader": self.leader,
                "follower": self.follower,
                "target_episodes": self.target_episodes,
                "fps": self.fps,
                "phase": self._phase,
                "episodes": [e.to_dict() for e in self._episodes]
                + ([self._current.to_dict()] if self._current else []),
                "error": self._last_error,
                # None unless a requested camera is missing: absent cameras
                # must be visible BEFORE 10 episodes are collected blind.
                "camera_notice": getattr(self._backend, "camera_notice", None),
            }

    # ------------------------------------------------------------ commands

    def start_episode(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            if self._phase == "recording":
                return self.session()  # idempotent, per contract
            self._current = EpisodeState(index=self._next_index())
            self._current.started_at = self._clock()
            self._last_error = None
            self._phase = "recording"
        return self.session()

    def stop_episode(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            if self._phase != "recording" or self._current is None:
                return self.session()
            ep = self._current
            self._phase = "idle"
            self._current = None
            if ep.frames == 0:
                # An empty episode cannot be saved (LeRobot rejects empty
                # buffers) and keeping a 0-frame entry would lie about the
                # dataset's contents. Surface it instead of half-keeping it.
                self._last_error = (
                    "episode had 0 frames - nothing was captured, check teleop"
                )
                return self.session()
            info = self._recorder.save_episode()
            if isinstance(info, dict) and info.get("status") == "error":
                self._last_error = str(info.get("message", "save_episode failed"))
                return self.session()
            ep.duration_s = self._clock() - ep.started_at
            self._episodes.append(ep)
        return self.session()

    def redo_episode(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            if self._phase != "recording":
                return self.session()
            self._recorder.clear_episode_buffer()
            self._current = None
            self._phase = "idle"
        return self.session()

    def discard(self, index: int) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            for ep in self._episodes:
                if ep.index == int(index):
                    ep.discarded = True
                    break
            else:
                raise KeyError(f"no saved episode with index {index}")
        return self.session()

    def close(
        self, *, upload: bool = False, repo_id: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"ok": True, "detail": "session already closed"}
            if self._phase == "recording":
                # closing mid-episode keeps nothing half-written
                self._recorder.clear_episode_buffer()
                self._current = None
                self._phase = "idle"
            self._closed = True
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        kept = [e for e in self._episodes if not e.discarded]
        dropped = [e for e in self._episodes if e.discarded]
        detail = f"{len(kept)} episode(s) kept"
        if dropped:
            # LeRobot v3 has no cheap in-place episode deletion; discarded
            # episodes are excluded from training via the metadata note the
            # controller writes. Saying so beats silently pretending.
            detail += f", {len(dropped)} discarded (excluded via metadata)"
        try:
            self._recorder.finalize()
        except Exception as exc:  # noqa: BLE001
            self._backend.close()
            return {"ok": False, "detail": f"finalize failed: {exc}"}
        if upload:
            try:
                self._recorder.push_to_hub(repo_id=repo_id or self.dataset)
                detail += f", pushed to {repo_id or self.dataset}"
            except Exception as exc:  # noqa: BLE001
                self._backend.close()
                return {"ok": False, "detail": f"dataset saved but upload failed: {exc}"}
        # The finished dataset's own completion sentence must carry the defect
        # it was born with. A "10 episodes kept" toast is a receipt, and a
        # receipt that omits "this dataset has no camera channel" sends the
        # operator to the training screen to discover it there.
        notice = getattr(self._backend, "camera_notice", None)
        self._backend.close()
        result = {"ok": True, "detail": detail, "discarded_indices": [e.index for e in dropped]}
        if notice:
            missing = ", ".join(notice.get("missing") or ())
            result["camera_notice"] = notice
            result["detail"] = (
                f"{detail} — WITHOUT camera(s) {missing}: "
                + ("no image channel at all, so this dataset cannot train a visual policy"
                   if not notice.get("present")
                   else "those image channels are missing from every episode")
            )
        return result

    # ------------------------------------------------------------- helpers

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")

    def _next_index(self) -> int:
        return (self._episodes[-1].index + 1) if self._episodes else 0

    # ---------------------------------------------------------------- loop

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop_evt.wait(timeout=period):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - loop must survive a bad read
                with self._lock:
                    self._last_error = f"control step failed: {exc}"
                logger.warning("record tick failed: %r", exc)

    def tick(self) -> bool:
        """One control step. Teleop runs in EVERY phase (the operator lines
        the arms up between episodes); frames are recorded only while
        ``recording``. Returns True when a frame was written."""
        action = self._backend.leader_action()
        sent = self._backend.follower_apply(action)
        with self._lock:
            if self._phase != "recording" or self._current is None:
                return False
            obs = self._backend.follower_observation()
            self._recorder.add_frame(obs, sent, task=self.task)
            ep = self._current
            ep.frames += 1
            ep.duration_s = self._clock() - ep.started_at
            if ep.frames == 1:
                for cam in self._backend.camera_keys:
                    if cam in obs:
                        p = self._thumb_dir / f"{ep.index}_{cam}.jpg"
                        if _save_thumbnail(obs[cam], p):
                            ep.thumbnails[cam] = f"/api/record/thumb/{ep.index}/{cam}"
            return True


def camera_verdict(requested, present) -> dict[str, Any] | None:
    """What the operator must be told about cameras BEFORE they collect.

    ``camera_keys`` is derived from the follower's first observation, so a
    camera the machine refuses to open (macOS TCC denies the daemon, a cable
    is out, another process holds it) is simply ABSENT — and lerobot's schema
    is then built from what is present. The session records happily, every
    episode reports success, and the dataset that comes out has no image
    channel at all: it cannot train the visual policy it was collected for,
    and nothing in the flow ever said so. Same failure shape as a recorder
    that reports success with 0 frames, one layer up.

    Returns None when there is nothing to say (every requested camera is
    present, or none was requested and none appeared).
    """
    req = sorted(str(c) for c in (requested or ()))
    got = sorted(str(c) for c in (present or ()))
    missing = [c for c in req if c not in got]
    if not missing:
        return None
    consequence = (
        "the dataset will have NO image channel and cannot train a visual policy"
        if not got
        else "the dataset will be missing those image channels"
    )
    return {
        "requested": req,
        "present": got,
        "missing": missing,
        "message": (
            f"{len(missing)} of {len(req)} requested cameras did not open "
            f"({', '.join(missing)}) — {consequence}. The follower's own log says "
            "why it dropped them; on macOS a server started by a background "
            "daemon can never be granted camera access."
        ),
    }


def hardware_backend(
    *,
    follower_name: str,
    follower_port: str,
    leader_type: str,
    leader_port: str,
    cameras: dict[str, Any] | None = None,
) -> TeleopBackend:
    """The real thing: strands_robots follower + lerobot leader.

    Imported lazily so the state machine stays testable on machines without
    lerobot or serial hardware.
    """
    from strands_robots.robot import Robot
    from strands_robots.teleoperator import Teleoperator

    class _Hardware:
        def __init__(self) -> None:
            self._robot = Robot(
                follower_name, mode="real", port=follower_port,
                cameras=cameras or {},
            )
            # HardwareRobot connects lazily on the first task; recording
            # reads observations directly, so connect eagerly the same way
            # the fleet spawner does (calibrate=False - the arm was
            # calibrated when its peer was set up).
            inner = getattr(self._robot, "robot", None)
            if inner is not None and not getattr(inner, "is_connected", False):
                inner.connect(False)
            self._leader = Teleoperator(leader_type, port=leader_port)
            if hasattr(self._leader, "connect"):
                # calibrate=False, same reason as the follower above — and
                # doubly so here: lerobot's calibrate() talks to a HUMAN via
                # input(), which in this stdin-less server dies as
                # "EOF when reading a line" (cagatay hit exactly that from
                # the record screen, 2026-08-19). A missing calibration must
                # be a readable refusal naming the file, never an EOF.
                try:
                    self._leader.connect(calibrate=False)
                except TypeError:  # keyboards/gamepads take no calibrate arg
                    self._leader.connect()
                if not getattr(self._leader, "is_calibrated", True):
                    fpath = getattr(self._leader, "calibration_fpath", "its calibration file")
                    self._leader.disconnect()
                    raise ValueError(
                        f"leader arm at {leader_port} has no usable calibration "
                        f"({fpath} is missing or does not match the motors). "
                        "Calibrate it once from a terminal (lerobot-calibrate "
                        f"--teleop.type={leader_type} --teleop.port={leader_port}) "
                        "or copy an existing calibration json to that path — "
                        "a headless server cannot run the interactive wizard."
                    )
            obs = self._robot.get_observation()
            self.camera_keys = sorted(
                k for k, v in obs.items() if getattr(v, "ndim", 0) == 3
            )
            self._camera_dims = {
                k: (obs[k].shape[0], obs[k].shape[1]) for k in self.camera_keys
            }
            # Requested vs actually-present, judged at OPEN time: the caller's
            # ``cameras`` dict is the intent, the first observation is reality.
            self.camera_notice = camera_verdict(cameras or {}, self.camera_keys)
            self._robot_type = follower_name

        def recorder_kwargs(self) -> dict[str, Any]:
            """What DatasetRecorder.create needs beyond repo_id/fps/task -
            derived from the live robot so the schema cannot drift from
            what the observations actually contain."""
            return {
                "robot_type": self._robot_type,
                "robot_features": getattr(self._robot, "observation_features", None),
                "action_features": getattr(self._robot, "action_features", None),
                "camera_keys": list(self.camera_keys),
                "camera_dims": dict(self._camera_dims),
            }

        def leader_action(self) -> dict[str, float]:
            return dict(self._leader.get_action())

        def follower_apply(self, action: dict[str, float]) -> dict[str, float]:
            sent = self._robot.send_action(action)
            return dict(sent) if isinstance(sent, dict) else dict(action)

        def follower_observation(self) -> dict[str, Any]:
            return dict(self._robot.get_observation())

        def close(self) -> None:
            for dev in (self._leader, self._robot):
                try:
                    if hasattr(dev, "disconnect"):
                        dev.disconnect()
                    elif hasattr(dev, "close"):
                        dev.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("device close failed: %r", exc)

    return _Hardware()
