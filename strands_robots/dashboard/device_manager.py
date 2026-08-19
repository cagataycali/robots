"""USB device + camera discovery and robot process lifecycle.

Detects:
* Serial servo buses (Feetech/Dynamixel USB adapters). Known VIDs:
  - 0x1a86 WCH CH34x  (SO-100/SO-101 controller boards - enumerate as
    "USB Single Serial" on macOS, so keyword matching alone misses them)
  - 0x0403 FTDI
* Local cameras via OpenCV index probe (the dashboard is the sole owner of
  local USB cams - two readers on one index steal each other's frames; robots
  opened by the dashboard get camera configs pointing at these indices).

Robot lifecycle: spawns `Robot(..., mode="real"|"sim").run()`-style child
processes and tracks them, so a detected arm becomes a mesh peer with one
click from the UI.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOG_TAIL_LINES = 200  # ring buffer per managed robot

#: How long /api/devices/spawn watches a new child before answering. A spawn
#: that reports success for a process that dies two seconds later is worse than
#: a slow spawn: the operator walks away believing an arm is live.
SPAWN_SETTLE_S = float(os.environ.get("STRANDS_DASHBOARD_SPAWN_SETTLE_S", "5") or 5)

#: Lines that name a *consequence*, not the cause. A child that dies while
#: building its cameras logs the real ValueError first and then a cleanup error
#: from the half-built object -- reporting the last line blames teardown.
_CONSEQUENCE_MARKERS = ("cleanup error", "during handling of the above")

# Where USB device profiles live. One JSON object keyed by USB serial number
# (the only stable identity a board keeps across replug and across port
# renumbering), value = the spawn payload that worked last time.
DEFAULT_PROFILES_PATH = os.path.join(Path.home(), ".strands_dashboard", "profiles.json")

# Auto-spawn watcher cadence and unplug debounce. USB enumeration flaps: a
# board can miss a single poll while the OS re-reads its descriptors, so one
# absent poll must never tear down a running robot.
AUTOSPAWN_POLL_S = 2.0
AUTOSPAWN_MISSING_POLLS = 2

# VIDs seen on servo-bus USB adapters. Keyword matching (feetech/dynamixel/...)
# still applies; VIDs catch the generic-description boards.
SERVO_VIDS = {0x1A86, 0x0403}
SERVO_KEYWORDS = ("feetech", "dynamixel", "sts3215", "xl430", "xl330", "usb single serial", "ch340", "ch343")
EXCLUDE_KEYWORDS = ("bluetooth", "debug", "internal", "apple", "modem-phone")


@dataclass
class ManagedRobot:
    peer_id: str
    robot_name: str
    mode: str                      # "real" | "sim"
    port: str | None = None
    cameras: dict[str, Any] = field(default_factory=dict)
    process: subprocess.Popen | None = None
    started_at: float = 0.0
    # Last N lines of child output. The pipe MUST be drained:
    # with stdout=PIPE and no reader, the child blocks in print() once the
    # OS pipe buffer (~64KB) fills and the peer silently freezes.
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def crash_reason(lines: Iterable[str]) -> str | None:
    """The line that explains why a child died, or None if nothing does.

    Two Python conventions decide where the answer lives, and both are easy to
    get backwards -- the live spawn probe reported "Traceback (most recent call
    last):" until this read them properly:

    1. A traceback prints its frames first and the MESSAGE LAST. The header is
       the first fault-shaped line and says nothing; the operator needs the
       tail::

           Traceback (most recent call last):
             File ".../hardware_robot.py", line 177, in _build_camera_config
               raise ValueError(
           ValueError: Camera 'main' config must be a mapping ... got int: 3.

    2. When exceptions chain ("During handling of the above exception..."), the
       FIRST block holds the root cause and later blocks are the fallout of
       unwinding a half-built object. Same for a trailing "Cleanup error", which
       is true and useless -- it blames teardown for a configuration mistake.

    So: the exception line of the first traceback, else the first fault-shaped
    line for children that die without one (a bare "error: unknown option", a
    usage message). Timestamps and ``LEVEL:logger:`` prefixes are stripped so the
    result can be shown to a person as-is.

    Args:
        lines: Child output, oldest first (the ring buffer's natural order).

    Returns:
        A one-line reason, or None when the output blames nothing (a silent
        exit, or a child killed from outside).
    """
    cleaned: list[str] = []
    for raw in lines:
        line = str(raw).strip()
        if not line:
            continue
        body = _strip_log_prefixes(line)
        low = body.lower()
        if low.startswith("during handling of the above") or low.startswith(
            "the above exception was the direct cause"
        ):
            # A chained block starts here: everything after it is fallout.
            break
        if "cleanup error" in low:
            continue
        cleaned.append(body)

    in_traceback = False
    for body in cleaned:
        if body.lower().startswith("traceback (most recent"):
            in_traceback = True
            continue
        if in_traceback:
            # Frames are indented; the exception line is not.
            if body != body.lstrip():
                continue
            if _looks_like_a_fault(body):
                return body[:400]
            continue
        if _looks_like_a_fault(body):
            return body[:400]

    if in_traceback:
        # A traceback whose tail never arrived (ring buffer cut it, child was
        # killed mid-print). Say that rather than quoting a frame.
        return "the process died with a traceback (see the log)"
    return None


def _strip_log_prefixes(line: str) -> str:
    """Drop the drain thread's ``HH:MM:SS`` stamp and any ``LEVEL:logger:``.

    Indentation after the stamp is preserved, because it is what distinguishes a
    traceback frame from the exception line that ends it.
    """
    body = line
    if len(body) > 9 and body[2] == ":" and body[5] == ":" and body[8] == " ":
        body = body[9:]
    for level in ("ERROR:", "CRITICAL:", "WARNING:"):
        if body.lstrip().startswith(level):
            rest = body.lstrip()[len(level):]
            body = rest.split(":", 1)[1].strip() if ":" in rest else rest.strip()
            break
    return body


def _looks_like_a_fault(text: str) -> bool:
    """True when a line reads like the reason a process stopped."""
    if not text:
        return False
    head = text.split(":", 1)[0].strip()
    if head.endswith(("Error", "Exception", "Interrupt")) and " " not in head:
        return True  # "ValueError: ..." / "ConnectionError: ..."
    low = text.lower()
    return low.startswith(("traceback (most recent", "fatal", "error:", "usage:"))


def _drain(proc: subprocess.Popen, logs: deque[str], peer_id: str) -> None:
    """Continuously read child stdout so the pipe never fills."""
    try:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if line:
                logs.append(f"{time.strftime('%H:%M:%S')} {line}")
    except Exception as e:  # reader must never take the server down
        logs.append(f"[drain error: {e}]")
    finally:
        try:
            code = proc.wait(timeout=5)  # EOF usually precedes reaping by a tick
        except Exception:
            code = proc.poll()
        logs.append(f"[process exited code={code}]")
        logger.info("robot %s child exited (code=%s)", peer_id, code)


def scan_serial_ports() -> list[dict[str, Any]]:
    """Enumerate candidate servo-bus serial ports."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    out: list[dict[str, Any]] = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        manu = (getattr(p, "manufacturer", None) or "").lower()
        text = desc + " " + manu
        if any(k in text for k in EXCLUDE_KEYWORDS):
            continue
        vid_match = p.vid in SERVO_VIDS if p.vid else False
        kw_match = any(k in text for k in SERVO_KEYWORDS)
        if not (vid_match or kw_match):
            continue
        # prefer /dev/cu.* on macOS (call-up device, non-blocking open)
        device = p.device
        if sys.platform == "darwin" and device.startswith("/dev/tty."):
            cu = device.replace("/dev/tty.", "/dev/cu.")
            if os.path.exists(cu):
                device = cu
        out.append({
            "device": device,
            "description": p.description,
            "vid": f"{p.vid:04x}" if p.vid else None,
            "pid": f"{p.pid:04x}" if p.pid else None,
            "serial_number": getattr(p, "serial_number", None),
            "likely_robot": "so101" if (p.vid == 0x1A86) else None,
        })
    return out


def scan_cameras(max_index: int = 4, skip: set[int] | None = None) -> list[dict[str, Any]]:
    """Probe OpenCV camera indices. Cheap open/read/release per index.

    ``skip`` indices are NOT opened (probing an index owned by a
    running robot's camera thread steals/flaps its frames mid-episode).
    """
    try:
        import cv2
    except ImportError:
        return []
    cams: list[dict[str, Any]] = []
    for i in range(max_index):
        if skip and i in skip:
            continue
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    cams.append({
                        "index": i, "width": w, "height": h,
                        "fps": round(fps, 1) if fps and fps > 0 else None,
                    })
        finally:
            cap.release()
    return cams


def scan_camera_names() -> list[dict[str, Any]]:
    """Human names of the attached cameras, best effort per platform.

    macOS: parsed from ffmpeg's AVFoundation device listing (if ffmpeg is
    installed). Linux: read from /sys/class/video4linux. IMPORTANT: the
    listing order is NOT guaranteed to match OpenCV index order (Continuity
    cameras in particular renumber), so a name here is a roster entry, not
    an index label - the preview endpoint is the authoritative way to tell
    which index is which camera.
    """
    names: list[dict[str, Any]] = []
    if sys.platform == "darwin":
        import re
        import shutil

        ffmpeg = shutil.which("ffmpeg") or next(
            (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)),
            None,
        )
        if not ffmpeg:
            return names
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True, text=True, timeout=10,
            ).stderr
        except Exception as e:  # noqa: BLE001 - enumeration is decoration, never fatal
            logger.debug("camera name scan failed: %r", e)
            return names
        in_video = False
        for line in out.splitlines():
            if "AVFoundation video devices" in line:
                in_video = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if in_video:
                m = re.search(r"\[(\d+)\]\s+(.+)$", line)
                if m:
                    names.append({"listing_index": int(m.group(1)), "name": m.group(2).strip()})
    elif sys.platform.startswith("linux"):
        import glob

        for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            try:
                idx = int(path.split("video")[-1].split("/")[0])
                with open(path) as f:
                    names.append({"listing_index": idx, "name": f.read().strip()})
            except (OSError, ValueError):
                continue
    return names


_SPAWNER = r'''
import os, sys, time, json
cfg = json.loads(sys.argv[1])
os.environ.setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", os.environ.get("STRANDS_MESH_LOCAL_DEV", "1"))
os.environ.setdefault("STRANDS_MESH_MULTICAST", "true")
os.environ.setdefault("STRANDS_MESH", "true")
os.environ.setdefault("STRANDS_MESH_CAMERA_HZ", os.environ.get("STRANDS_MESH_CAMERA_HZ", "5"))

from strands_robots import Robot

if cfg["mode"] == "real":
    kwargs = {}
    if cfg.get("robot_id"):
        kwargs["id"] = cfg["robot_id"]  # lerobot calibration identity
    robot = Robot(
        cfg["robot_name"], mode="real", port=cfg["port"],
        cameras=cfg.get("cameras") or None,
        mesh=True, peer_id=cfg["peer_id"], **kwargs,
    )
    # Connect eagerly so the mesh publishes joints + camera frames right away
    # (HardwareRobot otherwise connects lazily on the first task). Goes through
    # connect_eagerly so a camera this machine will not open costs the camera
    # and not the whole arm -- lerobot's own connect() gates is_connected on
    # every camera, so one refusal reported a healthy arm as dead.
    try:
        ok, degraded, err = robot.connect_eagerly()
        if ok and degraded:
            for cam_name, reason in degraded.items():
                print(f"camera {cam_name!r} unavailable, dropped: {reason}", flush=True)
            print(f"hardware connected WITHOUT camera(s): {', '.join(degraded)}", flush=True)
        elif ok:
            print("hardware connected", flush=True)
        else:
            print(f"eager connect failed (will retry on first task): {err}", flush=True)
    except Exception as e:
        print(f"eager connect failed (will retry on first task): {e}", flush=True)
    print(f"{cfg['peer_id']} (real @ {cfg['port']}) online", flush=True)
    while True:
        time.sleep(1)
else:
    sim = Robot(cfg["robot_name"], mesh=True, peer_id=cfg["peer_id"])
    n = cfg["robot_name"]
    sim.add_camera(name=f"{n}/front", position=[0.6, 0.4, 0.5], target=[0.0, 0.0, 0.15])
    try:
        sim.add_camera(name=f"{n}/wrist", position=[0.02, 0.0, 0.05], target=[0.2, 0.0, 0.0], parent_body=f"{n}/gripper")
    except Exception:
        pass
    print(f"{cfg['peer_id']} (sim) online", flush=True)
    while True:
        sim.step(5)
        time.sleep(0.2)
'''


_COLLECT_SPAWNER = r'''
import os, sys, time, json
cfg = json.loads(sys.argv[1])
os.environ.setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", os.environ.get("STRANDS_MESH_LOCAL_DEV", "1"))
os.environ.setdefault("STRANDS_MESH_MULTICAST", "true")
os.environ.setdefault("STRANDS_MESH", "true")
os.environ.setdefault("STRANDS_MESH_CAMERA_HZ", os.environ.get("STRANDS_MESH_CAMERA_HZ", "5"))

from strands_robots import Robot
from strands_robots.tools.run_policy import run_policy

robot_name = cfg.get("robot_name") or "so101"
sim = Robot(robot_name, mesh=True, peer_id=cfg["peer_id"])
n = robot_name
sim.add_camera(name=f"{n}/front", position=[0.6, 0.4, 0.5], target=[0.0, 0.0, 0.15])
try:
    sim.add_camera(name=f"{n}/wrist", position=[0.02, 0.0, 0.05], target=[0.2, 0.0, 0.0], parent_body=f"{n}/gripper")
except Exception:
    pass
print(f"{cfg['peer_id']} (collect sim) online - {cfg['n_episodes']} episodes", flush=True)

# run_policy drives the N-episode loop deterministically and reports
# parquet-truth (meta/info.json total_episodes) - no self-reporting.
# run_policy takes n_steps (not duration): steps = seconds * control Hz.
control_hz = 30.0
n_steps = max(1, int(float(cfg.get("duration", 10.0)) * control_hz))
res = run_policy(
    simulation=sim,
    robot_name=robot_name,
    policy_provider=cfg.get("policy_provider", "mock"),
    policy_config=cfg.get("policy_config") or None,
    instruction=cfg.get("instruction", ""),
    n_episodes=int(cfg.get("n_episodes", 5)),
    n_steps=n_steps,
    control_frequency=control_hz,
    dataset_root=cfg["dataset_root"],
    dataset_repo_id=cfg.get("dataset_repo_id", "local/collected"),
    dataset_task=cfg.get("instruction", ""),
    dataset_fps=int(cfg.get("fps", 30)),
)
status = res.get("status")
for block in res.get("content", []):
    if isinstance(block, dict) and block.get("json"):
        print(f"collect: {status}: {json.dumps(block['json'])[:400]}", flush=True)
        break
    if isinstance(block, dict) and block.get("text"):
        print(f"collect: {status}: {block['text'][:300]}", flush=True)
time.sleep(3)
os._exit(0)
'''


_REPLAY_SPAWNER = r'''
import os, sys, time, json
cfg = json.loads(sys.argv[1])
os.environ.setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", os.environ.get("STRANDS_MESH_LOCAL_DEV", "1"))
os.environ.setdefault("STRANDS_MESH_MULTICAST", "true")
os.environ.setdefault("STRANDS_MESH", "true")
os.environ.setdefault("STRANDS_MESH_CAMERA_HZ", os.environ.get("STRANDS_MESH_CAMERA_HZ", "5"))

from strands_robots import Robot

robot_name = cfg.get("robot_name") or "so101"
sim = Robot(robot_name, mesh=True, peer_id=cfg["peer_id"])
n = robot_name
sim.add_camera(name=f"{n}/front", position=[0.6, 0.4, 0.5], target=[0.0, 0.0, 0.15])
print(f"{cfg['peer_id']} (replay sim) online", flush=True)

res = sim.replay_episode(
    cfg["repo_id"],
    episode=int(cfg.get("episode", 0)),
    root=cfg.get("root"),
    speed=float(cfg.get("speed", 1.0)),
)
status = res.get("status")
for block in res.get("content", []):
    if isinstance(block, dict) and block.get("text"):
        print(f"replay: {status}: {block['text'][:300]}", flush=True)
        break
# linger briefly so the last camera frames reach subscribers, then exit.
# os._exit: the mesh session runs non-daemon threads that would otherwise
# keep this one-shot process alive forever after the script body returns.
time.sleep(3)
os._exit(0)
'''


def profile_key(port: dict[str, Any]) -> str:
    """Stable identity for a detected serial board.

    The USB serial number survives replug and port renumbering, so it is the
    key. Boards that report no serial number fall back to their device path,
    which is weaker (the path can move between reboots) but still lets a
    single-adapter setup be remembered.
    """
    serial = port.get("serial_number")
    if serial:
        return str(serial)
    return str(port.get("device") or "")


class ProfileStore:
    """USB device profiles on disk: serial number -> saved spawn payload.

    A profile is written whenever the operator successfully spawns a real
    (serial-port) robot, so the next time that exact board appears the
    dashboard can bring it up with the same calibration id, camera mapping
    and peer_id instead of asking again.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("STRANDS_DASHBOARD_PROFILES") or DEFAULT_PROFILES_PATH
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except Exception as e:
            # A corrupt or unreadable store must be visible, not silently
            # treated as "no profiles remembered".
            logger.warning("device profiles at %s unreadable (%r); starting empty", self.path, e)
            return {}
        if not isinstance(data, dict):
            logger.warning("device profiles at %s are not a JSON object; starting empty", self.path)
            return {}
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def all(self) -> dict[str, dict[str, Any]]:
        """Every remembered profile, keyed by USB serial number."""
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def get(self, key: str) -> dict[str, Any] | None:
        """One profile by key, or None when that board was never spawned."""
        with self._lock:
            entry = self._data.get(key)
            return dict(entry) if entry is not None else None

    def save(self, key: str, payload: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        """Remember ``payload`` as the way to spawn the board at ``key``."""
        entry = dict(payload)
        entry["name"] = name or entry.get("name") or entry.get("peer_id") or key
        entry["serial_number"] = key
        entry["saved_at"] = time.time()
        with self._lock:
            self._data[key] = entry
            snapshot = {k: dict(v) for k, v in self._data.items()}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception as e:
            # Kept in memory either way, but the operator has to know the
            # profile will not survive a restart.
            logger.warning("could not persist device profile %s to %s: %r", key, self.path, e)
        return dict(entry)


class AutoSpawnWatcher:
    """Brings known USB boards up (and unplugged ones down) on its own.

    One poll of the serial bus per :data:`AUTOSPAWN_POLL_S`:

    * a board APPEARS and has a saved profile -> spawn it with that payload
    * a board appears with NO profile -> left alone (it only shows in
      ``/api/devices`` as detected; auto-spawning an unknown board would
      energise hardware nobody configured)
    * a board this watcher spawned DISAPPEARS for
      :data:`AUTOSPAWN_MISSING_POLLS` consecutive polls -> its child process
      is terminated so the card goes away

    Dedupe is deliberately paranoid, because double-spawning a real arm means
    two processes driving one servo bus: a port already claimed by a managed
    robot, or a profile whose ``peer_id`` is already a mesh peer, is skipped.
    Robots the watcher did not spawn are never terminated by it - process
    lifecycle the operator started stays the operator's.

    ``STRANDS_DASHBOARD_AUTOSPAWN=0`` disables the whole thing.
    """

    def __init__(
        self,
        manager: DeviceManager,
        list_ports: Callable[[], list[dict[str, Any]]] | None = None,
        peer_ids: Callable[[], Iterable[str]] | None = None,
        missing_polls: int = AUTOSPAWN_MISSING_POLLS,
    ) -> None:
        self.manager = manager
        self.list_ports = list_ports or scan_serial_ports
        self.peer_ids = peer_ids or (lambda: ())
        self.missing_polls = max(1, int(missing_polls))
        # key -> peer_id we auto-spawned for it
        self.adopted: dict[str, str] = {}
        self._missing: dict[str, int] = {}
        self._stop = threading.Event()
        # While True the watcher observes but never spawns/despawns. The
        # record session controller sets this after it deliberately frees a
        # board's port to record with it - otherwise the watcher would
        # respawn the peer within one poll and two processes would drive
        # one servo bus.
        self.suspended = False

    @staticmethod
    def enabled() -> bool:
        """False when STRANDS_DASHBOARD_AUTOSPAWN is set to a falsey value."""
        raw = os.environ.get("STRANDS_DASHBOARD_AUTOSPAWN", "1").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def _claimed(self, port: dict[str, Any], profile: dict[str, Any]) -> str | None:
        """Reason this board must not be spawned, or None when it is free."""
        device = port.get("device")
        for m in self.manager.robots.values():
            if not m.alive():
                continue
            if device and m.port == device:
                return f"port {device} already claimed by managed robot {m.peer_id}"
        peer_id = profile.get("peer_id")
        if peer_id:
            managed = self.manager.robots.get(peer_id)
            if managed is not None and managed.alive():
                return f"peer {peer_id} already running locally"
            try:
                if peer_id in set(self.peer_ids()):
                    return f"peer {peer_id} already present on the mesh"
            except Exception as e:
                # Not knowing the mesh is a reason to hold back, not to guess.
                logger.warning("autospawn: mesh peer lookup failed (%r); skipping %s", e, peer_id)
                return f"mesh peer list unavailable, refusing to spawn {peer_id}"
        return None

    def poll(self) -> dict[str, Any]:
        """One appear/disappear pass. Returns what it did, for tests and logs."""
        if not self.enabled():
            return {"skipped": "autospawn disabled"}
        if self.suspended:
            return {"skipped": "autospawn suspended (record session owns the ports)"}
        ports = {profile_key(p): p for p in self.list_ports() if profile_key(p)}
        spawned: list[str] = []
        despawned: list[str] = []
        ignored: list[str] = []

        for key, port in ports.items():
            self._missing.pop(key, None)
            if key in self.adopted:
                continue
            profile = self.manager.profiles.get(key)
            if profile is None:
                ignored.append(key)
                continue
            reason = self._claimed(port, profile)
            if reason is not None:
                logger.debug("autospawn: %s", reason)
                ignored.append(key)
                continue
            res = self._spawn_from_profile(port, profile)
            peer_id = res.get("peer_id")
            if res.get("error") or not peer_id:
                logger.warning("autospawn: spawning %s failed: %s", key, res.get("error"))
                continue
            self.adopted[key] = peer_id
            spawned.append(peer_id)
            logger.info("autospawn: %s appeared, spawned %s", key, peer_id)

        for key, peer_id in list(self.adopted.items()):
            if key in ports:
                continue
            misses = self._missing.get(key, 0) + 1
            self._missing[key] = misses
            if misses < self.missing_polls:
                continue
            self._missing.pop(key, None)
            self.adopted.pop(key, None)
            self.manager.despawn(peer_id)
            despawned.append(peer_id)
            logger.info("autospawn: %s unplugged, stopped %s", key, peer_id)

        return {"spawned": spawned, "despawned": despawned, "detected_unknown": ignored}

    def _spawn_from_profile(self, port: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        return self.manager.spawn(
            robot_name=profile.get("robot_name") or "so101",
            mode=profile.get("mode") or "real",
            peer_id=profile.get("peer_id"),
            # The board may have re-enumerated on a new path; the live path
            # wins over whatever the profile remembered.
            port=port.get("device") or profile.get("port"),
            cameras=profile.get("cameras"),
            robot_id=profile.get("robot_id"),
            remember=False,
        )

    def run_forever(self, interval: float = AUTOSPAWN_POLL_S) -> None:
        """Poll until :meth:`stop`. For a thread; the server uses asyncio."""
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as e:  # a watcher crash must not be silent
                logger.warning("autospawn poll failed: %r", e)
            self._stop.wait(interval)

    def stop(self) -> None:
        """Ask :meth:`run_forever` to return after the current sleep."""
        self._stop.set()


class DeviceManager:
    """Owns local device discovery + robot child processes."""

    CAMERA_CACHE_TTL_S = 30.0  # don't re-open /dev cameras on every request

    def __init__(self, profiles_path: str | None = None) -> None:
        self.robots: dict[str, ManagedRobot] = {}
        self._lock = threading.Lock()
        self._camera_cache: list[dict[str, Any]] = []
        self._camera_cache_t = 0.0
        self._camera_names_cache: list[dict[str, Any]] = []
        self._camera_names_cache_t = 0.0
        # One preview at a time: two concurrent opens of the same device
        # wedge some UVC cameras until replug.
        self._preview_lock = threading.Lock()
        self.profiles = ProfileStore(profiles_path)
        self.autospawn: AutoSpawnWatcher | None = None

    def _claimed_camera_indices(self) -> dict[int, str]:
        """OpenCV indices owned by LIVE managed robots -> peer_id.

        Camera configs are lerobot-shaped: {name: {type: "opencv",
        index_or_path: <int|str>, ...}} (see robot.py docstring). Only
        integer / digit-string index_or_path values claim an index; device
        paths (/dev/video0) never collide with index probing on macOS.
        """
        claimed: dict[int, str] = {}
        for m in self.robots.values():
            if not m.alive():
                continue
            for cfg in (m.cameras or {}).values():
                iop = cfg.get("index_or_path") if isinstance(cfg, dict) else None
                if isinstance(iop, bool):
                    continue
                if isinstance(iop, int):
                    claimed[iop] = m.peer_id
                elif isinstance(iop, str) and iop.isdigit():
                    claimed[int(iop)] = m.peer_id
        return claimed

    def _cameras(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Cached camera list. Claimed indices are reported, never probed."""
        claimed = self._claimed_camera_indices()
        now = time.time()
        if refresh or (now - self._camera_cache_t) > self.CAMERA_CACHE_TTL_S:
            self._camera_cache = scan_cameras(skip=set(claimed))
            self._camera_cache_t = now
        cams = [c for c in self._camera_cache if c["index"] not in claimed]
        cams.extend(
            {"index": i, "claimed_by": peer} for i, peer in sorted(claimed.items())
        )
        return sorted(cams, key=lambda c: c["index"])

    def devices(self, refresh: bool = False) -> dict[str, Any]:
        return {
            "serial_ports": scan_serial_ports(),
            "cameras": self._cameras(refresh=refresh),
            "camera_names": self._camera_names(refresh=refresh),
            "managed": {
                pid: {
                    "peer_id": m.peer_id, "robot_name": m.robot_name, "mode": m.mode,
                    "port": m.port, "alive": m.alive(), "started_at": m.started_at,
                    "log_tail": list(m.logs)[-20:],
                }
                for pid, m in self.robots.items()
            },
        }

    def _camera_names(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Cached roster of camera names (see scan_camera_names on ordering)."""
        now = time.time()
        if refresh or (now - self._camera_names_cache_t) > self.CAMERA_CACHE_TTL_S:
            self._camera_names_cache = scan_camera_names()
            self._camera_names_cache_t = now
        return self._camera_names_cache

    def preview_frame(self, index: int) -> bytes:
        """One JPEG frame from an UNCLAIMED camera index.

        This is the authoritative "which camera is index N" tool - names are
        a roster in listing order, but a picture cannot lie. Refuses indices
        owned by a running robot (opening one steals its frames mid-stream).

        Raises:
            PermissionError: the index is claimed by a managed robot.
            RuntimeError: the index would not open or produced no frame.
        """
        claimed = self._claimed_camera_indices()
        if index in claimed:
            raise PermissionError(
                f"camera index {index} is streaming for {claimed[index]} - "
                f"watch it on that robot's card instead"
            )
        import cv2

        with self._preview_lock:
            cap = cv2.VideoCapture(index)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"camera index {index} would not open")
                # A couple of warm-up reads: first frames from a cold sensor
                # are often black or half-exposed.
                frame = None
                for _ in range(3):
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                        break
                if frame is None:
                    raise RuntimeError(f"camera index {index} produced no frame")
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                return bytes(buf.tobytes())
            finally:
                cap.release()

    def logs(self, peer_id: str) -> dict[str, Any]:
        """Full ring buffer for one managed robot."""
        m = self.robots.get(peer_id)
        if m is None:
            return {"error": f"unknown peer {peer_id}"}
        return {"peer_id": peer_id, "alive": m.alive(), "lines": list(m.logs)}

    def spawn(
        self,
        robot_name: str,
        mode: str = "sim",
        peer_id: str | None = None,
        port: str | None = None,
        cameras: dict[str, Any] | None = None,
        robot_id: str | None = None,
        remember: bool = True,
    ) -> dict[str, Any]:
        import json as _json

        if mode == "real" and not port:
            return {"error": "port required for mode=real"}
        peer_id = peer_id or f"{robot_name}-{mode}-{int(time.time()) % 10000}"
        with self._lock:
            if peer_id in self.robots and self.robots[peer_id].alive():
                return {"error": f"peer {peer_id} already running"}
            cfg = {"robot_name": robot_name, "mode": mode, "peer_id": peer_id, "port": port, "cameras": cameras, "robot_id": robot_id}
            proc = subprocess.Popen(
                [sys.executable, "-c", _SPAWNER, _json.dumps(cfg)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            managed = ManagedRobot(
                peer_id=peer_id, robot_name=robot_name, mode=mode, port=port,
                cameras=cameras or {}, process=proc, started_at=time.time(),
            )
            self.robots[peer_id] = managed
            # Drain stdout forever - daemon thread, ring buffer.
            threading.Thread(
                target=_drain, args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}", daemon=True,
            ).start()
        logger.info("spawned %s (%s, pid=%s)", peer_id, mode, proc.pid)
        if remember and mode == "real" and port:
            self.remember_profile(cfg)
        return {"peer_id": peer_id, "pid": proc.pid, "mode": mode}

    def settle(
        self,
        peer_id: str,
        *,
        timeout: float = SPAWN_SETTLE_S,
        is_up: Callable[[str], bool] | None = None,
        poll: float = 0.1,
        sleep: Callable[[float], None] | None = None,
        now: Callable[[], float] | None = None,
    ) -> dict[str, Any]:
        """Watch a just-spawned peer long enough to answer honestly.

        ``spawn()`` returns as soon as ``Popen`` hands back a pid, which is
        always -- a robot whose camera config is wrong, whose port is taken, or
        whose policy is not installed still gets a pid, dies a second later, and
        the dashboard shows a card for a peer that will never appear. This
        watches the gap.

        Returns as soon as the answer is known, so a healthy peer costs only the
        time it needs to announce itself:

        - ``running``: the child is alive and ``is_up`` confirms the mesh has
          seen it. The only status that means "it works".
        - ``failed``: the child exited. Carries ``exit_code`` and ``reason``
          (see :func:`crash_reason`).
        - ``starting``: still alive at the deadline, not yet announced. Not a
          failure and not a success -- slow hardware exists, and claiming
          either would be a guess.
        - ``gone``: no longer tracked (despawned while we watched).

        Args:
            peer_id: The peer returned by :meth:`spawn`.
            timeout: Seconds to watch before answering ``starting``.
            is_up: Optional presence check -- given the peer id, True when the
                mesh has heard from it. Without one, an alive child at the
                deadline is ``starting`` (a pid alone is not evidence).
            poll: Seconds between checks.
            sleep: Injected for tests.
            now: Injected for tests.

        Returns:
            A dict with ``status`` plus whatever that status carries.
        """
        _sleep = sleep or time.sleep
        _now = now or time.monotonic
        deadline = _now() + max(0.0, timeout)
        while True:
            managed = self.robots.get(peer_id)
            if managed is None:
                return {"status": "gone"}
            if not managed.alive():
                proc = managed.process
                code = proc.poll() if proc is not None else None
                reason = crash_reason(list(managed.logs))
                out: dict[str, Any] = {"status": "failed", "exit_code": code}
                # The log tail travels with the failure: a reason the operator
                # can act on beats "exit code 1", and they should not have to
                # go find a second endpoint while the card is still on screen.
                out["reason"] = reason or (
                    f"the process exited with code {code}" if code is not None
                    else "the process is gone"
                )
                out["log_tail"] = list(managed.logs)[-12:]
                return out
            if is_up is not None:
                try:
                    if is_up(peer_id):
                        return {"status": "running"}
                except Exception:  # noqa: BLE001 - a broken probe must not fail a good spawn
                    logger.debug("settle: presence probe raised for %s", peer_id, exc_info=True)
            if _now() >= deadline:
                # Alive but unannounced. Say exactly that.
                return {"status": "starting", "waited_s": round(max(0.0, timeout), 2)}
            _sleep(min(poll, max(0.0, deadline - _now())))

    def remember_profile(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Save this spawn payload as the profile for the board behind its port.

        Keyed by the port's USB serial number so a replug on a different
        ``/dev/cu.*`` path still finds it. Returns the stored profile, or None
        when the board is no longer enumerable (nothing to key on).
        """
        port = payload.get("port")
        if not port:
            return None
        try:
            detected = {p.get("device"): p for p in scan_serial_ports()}
        except Exception as e:
            logger.warning("could not rescan serial ports to save a profile: %r", e)
            return None
        info = detected.get(port)
        key = profile_key(info) if info else str(port)
        if not key:
            return None
        return self.profiles.save(key, payload)

    def start_autospawn(
        self,
        list_ports: Callable[[], list[dict[str, Any]]] | None = None,
        peer_ids: Callable[[], Iterable[str]] | None = None,
    ) -> AutoSpawnWatcher | None:
        """Create the USB auto-spawn watcher, or None when disabled by env."""
        if not AutoSpawnWatcher.enabled():
            logger.info("USB auto-spawn disabled (STRANDS_DASHBOARD_AUTOSPAWN=0)")
            return None
        self.autospawn = AutoSpawnWatcher(self, list_ports=list_ports, peer_ids=peer_ids)
        return self.autospawn

    def replay(
        self,
        repo_id: str,
        episode: int = 0,
        root: str | None = None,
        speed: float = 1.0,
        robot_name: str = "so101",
    ) -> dict[str, Any]:
        """One-shot replay sim: spawns a mesh sim peer, replays the episode
        with real physics + cameras streaming on the mesh, exits when done.

        The peer appears in the fleet grid like any sim while the replay
        runs - the operator literally watches the recorded episode through
        the mesh camera rail. replay_episode is in-process-only upstream
        (not a wire action), so a dedicated short-lived process is the way
        a robot-less dashboard can drive it.
        """
        import json as _json

        peer_id = f"replay-{int(time.time()) % 100000}"
        cfg = {
            "peer_id": peer_id, "repo_id": repo_id, "episode": episode,
            "root": root, "speed": speed, "robot_name": robot_name,
        }
        with self._lock:
            proc = subprocess.Popen(
                [sys.executable, "-c", _REPLAY_SPAWNER, _json.dumps(cfg)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            managed = ManagedRobot(
                peer_id=peer_id, robot_name=robot_name, mode="replay",
                process=proc, started_at=time.time(),
            )
            self.robots[peer_id] = managed
            threading.Thread(
                target=_drain, args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}", daemon=True,
            ).start()
        logger.info("replay %s ep%d as %s (pid=%s)", repo_id, episode, peer_id, proc.pid)
        return {"peer_id": peer_id, "pid": proc.pid, "repo_id": repo_id, "episode": episode}

    def collect(
        self,
        dataset_root: str,
        dataset_repo_id: str = "local/collected",
        robot_name: str = "so101",
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        n_episodes: int = 5,
        duration: float = 10.0,
        fps: int = 30,
    ) -> dict[str, Any]:
        """One-shot data collection: spawn a mesh sim, roll out a policy for
        N recorded episodes (parquet-truth verified by the run_policy tool),
        exit. The dataset lands where the Training tab's discovery scans.

        This is scripted collection (policy demos); teleop-driven human
        demos additionally need a leader arm attached to a real robot peer.
        """
        import json as _json

        peer_id = f"collect-{int(time.time()) % 100000}"
        cfg = {
            "peer_id": peer_id, "robot_name": robot_name,
            "policy_provider": policy_provider, "policy_config": policy_config,
            "instruction": instruction, "n_episodes": n_episodes,
            "duration": duration, "dataset_root": dataset_root,
            "dataset_repo_id": dataset_repo_id, "fps": fps,
        }
        with self._lock:
            proc = subprocess.Popen(
                [sys.executable, "-c", _COLLECT_SPAWNER, _json.dumps(cfg)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            managed = ManagedRobot(
                peer_id=peer_id, robot_name=robot_name, mode="collect",
                process=proc, started_at=time.time(),
            )
            self.robots[peer_id] = managed
            threading.Thread(
                target=_drain, args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}", daemon=True,
            ).start()
        logger.info("collect %d eps -> %s as %s (pid=%s)", n_episodes, dataset_root, peer_id, proc.pid)
        return {"peer_id": peer_id, "pid": proc.pid, "dataset_root": dataset_root, "n_episodes": n_episodes}

    def despawn(self, peer_id: str) -> dict[str, Any]:
        with self._lock:
            m = self.robots.get(peer_id)
            if m is None:
                return {"error": f"unknown peer {peer_id}"}
            if m.process is not None and m.alive():
                m.process.terminate()
                try:
                    m.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    m.process.kill()
            del self.robots[peer_id]
        return {"peer_id": peer_id, "stopped": True}

    def shutdown(self) -> None:
        for pid in list(self.robots):
            self.despawn(pid)
