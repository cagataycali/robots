"""USB device + camera discovery and robot process lifecycle.

Detects:
* Serial servo buses (Feetech/Dynamixel USB adapters). Known VIDs:
  - 0x1a86 WCH CH34x  (SO-100/SO-101 controller boards - enumerate as
    "USB Single Serial" on macOS, so keyword matching alone misses them;
    this is bug #5 in BUGS.md)
  - 0x0403 FTDI
* Local cameras via OpenCV index probe (dashboard is the sole owner of
  local USB cams - the neon lesson; robots opened by the dashboard get
  camera configs pointing at these indices).

Robot lifecycle: spawns `Robot(..., mode="real"|"sim").run()`-style child
processes and tracks them, so a detected arm becomes a mesh peer with one
click from the UI.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

LOG_TAIL_LINES = 200  # ring buffer per managed robot

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
    # Last N lines of child output. The pipe MUST be drained (BUGS.md #14):
    # with stdout=PIPE and no reader, the child blocks in print() once the
    # OS pipe buffer (~64KB) fills and the peer silently freezes.
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def _drain(proc: subprocess.Popen, logs: deque[str], peer_id: str) -> None:
    """Continuously read child stdout so the pipe never fills (bug #14)."""
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

    ``skip`` indices are NOT opened (BUGS.md #16: probing an index owned by a
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
                    cams.append({"index": i, "width": w, "height": h})
        finally:
            cap.release()
    return cams


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
    # (HardwareRobot otherwise connects lazily on the first task).
    inner = getattr(robot, "robot", None)
    if inner is not None and not getattr(inner, "is_connected", False):
        try:
            inner.connect(False)  # calibrate=False
            print("hardware connected", flush=True)
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


class DeviceManager:
    """Owns local device discovery + robot child processes."""

    CAMERA_CACHE_TTL_S = 30.0  # bug #16: don't re-open /dev cameras per request

    def __init__(self) -> None:
        self.robots: dict[str, ManagedRobot] = {}
        self._lock = threading.Lock()
        self._camera_cache: list[dict[str, Any]] = []
        self._camera_cache_t = 0.0

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
            "managed": {
                pid: {
                    "peer_id": m.peer_id, "robot_name": m.robot_name, "mode": m.mode,
                    "port": m.port, "alive": m.alive(), "started_at": m.started_at,
                    "log_tail": list(m.logs)[-20:],
                }
                for pid, m in self.robots.items()
            },
        }

    def logs(self, peer_id: str) -> dict[str, Any]:
        """Full ring buffer for one managed robot (drained per bug #14)."""
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
            # Drain stdout forever (bug #14) - daemon thread, ring buffer.
            threading.Thread(
                target=_drain, args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}", daemon=True,
            ).start()
        logger.info("spawned %s (%s, pid=%s)", peer_id, mode, proc.pid)
        return {"peer_id": peer_id, "pid": proc.pid, "mode": mode}

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
