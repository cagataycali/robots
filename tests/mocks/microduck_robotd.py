"""A mock ``robotd`` on a real unix socket, for the Microduck driver tests.

This is the driver's proof bar: ``Robot("microduck", mode="real")`` "works
as-is on the robot" only if the driver speaks a byte-faithful robotd protocol.
So the tests do not monkeypatch the driver's socket - they stand up a *real*
``AF_UNIX`` server that speaks the exact ``duck-ipc-proto`` wire format and let
the driver's own :class:`~strands_robots.drivers.microduck._RobotdClient` talk
to it over a genuine socket.

Faithfulness rules, so a mock frame that drifts from a real robotd frame fails
the test rather than passing a broken driver:

* The state frame is the **verbatim literal** from ``duck-ipc-proto/src/lib.rs``'s
  ``robot_state_uses_the_documented_field_names`` test - the wire keys ``move``
  and ``loop`` (never ``movement``/``control_loop``), the deadman-limited twist
  ``requested [0.4,0,0] applied [0.15,0,0]``, ``safety.gain 200``, ``loop.hz
  49.8``. Only the 15-wide ``joints``/``targets`` are made distinct
  (``0..14``) so the 15->14 mouth-drop is observable.
* Discrete calls answer an ``IntentResult`` ``{accepted, reason?}``; ``hello``
  answers ``HelloResult`` ``{api_version, daemon_version, revision}``;
  ``robot.health`` answers a ``Battery`` under ``battery``.
* Notifications (``robot.move``/``head``/``pose``/``mouth``) get **no** reply,
  as robotd gives none, and their raw bytes are recorded for the byte-exact
  assertions.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from typing import Any

from strands_robots.drivers.microduck import MICRODUCK_API_VERSION

# The exact RobotState literal from duck-ipc-proto's own serialization test,
# except joints/targets which carry 0..14 so the mouth-drop (index 9) is visible.
STATE_PARAMS: dict[str, Any] = {
    "t": 1.5,
    "move": {"requested": [0.4, 0.0, 0.0], "applied": [0.15, 0.0, 0.0], "limited_by": ["deadman"]},
    "head": [0.0, 0.0, 0.0, 0.0],
    "policy": "walk",
    "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0], "gain": 200},
    "loop": {"hz": 49.8, "missed": 0},
    "joints": [float(i) for i in range(15)],
    "targets": [float(i) for i in range(15)],
    "odom": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
}


class MockRobotd:
    """A one-connection robotd server on a real unix socket.

    Attributes:
        path: The socket path to hand a driver as ``port=``.
        received: Every raw line the client sent, in order (bytes).
        methods: Every method name received, in order.
    """

    def __init__(
        self,
        *,
        api_version: int = MICRODUCK_API_VERSION,
        state_interval: float = 0.01,
        skills: tuple[str, ...] = ("ground_pick", "kick_left", "kick_right", "sit_toggle", "roulade"),
        mode: str = "walk",
        sitting: bool | None = False,
        decline: dict[str, str] | None = None,
        path: str | None = None,
    ) -> None:
        """``decline`` maps a method name to the ``reason`` its IntentResult refuses with.

        ``path`` binds the socket at a caller-chosen location (a forward's local end).
        """
        self._api_version = api_version
        self._state_interval = state_interval
        self.skills = list(skills)
        self.mode = mode
        self.sitting = sitting
        self.decline = dict(decline or {})
        self._dir = tempfile.mkdtemp(prefix="mock-robotd-")
        self.path = path or os.path.join(self._dir, "robotd.sock")
        self.received: list[bytes] = []
        self.methods: list[str] = []
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(1)
        self._stop = threading.Event()
        self._streaming = threading.Event()
        self._conn: socket.socket | None = None
        self._thread = threading.Thread(target=self._serve, name="mock-robotd", daemon=True)

    def __enter__(self) -> MockRobotd:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        self._streaming.clear()
        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass  # teardown best-effort; an already-closed socket is fine
        try:
            os.unlink(self.path)
        except OSError:
            pass  # the socket file may never have been bound, or be gone already

    # -- server internals ------------------------------------------------ #

    def _send(self, conn: socket.socket, obj: dict[str, Any]) -> None:
        conn.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))

    def _serve(self) -> None:
        self._server.settimeout(5.0)
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        self._conn = conn
        conn.settimeout(0.2)
        streamer = threading.Thread(target=self._stream_states, args=(conn,), daemon=True)
        streamer.start()
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._handle(conn, line + b"\n")

    def _handle(self, conn: socket.socket, raw: bytes) -> None:
        obj = json.loads(raw)
        method = obj.get("method")
        with self._lock:
            self.received.append(raw)
            self.methods.append(method)
        request_id = obj.get("id")

        # Notifications (continuous intents) get no reply.
        if request_id is None:
            return

        if method == "hello":
            self._send(
                conn,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"api_version": self._api_version, "daemon_version": "1.2.3", "revision": None},
                },
            )
        elif method == "robot.subscribe":
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": self._policies()})
            self._streaming.set()
        elif method == "robot.policies":
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": self._policies()})
        elif method == "robot.mode":
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": {"mode": self.mode}})
        elif method == "robot.modelApi":
            self._send(
                conn, {"jsonrpc": "2.0", "id": request_id, "result": {"api_version": 3, "duck_control": "0.14.1"}}
            )
        elif method == "robot.model":
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": {"name": "microduck", "joints": 15}})
        elif method in self.decline:
            self._send(
                conn,
                {"jsonrpc": "2.0", "id": request_id, "result": {"accepted": False, "reason": self.decline[method]}},
            )
        elif method == "robot.look":
            params = obj.get("params") or {}
            self._send(
                conn,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "accepted": True,
                        "head": {"neck_pitch": 0.1, "head_pitch": 0.0, "head_yaw": 0.2, "head_roll": 0.0},
                        "clamped": abs(float(params.get("y", 0.0))) > 1.0,
                    },
                },
            )
        elif method == "robot.setMode":
            self.mode = str((obj.get("params") or {}).get("mode", self.mode))
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": {"accepted": True}})
        elif method == "robot.do":
            if (obj.get("params") or {}).get("skill") == "sit_toggle" and self.sitting is not None:
                self.sitting = not self.sitting
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": {"accepted": True}})
        elif method == "robot.health":
            self._send(
                conn,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"healthy": True, "battery": {"volts": 7.4, "percent": 82.0}, "bus": {}},
                },
            )
        else:  # robot.do / robot.enable / robot.relax / robot.stop / robot.init
            self._send(conn, {"jsonrpc": "2.0", "id": request_id, "result": {"accepted": True}})

    def _policies(self) -> dict[str, Any]:
        """The v28+ ``robot.policies``/``robot.subscribe`` result shape."""
        result: dict[str, Any] = {
            "accepted": True,
            "walk": "alpha_walking.onnx",
            "stand": "stand.onnx",
            "skills": list(self.skills),
            "mode": self.mode,
            "homed": True,
        }
        if self.sitting is not None:
            result["sitting"] = self.sitting
        return result

    def _stream_states(self, conn: socket.socket) -> None:
        if not self._streaming.wait(timeout=5.0):
            return
        while not self._stop.is_set():
            try:
                self._send(conn, {"jsonrpc": "2.0", "method": "robot.state", "params": STATE_PARAMS})
            except OSError:
                return
            time.sleep(self._state_interval)
