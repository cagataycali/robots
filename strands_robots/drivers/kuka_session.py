"""The FRI session process behind :class:`~strands_robots.drivers.kuka.KukaDriver`.

``pyfri``'s ``ClientApplication.step()`` blocks in ``recvfrom`` holding the GIL
until the controller's next datagram arrives, so it cannot share a process with
anything that must keep running when the controller goes quiet. This module runs
as its own script, run by file path: it imports only the standard
library and ``pyfri``, answers the controller every FRI cycle, and talks to the
driver over two pipes.

* Out, every cycle: one :data:`STATE` frame (measured, interpolated and
  commanded joint positions, measured and external torques, then the cycle
  count, session/safety/drive/quality/command-mode enums and the sample time).
  The pipe is non-blocking and a frame is smaller than ``PIPE_BUF``, so a frame
  is written whole or dropped, never torn, and a slow reader never stalls the
  FRI cycle.
* In, whenever the driver writes: one :data:`TARGET` frame (seven joint
  targets, a valid flag, a stop flag). Each cycle drains the pipe and keeps the
  newest frame.

In ``COMMANDING_ACTIVE`` the commanded position moves from the last one sent
towards the target by at most ``MAX_JOINT_SPEED * sample time`` per cycle; with
no valid target it holds the last position sent. Each new session phase starts
from the controller's interpolated position, and the target is dropped.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import importlib
import math
import os
import signal
import struct
import sys
from typing import Any

#: Joints per vector.
N_JOINTS = 7

#: Largest joint speed per axis, rad/s: the ``velocity`` rows (85, 85, 100, 75,
#: 130, 135, 135 deg/s) of lbr-stack's ``lbr_description/urdf/iiwa14/joint_limits.yaml``.
MAX_JOINT_SPEED: tuple[float, ...] = tuple(math.radians(d) for d in (85, 85, 100, 75, 130, 135, 135))

#: Session process -> driver: five joint vectors, then seven scalars.
STATE = struct.Struct(f"<{5 * N_JOINTS + 7}d")

#: Driver -> session process: the target, ``valid`` and ``stop``.
TARGET = struct.Struct(f"<{N_JOINTS + 2}d")

#: Exit code when the UDP port could not be bound.
EXIT_BIND_FAILED = 2

#: Exit code when ``pyfri`` reports every joint as the first one (see :func:`copies_the_first_joint`).
EXIT_BROKEN_BINDING = 3

_PR_SET_PDEATHSIG = 1


def _drain(fd: int, frame: struct.Struct, buffer: bytearray) -> tuple[float, ...] | None:
    """Read everything waiting on ``fd``; return the newest whole frame, or ``None``."""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            break
        if not chunk:
            break
        buffer.extend(chunk)
    whole = len(buffer) // frame.size
    if not whole:
        return None
    newest = frame.unpack_from(buffer, (whole - 1) * frame.size)
    del buffer[: whole * frame.size]
    return newest


def copies_the_first_joint(state: Any) -> bool:
    """Whether ``pyfri`` hands back the first joint's value for all seven.

    A ``pyfri`` built with the pybind11 it vendors (2.11) and run under numpy 2
    returns every joint array as its first element repeated, while the C++ SDK
    underneath decodes the message correctly. Commanding from such a reading
    would send all seven joints to A1's angle. A real arm never reports seven
    bitwise-equal positions together with seven bitwise-equal torques unless
    both are all zero, which this check lets through.
    """
    vectors = [[float(v) for v in getter()] for getter in (state.getMeasuredJointPosition, state.getMeasuredTorque)]
    return all(len(set(v)) == 1 for v in vectors) and any(v[0] != 0.0 for v in vectors)


def run_session(sdk: Any, inbox: int, outbox: int, host: str | None, fri_port: int) -> int:
    """Answer the controller every FRI cycle until told to stop or the session ends.

    Args:
        sdk: The ``pyfri`` module (a test passes a fake).
        inbox: File descriptor :data:`TARGET` frames arrive on.
        outbox: File descriptor :data:`STATE` frames are written to.
        host: The controller address to accept datagrams from, or ``None`` for any.
        fri_port: The local UDP port the controller sends to.

    Returns:
        ``0`` when the session ended, :data:`EXIT_BIND_FAILED` when the port could not be bound.
    """
    os.set_blocking(inbox, False)
    os.set_blocking(outbox, False)
    pending = bytearray()
    latest: dict[str, Any] = {"target": None, "stop": False, "cycles": 0, "broken": False}

    def poll_inbox() -> None:
        frame = _drain(inbox, TARGET, pending)
        if frame is not None:
            latest["target"] = list(frame[:N_JOINTS]) if frame[N_JOINTS] else None
            latest["stop"] = bool(frame[N_JOINTS + 1])

    class _Client(sdk.LBRClient):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.hold: list[float] | None = None

        def _cycle(self) -> Any:
            poll_inbox()
            latest["cycles"] += 1
            state = self.robotState()
            if latest["cycles"] == 1 and copies_the_first_joint(state):
                latest["broken"] = latest["stop"] = True
            return state

        def _publish(self, state: Any) -> None:
            commanded = self.hold if self.hold is not None else [float(v) for v in state.getIpoJointPosition()]
            values = [
                *(float(v) for v in state.getMeasuredJointPosition()),
                *(float(v) for v in state.getIpoJointPosition()),
                *commanded,
                *(float(v) for v in state.getMeasuredTorque()),
                *(float(v) for v in state.getExternalTorque()),
                float(latest["cycles"]),
                float(int(state.getSessionState())),
                float(int(state.getSafetyState())),
                float(int(state.getDriveState())),
                float(int(state.getConnectionQuality())),
                float(int(state.getClientCommandMode())),
                float(state.getSampleTime()),
            ]
            try:
                os.write(outbox, STATE.pack(*values))
            except BlockingIOError:
                pass  # the driver is behind; it reads the next frame instead
            except BrokenPipeError:
                latest["stop"] = True

        def _send(self, position: list[float]) -> None:
            if latest["broken"]:
                return
            self.hold = position
            self.robotCommand().setJointPosition(position)

        def onStateChange(self, old_state: Any, new_state: Any) -> None:
            # A new phase starts from the controller's pose, never from a stale target.
            self.hold = None
            latest["target"] = None

        def monitor(self) -> None:
            state = self._cycle()
            if not latest["broken"]:
                self._publish(state)

        def waitForCommand(self) -> None:
            state = self._cycle()
            self._send([float(v) for v in state.getIpoJointPosition()])
            if not latest["broken"]:
                self._publish(state)

        def command(self) -> None:
            state = self._cycle()
            hold = self.hold if self.hold is not None else [float(v) for v in state.getIpoJointPosition()]
            target = latest["target"]
            if target is None:
                self._send(hold)
            else:
                limit = float(state.getSampleTime())
                self._send(
                    [
                        now + max(-speed * limit, min(speed * limit, goal - now))
                        for now, goal, speed in zip(hold, target, MAX_JOINT_SPEED, strict=True)
                    ]
                )
            if not latest["broken"]:
                self._publish(state)

    client = _Client()
    app = sdk.ClientApplication(client)
    if not app.connect(fri_port, host):
        return EXIT_BIND_FAILED
    try:
        while not latest["stop"]:
            if not app.step():
                break
            if latest["cycles"] and int(client.robotState().getSessionState()) == 0:
                break  # the controller closed the session (IDLE)
            poll_inbox()
    finally:
        app.disconnect()
    return EXIT_BROKEN_BINDING if latest["broken"] else 0


def main(argv: list[str] | None = None) -> int:
    """Script entry point: ``--inbox FD --outbox FD --fri-port N [--host IP] [--parent PID]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inbox", type=int, required=True)
    parser.add_argument("--outbox", type=int, required=True)
    parser.add_argument("--fri-port", type=int, required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--parent", type=int, default=None)
    args = parser.parse_args(argv)
    if sys.platform.startswith("linux"):
        # Die with the driver: an orphan blocked in recvfrom would hold the FRI port forever.
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if args.parent is not None and os.getppid() != args.parent:
        return 0  # the driver exited before the death signal was armed
    try:
        return run_session(importlib.import_module("pyfri"), args.inbox, args.outbox, args.host, args.fri_port)
    except OSError as exc:
        if exc.errno == errno.EPIPE:
            return 0
        raise


if __name__ == "__main__":
    sys.exit(main())
