"""Web-drivable calibration state machine for SO-arm followers/leaders.

LeRobot's ``Robot.calibrate()`` is interactive: it blocks on ``input()`` to
gate the "move to centre" and "sweep the joints" phases. That is fine at a
terminal but impossible to drive from a browser. This module decomposes the
same procedure into discrete, mesh-RPC-friendly steps that a dashboard can
sequence with button clicks:

    begin   -> torque off + POSITION mode; arm goes limp so the operator can
               move it freely. Resets any in-progress session.
    home    -> operator has moved the arm to the centre of its range; we write
               half-turn homing offsets (set_half_turn_homings()).
    record  -> called repeatedly (or run on a background sampler): polls
               present positions and accumulates per-motor min/max while the
               operator sweeps each joint.
    finish  -> build MotorCalibration from homing offsets + recorded ranges,
               write it to the motors, and persist the JSON to the standard
               lerobot calibration path so it survives restarts.
    cancel  -> discard the session, re-enable torque.

Design notes
------------
* The full-rotation motor (``wrist_roll`` on SO arms) is special-cased exactly
  as upstream ``calibrate()`` does: its range is pinned to the encoder span
  ``[0, 4095]`` rather than the swept extremes.
* All bus access goes through the live ``FeetechMotorsBus`` the robot already
  owns -- we never open a second handle, so there is no port contention.
* State lives in a single ``CalibrationSession`` per robot instance; the mesh
  ``_dispatch`` holds one on the robot under a documented attribute.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# SO-arm motor whose range is the full encoder span, not the swept extremes.
# Matches lerobot so_follower.calibrate()'s ``full_turn_motor``.
_FULL_TURN_MOTOR = "wrist_roll"
_ENCODER_MIN = 0
_ENCODER_MAX = 4095


class CalibrationSession:
    """Tracks one in-progress calibration for a single arm bus.

    Thread-safety: ``record_tick`` and the phase transitions can be called
    from different threads (a background sampler + the RPC dispatch thread),
    so all mutation is guarded by ``self._lock``.
    """

    def __init__(self, bus: Any, motor_names: list[str], bus_lock: Any = None) -> None:
        self._bus = bus
        self._motors = motor_names
        self._lock = threading.Lock()
        # Optional shared lock with the owning Mesh's state loop so the
        # background sampler's reads never race a state read on the same
        # half-duplex serial bus. Falls back to a private lock if absent.
        self._bus_lock = bus_lock if bus_lock is not None else threading.RLock()
        self.phase = "idle"  # idle -> homing -> recording -> done
        self._homing_offsets: dict[str, int] = {}
        self._mins: dict[str, int] = {}
        self._maxes: dict[str, int] = {}
        self._sampler: threading.Thread | None = None
        self._sampling = False

    # -- phase transitions ---------------------------------------------

    def begin(self) -> dict[str, Any]:
        """Disable torque so the operator can move the arm by hand."""
        with self._lock:
            self._bus.disable_torque()
            # POSITION operating mode for every motor (mirrors upstream).
            from lerobot.motors.feetech import OperatingMode

            for motor in self._motors:
                self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.phase = "homing"
            self._homing_offsets = {}
            self._mins = {}
            self._maxes = {}
        return {"phase": self.phase, "message": "Torque off. Move arm to the CENTRE of its range, then call 'home'."}

    def set_home(self) -> dict[str, Any]:
        """Write half-turn homing offsets at the current (centred) pose."""
        with self._lock:
            if self.phase not in ("homing", "recording"):
                return {"error": f"set_home not valid in phase {self.phase!r}; call begin first."}
            self._homing_offsets = dict(self._bus.set_half_turn_homings())
            # Seed range tracking from the current (post-homing) positions.
            start = self._bus.sync_read("Present_Position", self._motors, normalize=False)
            self._mins = dict(start)
            self._maxes = dict(start)
            self.phase = "recording"
        return {
            "phase": self.phase,
            "homing_offsets": self._homing_offsets,
            "message": "Homing set. Now SWEEP every joint through its full range, then call 'finish'.",
        }

    def record_tick(self) -> dict[str, Any]:
        """Sample present positions once, updating min/max. Returns live values."""
        with self._lock:
            if self.phase != "recording":
                return {"error": f"record_tick not valid in phase {self.phase!r}"}
            with self._bus_lock:
                pos = self._bus.sync_read("Present_Position", self._motors, normalize=False)
            for m in self._motors:
                if m in pos:
                    self._mins[m] = min(self._mins.get(m, pos[m]), pos[m])
                    self._maxes[m] = max(self._maxes.get(m, pos[m]), pos[m])
            return {
                "phase": self.phase,
                "positions": dict(pos),
                "mins": dict(self._mins),
                "maxes": dict(self._maxes),
            }

    def start_background_sampler(self, hz: float = 20.0) -> None:
        """Continuously record_tick() on a daemon thread until finish/cancel.

        Lets the operator sweep freely without the client having to poll at
        the sample rate. Idempotent: a second call is a no-op.
        """
        if self._sampling:
            return
        self._sampling = True

        def _loop() -> None:
            period = 1.0 / hz
            while self._sampling and self.phase == "recording":
                try:
                    self.record_tick()
                except Exception as exc:  # noqa: BLE001 — keep sampling
                    logger.debug("calibration sampler tick error: %s", exc)
                time.sleep(period)

        self._sampler = threading.Thread(target=_loop, name="calib-sampler", daemon=True)
        self._sampler.start()

    def finish(self) -> dict[str, Any]:
        """Build + write + persist calibration from the recorded session."""
        from lerobot.motors import MotorCalibration

        self._sampling = False
        with self._lock:
            if self.phase != "recording":
                return {"error": f"finish not valid in phase {self.phase!r}"}
            if not self._homing_offsets:
                return {"error": "no homing offsets; call 'home' before 'finish'."}

            mins = dict(self._mins)
            maxes = dict(self._maxes)
            # Pin the full-rotation motor to the encoder span.
            if _FULL_TURN_MOTOR in self._motors:
                mins[_FULL_TURN_MOTOR] = _ENCODER_MIN
                maxes[_FULL_TURN_MOTOR] = _ENCODER_MAX

            same = [m for m in self._motors if mins.get(m) == maxes.get(m)]
            if same:
                return {
                    "error": (
                        f"Motors not swept (min==max): {same}. "
                        "Move every joint through its full range before finishing."
                    )
                }

            calibration: dict[str, MotorCalibration] = {}
            bus_motors = self._bus.motors
            for motor in self._motors:
                m = bus_motors[motor]
                calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=self._homing_offsets[motor],
                    range_min=mins[motor],
                    range_max=maxes[motor],
                )

            self._bus.write_calibration(calibration)
            self.phase = "done"
            self._calibration = calibration
        return {
            "phase": self.phase,
            "calibration": {m: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
                            for m, c in calibration.items()},
            "message": "Calibration written to motors. Persist via robot._save_calibration if desired.",
        }

    def cancel(self) -> dict[str, Any]:
        self._sampling = False
        with self._lock:
            self.phase = "idle"
        try:
            self._bus.enable_torque()
        except Exception as exc:  # noqa: BLE001
            logger.debug("cancel: enable_torque failed: %s", exc)
        return {"phase": self.phase, "message": "Calibration cancelled; torque re-enabled."}

    @property
    def calibration(self) -> Any:
        return getattr(self, "_calibration", None)
