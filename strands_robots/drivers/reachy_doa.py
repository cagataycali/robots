"""Turn toward whoever is talking - Direction of Arrival for the Reachy Mini.

The Wireless Mini's ReSpeaker array reports a Direction of Arrival that the
daemon serves at ``GET /api/state/doa`` (and inside ``/api/state/full`` with
``with_doa=true``): ``{"angle": radians, "speech_detected": bool}`` in Pollen's
convention - ``0`` is the robot's **left**, ``pi/2`` is straight ahead (or
straight behind: a linear array cannot tell front from back), ``pi`` is the
**right**. This module turns that stream into at most one smooth head turn per
utterance, so a Mini with no face in view still faces the person speaking.

Two pieces, split so the judgement is testable without a robot:

* :class:`DoaTurner` is a pure decision engine. Feed it state frames with
  :meth:`DoaTurner.observe` and it answers ``None`` (and says why in
  :attr:`DoaTurner.why`) or a :class:`TurnPlan` naming the head yaw, the body
  yaw and the pitch to command. It never touches a daemon.
* :class:`DoaLoop` runs a turner on a daemon thread: it polls a ``read_frame``
  callable, hands each frame to the turner, and calls a ``look`` callable with
  the plan. The driver that owns it stops it from ``cleanup()``/``stop()``, so
  the loop's lifetime is the driver's.

The rules are ported from ``cagataycali/tiny-the-reachy`` ``dashboard/doa.py``,
where they were tuned on the desk robot:

1. speech must be present on :data:`MIN_CONSEC` consecutive frames whose angles
   agree within :data:`TOL_DEG` - one loud click is not a speaker;
2. readings at the array's rails (``0`` / ``pi``, within :data:`RAIL_DEG`) are
   discarded - the daemon reports those for "no estimate" and for the robot's
   own speaker;
3. the yaw delta is ``sign * (pi/2 - angle)``; below :data:`MIN_DELTA_DEG` the
   robot is already facing the speaker and does not twitch;
4. the head takes up to :data:`HEAD_MAX_DEG` of the turn and the body carries
   the rest, bounded by :data:`BODY_MAX_DEG`, so a voice from behind swings the
   base rather than twisting the neck;
5. at most one turn per :data:`RATE_S`; a same-direction turn within
   :data:`WINDUP_S` whose bearing did not shrink is a *windup* (the sound is
   moving with the robot: its own speaker, an echo) and is refused;
6. a caller-supplied ``blocked()`` veto - the driver uses it to say "the face
   tracker has a fresh lock" or "a recorded move is playing", both of which
   outrank a bearing.

The sign is EMPIRICAL. Pollen's convention makes ``+yaw`` the robot's left and
``0`` rad the left mic, so the default ``+1`` is what the SDK implies, but the
array can be mounted either way round; :attr:`DoaTurner.sign` is a constructor
argument and the owner confirms it by speaking from the robot's left once.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from strands_robots.mesh.pacing import Ticker

logger = logging.getLogger(__name__)

#: Consecutive speech frames whose bearings must agree before a turn.
MIN_CONSEC: int = 6
#: Agreement tolerance between those frames, degrees.
TOL_DEG: float = 12.0
#: Minimum seconds between two turns.
RATE_S: float = 3.0
#: A bearing closer than this to straight ahead is "already facing".
MIN_DELTA_DEG: float = 10.0
#: How far the head alone may turn; the body carries the remainder.
HEAD_MAX_DEG: float = 45.0
#: How far a DoA turn may ever swing the body.
BODY_MAX_DEG: float = 60.0
#: Readings within this many degrees of ``0``/``pi`` are the array's rails, not bearings.
RAIL_DEG: float = 3.0
#: A same-direction turn within this window whose bearing did not shrink is a windup.
WINDUP_S: float = 10.0
#: "Did not shrink" means the new |delta| is at least this fraction of the last one.
WINDUP_RATIO: float = 0.6
#: A frame gap longer than this breaks the streak.
STALE_S: float = 1.0
#: Duration of the commanded turn, seconds.
TURN_S: float = 0.7
#: Pitch is carried over from the current pose but kept in this window.
PITCH_WINDOW_DEG: tuple[float, float] = (-20.0, 15.0)
#: Default polling period of :class:`DoaLoop`, seconds (the daemon's own DoA cadence is ~10 Hz).
POLL_S: float = 0.1


@dataclass(frozen=True)
class TurnPlan:
    """One turn the turner asks for: absolute degrees for the daemon's ``goto``.

    Attributes:
        yaw: Head yaw to command, degrees, within +/-:data:`HEAD_MAX_DEG`.
        body_yaw: Body yaw to command, degrees, or ``None`` when the head alone
            covers the bearing and the body is left where it is.
        pitch: Head pitch to command, degrees - the current pitch clamped into
            :data:`PITCH_WINDOW_DEG`, so a turn does not also nod.
        from_yaw: The head yaw the robot was at when the plan was made.
        angle_deg: The mean DoA bearing that produced the plan, degrees.
        delta_deg: The signed yaw delta the bearing implies, degrees.
        duration: Seconds the turn should take.
    """

    yaw: float
    body_yaw: float | None
    pitch: float
    from_yaw: float
    angle_deg: float
    delta_deg: float
    duration: float = TURN_S


def _finite(value: Any) -> float | None:
    """Return ``value`` as a float when it is a finite real number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _frame_doa(frame: Any) -> tuple[float, bool] | None:
    """Pull ``(angle_rad, speech_detected)`` out of a state frame, or ``None``.

    Accepts both shapes the daemon produces: the ``/api/state/full`` frame that
    nests ``doa`` under its own key, and the bare ``/api/state/doa`` body.
    """
    if not isinstance(frame, dict):
        return None
    doa = frame.get("doa", frame)
    if not isinstance(doa, dict):
        return None
    angle = _finite(doa.get("angle"))
    if angle is None:
        return None
    return angle, bool(doa.get("speech_detected"))


class DoaTurner:
    """Decide, frame by frame, whether the Mini should turn toward a voice."""

    def __init__(self, *, sign: float = 1.0, frame: str = "head", enabled: bool = True) -> None:
        """Build a turner.

        Args:
            sign: ``+1`` when ``+yaw`` is the side the array reports as ``0``
                rad (the SDK convention), ``-1`` for a mirrored mount. Only the
                sign of the value is used.
            frame: ``"head"`` adds the delta to the current head yaw (the mics
                move with the head, so the bearing is relative to it);
                ``"body"`` treats the delta as an absolute body-frame yaw.
            enabled: Whether the turner may plan turns. A disabled turner still
                tracks the bearing so :meth:`status` stays informative.

        Raises:
            ValueError: If ``frame`` is not ``"head"`` or ``"body"``.
        """
        if frame not in ("head", "body"):
            raise ValueError(f"DoaTurner: frame must be 'head' or 'body', not {frame!r}")
        self.sign = 1.0 if sign >= 0 else -1.0
        self.frame = frame
        self.enabled = bool(enabled)
        self._streak: deque[float] = deque(maxlen=max(MIN_CONSEC, 8))
        self._last_frame_t: float | None = None
        self.last_angle: float | None = None
        self.speech = False
        self.turns = 0
        self.windups = 0
        self.last_turn_t: float | None = None
        self.last_turn: TurnPlan | None = None
        self.why: str | None = None

    # ------------------------------------------------------------------ #
    # Public.                                                            #
    # ------------------------------------------------------------------ #

    def set_enabled(self, enabled: bool) -> None:
        """Switch planning on or off; either way the streak restarts."""
        self.enabled = bool(enabled)
        self._streak.clear()
        self.why = None if self.enabled else "disabled"

    def status(self) -> dict[str, Any]:
        """A JSON-ready snapshot of what the turner sees and last did."""
        return {
            "enabled": self.enabled,
            "armed": self._agree(),
            "speech": self.speech,
            "angle_deg": None if self.last_angle is None else round(math.degrees(self.last_angle), 1),
            "delta_deg": None if self.last_angle is None else round(self._delta_deg(self.last_angle), 1),
            "turns": self.turns,
            "windups": self.windups,
            "last_turn_t": self.last_turn_t,
            "last_turn": None if self.last_turn is None else asdict(self.last_turn),
            "why": self.why,
            "sign": self.sign,
            "frame": self.frame,
            "rules": {
                "min_consec": MIN_CONSEC,
                "tol_deg": TOL_DEG,
                "rate_s": RATE_S,
                "min_delta_deg": MIN_DELTA_DEG,
                "head_max_deg": HEAD_MAX_DEG,
                "body_max_deg": BODY_MAX_DEG,
                "rail_deg": RAIL_DEG,
                "windup_s": WINDUP_S,
            },
        }

    def observe(
        self,
        frame: dict[str, Any],
        *,
        now: float | None = None,
        blocked: Callable[[], str | None] | None = None,
    ) -> TurnPlan | None:
        """Take one state frame; return the turn it warrants, or ``None``.

        Args:
            frame: A daemon state frame carrying ``doa`` and, for the plan,
                ``head_pose`` (``{roll, pitch, yaw}`` in radians) and
                ``body_yaw`` (radians). Missing pose fields read as zero.
            now: Monotonic seconds, injectable for tests; defaults to
                :func:`time.monotonic`.
            blocked: Optional veto consulted only once the streak is armed - it
                returns a reason string to refuse this turn, or ``None``. Kept
                as a callable so the caller pays for its lookups (a tracked-face
                GET, a running-moves GET) only when a turn is imminent.

        Returns:
            A :class:`TurnPlan` when every rule passes, else ``None`` with the
            reason in :attr:`why`.
        """
        now = time.monotonic() if now is None else now
        reading = _frame_doa(frame)
        if reading is None:
            return None
        angle, speech = reading
        self.last_angle, self.speech = angle, speech
        if self._last_frame_t is not None and now - self._last_frame_t > STALE_S:
            self._streak.clear()
        self._last_frame_t = now

        if not speech:
            self._streak.clear()
            self.why = "no speech"
            return None
        deg = math.degrees(angle)
        if deg <= RAIL_DEG or deg >= 180.0 - RAIL_DEG:
            self._streak.clear()
            self.why = "rail reading"
            return None
        self._streak.append(angle)
        if not self.enabled:
            self.why = "disabled"
            return None
        if len(self._streak) < MIN_CONSEC:
            self.why = f"listening {len(self._streak)}/{MIN_CONSEC}"
            return None
        if not self._agree():
            self.why = "sound is moving"
            return None
        if self.last_turn_t is not None and now - self.last_turn_t < RATE_S:
            self.why = "rate limit"
            return None
        if blocked is not None and (reason := blocked()):
            self.why = reason
            return None

        tail = list(self._streak)[-MIN_CONSEC:]
        mean = sum(tail) / len(tail)
        delta = self._delta_deg(mean)
        if abs(delta) < MIN_DELTA_DEG:
            self.why = f"already facing ({delta:+.0f} deg)"
            return None
        last = self.last_turn
        if (
            last is not None
            and self.last_turn_t is not None
            and now - self.last_turn_t < WINDUP_S
            and last.delta_deg * delta > 0
            and abs(delta) >= WINDUP_RATIO * abs(last.delta_deg)
        ):
            self.windups += 1
            self._streak.clear()
            self.why = "windup guard"
            return None

        plan = self._plan(frame, mean, delta)
        self.last_turn_t = now
        self.last_turn = plan
        self.turns += 1
        self._streak.clear()
        self.why = None
        return plan

    # ------------------------------------------------------------------ #
    # Internals.                                                         #
    # ------------------------------------------------------------------ #

    def _delta_deg(self, angle: float) -> float:
        return self.sign * math.degrees(math.pi / 2 - angle)

    def _agree(self) -> bool:
        if len(self._streak) < MIN_CONSEC:
            return False
        tail = list(self._streak)[-MIN_CONSEC:]
        mean = sum(tail) / len(tail)
        return all(abs(math.degrees(a - mean)) <= TOL_DEG for a in tail)

    def _plan(self, frame: dict[str, Any], mean: float, delta: float) -> TurnPlan:
        pose = frame.get("head_pose")
        pose = pose if isinstance(pose, dict) else {}
        cur_yaw = math.degrees(_finite(pose.get("yaw")) or 0.0)
        cur_pitch = math.degrees(_finite(pose.get("pitch")) or 0.0)
        cur_body = math.degrees(_finite(frame.get("body_yaw")) or 0.0)
        target = (cur_yaw + delta) if self.frame == "head" else delta
        head = max(-HEAD_MAX_DEG, min(HEAD_MAX_DEG, target))
        rest = target - head
        body: float | None = None
        if abs(rest) > 1.0:
            body = round(max(-BODY_MAX_DEG, min(BODY_MAX_DEG, cur_body + rest)), 1)
        low, high = PITCH_WINDOW_DEG
        return TurnPlan(
            yaw=round(head, 1),
            body_yaw=body,
            pitch=round(max(low, min(high, cur_pitch)), 1),
            from_yaw=round(cur_yaw, 1),
            angle_deg=round(math.degrees(mean), 1),
            delta_deg=round(delta, 1),
        )


class DoaLoop:
    """Run a :class:`DoaTurner` on a daemon thread against live frames.

    The loop owns nothing but its thread: the frame source and the motion
    sink are callables the driver supplies, so the loop can be exercised with
    a scripted frame list and a recording ``look`` in a unit test, and on the
    robot it is the driver's ``GET /api/state/full?with_doa=true`` and its
    interpolated ``goto``.
    """

    def __init__(
        self,
        read_frame: Callable[[], dict[str, Any] | None],
        look: Callable[..., dict[str, Any]],
        *,
        blocked: Callable[[], str | None] | None = None,
        turner: DoaTurner | None = None,
        poll_s: float = POLL_S,
        name: str = "reachy-doa",
    ) -> None:
        """Build a stopped loop.

        Args:
            read_frame: Returns the latest daemon state frame, or ``None`` when
                it could not be read; exceptions are caught and counted.
            look: Called with ``yaw=, body_yaw=, pitch=, roll=0.0, duration=``
                in degrees/seconds; its envelope's ``status`` decides whether
                the turn is counted as sent.
            blocked: Forwarded to :meth:`DoaTurner.observe`.
            turner: The decision engine; a default one when ``None``.
            poll_s: Seconds between frames.
            name: Thread name, so a stuck loop is identifiable in a dump.
        """
        if not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError(f"DoaLoop: poll_s must be a positive finite number, not {poll_s!r}")
        self._read_frame = read_frame
        self._look = look
        self._blocked = blocked
        self.turner = turner or DoaTurner()
        self._poll_s = float(poll_s)
        self._name = name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        #: The running loop's pacer, so :meth:`stop` can interrupt its wait.
        self._ticker: Ticker | None = None
        self._lock = threading.Lock()
        self.read_errors = 0
        self.last_error: str | None = None
        self.last_sent: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        """Whether the polling thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start polling; a second call on a running loop is a no-op."""
        with self._lock:
            if self.running:
                return
            self._stop_event.clear()
            self.turner.set_enabled(True)
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop polling and wait for the thread. Idempotent; never raises."""
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
            ticker = self._ticker
        if ticker is not None:
            # Ring the doorbell so shutdown is immediate rather than within a
            # slice; wake() never raises, including after the ticker closed.
            ticker.wake()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("%s: the DoA loop did not stop within %.1fs", self._name, timeout)
        self.turner.set_enabled(False)

    def status(self) -> dict[str, Any]:
        """The turner's status plus the loop's own health."""
        return {
            **self.turner.status(),
            "running": self.running,
            "poll_s": self._poll_s,
            "read_errors": self.read_errors,
            "last_error": self.last_error,
            "last_sent": self.last_sent,
        }

    def _run(self) -> None:
        # Paced by Ticker, not by ``stop_event.wait(poll_s)``: every tick reads a
        # daemon state frame over the network, and a delay adds that read to the
        # period, so the nominal 10 Hz poll would run at 1 / (poll_s + read) and
        # the speech window would be sampled by fewer frames than the turner's
        # consecutive-frame rule counts on. A deadline subtracts the read instead.
        with Ticker(self._poll_s, self._stop_event) as ticker:
            self._ticker = ticker
            try:
                while not self._stop_event.is_set():
                    try:
                        frame = self._read_frame()
                    except Exception as exc:  # noqa: BLE001 - a frame source can fail in any way; the loop must outlive it
                        self.read_errors += 1
                        self.last_error = f"read_frame: {exc}"[:200]
                        frame = None
                    if frame is not None:
                        self.step(frame)
                    if ticker.wait():
                        break
            finally:
                self._ticker = None

    def step(self, frame: dict[str, Any], *, now: float | None = None) -> TurnPlan | None:
        """Process one frame and send the turn it warrants, if any.

        Exposed so a test can drive the loop without a thread.

        Args:
            frame: A daemon state frame.
            now: Forwarded to :meth:`DoaTurner.observe`.

        Returns:
            The plan that was sent (or refused by ``look``), or ``None``.
        """
        plan = self.turner.observe(frame, now=now, blocked=self._blocked)
        if plan is None:
            return None
        try:
            result = self._look(
                yaw=plan.yaw, body_yaw=plan.body_yaw, pitch=plan.pitch, roll=0.0, duration=plan.duration
            )
        except Exception as exc:  # noqa: BLE001 - motion refusal must not kill the loop
            self.last_error = f"look: {exc}"[:200]
            return plan
        self.last_sent = {"plan": asdict(plan), "result": result}
        if isinstance(result, dict) and result.get("status") != "success":
            self.last_error = f"look refused: {result}"[:200]
        else:
            self.last_error = None
        return plan
