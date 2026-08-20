"""Make a socket's DEATH visible — the missing half of the camera stream's story.

Q40 (a phone reopening the same camera socket 63,906 times in ten hours) hid for a full
day because of one absence: the log contained 63,906 ``connection open`` lines and
**zero** closes. ws_camera accepts, loops at 15fps, and its ``except
(WebSocketDisconnect, RuntimeError): pass`` swallows the end of every stream. So a
reconnect storm was indistinguishable from 63,906 healthy viewers arriving, and the one
question that would have solved it in a second — "did these sockets ever send a frame?"
— had no answer anywhere on the machine.

The fix is not "log every close": at the rate of the incident that IS the incident,
amplified. It is to log the close with the facts that distinguish the cases (frames
sent, how long it lived, and a verdict in words), rate-limited per socket identity, with
the suppressed count carried into the next line so a storm shows up as a storm rather
than as silence.

Deliberately a plain counter + timestamp rather than anything cleverer: this runs in the
close path of a socket that may be dying because the process is under load.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def close_verdict(
    *,
    frames_sent: int,
    lifetime_s: float,
    publishing: bool,
    bytes_sent: int = 0,
) -> str:
    """One sentence naming what this closed socket actually was.

    The three cases an operator needs told apart, because the tile looks identical in
    all of them: a stream that worked, a viewer who left immediately, and a camera that
    has nothing to give.
    """
    if frames_sent > 0:
        # Q51: the SIZE is the half that was missing. One live camera tile measured
        # 4.6 fps at ~97 KB a frame = 0.45 MB/s, and the phone hammering this
        # dashboard from cellular reopens the same tile 1.55x a second - so "how much
        # did this socket actually carry before it died" decides whether the remote
        # client is buggy or simply cannot drink from a firehose. A frame count alone
        # cannot tell those apart.
        rate = f", {_mb(bytes_sent) / lifetime_s:.2f} MB/s" if bytes_sent and lifetime_s >= 0.05 else ""
        size = f" / {_mb(bytes_sent):.1f} MB" if bytes_sent else ""
        fps = f" ({frames_sent / lifetime_s:.1f} fps{rate})" if lifetime_s >= 0.05 else ""
        return f"streamed {frames_sent} frames{size} over {lifetime_s:.1f}s{fps}"
    if not publishing:
        return (
            f"sent nothing in {lifetime_s:.1f}s - that camera is not publishing to the mesh; "
            "the robot may not be running, or it could not open the device"
        )
    if lifetime_s < 2.0:
        return (
            f"sent nothing and closed after {lifetime_s:.1f}s - the client hung up before a "
            "frame was due (a page reload, or a retry loop with no backoff)"
        )
    return f"sent nothing in {lifetime_s:.0f}s although the camera is publishing - frames are not reaching this socket"


@dataclass
class _Seen:
    last_logged_at: float
    suppressed: int = 0


class CloseLogThrottle:
    """Log the first close for a key, then at most one per ``window_s``.

    The suppressed count rides the next line it lets through, so the log never quietly
    drops the fact that something happened 4,000 times.
    """

    def __init__(self, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._window_s = float(window_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, _Seen] = {}
        # a bound, for the same reason ttl_cache exists: keys are peer/camera pairs and
        # a spawn loop can invent them
        self._max_keys = 256

    def should_log(self, key: str) -> tuple[bool, int]:
        """Return (log_now, suppressed_since_last_line)."""
        now = self._clock()
        with self._lock:
            entry = self._seen.get(key)
            if entry is None:
                if len(self._seen) >= self._max_keys:
                    self._seen.pop(next(iter(self._seen)))
                self._seen[key] = _Seen(last_logged_at=now)
                return True, 0
            if now - entry.last_logged_at >= self._window_s:
                suppressed = entry.suppressed
                entry.last_logged_at = now
                entry.suppressed = 0
                return True, suppressed
            entry.suppressed += 1
            return False, entry.suppressed


def close_line(*, peer_id: str, cam: str, verdict: str, suppressed: int) -> str:
    """The log line itself, with the storm count when there is one."""
    tail = f" [+{suppressed} more closes suppressed in the last minute]" if suppressed else ""
    return f"camera socket {peer_id}/{cam} closed: {verdict}{tail}"


# --- Q52: a viewer that cannot drink the firehose must be able to ask for less -----

#: The tile's own pacing (server side sends at most this) - 15 fps was the only rate on
#: offer, and one tile at 4.6 fps x ~97 KB measured 0.45 MB/s, ~1.7 GB/h to a phone.
MAX_CAP_FPS = 30.0
MIN_CAP_FPS = 0.1


def fps_cap(raw: str | None) -> float | None:
    """Parse a viewer's requested frame rate. ``None`` means "as fast as frames arrive".

    A cap is a REQUEST FOR LESS, so a nonsense value must never become a request for
    more: anything unparseable, zero, negative or NaN is ignored (no cap, today's
    behaviour) rather than guessed at, and an absurdly high number is clamped instead of
    trusted. Below MIN_CAP_FPS a "cap" would freeze the tile, which no viewer wants and
    an attacker might: it clamps up.
    """
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v or v <= 0:  # NaN or nonsense
        return None
    return max(MIN_CAP_FPS, min(MAX_CAP_FPS, v))


def cap_note(cap: float | None) -> str:
    """What the close verdict says about the rate this socket agreed to."""
    return "" if cap is None else f" [viewer capped at {cap:g} fps]"
