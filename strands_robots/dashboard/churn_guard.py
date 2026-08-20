"""A defence against a viewer that reopens the same camera forever.

MEASURED, TWICE, ON THE LIVE DASHBOARD (BUGS.md Q40/Q46/Q52): one browser tab opened
``/ws/camera/so101-arm-1/top`` **1.53 times per second for twelve hours** — 75,489 sockets,
21.4 GB out of the house at 441 KB/s — and it was still doing it after both client-side
cures landed, because a cockpit tab left open for a day never fetches a new bundle. Its
last asset request was an hour before the measurement while the churn continued unbroken.

That is the lesson this module exists for: **the client-side fix protected only clients
that reload.** A server whose only defence lives in the page it served cannot defend
itself against yesterday's page, a paused tab, a phone in someone's pocket, or a script.
The rate was a flat 1.53/s across the whole incident, which is what "backoff reset by a
handshake" looks like from the outside — no decay, no recovery, forever.

So the server now applies its OWN cap when it sees churn, and the rules are deliberately
conservative:

* It never refuses the connection. A refusal would blank an operator's tile and hide the
  robot, and a storming client would just... reconnect. The frames keep flowing, slower.
* The threshold is far above human behaviour: a person reloading a dashboard opens a
  camera a handful of times a minute, and ``CHURN_OPENS_PER_MIN`` is 20. The measured
  storm was 92/min.
* It is a sliding window, so the cap LIFTS by itself the moment the churn stops. Nothing
  to reset, no state an operator has to know about.
* It says so, in the close verdict and to the tile itself, naming the number it counted.
  A silent throttle is indistinguishable from a slow camera, which would send the next
  person debugging this into the USB cable for no reason.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

#: Opens per minute, per viewer per camera, above which this is churn and not a human.
CHURN_OPENS_PER_MIN = 20
#: The rate a churning viewer is served at. Enough that a tile still moves (a stale image
#: is worse than a slow one), little enough that the link stops drowning: measured ~97 KB
#: a frame, so 2 fps is ~0.19 MB/s instead of 0.45.
CHURN_CAP_FPS = 2.0
#: Bound on remembered identities, so the guard cannot become the leak it prevents.
MAX_TRACKED = 512


@dataclass(frozen=True)
class ChurnVerdict:
    """What the server decided about this socket, and why — in words."""

    opens_in_window: int
    cap_fps: float | None
    reason: str | None

    @property
    def throttled(self) -> bool:
        return self.cap_fps is not None


class ChurnGuard:
    """Counts recent opens per (viewer, camera) and caps a storm's frame rate."""

    def __init__(
        self,
        *,
        opens_per_min: int = CHURN_OPENS_PER_MIN,
        cap_fps: float = CHURN_CAP_FPS,
        max_tracked: int = MAX_TRACKED,
    ) -> None:
        self._opens_per_min = opens_per_min
        self._cap_fps = cap_fps
        self._max_tracked = max_tracked
        self._seen: dict[str, deque[float]] = {}

    def note_open(self, identity: str, *, now: float | None = None) -> ChurnVerdict:
        """Record one socket opening and judge the viewer behind it."""
        now = time.monotonic() if now is None else now
        window = self._seen.setdefault(identity, deque())
        window.append(now)
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        # Evict the QUIETEST identities, never this one: a flooder must not be able to
        # push the operator's own entry out and buy itself a clean slate (Q11's lesson,
        # one layer down).
        if len(self._seen) > self._max_tracked:
            for key in sorted(self._seen, key=lambda k: len(self._seen[k]))[
                : len(self._seen) - self._max_tracked
            ]:
                if key != identity:
                    self._seen.pop(key, None)
        count = len(window)
        if count <= self._opens_per_min:
            return ChurnVerdict(opens_in_window=count, cap_fps=None, reason=None)
        return ChurnVerdict(
            opens_in_window=count,
            cap_fps=self._cap_fps,
            reason=(
                f"this viewer opened this camera {count} times in the last minute, so the "
                f"server is pacing it at {self._cap_fps:g} fps until that settles — the "
                f"tile keeps updating, it just stops saturating the link. Reload the page "
                f"to pick up the client-side fix for the reconnect loop."
            ),
        )

    def forget(self, identity: str) -> None:
        """Drop an identity (tests, and an operator who wants a clean measurement)."""
        self._seen.pop(identity, None)


def viewer_identity(*, subject: str | None, host: str | None, peer_id: str, cam: str) -> str:
    """Who is watching what — the key the guard counts.

    The auth subject comes FIRST and the host second, because behind the Cloudflare
    tunnel every remote viewer arrives as 127.0.0.1: keying on the address alone would
    lump a storming phone together with the operator's laptop and throttle both. Falling
    back to the host keeps the guard working on a LAN with auth disabled.
    """
    who = subject or host or "unknown"
    return f"{who}|{peer_id}|{cam}"


def effective_cap(requested: float | None, churn: float | None) -> float | None:
    """The rate actually served: the LOWER of what the viewer asked for and what the
    server will give a storm. A viewer asking for more than it is being throttled to
    cannot talk its way out, and a viewer asking for less is honoured — it knows its own
    link better than this guard does."""
    caps = [c for c in (requested, churn) if c is not None]
    return min(caps) if caps else None
