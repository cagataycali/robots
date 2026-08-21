"""Refused handshakes, counted — so a retry storm is visible from /api/health (Q88).

Why this exists. A phone left open on the dashboard held a session JWT that lapsed, and its
camera tiles reopened their websockets for 19.3 HOURS. Every single attempt was correctly
refused (close 1008 / HTTP 401), so the server was never wrong — but nothing anywhere COUNTED
those refusals. /api/health reported ``status: ok`` with cheerful coalescer stats while a client
hammered it thousands of times, and the only way I found the incident was grepping a 34 MB log.
A failure that is invisible in the status endpoint gets debugged as a hardware problem.

The client-side fixes (be35c097 / fa9df6f3 / 9b3f364c) stop OUR page from storming. This module
is for the case those cannot cover: an old bundle pinned by a service worker, a script with a
stale token, a second browser, a phone that is asleep in a drawer — any client whose code the
dashboard does not control. It converts "the log knows" into "the status endpoint says so".

Design notes:
  * Counting only. Nothing here refuses, rate-limits or blocks — a security decision made from
    a counter is a way to lock out the legitimate operator, and the auth guard is already
    correct. This exists to make the server DESCRIBE what it is doing.
  * A bounded ring per client, not a growing list: a storm is exactly the case where an
    unbounded structure becomes the second bug. Old timestamps are dropped by the window, and
    the number of tracked clients is capped (with the overflow still counted in the total, so
    the summary can never claim fewer refusals than happened).
  * Identity is (client ip, path, kind) — what answers the operator's real question, "which of
    my devices, which screen, and why". `kind` is carried because the two refusals need
    different sentences: a lapsed sign-in is fixed by signing in again, a cross-origin refusal
    by naming the origin in settings, and telling someone to reload their tab when the real
    problem is CORS wastes the trip. No token, no fragment of a credential is ever stored: this
    text is designed to be readable in a public status payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: How far back "recent" reaches. Five minutes is short enough that a fixed client stops being
#: reported almost immediately, and long enough that a 30s-backoff loop still shows up.
WINDOW_S = 300.0

#: Refusals from ONE client inside the window that mean "this is a loop, not a person typing a
#: wrong token". A human retrying by hand does not reach 10 in five minutes; the measured
#: incident's camera tiles produced hundreds.
STORM_THRESHOLD = 10

#: Distinct clients tracked. Past this, refusals still count towards the total and the tracked
#: clients keep their history — an attacker rotating source ports must not be able to erase the
#: evidence of the storm that matters by filling this map.
MAX_CLIENTS = 64


@dataclass
class RefusalTally:
    """Every refused /api or /ws handshake, in memory, bounded."""

    total: int = 0
    #: (client, path, kind) -> ring of epoch seconds, newest last
    _recent: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    #: refusals dropped because MAX_CLIENTS was reached (counted, never hidden)
    untracked: int = 0

    def record(self, *, client: str, path: str, now: float, kind: str = "credential") -> None:
        self.total += 1
        key = (client or "?", path or "?", kind or "credential")
        seen = self._recent.get(key)
        if seen is None:
            if len(self._recent) >= MAX_CLIENTS:
                self.untracked += 1
                return
            seen = []
            self._recent[key] = seen
        seen.append(now)
        # Trim on write: the read path is /api/health, which must stay cheap even mid-storm.
        cutoff = now - WINDOW_S
        if seen[0] < cutoff:
            self._recent[key] = [t for t in seen if t >= cutoff]

    def summary(self, now: float, *, detailed: bool = True) -> dict[str, object] | None:
        """What /api/health should say, or ``None`` when there is nothing to report.

        None rather than a zeroed block on purpose: a health payload that always carries a
        refusals section trains the reader to skip it, and this number only matters when it is
        not zero.

        ``detailed=False`` is the answer for an UNAUTHENTICATED reader, because /api/health is
        deliberately public and this block is built from other people's addresses. The counts
        stay (a public "something is hammering me" is useful and gives nothing away), but the
        client address, the path and the naming sentence are withheld: a LAN address map plus
        "which screens are being refused" is reconnaissance, handed to exactly the caller who
        could not authenticate. Caught in review one iteration after shipping the counter —
        the counter was right, its audience was not.
        """
        cutoff = now - WINDOW_S
        live = {k: [t for t in v if t >= cutoff] for k, v in self._recent.items()}
        live = {k: v for k, v in live.items() if v}
        recent = sum(len(v) for v in live.values())
        if not self.total:
            return None
        out: dict[str, object] = {
            "total": self.total,
            "recent": recent,
            "window_s": int(WINDOW_S),
            "clients": len(live),
        }
        if self.untracked:
            out["untracked"] = self.untracked
        if not live:
            # Refused in the past, quiet now. Worth saying, because "the storm stopped" is the
            # answer to "did my fix work?" — and it says it without inventing a culprit.
            out["text"] = (
                f"{self.total} handshake(s) refused since start, none in the last "
                f"{int(WINDOW_S / 60)} minutes."
            )
            return out
        (client, path, kind), stamps = max(live.items(), key=lambda kv: len(kv[1]))
        worst = len(stamps)
        if not detailed:
            # Say THAT it is happening and how hard, never who or where.
            out["storm"] = worst >= STORM_THRESHOLD
            out["text"] = (
                f"{recent} handshake(s) refused in the last {int(WINDOW_S / 60)} minutes. "
                "Sign in to see which client and which path."
            )
            return out
        out["worst"] = {"client": client, "path": path, "kind": kind, "count": worst}
        if worst >= STORM_THRESHOLD:
            span = max(stamps) - min(stamps)
            rate = f"{worst / span * 60:.0f}/min" if span > 1 else f"{worst} in a burst"
            out["storm"] = True
            # The sentence names the cause the evidence supports (a retry loop with a credential
            # this server will not accept), the client, and the ONE action that ends it. It must
            # not say "an attack": a stale tab is overwhelmingly more likely, and calling the
            # operator's own phone an attacker is how a status page loses its credibility.
            cause = (
                "that page is holding an expired or wrong sign-in - reload it and sign in again"
                if kind == "credential"
                else "its Origin is not allow-listed - add it to security.cors_origins, or open "
                "the dashboard on its own address"
            )
            out["text"] = (
                f"{client} is retrying {path} and being refused ({kind}) {worst} times in the "
                f"last {int(WINDOW_S / 60)} minutes, ~{rate}. It will not recover by itself: "
                f"{cause}. Nothing is wrong with the robots."
            )
        else:
            out["storm"] = False
            out["text"] = (
                f"{recent} handshake(s) refused in the last {int(WINDOW_S / 60)} minutes "
                f"({len(live)} client(s), most from {client} on {path}, {kind}). Normal if someone is "
                f"signing in; a loop would show tens."
            )
        return out
