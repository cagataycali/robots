"""Is the arm being recorded actually MOVING, or is the dataset a still life?

BUGS.md Q35. The failure this module exists for was measured on this rig, not imagined:
a Feetech bus answers POSITION READS from the USB logic rail (5.5V) while torque needs
the 12V pack, so if the follower's supply trips mid-episode every surface in the record
path keeps reporting success -- ``send_action`` returns the action it was handed, the
next ``get_observation`` returns valid joint numbers, frames keep landing at the full
declared fps, and the episode's frame count and duration look perfect. What comes out is
hundreds of frames of one unchanging pose: a dataset that teaches a policy to hold still,
discovered an hour later at training time. The same shape as the missing-camera case one
layer up (``record_worker.camera_verdict``): a success report is not evidence that
anything was captured.

Deliberately a NOTICE and never a refusal. Standing still is legitimate -- the operator
lines the arms up, pauses to think, holds a grasp -- so this reports what it measured and
what it would mean, and lets the human decide. A guard that stopped recording here would
throw away real episodes to prevent a suspicion.

The units are DEGREES, because that is what an SO-101 reports (Q27 was exactly this
assumption made the other way, in radians, and it silently refused every real frame).
``EPSILON_DEG`` is therefore a hair above the sensor noise of a still, powered arm and far
below any hand-guided motion.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["EPSILON_DEG", "MIN_SAMPLES", "WINDOW_S", "joint_positions", "prune", "motion_verdict"]

#: How far a joint must travel inside the window to count as movement, in DEGREES.
#: A still but powered SO-101 jitters by ~0.1-0.3 deg on this rig; hand guiding moves
#: tens of degrees. 0.5 sits between those with room on both sides.
EPSILON_DEG = 0.5

#: No verdict before this many samples: two frames that happen to match are not evidence,
#: and the first tick of an episode has nothing to compare against at all.
MIN_SAMPLES = 10

#: How far back to look. Long enough that a deliberate hold does not trip it on the first
#: pause, short enough that an operator hand-guiding an arm learns about a dead supply
#: while the episode can still be redone rather than after twenty minutes of it.
WINDOW_S = 8.0


def joint_positions(obs: Mapping[str, Any] | None) -> dict[str, float]:
    """The finite numeric joint positions of one observation.

    Only ``*.pos`` keys, matching ``teleop_source.positions_from_observation``: an
    observation also carries camera frames (ndarray) and velocities, and neither says
    anything about where the arm IS. A non-finite value is dropped rather than kept as
    NaN, because NaN compares false against everything and would make a frozen arm look
    like a moving one -- the exact direction of error this module must not make.
    """
    out: dict[str, float] = {}
    for key, value in (obs or {}).items():
        if not isinstance(key, str) or not key.endswith(".pos"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        v = float(value)
        if math.isfinite(v):
            out[key] = v
    return out


def prune(
    samples: Sequence[tuple[float, Mapping[str, float]]], now: float, window_s: float = WINDOW_S
) -> list[tuple[float, Mapping[str, float]]]:
    """The samples still inside the window, oldest first.

    The caller records one sample per frame at up to 30fps for as long as the operator
    keeps recording, so the ring MUST be bounded by time here rather than by a count the
    fps would silently change the meaning of.
    """
    if not samples:
        return []
    cutoff = now - max(0.0, float(window_s))
    return [(t, p) for t, p in samples if t >= cutoff]


def _travel(samples: Iterable[tuple[float, Mapping[str, float]]]) -> tuple[str | None, float]:
    """The joint that moved most inside these samples, and by how much.

    Peak-to-peak per joint, not first-vs-last: an arm that swings out and comes back to
    the same pose HAS moved, and first-vs-last would call that frozen.
    """
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for _t, positions in samples:
        for joint, raw in positions.items():
            # Defensive on purpose: the caller extracts positions with
            # ``joint_positions``, but a future caller handing over a whole observation
            # would otherwise crash the record loop on comparing two camera frames -
            # and a crashed tick is a lost frame, which is worse than a missing notice.
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if not math.isfinite(value):
                continue
            if joint not in lo or value < lo[joint]:
                lo[joint] = value
            if joint not in hi or value > hi[joint]:
                hi[joint] = value
    best_joint: str | None = None
    best = 0.0
    for joint, high in hi.items():
        span = high - lo[joint]
        if best_joint is None or span > best:
            best_joint, best = joint, span
    return best_joint, best


def motion_verdict(
    samples: Sequence[tuple[float, Mapping[str, float]]],
    *,
    now: float,
    frames: int | None = None,
    window_s: float = WINDOW_S,
    epsilon_deg: float = EPSILON_DEG,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, Any] | None:
    """What to tell the operator about the follower's motion, or ``None``.

    ``None`` means "nothing can be said yet", which is a different state from "it is
    moving" and must not be rendered as reassurance: too few samples, a window not yet
    covered, or observations that carried no joint positions at all (a sim backend, or a
    schema this rule does not understand -- silence beats guessing).
    """
    # COVERAGE is judged on the whole history handed over, TRAVEL only on the window.
    # Doing both on the pruned window cannot work at any fps: pruning at ``now - window_s``
    # leaves the oldest kept sample just INSIDE the cutoff, so its span is always about one
    # sample interval short of the threshold and the verdict stays silent forever. I fixed
    # that once by measuring against ``now`` and it was still short - because the reach has
    # to come from a sample OLDER than the window, which pruning is exactly what removes.
    # Hence ``prune_window`` below, and the worker keeping a ring wider than the window.
    if not samples:
        return None
    reach_s = now - samples[0][0]
    if reach_s < window_s:
        # Less history than the window: a hold this short is ordinary.
        return None
    window = prune(samples, now, window_s)
    if len(window) < max(2, int(min_samples)):
        return None
    span_s = now - window[0][0]
    # A stream that STOPPED is a different defect (frames are not being recorded at all,
    # and the fps panel already says so). Reporting it as a motionless arm would send the
    # operator to check a power supply that is fine.
    if now - window[-1][0] > window_s / 2:
        return None
    joint, travelled = _travel(window)
    if joint is None:
        return None  # no joint positions in any sample - not our subject
    if travelled >= epsilon_deg:
        return None

    counted = f"{frames} frames of " if frames else ""
    return {
        "still": True,
        "seconds": round(span_s, 1),
        "samples": len(window),
        "max_travel_deg": round(travelled, 3),
        "quietest_joint": joint,
        "message": (
            f"the follower has not moved for {span_s:.0f}s - {counted}one unchanging pose "
            f"(largest joint travel {travelled:.2f} deg). If that is deliberate, ignore this. "
            "If it is not: a Feetech bus still REPORTS positions from the USB logic rail "
            "when the 12V supply is off, so a tripped supply looks exactly like this - "
            "valid numbers, full frame rate, an arm that never moves. Check the follower's "
            "power before collecting more, and redo this episode."
        ),
    }
