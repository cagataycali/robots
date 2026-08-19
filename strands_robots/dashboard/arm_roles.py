"""Which arm is the leader and which is the follower — measured, not labelled (U2).

The operator's report: "as I see the leader arm is follower, follower is leader on
the dashboard at the moment". Today nothing in the dashboard *measures* the
difference — the role comes from whatever name a profile was given, so a swap is
invisible and silently wrong in exactly the place it hurts (a record session
drives the follower).

There is a physical ground truth on an SO-100/SO-101 pair: the follower runs its
servo bus at 12V, the leader at 7.4V. Every Feetech servo reports its own supply
on the read-only ``Present_Voltage`` register, so the role can be read off the
hardware in a few milliseconds without moving anything.

This module is pure: it decides what a set of voltage readings MEANS. The bus
access lives in device_manager, and it is register reads only — never a torque
or position write.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Mapping

#: Above this, the bus is on the 12V supply → follower. Measured live on this
#: rig: a powered SO-101 follower reads 12.6-12.7V on every servo.
FOLLOWER_MIN_V = 9.0
#: Below this the arm is NOT on its own supply. Measured live: an SO-101 whose
#: power supply was off still answered 5.5-5.6V on all six servos - the USB/UART
#: logic rail, not a battery. That is why this floor is 6.5V and not 5.5V: a
#: 7.4V pack reads 6.6-8.4V, so 5.5V sits below every real supply, and the first
#: version of this threshold (5.5) called an UNPOWERED arm "leader" by a tenth
#: of a volt. Guessing a role there is how a follower gets driven as a
#: teleoperator - the exact swap this package exists to catch.
POWERED_MIN_V = 6.5
#: What an unpowered Feetech bus reads through USB alone, for the message.
USB_RAIL_V = 5.5
#: Wider than any real supply ripple; a spread this large means the readings are
#: not describing one bus.
SPREAD_MAX_V = 1.5


def classify_role(volts: float | None) -> tuple[str, str]:
    """One reading → (role, reason). Roles: follower / leader / unpowered / unknown."""
    if volts is None:
        return "unknown", "no voltage could be read from the bus"
    if volts >= FOLLOWER_MIN_V:
        return "follower", f"{volts:.1f}V bus — the 12V supply, which is the follower arm"
    if volts >= POWERED_MIN_V:
        return "leader", f"{volts:.1f}V bus — the 7.4V supply, which is the leader arm"
    if volts >= USB_RAIL_V - 0.5:
        return (
            "unpowered",
            f"{volts:.1f}V — that is the USB logic rail, not a supply: this arm's power "
            f"is off. A leader's 7.4V pack reads 6.6-8.4V and a follower's 12V reads "
            f"10.5-12.7V, so the role cannot be read until it is powered",
        )
    return (
        "unpowered",
        f"{volts:.1f}V — this arm is not on its power supply, so its role cannot be read "
        f"(an unpowered arm reads like nothing at all, not like a leader)",
    )


def role_verdict(readings: Mapping[Any, float | None]) -> dict[str, Any]:
    """Several servos on one bus → one verdict, or an honest refusal.

    The median is used rather than the mean because a single servo that answers
    0.0 (or 25.5 — a byte read that went wrong) would otherwise drag a 12V bus
    below the threshold and rename the arm.

    A spread wider than SPREAD_MAX_V is reported as ``mixed`` instead of being
    averaged into a confident answer: servos on one bus share one supply, so
    disagreement means a wiring or reading fault, and guessing a role from
    faulty data is worse than saying so.
    """
    good = {k: float(v) for k, v in readings.items() if isinstance(v, (int, float)) and v is not None}
    if not good:
        return {
            "role": "unknown",
            "reason": "no servo answered a voltage read",
            "remedy": "check the USB cable and that the arm's power supply is on, then retry",
            "volts": None,
            "readings": dict(readings),
        }
    values = sorted(good.values())
    mid = float(median(values))
    spread = values[-1] - values[0]
    if spread > SPREAD_MAX_V:
        return {
            "role": "mixed",
            "reason": (
                f"servos on this bus report different supplies ({values[0]:.1f}V to "
                f"{values[-1]:.1f}V) — they share one supply, so this is a fault, not a role"
            ),
            "remedy": "check the power wiring / daisy-chain, then retry before trusting any role",
            "volts": mid,
            "spread": round(spread, 2),
            "readings": good,
        }
    role, reason = classify_role(mid)
    verdict: dict[str, Any] = {
        "role": role,
        "reason": reason,
        "volts": round(mid, 2),
        "spread": round(spread, 2),
        "readings": good,
        "motors_answered": len(good),
    }
    if role == "unpowered":
        verdict["remedy"] = "switch on / plug in this arm's power supply, then retry"
    return verdict


def disagreement(profile_role: str | None, measured: Mapping[str, Any]) -> dict[str, Any] | None:
    """The point of the whole exercise: does the label match the hardware?

    Returns None when there is nothing to say (no label, or an unusable
    measurement — a fault must not be reported as "your label is wrong").
    """
    measured_role = measured.get("role")
    if not profile_role or measured_role not in ("leader", "follower"):
        return None
    if profile_role == measured_role:
        return None
    return {
        "labelled": profile_role,
        "measured": measured_role,
        "message": (
            f"this arm is labelled {profile_role} but its bus reads "
            f"{measured.get('volts')}V, which is the {measured_role} arm"
        ),
        "remedy": f"relabel it {measured_role} before teleop or recording — "
                  f"the roles decide which arm is driven",
    }
