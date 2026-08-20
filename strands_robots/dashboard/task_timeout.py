"""How long to wait for a task command's answer - and what a timeout means.

The task route sends ``action: "start"`` by default, which mesh core answers with
an immediate future-tracking ack (``execute`` is the variant that blocks until the
rollout ends). But the wait was sized for the blocking one:

    timeout_s = max(float(body.get("timeout", 60.0)), duration + 10.0)

so a 1-hour run meant the dashboard would sit on the ack for 3610 seconds. A peer
that never acks - wedged serial, dead child, a lost response - therefore left the Run button
spinning in "starting" for the whole nominal run length with nothing on screen,
which is the exact state where an operator starts clicking things.

Two different waits, sized separately:

* ``start`` waits for an ACK. Bounded and independent of duration, but generous,
  because a first-time checkpoint download happens BEFORE the ack and a premature
  "failed" on a policy that then loads and moves an arm is worse than waiting.
* ``execute`` waits for the ROLLOUT, so it keeps duration + a margin.

And a timeout on ``start`` is not "nothing happened": the command was delivered,
so the robot may be loading a policy and about to move. Saying so is the whole
point of this module.
"""

from __future__ import annotations

#: Ceiling for an ack wait, seconds. A cold HuggingFace checkpoint can take
#: minutes to fetch before the peer answers, so this is generous by design; it
#: only has to be shorter than "as long as the task itself".
DEFAULT_ACK_CAP_S = 120.0
#: Margin over ``duration`` for a blocking ``execute``.
ROLLOUT_MARGIN_S = 10.0


def task_ack_budget(
    action: str,
    requested_timeout: float | None,
    duration: float | None,
    ack_cap_s: float = DEFAULT_ACK_CAP_S,
) -> tuple[float, str]:
    """Return ``(timeout_s, kind)`` where kind is ``"ack"`` or ``"rollout"``.

    An explicit caller-supplied timeout is always honoured - it is the caller's
    business how long they wait - so this only decides the FLOOR.
    """
    try:
        asked = float(requested_timeout) if requested_timeout is not None else 0.0
    except (TypeError, ValueError):
        asked = 0.0
    try:
        dur = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        dur = 0.0
    if dur < 0 or dur != dur:  # negative or NaN
        dur = 0.0
    if asked < 0 or asked != asked:
        asked = 0.0

    if str(action).lower() == "execute":
        return max(asked, dur + ROLLOUT_MARGIN_S), "rollout"
    # "start" and anything unknown: wait for an ack, never for the whole run.
    # A short run still gets its full duration+margin, because for those the two
    # budgets agree and the longer one costs nothing.
    floor = min(max(dur + ROLLOUT_MARGIN_S, 0.0), max(ack_cap_s, 0.0))
    return max(asked, floor), "ack"


def timeout_verdict(kind: str, timeout_s: float, target: str = "") -> dict[str, object]:
    """What to tell a caller whose task command timed out.

    ``motion_possible`` is the field that matters: the command WAS delivered, so
    a UI must not render this as "nothing happened".
    """
    who = f" from {target}" if target else ""
    if kind == "rollout":
        return {
            "error": (
                f"no answer{who} within {timeout_s:g}s - the rollout was still running when the wait "
                "ended, so the task may be executing normally"
            ),
            "motion_possible": True,
            "timeout_kind": "rollout",
        }
    return {
        "error": (
            f"no acknowledgement{who} within {timeout_s:g}s - the command was delivered, so the robot "
            "may be loading a policy and about to move. Check its log before retrying."
        ),
        "motion_possible": True,
        "timeout_kind": "ack",
    }
