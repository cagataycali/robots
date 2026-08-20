"""Q80: the chat box could move a real arm with no confirmation of any kind.

Every motion gate this dashboard has grew on the play BUTTON. JOURNEYS #3 gave it a confirm sheet naming
the word "physical" (runRisk), Q79 made it refuse a policy that cannot drive the robot. None of that
touched the AGENT, whose ``fleet`` tool does this::

    fleet(action="task", target="so101-arm-1", instruction="pick up the red cube")
    -> send_cmd(target, {"action": "execute", ...})

Same command, same peer, same metal -- reached by typing a sentence into a chat box, with no dialog, no
physicality warning and no fit check. The dock's own placeholder even teaches it ("It can start and stop
real robots"), so this is not a hypothetical path: it is the advertised one. And an agent is precisely
the caller most likely to pick a target from a list and get it wrong, because it cannot see the room.

So the gate belongs on the CAPABILITY, not on one screen's button. This module decides, purely.

The rules, and why each one is shaped this way:

* STOPPING IS NEVER GATED. ``stop`` / ``stop_all`` / ``peers`` / ``status`` pass unconditionally --
  a safety brake that asks permission is not a safety brake, and "everyone stop" is the one sentence
  the operator most needs to work on the first try.
* only ``task`` on a PHYSICAL peer needs the grant. A sim peer moves pixels; refusing there would
  train the operator to grant reflexively, which is how a real gate dies.
* a peer we cannot classify counts as physical (runRisk's rule): a needless grant prompt costs one
  click, a missing one costs a collision.
* the grant is an ENV VAR, like every other consent in this dashboard, so it is visible on the
  permissions screen and revocable there rather than being a per-turn habit nobody can audit.
* the refusal NAMES the alternative that needs no permission (ask the agent for a sim peer, or press play
  yourself, where the confirm sheet and the Q79 fit check already live). A refusal without a next step
  is where an operator starts inventing workarounds.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

__all__ = ["MOTION_ENV", "GATED_ACTIONS", "agent_motion_allowed", "peer_is_physical"]

#: The grant. Set on the dashboard's process (or via the consent screen) to let the agent start
#: physical tasks by itself.
MOTION_ENV = "STRANDS_DASH_AGENT_PHYSICAL_MOTION"

#: Actions that can put a real robot in motion. Everything else -- including every way of STOPPING
#: one -- is deliberately outside this set.
GATED_ACTIONS: frozenset[str] = frozenset({"task"})

_TRUE = ("1", "true", "yes", "on")


def _granted(env: Mapping[str, str] | None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(MOTION_ENV, "")).strip().lower() in _TRUE


def peer_is_physical(peer: Mapping[str, Any] | None) -> tuple[bool, str]:
    """Is this peer metal? Returns (physical, why) -- the server-side twin of lib/runRisk.ts.

    ``hw`` is set by mesh/core.py to the inner lerobot device's name and exists only when a real
    device object is attached, so its presence is positive evidence. Its ABSENCE is only evidence
    when the peer said something else about itself; an empty or missing presence is unknown, and
    unknown is treated as metal.
    """
    if not peer:
        return True, "this peer is not on the fleet snapshot, so it cannot be shown to be a sim"
    presence = peer.get("presence") or {}
    hw = presence.get("hw")
    if isinstance(hw, str) and hw.strip():
        return True, f"it reports real hardware ({hw.strip()})"
    robot_type = str(presence.get("robot_type") or "").strip().lower()
    if robot_type in ("sim", "simulation", "mujoco"):
        return False, f"it reports itself as {robot_type}"
    if presence.get("sim") is True or presence.get("mode") == "sim":
        return False, "it reports itself as a simulation"
    if not presence:
        return True, "this peer has announced no presence yet, so it cannot be shown to be a sim"
    return True, "it did not say it was a simulation"


def agent_motion_allowed(
    action: str,
    *,
    peer: Mapping[str, Any] | None = None,
    target: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """May the agent perform ``action`` on ``target`` by itself?

    Returns ``{"allowed": bool, "physical": bool, "reason": str}``. ``reason`` on a refusal is the
    text handed to the model verbatim, so it must read as an instruction to the HUMAN -- the model
    relays it, and a refusal the operator cannot act on becomes "the dashboard is broken".
    """
    act = (action or "").strip()
    if act not in GATED_ACTIONS:
        return {"allowed": True, "physical": False, "reason": "", "gated": False}

    physical, why = peer_is_physical(peer)
    if not physical:
        return {"allowed": True, "physical": False, "reason": "", "gated": True}
    if _granted(env):
        return {
            "allowed": True,
            "physical": True,
            "reason": "",
            "gated": True,
            "granted": True,
        }

    shown = target.strip() or "that robot"
    return {
        "allowed": False,
        "physical": True,
        "gated": True,
        "granted": False,
        "reason": (
            f"refused: starting a task on {shown} would MOVE REAL HARDWARE ({why}), and this "
            f"dashboard does not let the agent start physical motion on its own. Nothing was sent. "
            f"The human can press play on {shown}'s card, which confirms the motion and checks that "
            f"the policy fits that robot, or ask for a simulated peer instead. To let the agent do "
            f"it unattended, grant it once: set {MOTION_ENV}=1 for the dashboard. Stopping robots "
            f"is never gated - 'everyone stop' always works."
        ),
    }


# --- the OTHER half of the same asymmetry: the HTTP route ------------------------------------------
#
# agent_motion_allowed() guards the in-process fleet tool. POST /api/robots/{peer}/task is guarded by
# nothing: the play button's confirmation lives in the browser, so anything holding the API token - a
# script, a shell, an LLM handed the token, or whoever finds it after the public tunnel leaks it -
# can start real motion with one curl and no confirmation step at all.
#
# That is DELIBERATELY still the default: the token is the operator, and breaking every existing
# caller (the deploy snippet, tests, remote scripts) to enforce a claim a client can simply assert
# would be theatre with an outage attached. So this is a LOCK THE OPERATOR CAN CHOOSE: when
# TASK_CONFIRM_ENV is set, a task POST that does not carry the browser's confirmation marker is
# refused before anything is sent. The marker is not a security boundary - a script can send it too -
# it is an ANTI-ACCIDENT boundary, which is the honest thing to claim for it.
TASK_CONFIRM_ENV = "STRANDS_DASH_TASK_REQUIRES_CONFIRM"


def task_confirm_required(env: Mapping[str, str] | None = None) -> bool:
    """Has the operator asked for real-motion task POSTs to carry a confirmation?"""
    env = env if env is not None else os.environ
    return str(env.get(TASK_CONFIRM_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def task_post_allowed(
    *,
    peer: Mapping[str, Any] | None,
    confirmed: bool,
    target: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verdict for one task POST. Same shape as ``agent_motion_allowed``.

    Off by default, and never in the way of a simulated peer or of an already-confirmed click.
    """
    if not task_confirm_required(env):
        return {"allowed": True, "physical": False, "reason": "", "gated": False}
    if confirmed:
        return {"allowed": True, "physical": True, "reason": "", "gated": True, "confirmed": True}

    physical, why = peer_is_physical(peer)
    if not physical:
        return {"allowed": True, "physical": False, "reason": "", "gated": True}

    shown = target.strip() or "that robot"
    return {
        "allowed": False,
        "physical": True,
        "gated": True,
        "confirmed": False,
        "reason": (
            f"refused: this dashboard is set to require a confirmation before a task starts real "
            f"motion, and this request did not carry one ({shown}: {why}). Nothing was sent. Press play "
            f"on {shown}'s card - the browser confirms there - or, for a script you trust, send "
            f'"confirmed": true in the body. Turn the requirement off by clearing '
            f"{TASK_CONFIRM_ENV}. Stopping is never gated."
        ),
    }
