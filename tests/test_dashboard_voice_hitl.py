"""Voice HITL posture: fail-closed, and never disarming anyone else's gate.

The voice agent runs under bidi, which CANNOT pause a tool for a human
answer - so voice gets no robot_mesh, and its physical-motion safety is the
fleet tool's agent_motion_allowed backstop (no confirm rail on voice means
no grant can ever be deposited, so a physical task is always refused with
the point-at-the-screen sentence the VOICE_PROMPT teaches).

The defect this file exists to keep dead: voice.py once ran
``os.environ.setdefault("STRANDS_MESH_HITL_ACTIONS", "none")`` - a
process-wide write that silently disarmed the CHAT agent's robot_mesh
confirm gate the moment one voice session was opened.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from strands_robots.dashboard import voice

_VOICE_SRC = Path(inspect.getsourcefile(voice)).read_text(encoding="utf-8")


def _code_lines(src: str) -> list[str]:
    """Source lines that are code (docstrings/comments stripped, crudely but safely)."""
    # Drop triple-quoted blocks first, then per-line comments.
    no_docs = re.sub(r'"""[\s\S]*?"""', "", src)
    return [ln.split("#", 1)[0] for ln in no_docs.splitlines()]


def test_voice_never_writes_the_mesh_hitl_env():
    """The process-wide disarm must not come back (it may appear in prose only)."""
    offenders = [ln.strip() for ln in _code_lines(_VOICE_SRC) if "STRANDS_MESH_HITL_ACTIONS" in ln]
    assert offenders == [], (
        f"voice.py writes STRANDS_MESH_HITL_ACTIONS again: {offenders}. One voice "
        "session would disarm the chat agent's robot_mesh confirm gate process-wide."
    )


def test_voice_never_grants_physical_motion():
    """Nor the fleet backstop's own grant env - same disarm, different gate."""
    offenders = [
        ln.strip()
        for ln in _code_lines(_VOICE_SRC)
        if "STRANDS_DASH_AGENT_PHYSICAL_MOTION" in ln
    ]
    assert offenders == []


def test_voice_carries_no_robot_mesh():
    """bidi blows up on tool interrupts, so the gated tool stays off voice."""
    offenders = [ln.strip() for ln in _code_lines(_VOICE_SRC) if "robot_mesh" in ln]
    assert offenders == [], f"voice.py references robot_mesh in code: {offenders}"


def test_the_bidi_limitation_this_design_rests_on_still_holds():
    """TRIPWIRE: fleet-only voice is correct BECAUSE bidi refuses tool interrupts.

    When this fails, the SDK has grown bidi tool-interrupt support - revisit
    voice: it can then carry robot_mesh and a spoken/on-screen confirm rail
    instead of the fail-closed backstop.
    """
    from strands.experimental.bidi.agent import loop as bidi_loop

    src = inspect.getsource(bidi_loop)
    assert "tool interrupts are not supported in bidi" in src


def test_voice_prompt_teaches_the_refusal_is_final():
    """The spoken agent must point at the screen, never retry or reword."""
    assert "A spoken yes cannot grant it" in voice.VOICE_PROMPT
    assert "do NOT retry" in voice.VOICE_PROMPT
