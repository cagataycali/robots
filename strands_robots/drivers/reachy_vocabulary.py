"""Pure helpers behind the Reachy Mini's expressive action vocabulary.

:class:`~strands_robots.drivers.reachy.ReachyDriver` grew the verbs a desk
robot is actually asked for - ``look``, ``express``, ``say``, ``volume`` - and
each one needs a small piece of judgement that is easier to read, test and
reuse when it has no driver around it: which library move a plain-English
emotion names, what the daemon's ``goto`` body looks like in its own units,
which hosts a zero-argument bring-up should knock on, and how a spoken volume
word becomes a level. Every function here is a pure function of its arguments,
so this module imports nothing from the driver, the transport or the daemon and
is testable on a machine with no Reachy attached.

The alias table and the resolution order are ported verbatim from
``cagataycali/tiny-the-reachy`` (``tools/reachy_expression.py``), where they
ended a run of 125 ``emotion 'happy' not found`` tool errors: the HuggingFace
library has no bare ``happy`` - every move carries a take number
(``cheerful1``, ``curious1``, ``no1`` ...) - so a model that asks for the plain
word needs a translation, not a refusal.
"""

from __future__ import annotations

import math
import os
from typing import Any

from strands_robots.utils import refusal_repr, refusal_str

#: The daemon's own default REST port, repeated here so the discovery
#: candidates can be built without importing the driver.
DEFAULT_API_PORT: int = 8000

#: Hosts a zero-argument ``Robot("reachy_mini", mode="real")`` knocks on, in
#: order. ``localhost`` is where a Lite's daemon runs and where any process on
#: the robot itself finds the Wireless daemon; ``reachy-mini.local`` is the
#: Wireless's factory mDNS name, which is what the Pollen SDK and every desk
#: install answer to until an operator renames it.
DISCOVERY_HOSTS: tuple[str, ...] = ("localhost", "reachy-mini.local")

#: Environment variables that pin the daemon address ahead of discovery. The
#: names match the ones the Pollen SDK examples and ``tiny-the-reachy`` already
#: read, so an operator who has exported them for those tools has already
#: configured this driver.
ENV_HOST: str = "REACHY_HOST"
ENV_PORT: str = "REACHY_PORT"

#: Where a Piper/``tiny-tts`` speech service listens when one is installed on
#: the robot. Speech synthesis is NOT part of the Reachy daemon; this is the
#: sidecar ``tiny-the-reachy`` ships as ``tiny-tts.service``. ``REACHY_TTS_URL``
#: overrides the whole URL; otherwise the daemon host is combined with this port.
ENV_TTS_URL: str = "REACHY_TTS_URL"
DEFAULT_TTS_PORT: int = 5002

#: Interpolation methods the daemon's ``/api/move/goto`` accepts
#: (``InterpolationTechnique`` in the daemon's ``io/protocol.py``).
INTERPOLATIONS: tuple[str, ...] = ("minjerk", "linear", "ease_in_out", "cartoon")

#: Bounds for a ``goto`` duration in seconds. The floor is what the SDK
#: documents as the shortest smooth gesture; the ceiling keeps a single agent
#: call from parking the head in motion for a minute.
GOTO_DURATION_S: tuple[float, float] = (0.1, 10.0)

#: Head translation travel in millimetres, the same figure ``tiny-the-reachy``'s
#: dashboard clamps to (``LIM_XYZ_MM``). Refused rather than clamped here, on
#: the shared envelope's reasoning.
HEAD_TRANSLATION_LIMIT_MM: float = 25.0

#: Antenna travel in degrees (dashboard ``LIM_ANTENNA``). The SDK itself accepts
#: wider values and lets the motor stall against its stop; refusing at the
#: dashboard's figure keeps the ears off the stop.
ANTENNA_LIMIT_DEG: float = 150.0

#: Plain-English emotion -> library move. Ported from tiny-the-reachy.
EMOTION_ALIASES: dict[str, str] = {
    "happy": "cheerful1",
    "joy": "cheerful1",
    "glad": "cheerful1",
    "cheerful": "cheerful1",
    "excited": "enthusiastic1",
    "enthusiastic": "enthusiastic1",
    "hello": "welcoming1",
    "welcome": "welcoming1",
    "greeting": "welcoming1",
    "curious": "curious1",
    "surprised": "surprised1",
    "amazed": "amazed1",
    "yes": "yes1",
    "nod": "yes1",
    "no": "no1",
    "shake": "no1",
    "sad": "sad1",
    "angry": "rage1",
    "mad": "furious1",
    "scared": "scared1",
    "afraid": "fear1",
    "tired": "tired1",
    "sleepy": "tired1",
    "sleep": "sleep1",
    "bored": "boredom1",
    "confused": "confused1",
    "thinking": "thoughtful1",
    "thoughtful": "thoughtful1",
    "shy": "shy1",
    "proud": "proud1",
    "grateful": "grateful1",
    "thanks": "grateful1",
    "love": "loving1",
    "loving": "loving1",
    "laugh": "laughing1",
    "laughing": "laughing1",
    "oops": "oops1",
    "sorry": "oops1",
    "relief": "relief1",
    "relieved": "relief1",
    "calm": "calming1",
    "serene": "serenity1",
    "lonely": "lonely1",
    "success": "success1",
    "win": "success1",
    "dance": "dance1",
    "attentive": "attentive1",
    "listening": "attentive1",
    "helpful": "helpful1",
    "impatient": "impatient1",
    "irritated": "irritated1",
    "frustrated": "frustrated1",
    "disgusted": "disgusted1",
    "uncertain": "uncertain1",
    "understanding": "understanding1",
    "come": "come1",
    "go_away": "go_away1",
    "lost": "lost1",
    "exhausted": "exhausted1",
    "electric": "electric1",
}

#: Spoken volume words -> level. ``quieter``/``louder`` are relative and handled
#: by :func:`resolve_volume_level` against the current level.
VOLUME_WORDS: dict[str, int] = {
    "silent": 0,
    "mute": 0,
    "muted": 0,
    "shush": 0,
    "quiet": 0,
    "off": 0,
    "low": 25,
    "normal": 60,
    "default": 60,
    "loud": 80,
    "max": 100,
    "full": 100,
}


def resolve_move_name(requested: str, catalogue: list[str] | tuple[str, ...]) -> str | None:
    """Map a requested emotion to a name the library actually serves, or ``None``.

    Resolution order, first hit wins: exact name; the alias table; the name with
    a ``1`` take suffix; the unique-prefix match (``"displeased"`` ->
    ``displeased1``). Every hit except the alias fallback is checked against
    ``catalogue``, so a stale alias that names a move the library dropped is
    not returned as if it existed. With an EMPTY catalogue - the daemon could
    not list the library - the alias table alone is trusted, because refusing
    every plain word for want of a catalogue read would be the failure this
    function exists to end.

    Args:
        requested: What the caller asked for - any case, ``-`` or spaces
            tolerated (``"go away"`` -> ``go_away1``).
        catalogue: The library's move names as the daemon lists them.

    Returns:
        A move name to send, or ``None`` when nothing matches.
    """
    names = list(catalogue)
    key = (requested or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return None
    if key in names:
        return key
    alias = EMOTION_ALIASES.get(key)
    if alias and (alias in names or not names):
        return alias
    if key + "1" in names:
        return key + "1"
    hits = sorted(name for name in names if name.startswith(key))
    return hits[0] if hits else None


def goto_body_error(
    *,
    head: dict[str, float] | None,
    body_yaw: float | None,
    antennas: tuple[float, float] | None,
    duration: float,
    interpolation: str,
    context: str,
) -> str | None:
    """Report why a ``goto`` request cannot be built, or ``None``.

    The rotational envelope (pitch/roll/yaw/body and the head-body coupling) is
    the shared :func:`~strands_robots.drivers.reachy_envelope.envelope_error`
    and is checked by the driver before this; what is judged here is the rest of
    the request - translation travel, antenna travel, the duration window and
    the interpolation name - so a refusal names the one field that is wrong.

    Args:
        head: Head pose in degrees and millimetres, keys ``pitch``, ``roll``,
            ``yaw``, ``x``, ``y``, ``z``; ``None`` for no head command.
        body_yaw: Body yaw in degrees, or ``None`` to leave the body alone.
        antennas: ``(right, left)`` in degrees, or ``None`` to leave them alone.
        duration: Seconds the interpolation takes.
        interpolation: One of :data:`INTERPOLATIONS`.
        context: Verb name to quote in the reason.

    Returns:
        A reason, or ``None`` when a body can be built.
    """
    if head is None and body_yaw is None and antennas is None:
        return f"{context}: nothing to move - name a head pose, a body_yaw or an antenna pair"
    if not isinstance(duration, int | float) or isinstance(duration, bool) or not math.isfinite(duration):
        return f"{context}: duration must be a finite number of seconds, got {duration!r}"
    low, high = GOTO_DURATION_S
    if not low <= float(duration) <= high:
        return f"{context}: duration {duration:g} s is outside [{low:g}, {high:g}] s"
    if interpolation not in INTERPOLATIONS:
        return f"{context}: interpolation {interpolation!r} is not one of {list(INTERPOLATIONS)}"
    if head is not None:
        for axis in ("x", "y", "z"):
            value = head.get(axis, 0.0)
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
                return f"{context}: head {axis} must be a finite number of millimetres, got {value!r}"
            if abs(float(value)) > HEAD_TRANSLATION_LIMIT_MM:
                return f"{context}: head {axis} {value:g} mm is outside +/-{HEAD_TRANSLATION_LIMIT_MM:g} mm"
    if antennas is not None:
        if len(antennas) != 2:
            return f"{context}: antennas must be a (right, left) pair, got {len(antennas)} values"
        for side, value in zip(("right", "left"), antennas, strict=True):
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
                return f"{context}: antenna_{side} must be a finite number of degrees, got {value!r}"
            if abs(float(value)) > ANTENNA_LIMIT_DEG:
                return f"{context}: antenna_{side} {value:g} deg is outside +/-{ANTENNA_LIMIT_DEG:g} deg"
    return None


def goto_body(
    *,
    head: dict[str, float] | None,
    body_yaw: float | None,
    antennas: tuple[float, float] | None,
    duration: float,
    interpolation: str,
) -> dict[str, Any]:
    """Build the daemon's ``POST /api/move/goto`` body from degrees and millimetres.

    The daemon's ``GotoModelRequest`` wants ``head_pose`` as an ``XYZRPYPose``
    in metres and radians, ``antennas`` as ``[right, left]`` radians and
    ``body_yaw`` in radians; only the groups the caller named are sent, so a
    lone antenna move does not re-send a head pose. Validate with
    :func:`goto_body_error` first - this function converts and does not judge.

    Args:
        head: Head pose in degrees and millimetres (see :func:`goto_body_error`).
        body_yaw: Body yaw in degrees, or ``None``.
        antennas: ``(right, left)`` degrees, or ``None``.
        duration: Seconds.
        interpolation: Daemon interpolation name.

    Returns:
        The JSON body to post.
    """
    body: dict[str, Any] = {"duration": float(duration), "interpolation": interpolation}
    if head is not None:
        body["head_pose"] = {
            "x": float(head.get("x", 0.0)) / 1000.0,
            "y": float(head.get("y", 0.0)) / 1000.0,
            "z": float(head.get("z", 0.0)) / 1000.0,
            "roll": math.radians(float(head.get("roll", 0.0))),
            "pitch": math.radians(float(head.get("pitch", 0.0))),
            "yaw": math.radians(float(head.get("yaw", 0.0))),
        }
    if body_yaw is not None:
        body["body_yaw"] = math.radians(float(body_yaw))
    if antennas is not None:
        body["antennas"] = [math.radians(float(antennas[0])), math.radians(float(antennas[1]))]
    return body


def discovery_candidates(
    api_port: int = DEFAULT_API_PORT, environ: dict[str, str] | None = None
) -> list[tuple[str, int]]:
    """The ``(host, port)`` pairs a zero-argument bring-up probes, in order.

    ``REACHY_HOST`` alone decides the list when it is set - an operator who
    named a host does not want two other hosts tried behind it. It may carry
    its own ``:port`` suffix; otherwise ``REACHY_PORT`` and then ``api_port``
    supply one. Without it the list is :data:`DISCOVERY_HOSTS` on ``api_port``.

    Args:
        api_port: The port to use for a host that names none.
        environ: The environment to read; ``None`` reads :data:`os.environ`.
            Injectable so a test does not have to touch the real environment.

    Returns:
        Candidates in probe order. Never empty.
    """
    env = os.environ if environ is None else environ
    host = (env.get(ENV_HOST) or "").strip()
    port_text = (env.get(ENV_PORT) or "").strip()
    port = int(port_text) if port_text.isdigit() else int(api_port)
    if host:
        head, separator, tail = host.rpartition(":")
        if separator and head and tail.isdigit():
            return [(head, int(tail))]
        return [(host, port)]
    return [(name, port) for name in DISCOVERY_HOSTS]


def daemon_answers(status: Any) -> bool:
    """Whether a ``/api/daemon/status`` body is a daemon worth connecting to.

    A body that is not an object, or that carries a non-null top-level
    ``error``, is not: the daemon reports its own start-up failures there
    (``"Multiple Reachy Mini serial ports found ..."`` on a desktop that also
    has other USB-serial devices attached), and a daemon in that state serves
    no robot. A ``state`` of ``"error"`` says the same thing in a second field.

    Args:
        status: The decoded body, or the transport's ``{"error": ...}`` envelope.

    Returns:
        ``True`` for a usable daemon.
    """
    if not isinstance(status, dict):
        return False
    if status.get("error") is not None:
        return False
    return status.get("state") != "error"


def resolve_volume_level(level: Any, current: int | None) -> int | str:
    """Turn a caller's volume request into a 0-100 level, or a reason.

    Accepts an integer, a numeric string with an optional ``%``, one of
    :data:`VOLUME_WORDS`, or the relative words ``quieter`` (half of the
    current level) and ``louder`` (current plus 20). The relative words need
    ``current``; without it they are refused rather than computed from a guess.

    Args:
        level: The request.
        current: The daemon's current level, or ``None`` when it is unknown.

    Returns:
        An ``int`` in ``[0, 100]``, or a ``str`` reason.
    """
    if isinstance(level, bool):
        return f"volume: level must be 0-100 or a word, got {refusal_repr(level)}"
    if isinstance(level, int | float):
        if not math.isfinite(level):
            return f"volume: level must be a finite number, got {refusal_repr(level)}"
        value = int(round(float(level)))
    else:
        text = refusal_str(level).strip().lower()
        if text in VOLUME_WORDS:
            value = VOLUME_WORDS[text]
        elif text in ("quieter", "louder"):
            if current is None:
                return f"volume: {text!r} is relative to the current level, which the daemon did not report"
            value = current // 2 if text == "quieter" else current + 20
        else:
            try:
                parsed = float(text.rstrip("%"))
            except ValueError:
                return f"volume: unknown level {refusal_repr(level)} - use 0-100, or one of {sorted(VOLUME_WORDS)} / quieter / louder"
            # ``float`` accepts "inf", "infinity", "nan" and an overflowing
            # "1e999", and ``round`` raises OverflowError on an infinity - so
            # the finiteness check the numeric branch makes has to run here
            # too, before anything rounds.
            if not math.isfinite(parsed):
                return f"volume: level must be a finite number, got {refusal_repr(level)}"
            value = int(round(parsed))
    if not 0 <= value <= 100:
        return f"volume: level {value} is outside 0-100"
    return value
