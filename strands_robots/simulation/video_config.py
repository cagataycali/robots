"""The optional MP4 recording configuration a rollout accepts.

:class:`VideoConfig` is the typed form of the ``video`` dict that
:meth:`~strands_robots.simulation.base.SimEngine.run_policy` and
:class:`~strands_robots.simulation.policy_runner.PolicyRunner` both accept. It
lives in its own module, below both of them, so the engine can name the type at
module level without importing the runner: the runner needs the engine only as
an annotation, and the engine reaches the runner inside the methods that run a
policy, so neither module imports the other at import time.

Importing it from ``strands_robots.simulation.policy_runner`` keeps working.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

from strands_robots.utils import positive_whole_number_error

# Canonical :class:`VideoConfig` field -> the dict keys accepted for it, canonical
# key first followed by the legacy/tool_spec aliases. Single source of truth for
# both the schema check (``VideoConfig.validation_error``) and the value lookup
# (``VideoConfig.from_dict``), so the accepted set cannot drift between the two.
_VIDEO_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("path", "record_video", "output_path"),
    "fps": ("fps", "video_fps"),
    "camera": ("camera", "video_camera", "camera_name"),
    "width": ("width", "video_width"),
    "height": ("height", "video_height"),
}

_VIDEO_ACCEPTED_KEYS: tuple[str, ...] = tuple(sorted(key for aliases in _VIDEO_KEY_ALIASES.values() for key in aliases))


def _video_values_agree(first: Any, second: Any) -> bool:
    """Whether two spellings of one ``video`` field carry the same value.

    Args:
        first: Value carried by the earlier-listed spelling.
        second: Value carried by the later one.

    Returns:
        ``True`` when the two are equal, so resolving the field discards
        nothing. A pair whose equality is not a single truth value (an array)
        counts as disagreeing: the discard would be real either way, and naming
        both keys is the answer a caller can act on.
    """
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class VideoConfig:
    """Configuration for optional MP4 recording during :meth:`PolicyRunner.run`.

    Consolidates the five formerly-flat video parameters on
    :meth:`SimEngine.run_policy` into one typed object. Recording is an
    opt-in feature - if ``path`` is falsy, no recording occurs and the
    other fields are ignored.

    Attributes:
        path: Output MP4 path. ``None``/empty string → recording disabled.
            LLM-supplied, so it is validated (no ``..`` traversal, backslash
            separators, shell metacharacters, or symlinked target) before a
            writer is opened; set ``STRANDS_ROBOTS_VIDEO_ROOT`` to confine it
            to a sandbox.
        fps: Frames per second to *request*. Capped at ``control_frequency``
            when it would exceed it, so the rollout always plays back at
            real time (a rollout renders at most one frame per control
            step and cannot be up-sampled). The rate the MP4 was actually
            written at is :attr:`_RolloutVideoWriter.write_fps`, reported as
            ``video_fps`` in the :meth:`PolicyRunner.run` result - read that,
            not this, to know how the file on disk plays.
        camera: Camera name to render from. ``None`` → backend default.
        width: Render width in pixels.
        height: Render height in pixels.
    """

    path: str | None = None
    fps: int = 30
    camera: str | None = None
    width: int = 640
    height: int = 480

    @property
    def enabled(self) -> bool:
        """Whether recording is on: ``True`` iff an output ``path`` was set.

        The other fields (``fps``, ``camera``, ``width``, ``height``) are
        ignored when this is ``False`` -- a falsy ``path`` opts the whole
        rollout out of MP4 capture.
        """
        return bool(self.path)

    @staticmethod
    def _pick(d: dict[str, Any], field: str, default: Any = None) -> Any:
        """First present, non-``None`` value among ``field``'s accepted keys.

        Looks the canonical key up first, then the legacy aliases. Two
        spellings that carry DIFFERENT values are refused by
        :meth:`_alias_conflict_error` before this runs, so no value reachable
        here is discarded.
        Membership - not truthiness - decides: a caller-supplied ``0`` is
        returned as ``0`` (and rejected by :meth:`validation_error`) instead of
        collapsing into ``default`` the way an ``or`` chain would.

        Args:
            d: The caller's video-config dict.
            field: Canonical field name (a key of the alias map).
            default: Returned when no accepted key carries a value.

        Returns:
            The caller's value, or ``default``.
        """
        for key in _VIDEO_KEY_ALIASES[field]:
            value = d.get(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def _positive_int_error(value: Any, key: str) -> str | None:
        """Error text when a ``video`` dict value is not a positive whole number.

        Thin binding of the shared frame/pixel-count domain
        (:func:`positive_whole_number_error`) to the ``video:`` message prefix,
        so this dict schema and the plain-MP4 recorder's keyword parameters
        cannot drift apart on what counts as a usable ``fps`` / ``width`` /
        ``height``.

        Args:
            value: The caller-supplied value.
            key: The dict key it came from, used in the message.

        Returns:
            An error message, or ``None`` when the value is usable.
        """
        return positive_whole_number_error(value, key, "video")

    @classmethod
    def _alias_conflict_error(cls, d: dict[str, Any]) -> str | None:
        """Error text when two spellings of one field carry different values.

        :meth:`_pick` resolves a field by taking the first spelling that carries
        a value, so a dict naming two of them honors one and discards the other
        - the silent drop this schema exists to refuse, reached through keys it
        accepts. A camera named twice recorded the rollout from one of the two
        views under ``status="success"``; a path named twice wrote one file and
        left the other absent. Two spellings carrying the SAME value discard
        nothing and are accepted.

        Args:
            d: The caller's video-config dict, already known to hold only
                accepted keys.

        Returns:
            A message naming both spellings and their values, or ``None`` when
            no field is spelled twice with a disagreement.
        """
        for field, aliases in _VIDEO_KEY_ALIASES.items():
            carried = [(key, d[key]) for key in aliases if d.get(key) is not None]
            if len(carried) < 2:
                continue
            winner, kept = carried[0]
            for key, value in carried[1:]:
                if not _video_values_agree(kept, value):
                    return (
                        f"video: {winner!r} and {key!r} are both spellings of {field}, and they "
                        f"disagree ({kept!r} vs {value!r}); {winner!r} wins, so {key!r} would be "
                        "discarded. Pass one spelling of it."
                    )
        return None

    @classmethod
    def validation_error(cls, d: Any) -> str | None:
        """Error text when ``d`` is not a video config this class can honor.

        Recording options arrive as a free-form dict (LLM tool call or direct
        API), so a mistyped key has no signature to bounce off. Silently
        ignoring one is the worst outcome: ``{"filename": "/tmp/a.mp4"}``
        leaves ``path`` unset and the rollout reports ``status="success"``
        with no MP4 anywhere, and ``{"path": p, "resolution": [320, 240]}``
        records at the default 640x480 while the caller believes otherwise.
        Two accepted spellings of one field are the same drop wearing an
        accepted key: ``{"camera": "top", "camera_name": "wrist"}`` recorded
        from ``top`` while the caller had also named ``wrist``, and ``{"path":
        a, "output_path": b}`` wrote ``a`` and left ``b`` absent. This rejects
        any key outside the accepted set (with a closest-match hint), any pair
        of spellings that disagree about one field, and any known key whose
        value cannot be honored.

        Args:
            d: The caller's ``video`` argument. ``None`` (recording off) and an
                empty dict are valid.

        Returns:
            An error message describing the first problem found, or ``None``
            when the config is usable.
        """
        if d is None:
            return None
        if not isinstance(d, dict):
            return f"video must be a dict of recording options, got {type(d).__name__}."
        accepted = ", ".join(_VIDEO_ACCEPTED_KEYS)
        for key in d:
            if key in _VIDEO_ACCEPTED_KEYS:
                continue
            # Match case-insensitively so "FPS"/"Path" suggest their canonical
            # spelling; the cutoff is deliberately tight so an unrelated key
            # ("filename", "resolution") gets the accepted list rather than a
            # misleading nearest-neighbour.
            close = difflib.get_close_matches(str(key).lower(), _VIDEO_ACCEPTED_KEYS, n=1, cutoff=0.7)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            return f"video: unknown key {key!r}.{hint} Accepted keys: {accepted}."
        # Two spellings of one field: the collision is reported rather than
        # resolved, for the reason an unknown key is. Ahead of the per-field
        # domains below, because those grade the value that WINS - run after a
        # collision they would pass over the discarded one in silence.
        if error := cls._alias_conflict_error(d):
            return error
        for field in ("path", "camera"):
            value = cls._pick(d, field)
            if value is not None and not isinstance(value, str):
                return f"video: {field} must be a string, got {value!r}."
        for field in ("fps", "width", "height"):
            value = cls._pick(d, field)
            if value is None:
                continue
            if error := cls._positive_int_error(value, field):
                return error
        return None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VideoConfig | None:
        """Build from a plain dict (tool_spec dispatcher path). ``None`` passthrough.

        Accepts both canonical keys and the legacy/tool_spec aliases listed in
        :meth:`validation_error`.

        Args:
            d: Video-config dict, or ``None``/empty for "no recording".

        Returns:
            The config, or ``None`` when ``d`` is empty.

        Raises:
            ValueError: When ``d`` carries a key or value that cannot be
                honored (see :meth:`validation_error`). Public entry points
                (``run_policy`` / ``eval_policy`` / ``evaluate_benchmark`` /
                ``start_policy``) check first and return a structured tool
                error, so this raise is the guard for direct construction.
        """
        if not d:
            return None
        if error := cls.validation_error(d):
            raise ValueError(error)
        return cls(
            path=cls._pick(d, "path"),
            fps=int(cls._pick(d, "fps", 30)),
            camera=cls._pick(d, "camera"),
            width=int(cls._pick(d, "width", 640)),
            height=int(cls._pick(d, "height", 480)),
        )
