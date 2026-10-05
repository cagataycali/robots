"""Reads of the user-local robot overlay (``$STRANDS_BASE_DIR/user_robots.json``).

The leaf that :mod:`strands_robots.registry.loader` and
:mod:`strands_robots.registry.user_registry` both stand on. The loader merges
the overlay into the ``robots`` registry and keys its hot-reload cache on the
overlay's bytes; ``user_registry`` writes it. Each used to import the other for
these reads, which made them an import cycle, and deferring an import into a
function body did not remove that edge from the graph - it only moved the moment
it was paid. The shared reads live here instead, and nothing in this module
imports a registry sibling.
"""

import json
import logging
from pathlib import Path
from typing import Any

from strands_robots.utils import get_base_dir

logger = logging.getLogger(__name__)


def user_registry_path() -> Path:
    """Path of the user-local robot overlay, under the Strands base directory.

    Returns:
        ``$STRANDS_BASE_DIR/user_robots.json``. The one spelling of the overlay
        path, so a reader and a writer cannot disagree about which file the
        registry overlay is.
    """
    return get_base_dir() / "user_robots.json"


def user_registry_source() -> bytes | None:
    """Contents of the user-local robot overlay.

    Returns:
        The file's bytes, or ``None`` when the overlay does not exist yet or
        cannot be read. This is what the loader keys its hot-reload cache on, so
        an external edit to ``user_robots.json`` (a second process or a manual
        edit) invalidates the merged ``robots`` cache without requiring a manual
        :func:`~strands_robots.registry.loader.invalidate_cache` - including an
        edit that lands inside one filesystem timestamp tick, which a stat
        cannot distinguish.
    """
    try:
        return user_registry_path().read_bytes()
    except OSError:
        return None


def parse_user_robots(source: bytes | None) -> dict[str, Any]:
    """Robot definitions held in raw user-overlay bytes.

    Args:
        source: Contents of ``user_robots.json``, or None when the overlay is
            absent. Taken as bytes so a caller that keys a cache on the file's
            contents parses exactly what it keyed on.

    Returns:
        Dict mapping robot names to their definitions; empty when the overlay
        is absent, unreadable, or does not declare ``"robots"``.
    """
    if source is None:
        return {}
    try:
        data = json.loads(source)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # The getter must not raise: ``get_robot()`` reads the overlay on every
        # lookup and a crashed writer could leave the file corrupt. The
        # ``Robot(...)`` 404 reads :func:`last_parse_error` and names the file
        # + exact problem, so the user who hand-edited the overlay is not left
        # with "Unknown robot" and a WARNING log line most runs don't surface.
        _last_parse_error[0] = (str(user_registry_path()), str(exc))
        logger.warning("Failed to load user registry %s: %s", user_registry_path(), exc)
        return {}
    _last_parse_error[0] = None
    robots = data.get("robots") if isinstance(data, dict) else None
    return robots if isinstance(robots, dict) else {}


# Mutable one-slot container so :func:`last_parse_error` reads the latest
# parse state without exposing a module-level variable readers would stomp.
# Written here, read by ``strands_robots/robot.py::_validate_known_robot``.
_last_parse_error: list[tuple[str, str] | None] = [None]


def last_parse_error() -> tuple[str, str] | None:
    """The most recent user-overlay parse failure, or ``None``.

    Returns:
        ``(path, reason)`` for the last call to :func:`parse_user_robots` that
        raised ``JSONDecodeError`` / ``UnicodeDecodeError`` and was swallowed,
        or ``None`` if the latest parse succeeded (or no overlay exists). The
        ``Robot(...)`` 404 reads this so a user whose hand-edit broke the
        overlay is told what to fix - the parse error itself is otherwise only
        in a WARNING log line.
    """
    return _last_parse_error[0]
