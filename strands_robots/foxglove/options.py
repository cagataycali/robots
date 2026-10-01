"""Resolve the ``foxglove=`` family of constructor keywords into one settings object.

``Robot(name, foxglove=...)``, ``MuJoCoSimEngine(foxglove=...)`` and the
real-hardware ``Robot`` all take the same three keywords and hand them here,
so the accepted spellings and every refusal sentence exist exactly once:

* ``foxglove``: ``False`` (default, nothing starts), ``True`` (a live server on
  ``127.0.0.1:8765``) or ``"host:port"`` (``"0.0.0.0:8765"`` to serve the LAN,
  ``":0"`` for an ephemeral port). ``STRANDS_ROBOTS_FOXGLOVE=1`` (or a
  ``host:port``) turns it on for a call that left the keyword at ``False``.
* ``foxglove_mcap``: a path the same channels are written to as an MCAP file.
  Requires the server to be on, and refuses to overwrite an existing file.
* ``foxglove_services``: ``True`` advertises the gated ``Call Service``
  surface. Default ``False``: the server then advertises no capability at all.

Nothing here imports ``foxglove`` or ``mcap``; the settings object is plain data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_robots.utils import boolean_flag_error, is_boolean, refusal_repr

#: Environment switch that turns the live server on for every ``Robot()``.
FOXGLOVE_ENV = "STRANDS_ROBOTS_FOXGLOVE"

#: Where the server listens when ``foxglove=True`` names no address.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: How many ports above the requested one are tried when it is busy.
PORT_SEARCH_WIDTH = 16

_OFF_SPELLINGS = frozenset({"", "0", "false", "no", "off"})
_ON_SPELLINGS = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class FoxgloveOptions:
    """The resolved ``foxglove=`` keywords, ready for :class:`~strands_robots.foxglove.FoxgloveBridge`."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    mcap: Path | None = None
    services: bool = False
    #: True when :data:`FOXGLOVE_ENV` switched the server on rather than the keyword.
    from_env: bool = False


def _split_host_port(value: str, param: str, context: str) -> tuple[str, int] | str:
    """Parse ``"host:port"`` / ``":port"`` / ``"host"``; return the pair or a refusal."""
    text = value.strip()
    host, sep, port_text = text.rpartition(":")
    if not sep:
        host, port_text = text, str(DEFAULT_PORT)
    host = host.strip() or DEFAULT_HOST
    port_text = port_text.strip() or str(DEFAULT_PORT)
    if not port_text.isdigit() or not 0 <= int(port_text) <= 65535:
        return (
            f"{context}: {param}={refusal_repr(value)} does not name a port: "
            "write 'host:port' with a port in 0-65535, e.g. '0.0.0.0:8765', "
            "or ':0' for an ephemeral port."
        )
    return host, int(port_text)


def foxglove_address_error(value: Any, param: str, context: str) -> str | None:
    """Refusal for a ``foxglove=`` value that is neither a boolean nor a ``host:port`` string.

    Args:
        value: The caller's value.
        param: The keyword it arrived as, for the message.
        context: The owner (``"Robot"``, ``"MuJoCoSimEngine"``), for the message.

    Returns:
        The refusal sentence, or ``None`` when the value can be honoured.
    """
    if is_boolean(value):
        return None
    if isinstance(value, str):
        parsed = _split_host_port(value, param, context)
        return parsed if isinstance(parsed, str) else None
    return (
        f"{context}: {param} must be True, False or a 'host:port' string, got {refusal_repr(value)}. "
        "True serves on 127.0.0.1:8765; '0.0.0.0:8765' serves the LAN; ':0' picks a free port."
    )


def foxglove_mcap_error(value: Any, param: str, context: str) -> str | None:
    """Refusal for a ``foxglove_mcap=`` value that cannot be written as a new file.

    A path that already exists is refused rather than overwritten: an MCAP is
    a recording, and the cheapest way to lose one is a second run with the
    same default path.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        return f"{context}: {param} must be a file path (str or PathLike), got {refusal_repr(value)}."
    path = Path(value)
    if path.exists():
        return (
            f"{context}: {param}={refusal_repr(str(path))} already exists and would be overwritten; "
            "pick a new path (the file is a recording, not a log)."
        )
    if not path.parent.exists():
        return f"{context}: {param}={refusal_repr(str(path))}: its directory does not exist."
    return None


def _env_switch(context: str) -> str | bool:
    """Read :data:`FOXGLOVE_ENV`: ``False`` when unset or off, ``True`` or a ``host:port`` otherwise."""
    raw = os.environ.get(FOXGLOVE_ENV)
    if raw is None:
        return False
    text = raw.strip()
    if text.lower() in _OFF_SPELLINGS:
        return False
    if text.lower() in _ON_SPELLINGS:
        return True
    if error := foxglove_address_error(text, FOXGLOVE_ENV, context):
        raise ValueError(error)
    return text


def resolve_foxglove_options(
    foxglove: Any,
    *,
    foxglove_mcap: Any = None,
    foxglove_services: Any = False,
    context: str,
) -> FoxgloveOptions | None:
    """Validate the three keywords and return the settings, or ``None`` when nothing should start.

    Args:
        foxglove: ``False``, ``True`` or ``"host:port"``. ``False`` still
            consults :data:`FOXGLOVE_ENV`.
        foxglove_mcap: Optional path for the MCAP sidecar.
        foxglove_services: Whether to advertise the gated service surface.
        context: Owner name for refusal sentences.

    Returns:
        A :class:`FoxgloveOptions`, or ``None`` when the server stays off.

    Raises:
        ValueError: A keyword that cannot be honoured, with the sentence naming it.
    """
    if error := foxglove_address_error(foxglove, "foxglove", context):
        raise ValueError(error)
    if error := boolean_flag_error(foxglove_services, "foxglove_services", context):
        raise ValueError(error)
    if error := foxglove_mcap_error(foxglove_mcap, "foxglove_mcap", context):
        raise ValueError(error)

    from_env = False
    if foxglove is False:
        switched = _env_switch(context)
        if switched is False:
            if foxglove_mcap is not None or foxglove_services:
                raise ValueError(
                    f"{context}: foxglove_mcap / foxglove_services require foxglove=True "
                    "(they configure the Foxglove server, which is off here)."
                )
            return None
        foxglove, from_env = switched, True

    host, port = DEFAULT_HOST, DEFAULT_PORT
    if isinstance(foxglove, str):
        parsed = _split_host_port(foxglove, "foxglove", context)
        if isinstance(parsed, str):  # pragma: no cover - screened above
            raise ValueError(parsed)
        host, port = parsed
    return FoxgloveOptions(
        host=host,
        port=port,
        mcap=Path(foxglove_mcap) if foxglove_mcap is not None else None,
        services=bool(foxglove_services),
        from_env=from_env,
    )
