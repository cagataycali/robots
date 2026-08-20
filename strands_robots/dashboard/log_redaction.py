"""Keep credentials out of the log file.

A browser cannot set headers on a WebSocket handshake, so every camera and mesh socket
carries its JWT in the QUERY STRING — and uvicorn's access log writes the request line
verbatim. Measured on this Mac after ten hours of uptime: 63,000 lines of
``"WebSocket /ws/camera/so101-arm-1/top?token=eyJhbGciOi…" [accepted]``, each one a
complete, still-valid bearer token, in a 21 MB file in /tmp with mode 0644. Anything
that can read that file — any local process, any log shipper, a pasted tail in a bug
report, this agent's own transcripts — holds a working credential for a dashboard that
is published through a public tunnel and can drive real robot arms.

The token is not the leak; logging it is. So the log line is redacted at the logging
layer, which covers every current and future call site at once: uvicorn's access log,
uvicorn.error's handshake messages, and anything the app logs that happens to contain a
URL. ``redact_secrets`` is pure and takes the whole formatted message, because the
secret can be anywhere in it.

What is deliberately kept: the parameter NAME, the path, the peer, the status. A
redacted log must still be a usable log — "which socket did this phone open 63,000
times" has to remain answerable, and the fingerprint (length + last 4) is enough to
tell two tokens apart without being enough to use one.
"""

from __future__ import annotations

import logging
import re

#: query parameters whose VALUE is a credential
_SECRET_QUERY_KEYS = ("token", "access_code", "api_key", "apikey", "password", "secret", "code")

_QUERY_RE = re.compile(
    r"(?P<key>\b(?:" + "|".join(_SECRET_QUERY_KEYS) + r")=)(?P<val>[^\s&\"'#]+)",
    re.IGNORECASE,
)
#: `Authorization: Bearer xyz`, and the bare `Bearer xyz` some clients log
_BEARER_RE = re.compile(r"(?i)(?P<key>bearer\s+)(?P<val>[A-Za-z0-9._\-~+/]{8,}=*)")
#: a JWT sitting loose in a message, with no key to hang the redaction on
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")


def fingerprint(secret: str) -> str:
    """A stable, non-usable label for a credential: its length and last 4 characters.

    Two sockets opened by two different phones must remain distinguishable in a log —
    that is most of what the log is FOR — and 4 characters of a 200-character JWT is
    not a credential.
    """
    tail = secret[-4:] if len(secret) >= 8 else ""
    return f"<redacted:{len(secret)}{':' + tail if tail else ''}>"


def redact_secrets(message: str) -> str:
    """Return ``message`` with every credential-shaped value replaced by a fingerprint."""
    if not message:
        return message

    def _q(m: re.Match[str]) -> str:
        return m.group("key") + fingerprint(m.group("val"))

    out = _QUERY_RE.sub(_q, message)
    out = _BEARER_RE.sub(lambda m: m.group("key") + fingerprint(m.group("val")), out)
    return _JWT_RE.sub(lambda m: fingerprint(m.group(0)), out)


class RedactingFilter(logging.Filter):
    """Redacts the FORMATTED message of every record that passes through.

    A filter rather than a formatter: filters are inherited by every handler on the
    logger, so one install covers handlers added later (uvicorn adds its own), and a
    record whose secret is in an ``args`` tuple is still caught because the record is
    collapsed with ``getMessage()`` first.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            original = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not break logging
            return True
        cleaned = redact_secrets(original)
        if cleaned != original:
            record.msg = cleaned
            record.args = ()
        return True


#: loggers that carry request lines; the root catches everything else
_TARGETS = ("", "uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "strands_robots")


def install_redaction(logger_names: tuple[str, ...] = _TARGETS) -> None:
    """Attach the filter to the loggers that can carry a URL. Idempotent."""
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactingFilter) for f in logger.filters):
            logger.addFilter(RedactingFilter())
        # uvicorn attaches its own handlers with their own filter chain; a handler-level
        # filter is what actually catches records logged directly to those handlers
        for handler in logger.handlers:
            if not any(isinstance(f, RedactingFilter) for f in handler.filters):
                handler.addFilter(RedactingFilter())
