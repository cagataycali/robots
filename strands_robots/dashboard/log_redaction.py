"""Keep credentials out of the log file. A browser cannot set headers on a WebSocket handshake, so
every camera and mesh socket carries its JWT in the QUERY STRING - and uvicorn's access log
writes the request line verbatim. A failing socket then writes that same URL a second time,
inside a traceback, which a formatter appends after every filter has already run.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

#: query parameters whose VALUE is a credential
_SECRET_QUERY_KEYS = ("token", "access_code", "api_key", "apikey", "password", "secret")
# : `code` is an oauth credential AND the commonest word in an HTTP log.
_LONG_ONLY_QUERY_KEYS = ("code",)

_QUERY_RE = re.compile(
    r"(?P<key>\b(?:" + "|".join(_SECRET_QUERY_KEYS) + r")=)(?P<val>[^\s&\"'#]+)",
    re.IGNORECASE,
)
_LONG_QUERY_RE = re.compile(
    r"(?P<key>\b(?:" + "|".join(_LONG_ONLY_QUERY_KEYS) + r")=)(?P<val>[^\s&\"'#]{8,})",
    re.IGNORECASE,
)
#: `Authorization: Bearer xyz`, and the bare `Bearer xyz` some clients log
_BEARER_RE = re.compile(r"(?i)(?P<key>bearer\s+)(?P<val>[A-Za-z0-9._\-~+/]{8,}=*)")
#: a JWT sitting loose in a message, with no key to hang the redaction on
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")

_KEYED_WORDS = (
    "token",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "api_key",
    "apikey",
    "access_code",
    "auth",
    "authorization",
    "credential",
)
_KEYED_RE = re.compile(
    r"(?i)(?P<key>[\w.\-]*(?:" + "|".join(_KEYED_WORDS) + r")[\w.\-]*\"?'?\s*[:=]\s*\"?'?)"
    r"(?P<val>[A-Za-z0-9._\-~+/]{8,}=*)"
)

# : Rail 2, and the one that cannot be out-guessed: the process's OWN credentials, registered
# as : literals when they are loaded.
_known: set[str] = set()

#: every rail that names its credential in a ``val`` group, so the values a message carries can be
#: read off with the same patterns that redact them - see :func:`_credential_values`.
_VALUE_RAILS = (_QUERY_RE, _LONG_QUERY_RE, _KEYED_RE, _BEARER_RE)


def register_secret(value: str | None) -> None:
    """Redact ``value`` from every future log line, whatever shape it appears in."""
    if value and len(value.strip()) >= 12:
        _known.add(value.strip())


def forget_secrets() -> None:
    """Test-only: drop the registered literals (a leaked registration outlives one test)."""
    _known.clear()


def fingerprint(secret: str) -> str:
    """A stable, non-usable label for a credential: its length and last 4 characters."""
    tail = secret[-4:] if len(secret) >= 8 else ""
    return f"<redacted:{len(secret)}{':' + tail if tail else ''}>"


def _credential_values(message: str) -> tuple[str, ...]:
    """Every credential-shaped value :func:`redact_secrets` replaces in ``message``.

    Answers "is this text still carrying a credential" for
    :meth:`RedactingFilter.filter`, which cannot ask by re-running the redaction: a
    fingerprint's own text matches the value pattern, so a second pass reports a
    difference for text that is already clean - the corruption the literals-last rail
    order exists to prevent, read as a verdict instead of written into the line.
    """
    found: list[str] = []
    for rail in _VALUE_RAILS:
        found.extend(m.group("val") for m in rail.finditer(message))
    found.extend(m.group(0) for m in _JWT_RE.finditer(message))
    found.extend(secret for secret in _known if secret in message)
    return tuple(found)


def redact_secrets(message: str) -> str:
    """Return ``message`` with every credential-shaped value replaced by a fingerprint."""
    if not message:
        return message

    def _q(m: re.Match[str]) -> str:
        return m.group("key") + fingerprint(m.group("val"))

    out = _QUERY_RE.sub(_q, message)
    out = _LONG_QUERY_RE.sub(_q, out)
    # The keyed rail runs AFTER the query rail so `?token=x` keeps its narrower, well-tested
    # handling (it must stop at & and #, which a generic value pattern does not know about).
    out = _KEYED_RE.sub(_q, out)
    out = _BEARER_RE.sub(lambda m: m.group("key") + fingerprint(m.group("val")), out)
    out = _JWT_RE.sub(lambda m: fingerprint(m.group(0)), out)
    # Rail 2 LAST, deliberately: a fingerprint's own text ("<redacted:43:8aco>") matches the value
    # pattern, so replacing literals FIRST let the keyed rail redact the LABEL and print a wrong
    # length ("?token=<redacted:18:aco>>").
    for secret in sorted(_known, key=len, reverse=True):
        if secret in out:
            out = out.replace(secret, fingerprint(secret))
    return out


#: renders ``exc_info`` exactly as the default formatter would, so the text this filter
#: redacts is the text that would otherwise have been written.
_EXC_RENDERER = logging.Formatter()

#: marks a record this filter has already redacted (see :class:`RedactingFilter`).
_DONE = "_strands_robots_redacted"

#: written in place of an appended part this filter could not redact. Withholding is the only
#: fail-closed answer - the part may hold the credential - and a marker says why it is absent.
_WITHHELD = "<withheld: this part of the record could not be redacted>"


class RedactingFilter(logging.Filter):
    """Redacts every part of a record a formatter renders, and redacts it once.

    ``logging.Formatter.format`` renders THREE parts and appends the last two: the
    message, ``exc_text`` (rendered from ``exc_info``), and ``stack_info``. Both are
    appended AFTER every filter has run, so a filter that reads ``getMessage()`` alone
    never sees them - and an exception message is exactly where a request URL ends up,
    so ``logger.exception(...)`` wrote the credential verbatim into the traceback while
    the request line above it was redacted. ``exc_info`` is rendered here instead of
    later because the formatter reuses an ``exc_text`` that is already set.

    A formatter may also render ``args`` DIRECTLY rather than through ``getMessage()``,
    and uvicorn's own ``AccessFormatter`` does: it unpacks five values out of
    ``record.args`` and builds the request line from them, never reading the message this
    filter had cleaned. Baking the redacted text into ``msg`` and clearing ``args`` - the
    stock way to freeze a redacted message - therefore left that formatter nothing to
    unpack, and it raised ``ValueError`` inside ``Handler.emit`` for exactly the records
    carrying a credential: the access log kept every ordinary request and dropped every
    authenticated socket handshake, which is the audit trail this module exists to keep
    readable. So the args are redacted in place and their arity preserved, and the message
    is baked only when the credential is not visible in any single arg - the case where no
    per-arg redaction could reach it, and where failing closed is the right answer.

    Redacting is idempotent per record because :func:`install_redaction` attaches this
    filter at BOTH the logger and its handlers (a handler-level filter is what catches
    records logged straight to a handler). Without the marker the handler pass redacted
    the logger pass's OUTPUT: a fingerprint's own text matches the value pattern, so the
    line printed a wrong length - ``?token=<redacted:18:aco>>`` for a 43-character token,
    the same corruption the literals-last rail order exists to prevent.

    An appended part this filter cannot render or redact is WITHHELD, not raised: a
    ``Formatter`` runs inside ``Handler.emit``, where ``handleError`` degrades a broken
    record to a note on stderr, and a filter has no such guard - so an escape here would
    come out of the caller's own logging call, on any logger in :data:`_TARGETS`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, _DONE, False):
            return True
        try:
            original: str | None = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not break logging
            original = None
        if original is not None:
            cleaned = redact_secrets(original)
            if cleaned != original:
                # ``args`` is a rendered part too: a formatter may read the credential straight
                # out of it rather than out of the message. Redact the parts and keep the arity,
                # so a formatter that unpacks them still has values to unpack.
                carried = _credential_values(original)
                original_args = record.args
                record.args = _redacted_args(original_args)
                try:
                    rerendered: str | None = record.getMessage()
                except Exception:  # noqa: BLE001 - a broken record must not break logging
                    rerendered = None
                if (
                    rerendered is None
                    or any(secret in rerendered for secret in carried)
                    or not all(_wholly_inside_one_str_arg(secret, original_args) for secret in carried)
                ):
                    # The credential is not wholly inside a single str arg - either it
                    # straddles two args, or it is only credential-shaped once format
                    # string and args are joined, or re-rendering failed.  Bake the
                    # redacted text and drop the args: a formatter reading them
                    # positionally then fails under ``Handler.emit``'s guard, which is
                    # the fail-closed answer.
                    record.msg = cleaned
                    record.args = ()
        # Each appended rail carries the message rail's guard, and carries it SEPARATELY: a
        # part that cannot be rendered must not exempt the part that can. Only the 3-tuple
        # that logging documents is rendered here, but a shape check is not a guard: `Logger._log`
        # accepts a 3-tuple whose middle value is not an exception, and rendering one raises.
        if isinstance(record.exc_info, tuple) and len(record.exc_info) == 3 and not record.exc_text:
            try:
                record.exc_text = _EXC_RENDERER.formatException(record.exc_info)
            except Exception:  # noqa: BLE001 - a broken record must not break logging
                # Left unset: the formatter's own attempt at it runs inside `Handler.emit`,
                # where `handleError` degrades to a note on stderr and the caller's logging
                # call returns - which is what stock logging does with the same record.
                record.exc_text = None
        if record.exc_text:
            record.exc_text = _redact_appended(record.exc_text)
        if record.stack_info:
            record.stack_info = _redact_appended(record.stack_info)
        setattr(record, _DONE, True)
        return True


_Args = tuple[object, ...] | Mapping[str, object] | None


def _wholly_inside_one_str_arg(secret: str, args: _Args) -> bool:
    """True when ``secret`` is a contiguous substring of a single ``str`` arg.

    If a credential straddles two args (``"%s%s" % (url[:40], url[-40:])``), per-arg
    redaction replaces only the half it sees in each arg, and the other half survives in
    plaintext once the format string joins them.  The caller uses this to decide whether
    per-arg redaction is sufficient or the whole message must be baked and the args
    dropped (fail-closed).
    """
    if isinstance(args, Mapping):
        return any(isinstance(v, str) and secret in v for v in args.values())
    if isinstance(args, tuple):
        return any(isinstance(v, str) and secret in v for v in args)
    return False


def _redacted_args(args: _Args) -> _Args:
    """Redact the credential out of a record's ``args`` without changing what they are.

    Only ``str`` values are rewritten: an ``int`` under a ``%d`` is not a credential and
    replacing it with a fingerprint would leave a format string its own args cannot render.
    """
    if isinstance(args, Mapping):
        return {key: redact_secrets(val) if isinstance(val, str) else val for key, val in args.items()}
    if isinstance(args, tuple):
        return tuple(redact_secrets(val) if isinstance(val, str) else val for val in args)
    return args


def _redact_appended(part: object) -> str:
    """Redact one appended part, withholding it rather than raising out of the logging call.

    A `Formatter` runs inside `Handler.emit`, so a part it cannot render degrades through
    `Handler.handleError`; a `Filter` has no such guard, and a hand-built non-str part
    (bytes, say) made `redact_secrets` raise out of the caller's own logging call.
    """
    try:
        return redact_secrets(part)  # type: ignore[arg-type]  # a hand-built part may be anything
    except Exception:  # noqa: BLE001 - a broken record must not break logging
        return _WITHHELD


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
