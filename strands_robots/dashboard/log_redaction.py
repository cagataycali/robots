"""Keep credentials out of the log file.

A browser cannot set headers on a WebSocket handshake, so every camera and mesh socket
carries its JWT in the QUERY STRING - and uvicorn's access log writes the request line
verbatim. Measured on this Mac after ten hours of uptime: 63,000 lines of
``"WebSocket /ws/camera/so101-arm-1/top?token=eyJhbGciOi…" [accepted]``, each one a
complete, still-valid bearer token, in a 21 MB file in /tmp with mode 0644. Anything
that can read that file - any local process, any log shipper, a pasted tail in a bug
report, this agent's own transcripts - holds a working credential for a dashboard that
is published through a public tunnel and can drive real robot arms.

The token is not the leak; logging it is. So the log line is redacted at the logging
layer, which covers every current and future call site at once: uvicorn's access log,
uvicorn.error's handshake messages, and anything the app logs that happens to contain a
URL. ``redact_secrets`` is pure and takes the whole formatted message, because the
secret can be anywhere in it.

What is deliberately kept: the parameter NAME, the path, the peer, the status. A
redacted log must still be a usable log - "which socket did this phone open 63,000
times" has to remain answerable, and the fingerprint (length + last 4) is enough to
tell two tokens apart without being enough to use one.
"""

from __future__ import annotations

import logging
import re

#: query parameters whose VALUE is a credential
_SECRET_QUERY_KEYS = ("token", "access_code", "api_key", "apikey", "password", "secret")
#: `code` is an oauth credential AND the commonest word in an HTTP log. Redacted only at credential
#: LENGTH so a status stays readable - measured, `response code=404 detail=...` was being logged as
#: `code=<redacted:3>`, which hides the one thing that line exists to say.
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

#: Q117. MEASURED against this machine's LIVE token in nine realistic log shapes: five printed it
#: verbatim. Every fixture this file was built from shared one incidental property - the secret sat
#: after `key=` or `Bearer ` - so the rules were tested against that property rather than against
#: "a credential must not reach a log". The shapes that leaked: an env assignment
#: (STRANDS_DASHBOARD_TOKEN=...), a JSON body ({"token": "..."}), a custom header
#: (X-Auth-Token: ...), an argv list (--token', '...') and prose with the value in parentheses.
#:
#: Rail 1 - the key may be PREFIXED (X-Auth-, STRANDS_DASHBOARD_) and separated by `=` or `:`, with
#: optional quotes around either side. `code` stays QUERY-ONLY on purpose: "code: 200" is an HTTP
#: status in a thousand log lines and redacting it would buy nothing and cost readability.
_KEYED_WORDS = ("token", "secret", "password", "passwd", "passphrase", "api_key", "apikey",
                "access_code", "auth", "authorization", "credential")
_KEYED_RE = re.compile(
    r"(?i)(?P<key>[\w.\-]*(?:" + "|".join(_KEYED_WORDS) + r")[\w.\-]*\"?'?\s*[:=]\s*\"?'?)"
    r"(?P<val>[A-Za-z0-9._\-~+/]{8,}=*)"
)

#: Rail 2, and the one that cannot be out-guessed: the process's OWN credentials, registered as
#: literals when they are loaded. A pattern must recognise a shape; this only has to match bytes it
#: was handed, so it covers argv, prose, a filename in parentheses - every shape nobody thought of.
#: Short values are ignored: redacting a 4-character string would scribble over ordinary words.
_known: set[str] = set()


def register_secret(value: str | None) -> None:
    """Redact ``value`` from every future log line, whatever shape it appears in."""
    if value and len(value.strip()) >= 12:
        _known.add(value.strip())


def forget_secrets() -> None:
    """Test-only: drop the registered literals (a leaked registration outlives one test)."""
    _known.clear()


def fingerprint(secret: str) -> str:
    """A stable, non-usable label for a credential: its length and last 4 characters.

    Two sockets opened by two different phones must remain distinguishable in a log -
    that is most of what the log is FOR - and 4 characters of a 200-character JWT is
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
    out = _LONG_QUERY_RE.sub(_q, out)
    # The keyed rail runs AFTER the query rail so `?token=x` keeps its narrower, well-tested
    # handling (it must stop at & and #, which a generic value pattern does not know about).
    out = _KEYED_RE.sub(_q, out)
    out = _BEARER_RE.sub(lambda m: m.group("key") + fingerprint(m.group("val")), out)
    out = _JWT_RE.sub(lambda m: fingerprint(m.group(0)), out)
    # Rail 2 LAST, deliberately: a fingerprint's own text ("<redacted:43:8aco>") matches the value
    # pattern, so replacing literals FIRST let the keyed rail redact the LABEL and print a wrong
    # length ("?token=<redacted:18:aco>>"). Patterns first, then whatever they missed - argv, prose,
    # a value in parentheses, any shape nobody thought of.
    for secret in sorted(_known, key=len, reverse=True):
        if secret in out:
            out = out.replace(secret, fingerprint(secret))
    return out


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
