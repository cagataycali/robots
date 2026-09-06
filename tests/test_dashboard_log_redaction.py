"""A log that leaks a working credential is a vulnerability, not a log.

Measured on the live dashboard: 63,000 access-log lines, each carrying a complete valid
JWT in a WebSocket query string, in a 21 MB world-readable file in /tmp - for a
dashboard published through a public tunnel that can drive real arms.
"""

from __future__ import annotations

import io
import logging

import pytest

from strands_robots.dashboard.log_redaction import (
    _WITHHELD,
    RedactingFilter,
    fingerprint,
    forget_secrets,
    install_redaction,
    redact_secrets,
    register_secret,
)

REAL_LINE = (
    '2600:4041:4256:7e00:0 - "WebSocket /ws/camera/so101-arm-1/top?'
    "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJyLVFOMmNiVDlIRXEt."
    'W1L2czgAYPYQO-zzCc-0C-HEDGwliiChAZR9jZuScQE" [accepted]'
)


class TestTheRealLine:
    def test_the_token_is_gone(self) -> None:
        out = redact_secrets(REAL_LINE)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
        assert "HEDGwliiChAZR9jZuScQE" not in out

    def test_the_line_is_still_a_useful_log_line(self) -> None:
        # the whole point of the access log is answering "which socket, from whom,
        # how many times" - redaction must not take that away
        out = redact_secrets(REAL_LINE)
        assert "/ws/camera/so101-arm-1/top" in out
        assert "2600:4041:4256:7e00:0" in out
        assert "[accepted]" in out
        assert "token=" in out, "the parameter NAME is not a secret and its absence hides the shape"

    def test_two_different_tokens_stay_distinguishable(self) -> None:
        a = redact_secrets("?token=" + "a" * 40 + "wxyz")
        b = redact_secrets("?token=" + "b" * 40 + "abcd")
        assert a != b


class TestWhatCountsAsASecret:
    def test_every_credential_query_key(self) -> None:
        for key in ("token", "access_code", "api_key", "password", "secret"):
            out = redact_secrets(f"GET /x?{key}=supersecretvalue123 HTTP/1.1")
            assert "supersecretvalue123" not in out, key

    def test_a_bearer_header(self) -> None:
        assert "kDD9toTMVDwOXYn5XfDI0v9nKGC6tSM8xPKcNaco" not in redact_secrets(
            "Authorization: Bearer kDD9toTMVDwOXYn5XfDI0v9nKGC6tSM8xPKcNaco"
        )

    def test_a_loose_jwt_with_no_key_to_hang_on(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYWdhdGF5In0.QWxsWW91ck1lc2g"
        assert jwt not in redact_secrets(f"auth failed for {jwt} from 10.0.0.5")

    def test_case_does_not_help_a_leak(self) -> None:
        assert "LEAKYVALUE" not in redact_secrets("?TOKEN=LEAKYVALUE")

    def test_ordinary_query_parameters_are_untouched(self) -> None:
        line = 'GET /api/training/datasets?q=so101&limit=12 HTTP/1.1" 200 OK'
        assert redact_secrets(line) == line

    def test_a_path_that_merely_contains_the_word_token_is_untouched(self) -> None:
        line = 'GET /api/auth/bootstrap_token HTTP/1.1" 200 OK'
        assert redact_secrets(line) == line

    def test_empty_and_none_ish_input_is_safe(self) -> None:
        assert redact_secrets("") == ""


class TestFingerprint:
    def test_it_reports_length_and_a_tail(self) -> None:
        fp = fingerprint("abcdefghijkl")
        assert "12" in fp and "ijkl" in fp

    def test_a_short_secret_gets_no_tail(self) -> None:
        # 4 of 200 JWT characters is not a credential; 4 of 6 might be
        assert fingerprint("abc123") == "<redacted:6>"


class TestFilterInstallation:
    def test_a_record_is_redacted_in_place(self, caplog) -> None:
        logger = logging.getLogger("test.redaction.inplace")
        logger.addFilter(RedactingFilter())
        with caplog.at_level(logging.INFO, logger="test.redaction.inplace"):
            logger.info("opened ?token=%s", "eyJa.bcdefgh.ijklmnop")
        assert "eyJa.bcdefgh.ijklmnop" not in caplog.text
        assert "token=" in caplog.text

    def test_install_is_idempotent(self) -> None:
        install_redaction(("test.redaction.twice",))
        install_redaction(("test.redaction.twice",))
        filters = logging.getLogger("test.redaction.twice").filters
        assert sum(isinstance(f, RedactingFilter) for f in filters) == 1

    def test_a_broken_record_does_not_break_logging(self) -> None:
        class _Bad:
            def __str__(self) -> str:
                raise RuntimeError("nope")

        record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", (_Bad(),), None)
        assert RedactingFilter().filter(record) is True


# --- Q117: the shapes the SHAPE-BASED rules missed ---------------------------------------------
#
# MEASURED against this machine's live token in nine realistic log lines: FIVE printed it verbatim.
# Every fixture above shares one incidental property - the secret sits after `key=` or `Bearer ` -
# so these rules were being tested against that property rather than against "a credential must not
# reach a log". That is the Q116 law a second time, and here it costs a live token instead of a
# cache header.
FAKE = "kDD6toTMVDwOXYn51XfDI0vNnKGC4tSM5xP5c858aco"


@pytest.mark.parametrize(
    "line",
    [
        f"env STRANDS_DASHBOARD_TOKEN={FAKE} inherited by child",  # prefixed key, `=`
        f"curl -H 'X-Auth-Token: {FAKE}' localhost:8090",  # prefixed key, `:`
        f'{{"token": "{FAKE}"}}',  # JSON body echoed into a log
        f'headers={{"authorization": "{FAKE}"}}',  # no "Bearer " to hang it on
        f"api_key = {FAKE}",  # spaces around the separator
    ],
)
def test_a_credential_is_redacted_whatever_holds_it(line: str) -> None:
    assert FAKE not in redact_secrets(line)


def test_a_registered_literal_is_redacted_in_a_shape_nobody_predicted() -> None:
    """The rail that cannot be out-guessed: argv and prose have no key at all."""
    argv = f"spawn argv: ['python', '-m', 'strands_robots.dashboard', '--token', '{FAKE}']"
    prose = f"wrote the token to ~/.strands_dashboard/local_api_token.txt ({FAKE})"
    try:
        assert FAKE in redact_secrets(prose)  # no key, no pattern: unredactable by shape alone
        register_secret(FAKE)
        for line in (argv, prose):
            out = redact_secrets(line)
            assert FAKE not in out
            assert "<redacted:43:8aco>" in out  # the fingerprint keeps the log useful
    finally:
        forget_secrets()


def test_a_registered_literal_does_not_corrupt_the_fingerprint_of_a_keyed_value() -> None:
    """Order matters: literals LAST.

    A fingerprint's own text matches the value pattern, so replacing literals first made the keyed
    rail redact the LABEL and print a wrong length: `?token=<redacted:18:aco>>`.
    """
    try:
        register_secret(FAKE)
        assert redact_secrets(f"?token={FAKE}&x=1") == "?token=<redacted:43:8aco>&x=1"
    finally:
        forget_secrets()


def test_a_short_value_is_never_registered() -> None:
    """Redacting a 4-character string would scribble over ordinary words in every line."""
    try:
        register_secret("abc")
        register_secret("")
        register_secret(None)
        assert redact_secrets("abc is a note about abcdef") == "abc is a note about abcdef"
    finally:
        forget_secrets()


def test_an_http_status_code_survives_but_an_oauth_code_does_not() -> None:
    """`code` is a credential key AND the commonest word in an HTTP log.

    Measured before this split: `response code=404` was logged as `code=<redacted:3>`, which hides
    the one thing that line exists to say.
    """
    assert redact_secrets("response code=404 detail='no endpoint'") == "response code=404 detail='no endpoint'"
    assert redact_secrets("HTTP status code: 200") == "HTTP status code: 200"
    assert "4/0AY0e-g7xQoauthgrant" not in redact_secrets("code=4/0AY0e-g7xQoauthgrant")


# --- what a formatter actually renders --------------------------------------------------------
#
# MEASURED through install_redaction() on a logger with one handler: the request line was
# redacted and the traceback underneath it printed the same token verbatim. `Formatter.format`
# renders three parts - the message, `exc_text` (from `exc_info`) and `stack_info` - and appends
# the last two after every filter has run, so reading `getMessage()` graded one of the three.
# The same measurement showed the OTHER half: install_redaction attaches the filter to the
# logger AND to its handlers, so a record was redacted twice and the second pass fingerprinted
# the first pass's fingerprint - `?token=<redacted:18:aco>>` for a 43-character token, the exact
# corruption test_a_registered_literal_does_not_corrupt_the_fingerprint_of_a_keyed_value forbids.
TOKEN = "kDD6toTMVDwOXYn51XfDI0vNnKGC4tSM5xP5c858aco"


@pytest.fixture
def rendered(request):
    """A logger wired the way install_redaction() wires one, plus what a formatter wrote.

    Yields ``(logger, read)``: ``read()`` returns the fully rendered log text, so the
    assertions grade what reaches the file rather than the record's attributes.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(f"test.redaction.rendered.{request.node.name}")
    logger.handlers[:] = [handler]
    logger.filters[:] = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    install_redaction((logger.name,))
    register_secret(TOKEN)
    try:
        yield logger, stream.getvalue
    finally:
        forget_secrets()
        logger.handlers[:] = []
        logger.filters[:] = []


def test_a_credential_in_a_traceback_is_redacted(rendered) -> None:
    """The exception message is where a request URL ends up, and it is rendered after the filters."""
    logger, read = rendered
    try:
        raise ConnectionError(f"upstream rejected /ws/camera?token={TOKEN}")
    except ConnectionError:
        logger.exception("camera socket failed")
    out = read()
    assert TOKEN not in out
    assert fingerprint(TOKEN) in out
    # Still a traceback: redacting must not cost the diagnosis it was written for.
    assert "ConnectionError" in out and "Traceback" in out


def test_a_credential_in_stack_info_is_redacted() -> None:
    """`stack_info` is the third part a formatter appends, verbatim, on the same footing as exc_text.

    Set on the record rather than through ``stack_info=True``, because that flag captures the
    stack of the logging call itself - source text, which holds a credential's NAME, not its value.
    """
    try:
        register_secret(TOKEN)
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "handshake failed", None, None)
        record.stack_info = f'  File "app.py", line 1, in ws\n    connect("?token={TOKEN}")'
        assert RedactingFilter().filter(record) is True
        out = logging.Formatter("%(message)s").format(record)
        assert TOKEN not in out
        assert fingerprint(TOKEN) in out
    finally:
        forget_secrets()


def test_the_second_installed_filter_does_not_redact_the_first_ones_fingerprint(rendered) -> None:
    """install_redaction attaches at the logger AND its handlers, so redaction runs twice.

    A fingerprint's own text matches the value pattern, so the second pass reported the
    fingerprint's length (18) instead of the credential's (43).
    """
    logger, read = rendered
    logger.info("WebSocket /ws/camera?token=%s [accepted]", TOKEN)
    assert read().strip() == f"WebSocket /ws/camera?token={fingerprint(TOKEN)} [accepted]"


def test_a_record_whose_message_cannot_be_formatted_still_has_its_traceback_redacted() -> None:
    """A broken message must not break logging - and must not exempt the rest of the record."""

    class _Bad:
        def __str__(self) -> str:
            raise RuntimeError("nope")

    try:
        register_secret(TOKEN)
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", (_Bad(),), None)
        record.exc_text = f"ConnectionError: /ws?token={TOKEN}"
        assert RedactingFilter().filter(record) is True
        assert record.exc_text is not None
        assert TOKEN not in record.exc_text
    finally:
        forget_secrets()


def test_an_exc_info_that_is_not_the_documented_tuple_does_not_break_logging() -> None:
    """Only the 3-tuple logging documents can be rendered; anything else is left alone."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    record.exc_info = True  # type: ignore[assignment]  # what a hand-built record may carry
    assert RedactingFilter().filter(record) is True
    assert record.exc_text is None


def test_a_malformed_exc_info_tuple_degrades_the_way_stock_logging_does() -> None:
    """A Filter has no `handleError` guard, so a part it cannot render must not escape as an exception.

    `Formatter.format` runs inside `Handler.emit`, which degrades a record it cannot render to a
    note on stderr and lets the logging call return. A filter runs in `Logger.handle`, with no such
    guard - so rendering `exc_info` here has to answer for the records `Logger._log` accepts but
    `formatException` cannot render, of which a 3-tuple whose middle value is not an exception is
    one. The default target set includes the root logger, so an escape here would come out of ANY
    logging call in the process, including the safety-adjacent code whose logs this filter cleans.
    """
    broken = (ValueError, "not an exception", None)

    def wired(name: str, *, redact: bool) -> logging.Logger:
        logger = logging.getLogger(name)
        # The handler list is set HERE rather than in a fixture on purpose: the test harness
        # attaches its own capture handlers to a non-propagating logger during setup, and their
        # `handleError` re-raises by design - which would grade the harness, not stock logging.
        logger.handlers[:] = [logging.StreamHandler(io.StringIO())]
        logger.filters[:] = []
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        if redact:
            install_redaction((name,))
        return logger

    stock = wired("test.redaction.malformed.stock", redact=False)
    filtered = wired("test.redaction.malformed.filtered", redact=True)
    try:
        stock.error("boom", exc_info=broken)  # type: ignore[arg-type]  # a caller bug stock logging survives
        filtered.error("boom", exc_info=broken)  # type: ignore[arg-type]  # ... and so must this one
        # the filter leaves the part it could not render for the formatter, under emit's guard
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "boom", None, broken)  # type: ignore[arg-type]
        assert RedactingFilter().filter(record) is True
        assert record.exc_text is None
    finally:
        for logger in (stock, filtered):
            logger.handlers[:] = []
            logger.filters[:] = []


@pytest.mark.parametrize("part", ["exc_text", "stack_info"])
def test_an_appended_part_that_cannot_be_redacted_is_withheld_not_raised(part: str) -> None:
    """Withholding is the fail-closed answer: the part may hold the credential, so it is not written.

    `redact_secrets` is a str operation and both appended attributes are whatever a caller set, so
    a hand-built non-str part made it raise out of the caller's own logging call. Each rail is
    guarded separately - the message keeps its redaction whatever the appended parts are.
    """
    try:
        register_secret(TOKEN)
        record = logging.LogRecord("x", logging.INFO, __file__, 1, f"?token={TOKEN}", None, None)
        setattr(record, part, b"handshake failed")  # bytes: what redact_secrets cannot read
        assert RedactingFilter().filter(record) is True
        assert getattr(record, part) == _WITHHELD
        # the record still renders, the message is still redacted, and the marker says why the
        # appended part is absent rather than leaving a reader to wonder
        out = logging.Formatter("%(message)s").format(record)
        assert TOKEN not in out
        assert fingerprint(TOKEN) in out
        assert _WITHHELD in out
    finally:
        forget_secrets()


# --- the record has to stay renderable by the formatter it was written for ----------------------
#
# MEASURED against uvicorn 0.41's own AccessFormatter: three requests logged, TWO lines in the
# access log. `AccessFormatter.formatMessage` unpacks five values out of `record.args` and builds
# the request line from them - it never reads the message this filter cleaned - so baking the
# redacted text into `msg` and clearing `args`, the stock way to freeze a redacted message, left
# it nothing to unpack. It raised ValueError inside `Handler.emit` for exactly the records
# carrying a credential, and this module's own docstring says every camera and mesh socket carries
# one: the access log kept every ordinary request and dropped every authenticated handshake. That
# is `test_the_line_is_still_a_useful_log_line` read one layer up - the line was not less useful,
# it was absent - and a dropped handshake is the audit trail this module exists to keep.

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYW1lcmEiLCJleHAiOjk5fQ.c2lnbmF0dXJlX2hlcmVfMTIzNA"

#: uvicorn's access-log call, verbatim: the credential rides in ``args[2]``, never in the format
#: string, and the last arg is an ``int`` under a ``%d``.
_ACCESS_MSG = '%s - "%s %s HTTP/%s" %d'


def _access_record(full_path: str) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        _ACCESS_MSG,
        ("127.0.0.1:52111", "GET", full_path, "1.1", 200),
        None,
    )


class _PositionalFormatter(logging.Formatter):
    """Reads ``record.args`` positionally, which is what uvicorn's AccessFormatter does."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        client_addr, method, full_path, http_version, status_code = record.args  # type: ignore[misc]
        return f'{client_addr} - "{method} {full_path} HTTP/{http_version}" {status_code}'


class TestARedactedRecordIsStillRenderable:
    """``args`` is a rendered part: redact it, do not discard it."""

    def test_a_formatter_that_reads_args_positionally_still_has_them(self) -> None:
        record = _access_record(f"/ws/camera?token={JWT}")
        RedactingFilter().filter(record)
        line = _PositionalFormatter().format(record)
        assert JWT not in line
        # and it is still the log line the access log exists for
        assert "/ws/camera?token=" in line
        assert '"GET' in line and "200" in line

    def test_the_arity_the_formatter_unpacks_is_preserved(self) -> None:
        record = _access_record(f"/ws/camera?token={JWT}")
        RedactingFilter().filter(record)
        assert isinstance(record.args, tuple)
        assert len(record.args) == 5

    def test_the_credential_is_gone_from_the_args_themselves(self) -> None:
        # A formatter reading args raw never sees the cleaned message, so the redaction has
        # to have happened in the values, not only in the rendered text.
        record = _access_record(f"/ws/camera?token={JWT}")
        RedactingFilter().filter(record)
        assert not any(JWT in arg for arg in record.args if isinstance(arg, str))  # type: ignore[union-attr]

    def test_an_arg_that_is_not_a_string_is_left_as_it_is(self) -> None:
        # ``%d`` needs an int: a fingerprint there would leave a format string its own args
        # cannot render, which is the failure this fix is removing, reintroduced.
        record = _access_record(f"/ws/camera?token={JWT}")
        RedactingFilter().filter(record)
        assert record.args[-1] == 200  # type: ignore[index]
        assert logging.Formatter("%(message)s").format(record).endswith(" 200")

    def test_the_message_rail_is_redacted_as_before(self) -> None:
        record = _access_record(f"/ws/camera?token={JWT}")
        RedactingFilter().filter(record)
        assert JWT not in record.getMessage()
        assert f"token={fingerprint(JWT)}" in record.getMessage()

    def test_dict_style_args_are_redacted_too(self) -> None:
        # a mapping is passed INSIDE the args tuple, which is what ``Logger._log`` hands over;
        # ``LogRecord`` then unwraps it onto ``record.args``.
        record = logging.LogRecord(
            "x", logging.INFO, __file__, 1, "socket %(path)s", ({"path": f"/ws?token={JWT}"},), None
        )
        RedactingFilter().filter(record)
        assert JWT not in record.getMessage()
        if isinstance(record.args, dict):  # arity preserved -> the key is still there
            assert JWT not in record.args["path"]

    def test_a_record_carrying_no_credential_is_untouched(self) -> None:
        record = _access_record("/api/status")
        RedactingFilter().filter(record)
        assert record.args == ("127.0.0.1:52111", "GET", "/api/status", "1.1", 200)
        assert record.msg == _ACCESS_MSG


class TestACredentialVisibleOnlyOnceJoinedStillFailsClosed:
    """No per-arg redaction can reach it, so the message is baked and the args dropped."""

    def test_the_message_is_redacted_and_the_args_are_dropped(self) -> None:
        # ``supersecretvalue123`` is credential-shaped only because ``?token=`` precedes it in
        # the FORMAT STRING - the arg on its own is an ordinary word.
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "?token=%s", ("supersecretvalue123",), None)
        RedactingFilter().filter(record)
        assert "supersecretvalue123" not in record.getMessage()
        assert record.args == ()

    def test_a_registered_literal_is_redacted_without_fingerprinting_the_fingerprint(self) -> None:
        # The arg IS the credential here, so the args are kept - and the verdict must not be
        # taken by re-running the redaction, whose own fingerprint matches the value pattern
        # and would report a 18-character token for a 43-character one.
        try:
            register_secret(TOKEN)
            record = logging.LogRecord(
                "x", logging.INFO, __file__, 1, "WebSocket /ws/camera?token=%s [accepted]", (TOKEN,), None
            )
            RedactingFilter().filter(record)
            assert record.getMessage() == f"WebSocket /ws/camera?token={fingerprint(TOKEN)} [accepted]"
        finally:
            forget_secrets()

    def test_a_record_whose_message_cannot_be_rendered_does_not_break_logging(self) -> None:
        class _Bad:
            def __str__(self) -> str:
                raise RuntimeError("nope")

        record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", (_Bad(),), None)
        assert RedactingFilter().filter(record) is True


class TestTheAccessLogKeepsEveryRequest:
    """The whole defect, end to end, through uvicorn's own formatter."""

    def _access_logger(self, name: str, formatter: logging.Formatter) -> tuple[logging.Logger, io.StringIO]:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.handlers[:] = [handler]
        logger.filters[:] = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        install_redaction((name,))
        return logger, stream

    @pytest.mark.parametrize("real_uvicorn", [False, True], ids=["positional", "uvicorn"])
    def test_three_requests_produce_three_lines(self, real_uvicorn: bool) -> None:
        if real_uvicorn:
            uvicorn_logging = pytest.importorskip("uvicorn.logging")
            formatter: logging.Formatter = uvicorn_logging.AccessFormatter(
                '%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False
            )
        else:
            formatter = _PositionalFormatter()
        logger, stream = self._access_logger(f"test.redaction.access.{real_uvicorn}", formatter)
        try:
            for path in ("/api/status", f"/ws/camera?token={JWT}", "/api/robots"):
                logger.info(_ACCESS_MSG, "127.0.0.1:52111", "GET", path, "1.1", 200)
            written = stream.getvalue()
        finally:
            logger.handlers[:] = []
            logger.filters[:] = []
        assert len(written.strip().splitlines()) == 3, "the handshake carrying a credential was dropped"
        assert JWT not in written
        assert "/ws/camera?token=" in written, "which socket was opened is what the access log is for"
