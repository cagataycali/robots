"""A log that leaks a working credential is a vulnerability, not a log.

Measured on the live dashboard: 63,000 access-log lines, each carrying a complete valid
JWT in a WebSocket query string, in a 21 MB world-readable file in /tmp — for a
dashboard published through a public tunnel that can drive real arms.
"""

from __future__ import annotations

import logging

from strands_robots.dashboard.log_redaction import (
    RedactingFilter,
    fingerprint,
    install_redaction,
    redact_secrets,
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
