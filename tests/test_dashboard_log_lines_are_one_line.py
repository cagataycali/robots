"""A request header is chosen by whoever sends the request, and a log file is
parsed by line. `log_redaction.one_line` is the step between the two: every
dashboard log statement that quotes a header value goes through it, so a CRLF
inside Host, Origin or X-Forwarded-For cannot forge a second log entry.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

from strands_robots.dashboard import auth, settings  # noqa: E402
from strands_robots.dashboard.log_redaction import one_line  # noqa: E402

FORGED = "evil.example\r\nWARNING strands_robots.dashboard.auth: passkey enrolled for root"


class TestOneLine:
    def test_a_crlf_becomes_its_escape_and_the_forged_line_stays_inside_the_value(self) -> None:
        line = one_line(FORGED)
        assert "\r" not in line and "\n" not in line
        assert line == "evil.example\\r\\nWARNING strands_robots.dashboard.auth: passkey enrolled for root"

    def test_other_control_characters_are_escaped_not_dropped(self) -> None:
        assert one_line("a\x1b[31mred\x00") == "a\\x1b[31mred\\x00"

    def test_a_plain_value_is_untouched(self) -> None:
        assert one_line("203.0.113.9") == "203.0.113.9"
        assert (
            one_line("agent: expected a mapping of agent keys, got str")
            == "agent: expected a mapping of agent keys, got str"
        )

    def test_an_endless_value_is_cut(self) -> None:
        assert len(one_line("h" * 10_000)) == 200
        assert one_line("h" * 10_000).endswith("…")

    def test_any_object_is_accepted(self) -> None:
        assert one_line(None) == "None"
        assert one_line(42) == "42"


class _Url:
    scheme = "http"


class _Req:
    def __init__(self, **headers: str) -> None:
        self.headers = headers
        self.client = None
        self.cookies: dict[str, str] = {}
        self.url = _Url()


class TestTheDashboardLogsThroughIt:
    def test_a_refused_origin_is_logged_on_one_line(self, caplog, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_DASH_AUTH_ORIGIN", raising=False)
        monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
        request = _Req(host="localhost:8090", origin="http://" + FORGED)
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.auth"), pytest.raises(HTTPException):
            auth._derive_origin(request)
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "\n" not in message and "\r" not in message
        assert "evil.example" in message

    def test_a_lenient_settings_patch_that_goes_nowhere_is_logged_on_one_line(
        self, caplog, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
        settings.clear_overrides()
        settings.load(refresh=True)
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.settings"):
            settings.update({"agent": "not a mapping\r\nforged"})
        assert len(caplog.records) == 1
        assert "\n" not in caplog.records[0].getMessage()
