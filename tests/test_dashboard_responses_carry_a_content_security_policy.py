"""Every dashboard response carries a Content-Security-Policy, and the shell it serves fits inside it.

The dashboard drives robot hardware from a browser and had no policy at all: no
header from ``create_app()``, no ``<meta http-equiv>`` in either HTML entry
point. Nothing constrained a script that reached the origin, and nothing made
reaching it harder (finding f015, the defence-in-depth half).

The policy is a response header, so it covers every document the server
answers, not only the one shell. It is as tight as the product allows: scripts,
styles, fonts and workers only from this origin; no plugins, no ``<base>``, no
framing; images also from ``data:`` and ``blob:`` (camera previews are object
URLs). ``connect-src`` stays open to http(s) and ws(s) on purpose: the
dashboard is a mesh peer that legitimately dials a robot on another host (the
Settings drawer, ``?backend=``), and the token those requests may carry is
bound to that host by the ``?backend=`` fix (f003); a tighter list is an
operator knob for a follow-up, named in the PR.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard.server import create_app  # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "static"

DOCUMENTS = ["/", "/static/app.js", "/api/health", "/api/auth/status", "/no/such/path"]


def _policy(response) -> dict[str, list[str]]:
    header = response.headers.get("content-security-policy")
    assert header, f"{response.url} answered without a Content-Security-Policy"
    out: dict[str, list[str]] = {}
    for directive in header.split(";"):
        parts = directive.split()
        if parts:
            out[parts[0]] = parts[1:]
    return out


class TestEveryAnswerCarriesThePolicy:
    @pytest.mark.parametrize("path", DOCUMENTS)
    def test_the_header_is_on_documents_assets_api_answers_and_refusals(self, path: str) -> None:
        client = TestClient(create_app())
        response = client.get(path)
        policy = _policy(response)
        assert policy["default-src"] == ["'self'"]
        assert policy["script-src"] == ["'self'"], "a script from anywhere but this origin may not run"
        assert policy["object-src"] == ["'none'"]
        assert policy["base-uri"] == ["'none'"]
        assert policy["frame-ancestors"] == ["'none'"], "the cockpit for a robot arm is not embeddable"
        assert set(policy["img-src"]) == {"'self'", "data:", "blob:"}
        assert set(policy["connect-src"]) >= {"'self'", "ws:", "wss:"}
        assert "'unsafe-eval'" not in " ".join(sum(policy.values(), []))
        assert response.headers.get("referrer-policy") == "no-referrer", "a token in a URL must not leak in Referer"


class TestTheShellFitsInsideThePolicy:
    def test_the_built_shell_has_no_inline_script_or_style(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), (
            "an inline script would be refused by script-src 'self'"
        )
        assert "<style" not in html, "an inline style block would be refused by style-src 'self'"
        assert not re.search(r"\bon[a-z]+=", html), "an inline handler would be refused"
        assert 'src="/static/' in html and 'href="/static/' in html, "the shell loads from this origin"

    def test_the_bundle_does_not_build_code_from_strings(self) -> None:
        """script-src without 'unsafe-eval' is only honest if the bundle never needs it."""
        for name in ("app.js", "twin.js"):
            code = (STATIC / name).read_text(encoding="utf-8")
            assert not re.search(r"\beval\(|new Function\(", code), f"static/{name} builds code from a string"
