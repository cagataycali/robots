# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ownership-granting first enrollment refuses a loopback peer that came through a proxy.

Finding F-007 (CWE-290 / CWE-348). ``begin_registration`` admits the first
passkey - the one that seals the dashboard and owns the fleet - from the
machine itself when no bootstrap token is configured, and decides "the machine"
from the socket peer. That is the right reader for a direct bind. Behind the
same-host tunnel the docs describe (``cloudflared`` -> ``http://localhost:8090``)
without ``--proxy-headers``, however, EVERY remote visitor's socket peer is
``127.0.0.1``, so during the pre-enrollment window a stranger who reached the
tunnel first could enroll the owner passkey.

A proxy adds its forwarding headers to everything it relays and a browser on
the machine sends none of them, so the presence of any such header is the fact
that tells the two apart. The values stay untrusted: a header's presence is
evidence of a hop, never an address. Loopback with no header is still let in;
the bootstrap token is still the way in from anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth

PROXY_HEADERS = [
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-forwarded-host",
    "x-real-ip",
    "cf-connecting-ip",
    "cf-ray",
    "forwarded",
]


class FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None, client_host: str = "127.0.0.1") -> None:
        self.headers = {"host": "localhost:8090", **(headers or {})}
        self.client = type("C", (), {"host": client_host})()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


def test_the_roster_is_the_one_the_guard_reads() -> None:
    assert tuple(PROXY_HEADERS) == auth._PROXY_EVIDENCE_HEADERS


class TestAProxiedLoopbackPeerIsNotTheMachine:
    @pytest.mark.parametrize("header", PROXY_HEADERS)
    def test_each_forwarding_header_refuses_the_first_enrollment(self, header: str) -> None:
        request = FakeRequest({header: "anything"}, client_host="127.0.0.1")
        with pytest.raises(HTTPException) as raised:
            auth.begin_registration(request, label="stranger-through-the-tunnel")
        assert raised.value.status_code == 403
        assert header in raised.value.detail
        assert "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN" in raised.value.detail
        assert "--proxy-headers" in raised.value.detail

    def test_the_header_value_is_not_read_so_a_loopback_claim_is_no_better(self) -> None:
        """A proxied stranger who spells the forwarded address as loopback is still proxied."""
        request = FakeRequest({"x-forwarded-for": "127.0.0.1"}, client_host="127.0.0.1")
        with pytest.raises(HTTPException) as raised:
            auth.begin_registration(request, label="stranger")
        assert raised.value.status_code == 403

    def test_header_names_are_matched_case_insensitively(self) -> None:
        request = FakeRequest({"X-Forwarded-For": "203.0.113.9"}, client_host="127.0.0.1")
        with pytest.raises(HTTPException) as raised:
            auth.begin_registration(request, label="stranger")
        assert raised.value.status_code == 403

    @pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
    def test_an_ipv6_or_ipv4_loopback_peer_is_refused_alike(self, peer: str) -> None:
        request = FakeRequest({"cf-ray": "8a1b2c3d4e5f-IAD"}, client_host=peer)
        with pytest.raises(HTTPException):
            auth.begin_registration(request, label="stranger")

    def test_no_user_id_is_minted_and_no_challenge_stashed_on_refusal(self, tmp_path: Path) -> None:
        """The refusal happens before ``user_id`` is minted or a challenge stashed."""
        request = FakeRequest({"x-forwarded-proto": "https"}, client_host="127.0.0.1")
        before = len(auth._challenges)
        with pytest.raises(HTTPException):
            auth.begin_registration(request, label="stranger")
        assert not auth._load().get("user_id")
        assert len(auth._challenges) == before


class TestTheMachineItselfIsStillLetIn:
    @pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
    def test_loopback_with_no_forwarding_header_enrolls(self, peer: str) -> None:
        opts = auth.begin_registration(FakeRequest(client_host=peer), label="owner")
        assert opts.get("challenge_id")

    def test_ordinary_browser_headers_are_not_proxy_evidence(self) -> None:
        request = FakeRequest(
            {"user-agent": "Mozilla/5.0", "accept": "*/*", "origin": "http://localhost:8090", "cookie": "a=b"},
            client_host="127.0.0.1",
        )
        opts = auth.begin_registration(request, label="owner")
        assert opts.get("challenge_id")


class TestTheBootstrapTokenIsStillTheWayInFromAnywhere:
    def test_a_proxied_request_with_the_token_enrolls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "correct-horse")
        request = FakeRequest({"x-forwarded-for": "203.0.113.9", "cf-ray": "x"}, client_host="127.0.0.1")
        opts = auth.begin_registration(request, label="owner-remote", bootstrap="correct-horse")
        assert opts.get("challenge_id")

    def test_a_proxied_request_with_the_wrong_token_is_refused_for_the_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "correct-horse")
        request = FakeRequest({"x-forwarded-for": "203.0.113.9"}, client_host="127.0.0.1")
        with pytest.raises(HTTPException) as raised:
            auth.begin_registration(request, label="stranger", bootstrap="wrong")
        assert raised.value.status_code == 403
        assert "bootstrap token" in raised.value.detail


class TestTheGuardOnlyBitesTheFirstEnrollment:
    def test_a_later_enrollment_is_not_refused_for_a_forwarding_header(self, tmp_path: Path) -> None:
        """Once an owner exists the route enforces a session; the proxy check is not that gate."""
        store = {"user_id": "AAAA", "credentials": [{"id": "AQID", "public_key": "x", "sign_count": 0, "label": "k"}]}
        auth._save(store)
        request = FakeRequest({"x-forwarded-for": "203.0.113.9"}, client_host="127.0.0.1")
        opts = auth.begin_registration(request, label="second-device")
        assert opts.get("challenge_id")


def test_arrived_through_a_proxy_is_a_pure_function_of_the_request() -> None:
    assert auth._arrived_through_a_proxy(FakeRequest()) is None
    assert auth._arrived_through_a_proxy(FakeRequest({"Forwarded": "for=1.2.3.4"})) == "forwarded"
    assert auth._arrived_through_a_proxy(object()) is None
