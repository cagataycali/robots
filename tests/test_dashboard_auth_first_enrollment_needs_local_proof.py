# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ownership-granting first enrollment needs proof from the disk, never from the wire.

Finding F-007, follow-up (CWE-290 / CWE-348). The first fix refused a loopback
peer that carried a proxy header, and kept admitting a loopback peer that
carried none. A same-host Layer-4 forwarder - ``socat TCP-LISTEN:8443,fork
TCP:127.0.0.1:8090``, ``ssh -L``, nginx ``stream{}``, HAProxy ``mode tcp``, an
iptables DNAT rule, ``kubectl port-forward`` - relays raw bytes, adds no HTTP
header, and hands every remote client a ``127.0.0.1`` socket peer. Both
heuristics passed, and a stranger completed the first enrollment and owned a
dashboard that authorizes physical motion. No header roster can close that: the
property the guard needs is "this party can act as the service user on this
machine", and no request can assert it.

So the first enrollment is admitted on PROOF: the configured
``STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`` or, when none is set, a token the module
mints into a ``0600`` file beside the credential store. Reading that file is
the local act a remote peer cannot perform. These cells send exactly the
request the old rule admitted - loopback peer, no forwarding header, no token -
and grade the observable: no challenge is issued.
"""

from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from webauthn.helpers import bytes_to_base64url

from strands_robots.dashboard import auth

_STRANGER = "203.0.113.9"  # TEST-NET-3, never a real peer

#: Forwarding headers the proxy roster does not list. A proxy emitting only one
#: of these re-opened the L7 path under the header-presence rule; under the
#: proof rule they are irrelevant, and these cells pin that.
_UNLISTED_FORWARDING_HEADERS = [
    "x-client-ip",
    "true-client-ip",
    "x-cluster-client-ip",
    "via",
    "fastly-client-ip",
    "x-original-forwarded-for",
]


class FakeRequest:
    def __init__(
        self,
        headers: dict[str, str] | None = None,
        client_host: str | None = "127.0.0.1",
        scheme: str = "http",
    ) -> None:
        self.headers = {"host": "localhost:8090", **(headers or {})}
        self.url = SimpleNamespace(scheme=scheme)
        self.client = None if client_host is None else type("C", (), {"host": client_host})()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in (
        "STRANDS_DASH_AUTH_ENABLED",
        "STRANDS_DASH_AUTH_RP_ID",
        "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN",
        "STRANDS_DASH_AUTH_ENROLL_TOKEN_FILE",
    ):
        monkeypatch.delenv(k, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


def _token_file(tmp_path: Path) -> Path:
    return tmp_path / "enroll_token"


class TestTheL4ForwarderCaseIsRefused:
    """Loopback peer, no forwarding header, no token: the request the old rule let in."""

    @pytest.mark.parametrize("peer", ["127.0.0.1", "127.0.0.53", "::1", "localhost"])
    def test_a_bare_loopback_peer_does_not_enroll(self, peer: str) -> None:
        request = FakeRequest(client_host=peer)
        assert auth._arrived_through_a_proxy(request) is None, "this must be the header-free case"
        before = len(auth._challenges)
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(request, label="stranger-through-socat")
        assert e.value.status_code == 403
        assert len(auth._challenges) == before, "a refused enrollment must not issue a challenge"
        assert not auth._load().get("user_id"), "nor mint a user id"

    def test_the_refusal_names_the_forwarder_class_and_where_the_token_is(self, tmp_path: Path) -> None:
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger")
        detail = e.value.detail
        assert "127.0.0.1" in detail and "forward" in detail
        assert str(_token_file(tmp_path)) in detail
        assert "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN" in detail

    def test_the_refusal_never_carries_the_token(self, tmp_path: Path) -> None:
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger")
        token = _token_file(tmp_path).read_text().strip()
        assert token and token not in e.value.detail

    @pytest.mark.parametrize("header", _UNLISTED_FORWARDING_HEADERS)
    def test_a_forwarding_header_the_roster_omits_changes_nothing(self, header: str) -> None:
        """The proxy roster is diagnosis, not defence: an unlisted header is refused all the same."""
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest({header: _STRANGER}), label="stranger")
        assert e.value.status_code == 403

    def test_a_wrong_token_is_refused_in_constant_time_shape(self, tmp_path: Path) -> None:
        auth._local_enroll_token()
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger", bootstrap="guess")
        assert e.value.status_code == 403

    def test_a_non_ascii_guess_is_a_mismatch_not_a_server_error(self) -> None:
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger", bootstrap="pässwörd")
        assert e.value.status_code == 403

    def test_a_connection_with_no_peer_is_refused(self) -> None:
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(client_host=None), label="unknown")
        assert e.value.status_code == 403


class TestTheLocalTokenIsTheProof:
    def test_the_token_admits_the_first_enrollment(self, tmp_path: Path) -> None:
        opts = auth.begin_registration(FakeRequest(), label="owner", bootstrap=auth._local_enroll_token())
        assert opts.get("challenge_id")

    def test_the_token_is_read_off_disk_not_off_memory(self, tmp_path: Path) -> None:
        """What the operator ``cat``s is what the gate checks - the file is the contract."""
        auth._local_enroll_token()
        on_disk = _token_file(tmp_path).read_text().strip()
        opts = auth.begin_registration(FakeRequest(), label="owner", bootstrap=on_disk)
        assert opts.get("challenge_id")

    def test_whoever_holds_the_token_is_the_operator_wherever_they_connect_from(self) -> None:
        """A headless robot has no browser: reading the file over ssh and pasting it is the remote path."""
        opts = auth.begin_registration(
            FakeRequest(client_host=_STRANGER), label="remote-owner", bootstrap=auth._local_enroll_token()
        )
        assert opts.get("challenge_id")

    def test_the_file_is_created_owner_only_beside_the_store(self, tmp_path: Path) -> None:
        with pytest.raises(HTTPException):
            auth.begin_registration(FakeRequest(), label="stranger")
        path = _token_file(tmp_path)
        assert path.is_file()
        assert path.parent == Path(auth._store_path()).parent
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert len(path.read_text().strip()) >= 32

    def test_the_token_is_stable_across_reads_until_spent(self, tmp_path: Path) -> None:
        """An operator who read it a minute ago must still be able to use it."""
        assert auth._local_enroll_token() == auth._local_enroll_token()

    def test_a_file_readable_by_others_is_replaced_not_honoured(self, tmp_path: Path) -> None:
        first = auth._local_enroll_token()
        _token_file(tmp_path).chmod(0o644)
        second = auth._local_enroll_token()
        assert second != first
        assert stat.S_IMODE(_token_file(tmp_path).stat().st_mode) == 0o600
        with pytest.raises(HTTPException):
            auth.begin_registration(FakeRequest(), label="stranger", bootstrap=first)

    def test_an_empty_or_missing_file_is_reminted(self, tmp_path: Path) -> None:
        auth._local_enroll_token()
        _token_file(tmp_path).write_text("")
        assert auth._local_enroll_token()
        _token_file(tmp_path).unlink()
        assert auth._local_enroll_token()

    def test_the_location_can_be_relocated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        elsewhere = tmp_path / "run" / "dash.token"
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENROLL_TOKEN_FILE", str(elsewhere))
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger")
        assert elsewhere.is_file() and str(elsewhere) in e.value.detail
        assert not _token_file(tmp_path).exists()

    def test_the_token_is_retired_once_a_passkey_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With an owner enrolled the file guards nothing and would only be a secret left lying around."""
        request = FakeRequest()
        begun = auth.begin_registration(request, label="owner", bootstrap=auth._local_enroll_token())
        assert _token_file(tmp_path).is_file()
        monkeypatch.setattr(
            auth,
            "verify_registration_response",
            lambda **kw: SimpleNamespace(credential_id=b"\x01" * 16, credential_public_key=b"\x02" * 32, sign_count=0),
        )
        out = auth.finish_registration(request, begun["challenge_id"], {"id": bytes_to_base64url(b"\x01" * 16)})
        assert out["ok"] is True
        assert not _token_file(tmp_path).exists()

    def test_a_later_enrollment_ignores_the_bootstrap_value(self, tmp_path: Path) -> None:
        """The gate is on the one-way door only; the route guards later enrollments with a session."""
        auth._save({"credentials": [{"id": "AAAA", "name": "existing"}]})
        auth._cache = {}
        opts = auth.begin_registration(FakeRequest({"x-forwarded-for": _STRANGER}, client_host=_STRANGER))
        assert opts.get("challenge_id")
        assert not _token_file(tmp_path).exists(), "no token is minted when there is nothing to guard"


class TestTheConfiguredTokenTakesPrecedence:
    def test_no_file_is_minted_when_the_operator_set_a_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger")
        assert "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN" in e.value.detail
        assert not _token_file(tmp_path).exists()
        assert auth._first_enrollment_proof() == ("env", "let-me-in")

    def test_the_configured_token_is_required_even_on_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
        with pytest.raises(HTTPException):
            auth.begin_registration(FakeRequest(client_host="127.0.0.1"), label="owner")
        assert auth.begin_registration(FakeRequest(client_host=_STRANGER), bootstrap="let-me-in")["challenge_id"]


class TestTheOrderingOfRefusals:
    def test_an_unusable_rp_id_is_still_reported_first(self) -> None:
        """A bare-IP Host cannot hold a passkey from anywhere; that diagnosis beats the ownership one."""
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest({"host": "192.168.1.166:8090"}), label="lan")
        assert e.value.status_code == 400

    def test_a_damaged_store_is_named_in_the_refusal(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text('{"credentials": [{"id": "AAA')
        auth._cache = {}
        auth._load()
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="stranger")
        assert "unreadable" in e.value.detail and "corrupt-" in e.value.detail


class TestStatusTellsTheLoginScreenWhatToAskFor:
    def test_setup_needs_a_bootstrap_and_says_which_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = auth.status()
        assert out["setup_required"] is True
        assert out["bootstrap_required"] is True
        assert out["bootstrap_source"] == "file"
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
        assert auth.status()["bootstrap_source"] == "env"

    def test_status_never_carries_the_token_or_its_path(self, tmp_path: Path) -> None:
        auth._local_enroll_token()
        blob = repr(auth.status())
        assert _token_file(tmp_path).read_text().strip() not in blob
        assert str(tmp_path) not in blob

    def test_nothing_is_required_once_enrolled(self) -> None:
        auth._save({"credentials": [{"id": "AAAA", "name": "existing"}]})
        auth._cache = {}
        out = auth.status()
        assert out["bootstrap_required"] is False and out["bootstrap_source"] is None
