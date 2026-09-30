# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session token is only as good as the passkey it was minted for: remove the passkey, the session ends.

Finding f016 (CWE-613, CWE-672). ``verify_token`` was one ``jwt.decode``: the
signature against the store's ``jwt_secret`` and the ``exp`` claim, nothing
else. ``delete_credential`` removed a passkey from the store and rotated
nothing, so every token minted for that passkey kept working until it expired,
and could renew itself through ``/api/auth/renew`` and mint handoffs on the way.
The operator who removed a stolen phone's passkey had revoked nothing.

Now ``verify_token`` reads the token's ``sub`` (the credential id every issuer
stamps) against the credentials currently enrolled, from the very store it
already loads for the secret, and refuses a token whose passkey is gone with its
own sentence, so a revoked session reads differently from a forged or expired one
in the log. Renewal and handoff go through ``verify_token`` and inherit it. The
duration knobs are untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth


def _seal(tmp_path: Path, *cred_ids: str) -> None:
    """A store with these passkeys enrolled, written the way the module writes one."""
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "jwt_secret": "s" * 64,
                "created": 1789500000,
                "credentials": [
                    {"id": cid, "public_key": "AA", "sign_count": 0, "name": cid, "created": 1789500000}
                    for cid in cred_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    auth._load()


class TestARemovedPasskeyEndsItsSessions:
    def test_a_token_for_a_removed_passkey_is_refused(self, tmp_path: Path) -> None:
        _seal(tmp_path, "cred-phone", "cred-laptop")
        token = auth.issue_token("cred-phone", name="phone")
        assert auth.verify_token(token)["sub"] == "cred-phone"

        auth.delete_credential("cred-phone")

        with pytest.raises(HTTPException) as raised:
            auth.verify_token(token)
        assert raised.value.status_code == 401
        assert "revoked" in str(raised.value.detail)

    def test_the_other_passkeys_sessions_survive(self, tmp_path: Path) -> None:
        _seal(tmp_path, "cred-phone", "cred-laptop")
        keep = auth.issue_token("cred-laptop", name="laptop")
        auth.delete_credential("cred-phone")
        assert auth.verify_token(keep)["sub"] == "cred-laptop"

    def test_a_revoked_session_cannot_renew_itself(self, tmp_path: Path) -> None:
        _seal(tmp_path, "cred-phone", "cred-laptop")
        now = time.time()
        ttl = auth._token_ttl()
        stale = auth.issue_token("cred-phone", "phone", iat0=int(now - 0.6 * ttl), exp=int(now + 0.4 * ttl))
        assert auth.renew_if_due(stale, now=now) is not None, "past the half-life, so renewable while enrolled"
        auth.delete_credential("cred-phone")
        assert auth.renew_if_due(stale, now=now) is None

    def test_a_revoked_session_cannot_mint_a_handoff(self, tmp_path: Path) -> None:
        _seal(tmp_path, "cred-phone", "cred-laptop")
        token = auth.issue_token("cred-phone", name="phone")
        auth.delete_credential("cred-phone")
        with pytest.raises(HTTPException):
            auth.verify_token(token)  # the route resolves the session before it hands off; it never gets there
        assert auth.session_is_valid(token) is False

    def test_a_token_whose_subject_was_never_enrolled_is_refused(self, tmp_path: Path) -> None:
        """Forged by someone who has the secret but no passkey, or minted for a subject no passkey has."""
        _seal(tmp_path, "cred-laptop")
        token = auth.issue_token("nobody")
        with pytest.raises(HTTPException) as raised:
            auth.verify_token(token)
        assert raised.value.status_code == 401

    def test_a_dashboard_with_no_passkey_honours_no_session(self, tmp_path: Path) -> None:
        """Nothing enrolled means nothing to be signed in as; only the bootstrap proof or a static token admits."""
        auth._load()
        token = auth.issue_token("cred-phone")
        with pytest.raises(HTTPException):
            auth.verify_token(token)


class TestTheOtherRefusalsKeepTheirSentences:
    """An operator reading the log must still tell an expired session from a forged one."""

    def test_expired_is_still_expired(self, tmp_path: Path) -> None:
        _seal(tmp_path, "cred-laptop")
        token = auth.issue_token("cred-laptop", exp=int(time.time()) - 10)
        with pytest.raises(HTTPException) as raised:
            auth.verify_token(token)
        assert raised.value.detail == "session expired"

    def test_forged_is_still_invalid(self, tmp_path: Path) -> None:
        _seal(tmp_path, "cred-laptop")
        with pytest.raises(HTTPException) as raised:
            auth.verify_token("eyJ.not.real")
        assert raised.value.detail == "invalid session"
