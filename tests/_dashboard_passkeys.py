# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enrol passkey records the way the module writes them, for tests that mint sessions.

Since f016 ``auth.verify_token`` admits a token only while its ``sub`` names an
enrolled credential, so a cell that mints a session for ``cred1`` must have a
``cred1`` in the store first. The record carries the fields ``finish_registration``
writes; the public key is a placeholder no ceremony will ever verify against.
"""

from __future__ import annotations

import time
from unittest import mock

from strands_robots.dashboard import auth


def enroll(*cred_ids: str, name: str | None = None) -> None:
    """Add these credential ids to the current store (idempotent by id)."""
    store = auth._load()
    present = {c.get("id") for c in store.get("credentials", [])}
    for cid in cred_ids:
        if cid in present:
            continue
        store.setdefault("credentials", []).append(
            {"id": cid, "public_key": "AA", "sign_count": 0, "name": name or cid, "created": time.time()}
        )
    auth._save(store)


def issue_enrolled(subject: str, *args, **kwargs) -> str:
    """``auth.issue_token`` for a subject that is enrolled first."""
    enroll(subject)
    return auth.issue_token(subject, *args, **kwargs)


def handed_off(session: dict, now: float | None = None) -> dict:
    """What ``auth.issue_handoff`` answers for *session*, plus the ``token`` its code redeems for.

    The passkey assertion a real handoff needs is taken as verified (no
    authenticator in a test); everything after it is the module's own path.
    """
    with mock.patch.object(auth, "_verified_assertion", lambda *a, **k: {"id": session.get("sub")}):
        minted = auth.issue_handoff(None, session, "challenge", {}, now=now)
    return {**minted, "token": auth.redeem_handoff(minted["code"], now=now)}
