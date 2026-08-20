"""A credential store that cannot be parsed must not quietly unseal the dashboard.

`auth_enabled()` IS `has_credentials()`, and `_load()` used to answer an unreadable store by
writing a fresh default one over it. So one truncated write — a crash mid-save, a full disk —
did two things nobody would see: it dropped auth on every /api and /ws route (through the
tunnel: the public internet), and it destroyed the only record of the operator's passkey, so
even repairing the JSON by hand could not bring it back.

These tests pin the recovery posture instead: keep the bytes, say so, and let the person AT the
machine re-enroll while a stranger who merely benefited from a disk error cannot.
"""
import json

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth


class FakeRequest:
    def __init__(self, headers=None, client_host="127.0.0.1"):
        self.headers = headers or {"host": "localhost:8090"}
        self.client = type("C", (), {"host": client_host})()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID",
              "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    auth._cache_key = None
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


def _corrupt_store(tmp_path, body: str = '{"credentials": [{"id": "AAA'):
    path = tmp_path / "auth.json"
    path.write_text(body)          # truncated JSON: exactly what a killed process leaves
    return path


def test_the_unreadable_bytes_are_kept_not_clobbered(tmp_path):
    path = _corrupt_store(tmp_path)
    original = path.read_text()

    auth._load()

    backups = list(tmp_path.glob("auth.json.corrupt-*"))
    assert len(backups) == 1, "the operator's only credential record must survive"
    assert backups[0].read_text() == original, "kept verbatim — it may hold the credential id"
    # And the working store is valid again, so the dashboard still comes up.
    assert json.loads(path.read_text())["credentials"] == []


def test_the_corruption_is_reported_not_swallowed(tmp_path):
    _corrupt_store(tmp_path)
    auth._load()

    damage = auth.store_corruption()
    assert damage, "a silent recovery is how this stayed invisible"
    assert "corrupt-" in damage["backup"]
    assert "Error" in damage["reason"] or "Decode" in damage["reason"]


def test_a_healthy_store_reports_no_damage(tmp_path):
    auth._load()
    assert auth.store_corruption() is None
    assert not list(tmp_path.glob("auth.json.corrupt-*"))


def test_a_stranger_cannot_seize_the_dashboard_through_a_disk_error(tmp_path):
    _corrupt_store(tmp_path)
    auth._load()

    with pytest.raises(HTTPException) as e:
        auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="attacker")
    assert e.value.status_code == 403
    # The refusal must say what happened and where the bytes went — an operator reading only
    # this message has to be able to recover.
    assert "unreadable" in e.value.detail and "corrupt-" in e.value.detail
    assert "BOOTSTRAP_TOKEN" in e.value.detail


def test_the_person_at_the_machine_can_still_recover(tmp_path):
    _corrupt_store(tmp_path)
    auth._load()

    opts = auth.begin_registration(FakeRequest(client_host="127.0.0.1"), label="recovery")
    assert opts.get("challenge_id"), "recovery must not be a dead end for the owner"


def test_bootstrap_token_still_works_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
    _corrupt_store(tmp_path)
    auth._load()

    with pytest.raises(HTTPException):
        auth.begin_registration(FakeRequest(client_host="203.0.113.9"), bootstrap="wrong")
    opts = auth.begin_registration(FakeRequest(client_host="203.0.113.9"), bootstrap="let-me-in")
    assert opts.get("challenge_id")


def test_a_genuinely_new_dashboard_is_unaffected(tmp_path):
    # No corruption: first enrollment from anywhere keeps working exactly as before, so this
    # change narrows an accident rather than adding a gate to the normal path.
    opts = auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="fresh")
    assert opts.get("challenge_id")
    assert auth.store_corruption() is None
