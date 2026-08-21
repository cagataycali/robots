"""A credential store that cannot be parsed must not quietly unseal the dashboard.

`auth_enabled()` IS `has_credentials()`, and `_load()` used to answer an unreadable store by
writing a fresh default one over it. So one truncated write -- a crash mid-save, a full disk --
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
    assert backups[0].read_text() == original, "kept verbatim -- it may hold the credential id"
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
    # The refusal must say what happened and where the bytes went -- an operator reading only
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


class TestTheLastNineLines:
    """The defensive branches of this module, which had no coverage at all (2026-08-21).

    Each of these is a filesystem or request failing in a way that is rare and consequential: the
    module's whole promise is that auth cannot be dropped by an accident, and an untested `except`
    is exactly where that promise is usually broken.
    """

    def test_a_quarantine_that_CANNOT_move_the_file_still_records_the_damage(
        self, tmp_path, monkeypatch
    ):
        """The flag, not the backup, is what re-seals enrollment.

        On a read-only or full volume the rename fails - and if the code gave up there, the
        credential-less window would open with nobody recorded as responsible: a stranger through the
        tunnel could seize a dashboard whose owner still has a passkey on disk. So the move is
        best-effort and the FLAG is mandatory.
        """
        _corrupt_store(tmp_path)
        monkeypatch.setattr(
            auth.os, "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError(30, "Read-only file system")),
        )
        auth._cache_key = None
        auth._load()
        damage = auth.store_corruption()
        assert damage, "an unmovable corrupt store must still be REPORTED as corrupt"
        assert damage["backup"] == "", "and honest that no backup file exists"
        assert "JSONDecodeError" in damage["reason"] or "Expecting" in damage["reason"]
        # ... and the narrowing it exists for still applies, naming the missing backup gracefully.
        with pytest.raises(HTTPException) as err:
            auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="stranger")
        assert err.value.status_code == 403
        assert "a backup" in str(err.value.detail), (
            "the refusal must read sensibly when there is no backup path to name"
        )

    def test_a_store_that_cannot_be_chmodded_is_still_saved(self, tmp_path, monkeypatch):
        """0600 is a wish, not a precondition - some filesystems have no such concept.

        Refusing to save would mean a dashboard that cannot enroll a passkey at all on such a
        volume, which is a worse outcome than a file with the volume's own permissions.
        """
        monkeypatch.setattr(
            auth.os, "chmod",
            lambda *a, **k: (_ for _ in ()).throw(OSError(45, "Operation not supported")),
        )
        auth._save({"credentials": [], "note": "kept"})
        assert json.loads((tmp_path / "auth.json").read_text())["note"] == "kept"

    def test_a_store_that_vanishes_between_write_and_stat_invalidates_the_cache(
        self, tmp_path, monkeypatch
    ):
        """The cache key is (path, mtime, size). If stat fails there is no key to trust.

        Setting it to None means the next _load re-reads from disk - the safe direction. Keeping a
        stale key would serve a remembered store for a file somebody else has replaced.
        """
        real_stat = auth.Path.stat

        def flaky_stat(self, *a, **k):
            if self.name == "auth.json":
                raise OSError(2, "No such file or directory")
            return real_stat(self, *a, **k)

        monkeypatch.setattr(auth.Path, "stat", flaky_stat)
        auth._save({"credentials": []})
        assert auth._cache_key is None
        assert auth._cache == {"credentials": []}, "the write itself still happened"

    def test_a_request_whose_headers_explode_still_returns_a_status(self, tmp_path):
        """status() feeds the login screen. A diagnostic that raises takes the screen with it."""

        class Hostile:
            @property
            def headers(self):
                raise RuntimeError("no headers on this transport")

        out = auth.status(Hostile())
        assert out["setup_required"] is True and out["enabled"] is False
        assert "rp_id" not in out, "an undiscoverable rp_id is absent, never guessed"
