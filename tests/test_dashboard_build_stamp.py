"""/api/health says which build is answering — and admits when it cannot tell.

The value of this stamp is entirely in its honesty: a UI is going to decide "your server predates
this feature, restart it" from the absence of the key and from the commit it reports, so a wrong
commit is worse than no commit. Every test below is therefore about a case where the truth is NOT
available and the module must say None instead of inventing a plausible string.
"""
from __future__ import annotations

from strands_robots.dashboard.build_info import build_info, read_commit, stamp

SHA = "a2d7da05f00d1234"


_N = [0]


def _repo(tmp_path, head: str, *, ref_at: str | None = None, ref_body: str | None = None):
    # A fresh directory per call: several cases in one test would otherwise share a .git and the
    # second one dies in the helper, which reads like a module failure (it cost me a run).
    _N[0] += 1
    tmp_path = tmp_path / f"repo{_N[0]}"
    git = tmp_path / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(head, encoding="utf-8")
    if ref_at is not None:
        target = git / ref_at
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(ref_body or "", encoding="utf-8")
    return tmp_path


def test_a_branch_checkout_reports_the_short_sha(tmp_path):
    root = _repo(tmp_path, "ref: refs/heads/dashboard\n", ref_at="refs/heads/dashboard", ref_body=SHA + "\n")
    assert read_commit(root) == SHA[:12]


def test_a_detached_head_reports_its_sha(tmp_path):
    assert read_commit(_repo(tmp_path, SHA + "\n")) == SHA[:12]


def test_a_packed_ref_is_admitted_as_unknown_not_guessed(tmp_path):
    """refs/heads/<branch> absent means the ref is packed. Chasing packed-refs is a parser this
    module deliberately does not have, so the answer is None - a stamp that reports the wrong
    commit would send an operator to restart a server that is already current."""
    root = _repo(tmp_path, "ref: refs/heads/dashboard\n")
    assert read_commit(root) is None


def test_no_git_directory_at_all_is_none(tmp_path):
    assert read_commit(tmp_path) is None
    assert read_commit(None) is None


def test_a_head_that_tries_to_escape_the_git_dir_reads_nothing(tmp_path):
    root = _repo(tmp_path, "ref: ../../../../etc/passwd\n")
    assert read_commit(root) is None


def test_garbage_and_truncation_are_not_shas(tmp_path):
    assert read_commit(_repo(tmp_path, "not a sha at all\n")) is None
    assert read_commit(_repo(tmp_path, "a2d7d\n")) is None, "too short to be a sha"
    assert read_commit(_repo(tmp_path, "zzzzzzzzzzzz\n")) is None, "not hex"


def test_the_payload_shape_is_the_three_fields_a_client_reads():
    s = stamp(commit=None, version=None, started=1.0)
    assert s == {"commit": None, "version": None, "started": 1.0}


def test_this_checkout_stamps_itself_and_is_cached():
    first = build_info()
    assert first is build_info(), "cached: the build cannot change without a restart"
    assert set(first) == {"commit", "version", "started"}
    # This test runs inside the repo, so the commit must actually resolve - if this ever fails the
    # module has stopped finding its own checkout, which is the one case the stamp cannot report.
    assert first["commit"] and len(first["commit"]) == 12, first


def test_health_carries_the_build_and_a_client_can_tell_an_old_server_apart(tmp_path, monkeypatch):
    """The contract the UI will lean on: `build` is ALWAYS in /api/health, so a response WITHOUT it
    is an older server rather than a healthy one with nothing to say."""
    from fastapi.testclient import TestClient

    from strands_robots.dashboard.server import create_app

    app = create_app()
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert "build" in body, "absence is the old-server signal, so presence must be unconditional"
    assert set(body["build"]) == {"commit", "version", "started"}
    assert body["build"]["started"] <= body["t"]
