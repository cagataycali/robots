"""A GitHub robot whose meshes live in Git LFS downloads real meshes without git-lfs.

``git clone`` on a machine without git-lfs writes each LFS-tracked file as a text
pointer and exits 0. ``reachy_mini``'s upstream keeps every mesh in LFS, so its
download "succeeded" with 49 pointers MuJoCo could not decode, and the only
remedy was installing git-lfs and fetching again. The fetcher now resolves each
pointer from GitHub's LFS media endpoint at the cloned commit and keeps the
object only when its SHA-256 and size match the pointer.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import urllib.error
from pathlib import Path

import pytest

from strands_robots.assets import download

_MESH = b"\x00" * 80 + (1).to_bytes(4, "little") + b"\x00" * 50


def _pointer(data: bytes) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{hashlib.sha256(data).hexdigest()}\nsize {len(data)}\n".encode()
    )


@pytest.fixture
def clone(tmp_path) -> Path:
    repo = tmp_path / "repo"
    (repo / "desc" / "assets").mkdir(parents=True)
    (repo / "desc" / "assets" / "part one.stl").write_bytes(_pointer(_MESH))
    (repo / "desc" / "robot.xml").write_text("<mujoco/>")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True
    )
    return repo


def _serve(monkeypatch, body: bytes | None, seen: list[str]) -> None:
    def fake_urlopen(url, timeout=None):
        seen.append(url)
        if body is None:
            raise urllib.error.URLError("offline")
        return io.BytesIO(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_a_pointer_is_replaced_by_the_object_it_names(clone, monkeypatch) -> None:
    seen: list[str] = []
    _serve(monkeypatch, _MESH, seen)

    assert download._fetch_lfs_objects(clone, "owner/repo", clone / "desc") is None

    assert (clone / "desc" / "assets" / "part one.stl").read_bytes() == _MESH
    sha = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert seen == [f"https://media.githubusercontent.com/media/owner/repo/{sha}/desc/assets/part%20one.stl"]


def test_an_object_that_does_not_match_its_pointer_is_refused(clone, monkeypatch) -> None:
    _serve(monkeypatch, b"<html>not the mesh</html>", [])

    error = download._fetch_lfs_objects(clone, "owner/repo", clone / "desc")

    assert error is not None and "does not match its pointer" in error
    assert download._is_lfs_pointer(clone / "desc" / "assets" / "part one.stl"), "a bad object must not be kept"


def test_an_unreachable_endpoint_is_a_failure_not_a_silent_pointer(clone, monkeypatch) -> None:
    _serve(monkeypatch, None, [])
    error = download._fetch_lfs_objects(clone, "owner/repo", clone / "desc")
    assert error is not None and "could not be fetched" in error


def test_a_tree_without_pointers_fetches_nothing(tmp_path, monkeypatch) -> None:
    (tmp_path / "robot.xml").write_text("<mujoco/>")
    seen: list[str] = []
    _serve(monkeypatch, b"", seen)
    assert download._fetch_lfs_objects(tmp_path, "owner/repo", tmp_path) is None
    assert seen == []


def test_an_oversized_lfs_store_is_refused_before_any_fetch(clone, monkeypatch) -> None:
    seen: list[str] = []
    _serve(monkeypatch, _MESH, seen)
    monkeypatch.setattr(download, "_LFS_MAX_TOTAL_BYTES", 10)
    error = download._fetch_lfs_objects(clone, "owner/repo", clone / "desc")
    assert error is not None and "over the 10-byte limit" in error
    assert seen == []
