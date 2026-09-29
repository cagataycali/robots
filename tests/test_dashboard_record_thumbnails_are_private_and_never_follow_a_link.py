# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode thumbnails live in a private directory and the thumb route never follows a link.

Finding f006 (CWE-59, CWE-377, CWE-552). ``RecordController`` fixed its thumbnail
root to ``<tmpdir>/strands-record-thumbs``: a guessable name in the shared temp
directory, created lazily by the first recorded frame with ``exist_ok=True``, so
any other account on the host could create it first and fill it with symlinks.
``GET /api/record/thumb/{episode}/{camera}`` composed ``<root>/<episode>_<camera>.jpg``
from the URL's own segments (so the attacker chose the file name too), tested it
with ``Path.is_file`` and served it with ``FileResponse``, both of which follow
symlinks. The bytes of ``~/.strands_dashboard/auth.json`` (the session-signing
secret) or ``enroll_token`` came back labelled ``image/jpeg``.

Now the default root is a per-process ``mkdtemp`` directory (``0700``, unguessable);
an explicit root is created ``0700`` and refused unless it is a real directory the
service user owns that nobody else can write; the read route opens the file with
``O_NOFOLLOW``, checks it is a regular file inside the root, and serves the bytes
it read from that descriptor; and the writer refuses to write through a link.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import record_api, record_worker  # noqa: E402
from strands_robots.dashboard.record_api import RecordController  # noqa: E402

SECRET = '{"jwt_secret": "the-signing-secret-that-must-stay-on-disk"}'

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="symlinks and mode bits are POSIX facts")


@pytest.fixture()
def secret_file(tmp_path: Path) -> Path:
    """Stands in for ``~/.strands_dashboard/auth.json``: 0600, owned by the service user."""
    target = tmp_path / "state" / "auth.json"
    target.parent.mkdir()
    target.write_text(SECRET, encoding="utf-8")
    target.chmod(0o600)
    return target


def _client(ctl: RecordController, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The record router behind the real session dependency, called as the operator at the machine."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "test-fixture-bootstrap-proof")
    app = FastAPI()
    app.include_router(record_api.build_router(ctl))
    return TestClient(app, headers={"authorization": "Bearer test-fixture-bootstrap-proof"})


class TestTheDefaultRootIsPrivateAndUnguessable:
    def test_the_default_root_is_not_the_fixed_shared_name(self) -> None:
        import tempfile

        ctl = RecordController(devices=object())
        root = ctl.thumb_dir
        assert root != Path(tempfile.gettempdir()) / "strands-record-thumbs"
        assert root.is_dir() and not root.is_symlink()
        mode = stat.S_IMODE(root.stat().st_mode)
        assert mode == 0o700, oct(mode)
        assert root.stat().st_uid == os.geteuid()

    def test_two_controllers_do_not_share_a_root(self) -> None:
        assert RecordController(devices=object()).thumb_dir != RecordController(devices=object()).thumb_dir

    def test_an_explicit_root_is_created_private(self, tmp_path: Path) -> None:
        root = tmp_path / "thumbs"
        ctl = RecordController(devices=object(), thumb_root=str(root))
        assert ctl.thumb_dir == root
        assert stat.S_IMODE(root.stat().st_mode) == 0o700

    def test_an_explicit_root_that_is_a_link_is_refused(self, tmp_path: Path) -> None:
        """The attacker's move: the path exists already and points somewhere of their choosing."""
        elsewhere = tmp_path / "theirs"
        elsewhere.mkdir()
        (tmp_path / "thumbs").symlink_to(elsewhere)
        with pytest.raises(PermissionError, match="thumbnail"):
            RecordController(devices=object(), thumb_root=str(tmp_path / "thumbs")).thumb_dir  # noqa: B018

    def test_an_explicit_root_others_can_write_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "thumbs"
        root.mkdir(mode=0o777)
        root.chmod(0o777)
        with pytest.raises(PermissionError, match="thumbnail"):
            RecordController(devices=object(), thumb_root=str(root)).thumb_dir  # noqa: B018


class TestTheReadRouteNeverFollowsALink:
    def test_a_planted_link_is_404_and_leaks_nothing(
        self, tmp_path: Path, secret_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The attacker's order of events: the directory exists before the dashboard ever
        # writes a frame, and the link inside it is named after the URL they will request.
        root = tmp_path / "thumbs"
        root.mkdir()
        (root / "0_x.jpg").symlink_to(secret_file)
        ctl = RecordController(devices=object(), thumb_root=str(root))
        response = _client(ctl, monkeypatch).get("/api/record/thumb/0/x")
        assert response.status_code == 404
        assert "jwt_secret" not in response.text

    def test_a_link_to_a_directory_is_404(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ctl = RecordController(devices=object(), thumb_root=str(tmp_path / "thumbs"))
        (ctl.thumb_dir / "0_x.jpg").symlink_to(tmp_path)
        assert _client(ctl, monkeypatch).get("/api/record/thumb/0/x").status_code == 404

    def test_a_real_thumbnail_is_still_served(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ctl = RecordController(devices=object(), thumb_root=str(tmp_path / "thumbs"))
        (ctl.thumb_dir / "3_front.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg-bytes")
        response = _client(ctl, monkeypatch).get("/api/record/thumb/3/front")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"\xff\xd8\xff\xe0jpeg-bytes"

    def test_a_missing_thumbnail_is_still_404(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ctl = RecordController(devices=object(), thumb_root=str(tmp_path / "thumbs"))
        assert _client(ctl, monkeypatch).get("/api/record/thumb/9/none").status_code == 404


class TestTheWriterRefusesToWriteThroughALink:
    def test_a_link_at_the_thumbnail_name_is_not_written_through(self, tmp_path: Path, secret_file: Path) -> None:
        np = pytest.importorskip("numpy")
        before = secret_file.read_text(encoding="utf-8")
        link = tmp_path / "0_front.jpg"
        link.symlink_to(secret_file)
        frame = np.zeros((8, 8, 3), dtype="uint8")
        assert record_worker._save_thumbnail(frame, link) is False
        assert secret_file.read_text(encoding="utf-8") == before

    def test_a_plain_path_is_still_written(self, tmp_path: Path) -> None:
        np = pytest.importorskip("numpy")
        if not any(__import__("importlib").util.find_spec(m) for m in ("cv2", "PIL")):
            pytest.skip("no image writer installed")
        out = tmp_path / "0_front.jpg"
        assert record_worker._save_thumbnail(np.zeros((8, 8, 3), dtype="uint8"), out) is True
        assert out.is_file()


class TestTheMintedRootIsRemovedAtShutdown:
    def test_close_removes_a_minted_root_and_leaves_a_configured_one(self, tmp_path: Path) -> None:
        minted = RecordController(devices=object())
        root = minted.thumb_dir
        assert root.is_dir()
        minted.close_thumbs()
        assert not root.exists()

        configured = RecordController(devices=object(), thumb_root=str(tmp_path / "thumbs"))
        kept = configured.thumb_dir
        configured.close_thumbs()
        assert kept.is_dir()

    def test_the_app_removes_it_when_its_lifespan_ends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.dashboard.server import create_app

        app = create_app()
        with TestClient(app):
            root = app.state.record.thumb_dir
            assert root.is_dir()
        assert not root.exists()
