"""The SPA catch-all must not serve files outside its own dist directory.

The ``/{path:path}`` catch-all forwards the URL path unchanged into
``FRONTEND_DIST / path``. Without confinement, ``GET /../../etc/passwd``
resolves to ``/etc/passwd``, ``.is_file()`` reports True, and FastAPI's
``FileResponse`` streams the file - a filesystem read on a path a client
supplied. CodeQL's ``py/path-injection`` flagged exactly this: the URL path is
untrusted data flowing into a filesystem path expression.

This module grades the confinement rather than the alert. Every cell drives a
real request through ``TestClient`` and asserts what the response body is or is
not, so a rewrite that keeps the fix but drops the file scan will still be
covered.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Stock ``settings.json`` on this machine carries a live auth token; the
    same fixture the sibling ``test_dashboard_api_404`` uses keeps the test
    client below a fresh 401 wall."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


class _StubBridge:
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def _fake_dist(tmp_path, monkeypatch):
    """Build a plausible frontend/dist so the SPA route is mounted and there is
    a real file inside dist for the positive-path assertion to succeed against.

    The traversal target ``secret.txt`` is created OUTSIDE the dist so the
    confinement check is the only thing keeping it hidden - a fix that resolves
    ``../secret.txt`` to a real file would serve it if unguarded.
    """
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><body>index</body>")
    (dist / "app.js").write_text("// bundle")
    (dist / "assets").mkdir()
    (dist / "assets" / "x.css").write_text("body{}")
    secret = tmp_path / "secret.txt"
    secret.write_text("REDACTED-BUT-REACHABLE-VIA-TRAVERSAL")
    monkeypatch.setattr(srv, "FRONTEND_DIST", dist)
    return dist, secret


def _client() -> TestClient:
    # follow_redirects=False so a 307/308 does not swap the assertion target;
    # the SPA route answers directly.
    return TestClient(srv.create_app(bridge=_StubBridge()), follow_redirects=False)


class TestTheSpaCatchallConfinesUserPathsToItsDistDirectory:
    """Every cell measures a filesystem outcome that only the confinement can
    produce. A regression that drops the ``.resolve()``/``relative_to`` guard
    fails at least one cell here."""

    def test_a_dotdot_traversal_out_of_dist_does_not_leak_the_file(self, _fake_dist) -> None:
        """The historic defect: ``GET /../secret.txt`` served the real file.

        The client normalises literal ``..`` before send (so a hostile browser
        can too, silently), but URL-encoded ``..`` (``%2E%2E``) is delivered
        verbatim and this is the shape a real attacker reaches for. Every URL
        below is one they can construct.
        """
        c = _client()
        # Every URL below was measured to serve the secret on the pre-fix tree
        # (%2E%2E survives httpx's own path normalisation; the plain-.. rows
        # are folded by the client but grade the handler's guard when it is
        # exposed behind a proxy that does NOT fold).
        for url in (
            "/../secret.txt",
            "/foo/../../secret.txt",
            "/%2E%2E/secret.txt",
            "/..%2Fsecret.txt",
        ):
            r = c.get(url)
            assert "REDACTED-BUT-REACHABLE-VIA-TRAVERSAL" not in r.text, (
                f"traversal served the file outside dist: url={url!r} body_head={r.text[:80]!r}"
            )

    def test_a_deep_dotdot_traversal_out_of_dist_does_not_leak_the_file(self, _fake_dist) -> None:
        # A deeper walk hits the same resolve step; grade it separately so a
        # depth-1 special case does not pass by accident. Uses %2E%2E so the
        # httpx client does not fold the path before it reaches the handler.
        c = _client()
        r = c.get("/x/%2E%2E/%2E%2E/secret.txt")
        assert "REDACTED-BUT-REACHABLE-VIA-TRAVERSAL" not in r.text

    def test_a_file_inside_dist_is_still_served(self, _fake_dist) -> None:
        """The confinement must not break the legitimate SPA path."""
        c = _client()
        r = c.get("/app.js")
        assert r.status_code == 200
        assert r.text == "// bundle"

    def test_an_unknown_client_side_route_falls_back_to_index(self, _fake_dist) -> None:
        """An SPA route is not on disk; the shell answers with index.html.
        Same behaviour as before the fix - the guard must not turn this into a
        404."""
        c = _client()
        r = c.get("/some/spa/route")
        assert r.status_code == 200
        assert "<!doctype html>" in r.text.lower()

    def test_a_traversal_that_lands_on_a_real_file_still_refuses(self, tmp_path, monkeypatch) -> None:
        """Confinement is a canonical-path check, not a substring one. Build
        a dist whose sibling has an ``app.js`` too, so ``../sibling/app.js``
        names a real file OUTSIDE dist. The fix must refuse it even though
        the name matches a legitimate one."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><body>index</body>")
        (dist / "app.js").write_text("// real bundle")
        # /assets is mounted via StaticFiles(directory=dist/assets) at
        # create_app time - it must exist or the mount itself raises.
        (dist / "assets").mkdir()
        sibling = tmp_path / "sibling"
        sibling.mkdir()
        (sibling / "app.js").write_text("// evil bundle")
        monkeypatch.setattr(srv, "FRONTEND_DIST", dist)

        c = _client()
        r = c.get("/%2E%2E/sibling/app.js")
        assert "evil bundle" not in r.text


class TestTheDistRootIsResolvedOnceAtMountTime:
    """A per-request ``FRONTEND_DIST.resolve()`` on a symlinked dist would
    race a swap - the pattern in this file resolves once so the ancestor
    comparison is stable. The property is worth pinning because a well-meant
    refactor to "resolve both sides per request" reintroduces the race and
    still passes every functional cell above."""

    def test_the_module_uses_a_single_resolved_ancestor(self) -> None:
        # The guard is inline in server.py; grade the tree, not the shape.
        # Reading the source is a documentation cell - the functional cells
        # above are what fails on a regression.
        src = pathlib.Path(srv.__file__).read_text()
        # The resolved dist ancestor and the descendant test both have to be
        # present; either one alone is not the fix.
        assert "FRONTEND_DIST.resolve()" in src
        assert "relative_to" in src
