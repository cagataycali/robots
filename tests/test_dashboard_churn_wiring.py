"""The churn guard is actually WIRED to /ws/camera - not merely importable (Q53).

The pure guard has its own unit tests; this one exists because of a lesson from this
project's own history (U2 piece 3): I once enriched a ROUTE and proved it with curl while
the screen kept rendering from a different code path. A defence nobody reaches is not a
defence, so the acceptance test here is the socket's own behaviour: open a camera faster
than a human ever would and require the server to SAY it is pacing us.

Aimed at a peer/camera that publishes nothing, exactly like the live storm - the guard
counts ACCEPTED sockets, which is what a storm is made of.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from strands_robots.dashboard.churn_guard import CHURN_CAP_FPS, CHURN_OPENS_PER_MIN


def _app(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings
    from strands_robots.dashboard.server import create_app

    monkeypatch.setenv("STRANDS_MESH", "false")
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    # settings resolves defaults THROUGH the environment, so a token another module leaves
    # behind would close these sockets with 1008 and read as "the guard never fired".
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}
    return create_app()


#: A frame is ALWAYS available in these tests, deliberately: a socket that is not
#: throttled then answers with a binary image straight away, so a broken guard makes the
#: assertion fail instead of making the test hang on a read that never returns. (First
#: draft hung for two minutes on exactly that - a test whose failure mode is "forever" is
#: not a test.)
def _publishing(app, monkeypatch) -> None:
    frame = {"t": 1.0, "jpeg": b"\xff\xd8not-a-real-jpeg\xff\xd9"}
    monkeypatch.setattr(app.state.bridge, "latest_frame", lambda *a, **k: frame)


def _first_frame(ws) -> dict | None:
    """The socket's first frame, as a dict when it is text, None when it is an image."""
    msg = ws.receive()
    text = msg.get("text") if isinstance(msg, dict) else None
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


class TestTheGuardReachesTheSocket:
    def test_a_storm_is_told_it_is_being_paced(self, monkeypatch, tmp_path) -> None:
        app = _app(monkeypatch, tmp_path)
        _publishing(app, monkeypatch)
        client = TestClient(app)
        # The storm itself: opens counted, nothing read - a viewer in a reconnect loop is
        # not waiting around for frames either.
        for _ in range(CHURN_OPENS_PER_MIN + 1):
            with client.websocket_connect("/ws/camera/so101-arm-1/top"):
                pass
        with client.websocket_connect("/ws/camera/so101-arm-1/top") as ws:
            notice = _first_frame(ws)
        assert notice is not None, "the storm's next socket got an image, not a notice"
        assert notice["type"] == "camera_error", notice
        # An OLD bundle renders `camera_error` text, which is the whole reason this is not
        # a new frame type: the tab that caused the storm can still explain itself.
        assert f"{CHURN_CAP_FPS:g} fps" in notice["error"], notice["error"]
        assert notice["peer_id"] == "so101-arm-1" and notice["cam"] == "top"

    def test_a_human_opening_a_tile_is_never_told_anything(self, monkeypatch, tmp_path) -> None:
        """The false positive that would matter: a throttle notice on an operator's tile,
        for a camera that is merely not publishing yet."""
        app = _app(monkeypatch, tmp_path)
        _publishing(app, monkeypatch)
        client = TestClient(app)
        for _ in range(3):
            with client.websocket_connect("/ws/camera/so101-arm-2/wrist") as ws:
                assert _first_frame(ws) is None, "an operator's tile was told it is a storm"
