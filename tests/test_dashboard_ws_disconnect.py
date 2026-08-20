"""A send-only websocket must notice that its viewer left (Q50).

MEASURED on the live dashboard: 71,798 "connection open" lines in 11.5 hours and not one
close verdict. A handler that only ever SENDS learns about a disconnect from a failing
send, so a camera that publishes nothing never learns at all - the loop spun at 15Hz for
the life of the process and the `finally` block never ran. That is a leaked coroutine per
socket, and it is also why Q42's close verdict could not fire for the very failure it was
written to explain ("sent nothing - that camera is not publishing").
"""
from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from starlette.websockets import WebSocketDisconnect

from strands_robots.dashboard.server import _client_gone


class _FakeWS:
    def __init__(self, script):
        self.script = list(script)
        self.reads = 0

    async def receive(self):
        self.reads += 1
        if not self.script:
            await asyncio.sleep(3600)  # a real socket parks here
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_returns_on_the_disconnect_message():
    ws = _FakeWS([{"type": "websocket.receive", "text": "hi"}, {"type": "websocket.disconnect", "code": 1001}])
    asyncio.run(asyncio.wait_for(_client_gone(ws), 2))
    assert ws.reads == 2, "ordinary inbound messages must be consumed, not mistaken for a disconnect"


@pytest.mark.parametrize("exc", [WebSocketDisconnect(1006), RuntimeError("after close")])
def test_returns_when_the_channel_itself_dies(exc):
    asyncio.run(asyncio.wait_for(_client_gone(_FakeWS([exc])), 2))


def test_does_not_return_while_the_client_is_still_there():
    async def go():
        ws = _FakeWS([{"type": "websocket.receive", "bytes": b"x"}])
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_client_gone(ws), 0.25)
    asyncio.run(go())


def test_camera_socket_logs_its_verdict_after_the_client_hangs_up(monkeypatch, caplog, tmp_path):
    """End to end, with a bridge that publishes NOTHING - the leaking case."""
    from fastapi.testclient import TestClient

    from strands_robots.dashboard import server as srv

    from strands_robots.dashboard import settings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("STRANDS_MESH", "false")
    # This machine has a real token in settings.json, and an app that demands one would
    # refuse the socket before the handler ever ran - the test would then "pass" for the
    # wrong reason. settings.override is process-scoped (Q49) and never touches the file.
    settings.override("security", "auth_token", "")
    app = srv.create_app()

    class _Silent:
        def latest_frame(self, *a, **k):
            return None

        def snapshot(self):
            return {"type": "snapshot", "peers": {}}

        def attach_queue(self):
            return asyncio.Queue()

        def detach_queue(self, q):
            pass

        def start(self, loop):
            return False

        def stop(self):
            pass

    app.state.bridge = _Silent()
    srv._CAMERA_CLOSE_LOG.__init__()  # a fresh throttle, so this close is the first

    done = threading.Event()

    def run():
        with TestClient(app) as client, caplog.at_level(logging.INFO):
            with client.websocket_connect("/ws/camera/p1/wrist"):
                pass  # open, then hang up immediately - like a page reload
        done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(15)
    assert done.is_set(), "the handler never returned: the disconnect went unnoticed (Q50 regression)"
    settings.clear_overrides()
    lines = [r.message for r in caplog.records if "camera socket" in r.message]
    assert lines, f"no close verdict logged; saw {[r.message for r in caplog.records][-5:]}"
    assert "p1/wrist" in lines[0] and "not publishing" in lines[0]
