"""Websocket surface of the operator dashboard (consolidated).

Merged verbatim from test_dashboard_ws_frame_contract / _ws_chat_frames /
_ws_close_log / _ws_disconnect. Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import pathlib
import threading

import pytest
from starlette.websockets import WebSocketDisconnect

from strands_robots.dashboard import server as srv
from strands_robots.dashboard.server import CHAT_MAX_FRAME_BYTES, _client_gone, parse_chat_frame
from strands_robots.dashboard.ws_observability import (
    CloseLogThrottle,
    close_line,
    close_verdict,
)

# ============================================================================
# from tests/test_dashboard_ws_frame_contract.py
# Every frame type the server can emit must be handled by the frontend (Q80).
# ============================================================================

_DASH = pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard"
_EMITTERS = (_DASH / "server.py", _DASH / "mesh_bridge.py")
_FRONTEND_SRC = _DASH / "frontend" / "src"

#: Types that exist for a machine, not for the UI: the bundle never branches on them and must not be
#: forced to. Each needs a reason, so "add it to the ignore list" stays a visible decision.
_NOT_FOR_THE_UI = {
    # A heartbeat reply the socket layer answers by existing: the client's liveness check is that a
    # frame came back at all, so no component branches on the word.
    "pong",
    # ASGI's own vocabulary, not ours: `{"type": "websocket.close"}` is spoken to the SERVER by the app
    # (the middleware that refuses an unauthorised socket), and never reaches a browser as a frame.
    "websocket.close",
}


def _emitted_types() -> dict[str, set[str]]:
    """Every ``{"type": "<literal>"}`` dict built in real code, by file.

    ast, deliberately: the first version of this test grepped, and matched a docstring that spells out
    ``{"type": "response", ...}`` as documentation of the mesh RPC layering. It then demanded a frontend
    reader for a frame nothing ever sends -- a test failing on prose is worse than no test.
    """
    out: dict[str, set[str]] = {}
    for path in _EMITTERS:
        tree = ast.parse(path.read_text())
        found: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "type"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    found.add(value.value)
        out[path.name] = found
    return out


@pytest.fixture(scope="module")
def frontend_text() -> str:
    if not _FRONTEND_SRC.exists():
        pytest.skip(f"no frontend sources at {_FRONTEND_SRC}")
    return "\n".join(
        p.read_text()
        for p in _FRONTEND_SRC.rglob("*")
        if p.suffix in {".ts", ".tsx"} and not p.name.endswith(".test.mjs")
    )


def test_every_emitted_frame_type_has_a_reader(frontend_text: str) -> None:
    unread: dict[str, list[str]] = {}
    for filename, types in _emitted_types().items():
        missing = sorted(
            t
            for t in types
            if t not in _NOT_FOR_THE_UI and f"'{t}'" not in frontend_text and f'"{t}"' not in frontend_text
        )
        if missing:
            unread[filename] = missing
    assert not unread, (
        "frame types the backend can send that no frontend source mentions -- a websocket frame has no "
        f"status code, so these are dropped silently at their send rate: {unread}"
    )


def test_the_probe_found_the_types_this_dashboard_actually_streams(frontend_text: str) -> None:
    """The three frames a live fleet produces every second are the ones worth naming explicitly."""
    emitted = set().union(*_emitted_types().values())
    for t in ("snapshot", "state", "presence"):
        assert t in emitted, f"{t} is what /ws/mesh streamed on the live dashboard"
        assert f"'{t}'" in frontend_text, f"nothing in the bundle reads {t}"


def test_the_ignore_list_only_holds_types_that_are_really_emitted() -> None:
    """A stale exemption would hide the next real hole behind a name that no longer exists."""
    emitted = set().union(*_emitted_types().values())
    assert _NOT_FOR_THE_UI <= emitted, f"exempted but never sent: {sorted(_NOT_FOR_THE_UI - emitted)}"


# --- the other direction: frames the BUNDLE sends must be implemented (Q81) ----

#: Every ``{type: '...'}`` the frontend sends on a websocket, and where the server implements it. A new
#: entry belongs here in the same commit as the UI that sends it -- the pointer is the point, because the
#: failure this guards is "the button does nothing and no log mentions it".
_CLIENT_FRAMES = {
    "chat": ("server.py", "_CHAT_FRAME_TYPES"),
    "ping": ("server.py", "_CHAT_FRAME_TYPES"),
    "interrupt_response": ("server.py", "_CHAT_FRAME_TYPES"),
    "stop": ("voice.py", '"stop"'),
}


def _frames_the_bundle_sends() -> set[str]:
    import re

    sent: set[str] = set()
    for path in _FRONTEND_SRC.rglob("*"):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        text = path.read_text()
        for m in re.finditer(r"send\(JSON\.stringify\(\{([^}]*)\}", text):
            for t in re.finditer(r"type:\s*'([a-z_]+)'", m.group(1)):
                sent.add(t.group(1))
    return sent


def test_every_frame_the_bundle_sends_is_implemented_server_side() -> None:
    sent = _frames_the_bundle_sends()
    assert sent, "the scan found no client frames at all -- the pattern it looks for must have changed"
    unimplemented = sorted(sent - set(_CLIENT_FRAMES))
    assert not unimplemented, (
        "the UI sends these websocket frames and this test cannot see where the server handles them; "
        f"a frame the server does not implement is dropped in silence, with the socket still open: {unimplemented}"
    )
    for frame, (filename, marker) in _CLIENT_FRAMES.items():
        if frame not in sent:
            continue  # the UI stopped sending it; the handler may stay for older clients
        assert marker in (_DASH / filename).read_text(), f"{frame}: {filename} no longer mentions {marker}"


# ============================================================================
# from tests/test_dashboard_ws_chat_frames.py
# Q17/Q18/Q19: /ws/chat survives hostile or accidental frames.
# ============================================================================


def _text(payload) -> dict:
    return {"type": "websocket.receive", "text": payload}


# --- Q17: frames that used to kill the socket --------------------------------


def test_binary_frame_is_a_typed_error_not_a_crash():
    prompt, reply = parse_chat_frame({"type": "websocket.receive", "bytes": b"\x00\x01\x02"})
    assert prompt is None
    assert reply["type"] == "error" and "binary" in reply["error"]


def test_non_string_text_is_a_typed_error():
    for bad in (123, 1.5, True, ["x"], {"a": 1}):
        prompt, reply = parse_chat_frame(_text(json.dumps({"type": "chat", "text": bad})))
        assert prompt is None, bad
        assert reply["type"] == "error" and "string" in reply["error"], bad


# --- Q18: junk never becomes a paid model turn --------------------------------


def test_non_json_frame_is_an_error_not_a_prompt():
    prompt, reply = parse_chat_frame(_text("this is not json at all"))
    assert prompt is None
    assert reply["type"] == "error" and "JSON" in reply["error"]


def test_json_scalar_frame_is_an_error_not_a_prompt():
    prompt, reply = parse_chat_frame(_text('"just a string"'))
    assert prompt is None
    assert reply["type"] == "error" and "object" in reply["error"]


def test_oversized_frame_is_capped_before_parsing():
    huge = json.dumps({"type": "chat", "text": "x" * (2 * 1024 * 1024)})
    prompt, reply = parse_chat_frame(_text(huge))
    assert prompt is None
    assert reply["type"] == "error" and "KB" in reply["error"]


def test_cap_is_generous_enough_for_a_real_turn():
    fine = json.dumps({"type": "chat", "text": "x" * 8000})
    assert len(fine.encode()) < CHAT_MAX_FRAME_BYTES
    prompt, reply = parse_chat_frame(_text(fine))
    assert prompt == "x" * 8000 and reply is None


# --- normal protocol unchanged -------------------------------------------------


def test_ping_pong():
    prompt, reply = parse_chat_frame(_text(json.dumps({"type": "ping"})))
    assert prompt is None and reply == {"type": "pong"}


def test_valid_chat_frame_yields_prompt():
    prompt, reply = parse_chat_frame(_text(json.dumps({"type": "chat", "text": "  hello  "})))
    assert prompt == "hello" and reply is None


def test_empty_or_missing_text_is_silently_ignored():
    for payload in (
        {"type": "chat"},
        {"type": "chat", "text": ""},
        {"type": "chat", "text": "   "},
        {"type": "chat", "text": None},
    ):
        prompt, reply = parse_chat_frame(_text(json.dumps(payload)))
        assert prompt is None and reply is None, payload


def test_disconnect_message_raises_disconnect():
    with pytest.raises(WebSocketDisconnect):
        parse_chat_frame({"type": "websocket.disconnect", "code": 1001})


# --- through the real route: socket SURVIVES the bad frame --------------------


class _StubBridge:
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}


def _isolate(monkeypatch, tmp_path):
    """The real machine's settings.json carries a live auth token; a test
    client that reads it gets 1008 at the handshake."""
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None


def test_socket_survives_binary_then_answers_ping(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from starlette.testclient import TestClient

    from strands_robots.dashboard import auth

    auth._cache_key = None
    auth._cache = {}

    app = srv.create_app(bridge=_StubBridge())
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_bytes(b"\x00\x01\x02binary")
        ev = ws.receive_json()
        assert ev["type"] == "error" and "binary" in ev["error"]
        # Q17's whole point: the socket is still alive afterwards
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "pong"}
        ws.send_text("not json {{{")
        ev = ws.receive_json()
        assert ev["type"] == "error" and "JSON" in ev["error"]
        ws.send_text(json.dumps({"type": "chat", "text": 42}))
        ev = ws.receive_json()
        assert ev["type"] == "error" and "string" in ev["error"]
        # still alive
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "pong"}


def test_queued_notice_when_turn_lock_is_held(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from starlette.testclient import TestClient

    from strands_robots.dashboard import agent_bridge, auth

    auth._cache_key = None
    auth._cache = {}

    def fake_turn(prompt, q, cancel=None):
        q.put({"type": "done", "text": "ok"})
        q.put({"type": "__END__"})

    monkeypatch.setattr(agent_bridge, "run_turn_blocking", fake_turn)
    app = srv.create_app(bridge=_StubBridge())
    client = TestClient(app)

    assert agent_bridge._turn_lock.acquire(blocking=False), "lock must be free at test start"
    try:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "chat", "text": "hi"}))
            ev = ws.receive_json()
            assert ev["type"] == "notice" and "queued" in ev["text"]
    finally:
        agent_bridge._turn_lock.release()


# --- an unrecognised type is refused, not dropped (Q81) ------------------------
#
# The send direction of the websocket contract, audited 2026-08-20: this bundle sends {"type":"chat"}
# here and {"type":"stop"} on /ws/voice, and both are implemented -- so this is a guard against the next
# frame type, which will be added to the UI before the server learns it. A websocket has no status code:
# without this the operator taps a button, the socket stays healthily open, and nothing happens anywhere.


def test_unknown_frame_type_is_answered_by_name():
    prompt, reply = parse_chat_frame(_text(json.dumps({"type": "cancel", "run": 3})))
    assert prompt is None, "an unimplemented verb must never be promoted to a billed model turn (Q18)"
    assert reply is not None and reply["type"] == "error"
    assert "cancel" in reply["error"], "the operator needs the type named to report it"
    assert "chat" in reply["error"] and "ping" in reply["error"], "and what this server does accept"


def test_an_unknown_type_carrying_text_is_still_refused():
    """The type IS the contract: a confused client's text must not become an instruction."""
    prompt, reply = parse_chat_frame(_text(json.dumps({"type": "stop", "text": "move the arm"})))
    assert prompt is None and reply is not None and reply["type"] == "error"


def test_a_frame_with_no_type_but_text_still_works():
    """Compatibility: refusing this would break a working path in order to fix a silent one."""
    prompt, reply = parse_chat_frame(_text(json.dumps({"text": "hello"})))
    assert prompt == "hello" and reply is None


def test_known_types_keep_their_silence():
    """An empty submit stays silent -- it must not be billed, and it must not scold either."""
    for payload in ({"type": "chat"}, {"type": "chat", "text": "   "}):
        prompt, reply = parse_chat_frame(_text(json.dumps(payload)))
        assert prompt is None and reply is None, payload


# --- interrupt_response: answering a motion confirm ----------------------------


def test_interrupt_response_frame_yields_a_resume_turn():
    turn, reply = parse_chat_frame(
        _text(
            json.dumps(
                {"type": "interrupt_response", "id": " int-1 ", "response": {"approve": True}},
            )
        )
    )
    assert reply is None
    assert turn == {"interrupt_id": "int-1", "response": {"approve": True}}


def test_interrupt_response_no_is_carried_verbatim():
    turn, reply = parse_chat_frame(
        _text(
            json.dumps(
                {"type": "interrupt_response", "id": "int-1", "response": False},
            )
        )
    )
    assert reply is None and turn["response"] is False


def test_interrupt_response_requires_a_string_id():
    for bad in ({}, {"id": ""}, {"id": 42}, {"id": None}):
        turn, reply = parse_chat_frame(
            _text(
                json.dumps(
                    {"type": "interrupt_response", "response": True, **bad},
                )
            )
        )
        assert turn is None, bad
        assert reply["type"] == "error" and "id" in reply["error"], bad


def test_interrupt_response_requires_a_response_field():
    turn, reply = parse_chat_frame(_text(json.dumps({"type": "interrupt_response", "id": "int-1"})))
    assert turn is None
    assert reply["type"] == "error" and "response" in reply["error"]


def test_interrupt_response_reaches_resume_not_run(monkeypatch, tmp_path):
    """Through the real route: the frame dispatches to resume_interrupt_blocking."""
    _isolate(monkeypatch, tmp_path)
    from starlette.testclient import TestClient

    from strands_robots.dashboard import agent_bridge, auth

    auth._cache_key = None
    auth._cache = {}

    resumed: list = []

    def fake_resume(interrupt_id, response, q, cancel=None):
        resumed.append((interrupt_id, response))
        q.put({"type": "done", "text": "continued"})
        q.put({"type": "__END__"})

    def fake_run(prompt, q, cancel=None):
        raise AssertionError("a confirm answer must never become a fresh prompt")

    monkeypatch.setattr(agent_bridge, "resume_interrupt_blocking", fake_resume)
    monkeypatch.setattr(agent_bridge, "run_turn_blocking", fake_run)
    app = srv.create_app(bridge=_StubBridge())
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "interrupt_response", "id": "int-9", "response": True}))
        ev = ws.receive_json()
        assert ev == {"type": "done", "text": "continued"}
    assert resumed == [("int-9", True)]


# ============================================================================
# from tests/test_dashboard_ws_close_log.py
# A socket's death has to be visible, or a storm looks like popularity.
# ============================================================================


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestVerdict:
    def test_a_stream_that_worked_says_so_with_numbers(self) -> None:
        v = close_verdict(frames_sent=1800, lifetime_s=120.0, publishing=True)
        # Q51 gave the lifetime a decimal (a socket that lived 0.4s used to read as "0s")
        assert "1800 frames" in v and "120.0s" in v and "15.0 fps" in v

    def test_the_q40_case_names_the_cause_and_not_the_symptom(self) -> None:
        # exactly the incident: accepted, nothing published, closed in milliseconds
        v = close_verdict(frames_sent=0, lifetime_s=0.05, publishing=False)
        assert "not publishing" in v
        assert "robot may not be running" in v, "the operator needs the next step, not a status"

    def test_a_client_that_hung_up_is_not_blamed_on_the_camera(self) -> None:
        v = close_verdict(frames_sent=0, lifetime_s=0.4, publishing=True)
        assert "client hung up" in v
        assert "retry loop with no backoff" in v, "name the bug this log is here to catch"

    def test_publishing_but_nothing_delivered_is_its_own_case(self) -> None:
        # the honest "I do not know why" bucket - a real state, not a guess
        v = close_verdict(frames_sent=0, lifetime_s=45.0, publishing=True)
        assert "not reaching this socket" in v

    def test_every_verdict_mentions_how_long_it_lived(self) -> None:
        for kwargs in (
            {"frames_sent": 5, "lifetime_s": 3.0, "publishing": True},
            {"frames_sent": 0, "lifetime_s": 3.0, "publishing": False},
            {"frames_sent": 0, "lifetime_s": 0.2, "publishing": True},
            {"frames_sent": 0, "lifetime_s": 99.0, "publishing": True},
        ):
            assert "s" in close_verdict(**kwargs)  # type: ignore[arg-type]


class TestThrottle:
    def test_the_first_close_is_always_logged(self) -> None:
        t = CloseLogThrottle(window_s=60.0, clock=_Clock())
        assert t.should_log("so101-arm-1/top") == (True, 0)

    def test_a_storm_is_collapsed_but_counted(self) -> None:
        clock = _Clock()
        t = CloseLogThrottle(window_s=60.0, clock=clock)
        t.should_log("a/top")
        for _ in range(4000):  # the incident's rate
            log_now, _ = t.should_log("a/top")
            assert log_now is False
        clock.t += 60.0
        log_now, suppressed = t.should_log("a/top")
        assert log_now is True
        assert suppressed == 4000, "the count is the whole point: silence would hide the storm"

    def test_the_counter_resets_after_it_is_reported(self) -> None:
        clock = _Clock()
        t = CloseLogThrottle(window_s=10.0, clock=clock)
        t.should_log("k")
        t.should_log("k")
        clock.t += 10.0
        assert t.should_log("k") == (True, 1)
        clock.t += 10.0
        assert t.should_log("k") == (True, 0)

    def test_cameras_are_throttled_independently(self) -> None:
        t = CloseLogThrottle(window_s=60.0, clock=_Clock())
        assert t.should_log("arm-1/top")[0] is True
        assert t.should_log("arm-1/wrist")[0] is True, "one noisy tile must not silence another"

    def test_the_key_set_is_bounded(self) -> None:
        t = CloseLogThrottle(window_s=60.0, clock=_Clock())
        for i in range(2000):  # a spawn loop can invent peer ids
            t.should_log(f"peer-{i}/top")
        assert len(t._seen) <= 256


class TestLine:
    def test_it_names_the_socket_and_the_verdict(self) -> None:
        line = close_line(peer_id="so101-arm-1", cam="top", verdict="sent nothing in 0.1s", suppressed=0)
        assert "so101-arm-1/top" in line and "sent nothing" in line
        assert "suppressed" not in line

    def test_a_storm_count_is_in_the_line_itself(self) -> None:
        line = close_line(peer_id="p", cam="c", verdict="v", suppressed=4000)
        assert "+4000 more closes suppressed" in line


# --- Q51: bytes, not just frames ------------------------------------------------
def test_verdict_carries_the_volume_a_socket_actually_moved():
    """A frame count cannot tell a buggy client from a link that cannot keep up."""
    line = close_verdict(frames_sent=42, lifetime_s=9.0, publishing=True, bytes_sent=4_075_761)
    assert "42 frames" in line and "3.9 MB" in line and "4.7 fps" in line and "0.43 MB/s" in line


def test_volume_is_optional_and_never_divides_by_a_zero_lifetime():
    """Callers that do not measure bytes, and the socket that closed instantly."""
    assert close_verdict(frames_sent=3, lifetime_s=2.0, publishing=True) == "streamed 3 frames over 2.0s (1.5 fps)"
    instant = close_verdict(frames_sent=1, lifetime_s=0.0, publishing=True, bytes_sent=1000)
    assert "streamed 1 frames" in instant and "fps" not in instant and "MB/s" not in instant


# --- Q52: a viewer may ask for fewer frames; it may never ask for more -------------
def test_a_cap_is_honoured_within_sane_bounds():
    from strands_robots.dashboard.ws_observability import MAX_CAP_FPS, MIN_CAP_FPS, fps_cap

    assert fps_cap("1") == 1.0
    assert fps_cap("0.5") == 0.5
    assert fps_cap("1000") == MAX_CAP_FPS, "an absurd number is clamped, not trusted"
    assert fps_cap("0.0001") == MIN_CAP_FPS, "a 'cap' that freezes the tile helps nobody"


def test_nonsense_never_becomes_a_request_for_more():
    """The failure mode that matters: a bad value must fall back to today's behaviour."""
    from strands_robots.dashboard.ws_observability import fps_cap

    for raw in (None, "", "abc", "-5", "0", "nan", "1e400x"):
        assert fps_cap(raw) is None, raw
    assert fps_cap("inf") == 30.0, "infinity clamps to the ceiling rather than dividing by nothing"


def test_the_verdict_says_which_rate_the_socket_agreed_to():
    from strands_robots.dashboard.ws_observability import cap_note

    assert cap_note(None) == ""
    assert "1 fps" in cap_note(1.0) and "2.5 fps" in cap_note(2.5)


class TestTheLineActuallyReachesTheLog:
    """MEASURED 2026-08-20 on the live dashboard: 75,489 `connection open` lines and
    ZERO `connection closed` lines in one process lifetime.

    The two strings both come from the `websockets` library (server.py logs the open,
    protocol.discard() logs the close), and the close only fires when the sans-io close
    path reaches EOF - which in this deployment it never did, not once in 75k sockets.
    That asymmetry is exactly why a storm burning 20.7 GB stayed invisible for 12 hours:
    the log recorded every socket's birth and no socket's death, so churn, lifetime and
    cause were all unanswerable from it.

    So our close verdict must NOT depend on the library's logging. It is emitted from the
    handler's own `finally`, and this test pins that - including the abrupt-disconnect case,
    which is the only case the live rig has ever actually produced.
    """

    def _app(self, monkeypatch, tmp_path):
        from strands_robots.dashboard import auth
        from strands_robots.dashboard import settings as dsettings
        from strands_robots.dashboard.server import create_app

        monkeypatch.setenv("STRANDS_MESH", "false")
        monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
        # In a full sweep another module leaves DASHBOARD_AUTH_TOKEN in the environment,
        # and settings resolves defaults THROUGH the environment - so this app inherited a
        # token, the middleware closed my socket with 1008, and the failure looked like the
        # close verdict was missing. The test now owns its auth posture instead of assuming
        # a clean environment (same class of leak as tests/test_dashboard_env_allowlist).
        monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
        dsettings._cache = None
        auth._cache_key = None
        auth._cache = {}
        app = create_app()
        return app

    def test_an_abrupt_disconnect_still_logs_a_verdict(self, monkeypatch, tmp_path, caplog):
        import logging

        from fastapi.testclient import TestClient

        app = self._app(monkeypatch, tmp_path)
        # A camera that publishes nothing: the Q50 shape, and the one where the handler
        # has no failing send to notice the client left.
        monkeypatch.setattr(app.state.bridge, "latest_frame", lambda *a, **k: None)
        client = TestClient(app)
        with caplog.at_level(logging.INFO):
            with client.websocket_connect("/ws/camera/so101-arm-1/top"):
                pass  # closes immediately - the abrupt case
        lines = [r.getMessage() for r in caplog.records]
        assert any("camera socket so101-arm-1/top closed" in m for m in lines), lines[-6:]

    def test_a_socket_that_sent_nothing_is_logged_as_a_WARNING(self, monkeypatch, tmp_path, caplog):
        """Severity is the whole point: a storm of zero-frame sockets must be visible at
        the level an operator actually reads, not buried in INFO next to 75k opens.

        Uses a DIFFERENT camera than the test above on purpose: the per-identity rate
        limiter suppressed this close when both tests shared `arm-1/top`, which is the
        limiter doing exactly its job (one storm, one line) - and is worth stating, since
        a future reader will otherwise 'fix' the flake by weakening it.
        """
        import logging

        from fastapi.testclient import TestClient

        app = self._app(monkeypatch, tmp_path)
        monkeypatch.setattr(app.state.bridge, "latest_frame", lambda *a, **k: None)
        client = TestClient(app)
        with caplog.at_level(logging.INFO):
            with client.websocket_connect("/ws/camera/so101-arm-2/wrist"):
                pass
        closes = [r for r in caplog.records if "closed" in r.getMessage()]
        assert closes and closes[-1].levelno == logging.WARNING


# ============================================================================
# from tests/test_dashboard_ws_disconnect.py
# A send-only websocket must notice that its viewer left (Q50).
# ============================================================================


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
