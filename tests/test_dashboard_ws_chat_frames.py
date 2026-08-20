"""Q17/Q18/Q19: /ws/chat survives hostile or accidental frames.

A binary frame used to die with KeyError('text'), a non-string text with
AttributeError('int' has no 'strip') - client saw a bare connection drop.
And a NON-JSON frame was promoted to a prompt: junk billed as a real model
turn, with no size cap (a 2 MB frame ran). parse_chat_frame is the pure
gate: protocol errors become typed 'error' events, never prompts.
"""

from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

from strands_robots.dashboard import server as srv
from strands_robots.dashboard.server import CHAT_MAX_FRAME_BYTES, parse_chat_frame


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
    for payload in ({"type": "chat"}, {"type": "chat", "text": ""}, {"type": "chat", "text": "   "}, {"type": "chat", "text": None}):
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
