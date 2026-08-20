"""Every frame type the server can emit must be handled by the frontend (Q80).

The UI renders from the STREAM, not from the routes — that is the lesson the U2 role badge taught the
hard way (a route enriched, proved with curl, and the card still blank because the card reads the
websocket snapshot). Q79 gave the HTTP half a way to notice a server older than the bundle. The websocket
half has no status code at all: a frame type the bundle does not handle is simply dropped, in silence, at
whatever rate the server sends it. Nothing fails, nothing logs, the feature is just absent.

Measured against the live dashboard 2026-08-20 (probe on /ws/mesh, 12 frames): types snapshot/state/
presence arrive, and every type the source can emit IS handled today. `response` looked like a hole and
is not — it appears only inside a docstring describing the mesh RPC wire. Hence the AST: a regex over
these files reads prose as code and would have failed on that sentence.

Direction pinned here: server -> client. A type added to the backend with no reader lands as invisible
dead data; this test asks for the reader in the same commit.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

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
    reader for a frame nothing ever sends — a test failing on prose is worse than no test.
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
        "frame types the backend can send that no frontend source mentions — a websocket frame has no "
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
#: entry belongs here in the same commit as the UI that sends it — the pointer is the point, because the
#: failure this guards is "the button does nothing and no log mentions it".
_CLIENT_FRAMES = {
    "chat": ("server.py", "_CHAT_FRAME_TYPES"),
    "ping": ("server.py", "_CHAT_FRAME_TYPES"),
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
    assert sent, "the scan found no client frames at all — the pattern it looks for must have changed"
    unimplemented = sorted(sent - set(_CLIENT_FRAMES))
    assert not unimplemented, (
        "the UI sends these websocket frames and this test cannot see where the server handles them; "
        f"a frame the server does not implement is dropped in silence, with the socket still open: {unimplemented}"
    )
    for frame, (filename, marker) in _CLIENT_FRAMES.items():
        if frame not in sent:
            continue  # the UI stopped sending it; the handler may stay for older clients
        assert marker in (_DASH / filename).read_text(), f"{frame}: {filename} no longer mentions {marker}"
