"""``/ws/agent``: one operator conversation with the dashboard agent.

Client -> server frames: ``{"type":"say","text":...}`` starts a turn;
``{"type":"resume","id":...,"approve":bool,"always":bool}`` answers a consent
card. Server -> client frames are the console's events (text, tool_use,
tool_result, interrupt, done, error). One turn at a time per socket: a second
``say`` while a turn streams is refused with an error frame, not queued.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from strands_robots.dashboard import access, agent_console, agent_hitl

logger = logging.getLogger(__name__)
router = APIRouter()


def _console_factory(app: Any) -> Any:
    """Tests install their own; production builds a Console over the app's Safety."""
    factory = getattr(app.state, "console_factory", None)
    if factory is not None:
        return factory
    return lambda: agent_console.Console(
        app.state.safety,
        bridge=getattr(app.state, "bridge", None),
        devices=getattr(app.state, "devices", None),
    )


@router.get("/api/agent")
async def agent_info(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Which model the console will use, the tools it will hold, and which ask first."""
    bridge = getattr(request.app.state, "bridge", None)
    return {
        "model": agent_console.model_id(),
        "asks_first": agent_console.asks_first(bridge),
        "interrupt": agent_hitl.INTERRUPT_NAME,
        "tools": agent_console.expected_tool_names(bridge),
        "fleet_aware": bridge is not None,
    }


@router.post("/api/agent/reset")
async def agent_reset(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Consoles are per socket, so a reset is the page reconnecting its dock; this says so and what it will get."""
    body: Any = {}
    with contextlib.suppress(Exception):
        body = await request.json()
    clear = bool(isinstance(body, dict) and body.get("clear_history"))
    return {"reset": True, "history_cleared": clear, "reconnect": True, "model": agent_console.model_id()}


def strict_flag(value: Any) -> bool:
    """A consent flag from the page: the JSON boolean ``true`` or the string ``"true"`` is yes, anything else is no.

    ``bool()`` read the string ``"false"``, ``"no"``, ``1`` and any non-empty
    list as consent (f025). A flag that answers a motion interrupt is held to the
    strict spelling, so a misspelt refusal never becomes a yes.
    """
    return value is True or value == "true"


@router.websocket("/ws/agent")
async def agent_socket(ws: WebSocket) -> None:
    """A conversation. Admission is the same as every socket: Origin first, then the credential.

    The credential is re-checked while the socket lives (``access.serve_while_admitted``)
    and before every frame is acted on, so a consent answer from a session that
    has since signed out or lost its passkey is never delivered.
    """
    who = await access.admit_socket(ws)
    if who is None:
        return
    await ws.accept()
    try:
        console = _console_factory(ws.app)()
    except Exception as exc:  # noqa: BLE001 - no model, no creds: the operator reads why
        await ws.send_json({"type": "error", "message": f"agent unavailable: {type(exc).__name__}: {exc}"})
        await ws.close(code=4503)
        return
    with contextlib.suppress(WebSocketDisconnect):
        await access.serve_while_admitted(ws, who, _converse(ws, who, console))


async def _converse(ws: WebSocket, who: dict[str, Any], console: Any) -> None:
    """The frame loop of one conversation; ends with 4401 when a frame arrives after the caller was revoked."""
    busy = asyncio.Lock()
    while True:
        frame = await ws.receive_json()
        if not access.still_admitted(ws, who):
            await ws.close(code=4401)
            return
        kind = frame.get("type") if isinstance(frame, dict) else None
        if kind == "say":
            text = str(frame.get("text") or "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "say what?"})
                continue
            if len(text) > agent_console.MAX_PROMPT_CHARS:
                await ws.send_json(
                    {"type": "error", "message": f"prompt longer than {agent_console.MAX_PROMPT_CHARS} chars"}
                )
                continue
            prompt: Any = text
        elif kind == "resume":
            prompt = agent_console.Console.resume(
                str(frame.get("id") or ""), strict_flag(frame.get("approve")), strict_flag(frame.get("always"))
            )
        else:
            await ws.send_json({"type": "error", "message": "frames are {type: say|resume}"})
            continue
        if busy.locked():
            await ws.send_json({"type": "error", "message": "a turn is still streaming"})
            continue
        async with busy:
            async for event in console.run(prompt):
                await ws.send_json(event)
