"""``/ws/voice`` - speech-to-speech fleet control.

The browser streams PCM16 (16 kHz mono) frames in and receives the agent's audio
and transcript back. Admission is the same as every other socket. The voice agent
holds exactly one tool, :func:`voice.make_fleet_tool`, which fails closed on any
physical peer because a spoken conversation has no confirm rail to pause on.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from strands_robots.dashboard import access

router = APIRouter(tags=["voice"])

_VOICE_EXTRA = (
    "strands-agents[bidi] plus a voice provider key (OPENAI_API_KEY, AWS creds for nova_sonic, or GEMINI_API_KEY)"
)


@router.websocket("/ws/voice")
async def voice_socket(ws: WebSocket) -> None:
    """One spoken conversation; strangers are closed with 4401, a missing provider with 4503."""
    try:
        access.caller(ws)  # type: ignore[arg-type]
    except HTTPException:
        await access.refuse_socket(ws, 4401)
        return
    await ws.accept()
    try:
        from strands_robots.dashboard.voice import run_voice_session
    except ImportError as exc:
        await ws.send_json({"type": "error", "message": f"voice unavailable: {exc}. Needs {_VOICE_EXTRA}."})
        await ws.close(code=4503)
        return
    bridge: Any = getattr(ws.app.state, "bridge", None)
    try:
        await run_voice_session(ws, bridge=bridge)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:  # noqa: BLE001 - the operator hears why, the socket closes clean
        try:
            await ws.send_json({"type": "error", "message": f"voice session failed: {type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001 - already gone
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001 - already closed by the peer
            pass
