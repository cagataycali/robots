"""Speech-to-speech fleet control - browser mic <-> Strands bidi agent.

PCM16 audio flows over /ws/voice (binary in, base64 JSON out). The voice
agent carries the same robot_mesh toolset as
the chat agent, so "hey, make arm one pick up the red cube" actuates the
mesh directly from speech.

Providers (VOICE_PROVIDER env): openai (Realtime), nova_sonic, gemini.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

VOICE_PROMPT = """You are the Strands Robots fleet voice operator. You control real robots and
simulations on a mesh via the fleet tool. Keep spoken replies SHORT - one or
two sentences. Confirm before actuating real hardware (peer ids without 'sim'
in them). fleet(action='peers') shows who's online; fleet(action='task',
target=..., instruction=..., duration=...) runs a task;
fleet(action='stop_all') stops everything - use it immediately when asked to
stop."""

_DEFAULT_VOICES = {"openai": "marin", "nova_sonic": "tiffany", "gemini": "Kore"}


def _build_bidi_model(provider: str, voice: Optional[str] = None) -> Any:
    provider = provider.lower()
    v = voice or _DEFAULT_VOICES.get(provider)

    if provider in ("nova_sonic", "novasonic", "nova"):
        from strands.experimental.bidi.models import BidiNovaSonicModel

        region = os.getenv("AWS_REGION", "us-east-1")
        cfg = {"audio": {"voice": v}} if v else None
        return BidiNovaSonicModel(provider_config=cfg, client_config={"region": region})

    if provider in ("openai", "openai_realtime"):
        from strands.experimental.bidi.models import BidiOpenAIRealtimeModel

        kwargs: dict[str, Any] = {}
        if v:
            kwargs["provider_config"] = {"audio": {"voice": v}}
        if os.getenv("VOICE_MODEL"):
            kwargs["model_id"] = os.environ["VOICE_MODEL"]
        if os.getenv("OPENAI_API_KEY"):
            kwargs["client_config"] = {"api_key": os.environ["OPENAI_API_KEY"]}
        return BidiOpenAIRealtimeModel(**kwargs)

    if provider in ("gemini", "gemini_live"):
        from strands.experimental.bidi.models import BidiGeminiLiveModel

        kwargs = {}
        if v:
            kwargs["provider_config"] = {"audio": {"voice": v}}
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            kwargs["client_config"] = {"api_key": api_key}
        return BidiGeminiLiveModel(**kwargs)

    raise ValueError(f"unknown voice provider: {provider!r} (openai | nova_sonic | gemini)")


def build_voice_agent(provider: str | None = None, voice: str | None = None) -> Any:
    """BidiAgent with the fleet toolset. Caller supplies browser audio IO."""
    os.environ.setdefault("STRANDS_MESH_HITL_ACTIONS", "none")
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

    from strands.experimental.bidi import BidiAgent
    from strands.experimental.bidi.tools import stop_conversation

    from strands_robots.dashboard.agent_bridge import _make_fleet_tool

    provider = provider or os.getenv("VOICE_PROVIDER", "openai")
    voice = voice or os.getenv("VOICE_NAME") or None
    model = _build_bidi_model(provider, voice)
    return BidiAgent(
        model=model,
        tools=[_make_fleet_tool(), stop_conversation],
        system_prompt=os.getenv("DASHBOARD_VOICE_PROMPT", VOICE_PROMPT),
    )


async def run_voice_session(ws: Any) -> None:
    """Bridge one /ws/voice websocket to a fresh BidiAgent session.

    Browser -> binary PCM16 frames (16 kHz mono) or {"type":"stop"} text.
    Server -> {"type":"voice_meta","rate":N} then {"type":"audio","data":b64}
              plus {"type":"transcript","role":...,"text":...} when available.
    """
    import asyncio
    import json

    from starlette.websockets import WebSocketDisconnect

    try:
        from strands.experimental.bidi.types.events import (
            BidiAudioInputEvent,
            BidiAudioStreamEvent,
            BidiTranscriptStreamEvent,
        )
    except ImportError:
        BidiTranscriptStreamEvent = None  # older strands
        from strands.experimental.bidi.types.events import (  # type: ignore[no-redef]
            BidiAudioInputEvent,
            BidiAudioStreamEvent,
        )

    in_q: asyncio.Queue[bytes] = asyncio.Queue()
    stop_evt = asyncio.Event()

    class _BrowserInput:
        async def start(self, agent: Any) -> None:
            self._cfg = agent.model.config["audio"]

        async def stop(self) -> None:
            pass

        async def __call__(self) -> Any:
            data = await in_q.get()
            return BidiAudioInputEvent(
                audio=base64.b64encode(data).decode(),
                channels=self._cfg.get("channels", 1),
                format=self._cfg.get("format", "pcm"),
                sample_rate=self._cfg.get("input_rate", 16000),
            )

    class _BrowserOutput:
        async def start(self, agent: Any) -> None:
            rate = agent.model.config["audio"]["output_rate"]
            await ws.send_text(json.dumps({"type": "voice_meta", "rate": rate}))

        async def stop(self) -> None:
            pass

        async def __call__(self, event: Any) -> None:
            if isinstance(event, BidiAudioStreamEvent):
                await ws.send_text(json.dumps({"type": "audio", "data": event["audio"]}))
            elif BidiTranscriptStreamEvent is not None and isinstance(event, BidiTranscriptStreamEvent):
                try:
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "role": event.get("role", ""),
                        "text": event.get("text", ""),
                    }))
                except Exception:
                    pass

    try:
        agent = build_voice_agent()
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "error": f"voice agent: {e}"}))
        return

    async def _reader() -> None:
        while not stop_evt.is_set():
            try:
                raw = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                break
            if raw.get("bytes") is not None:
                await in_q.put(raw["bytes"])
            elif raw.get("text"):
                try:
                    if json.loads(raw["text"]).get("type") == "stop":
                        break
                except Exception:
                    pass
        stop_evt.set()

    import asyncio as _a

    reader_task = _a.create_task(_reader())
    runner = _a.create_task(agent.run(inputs=[_BrowserInput()], outputs=[_BrowserOutput()]))
    try:
        await stop_evt.wait()
    finally:
        runner.cancel()
        reader_task.cancel()
