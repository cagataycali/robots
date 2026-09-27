"""Speech-to-speech fleet control - browser mic <-> Strands bidi agent. PCM16 audio flows over
/ws/voice (binary in, base64 JSON out).
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

VOICE_PROMPT = """You are the Strands Robots fleet voice operator. You control real robots and
simulations on a mesh via the fleet tool. Keep spoken replies SHORT - one or
two sentences. Confirm before actuating real hardware (peer ids without 'sim'
in them). fleet(action='peers') shows who's online; fleet(action='task',
target=..., instruction=..., duration=...) runs a task;
fleet(action='stop_all') stops everything - use it immediately when asked to
stop.
Starting a task on a REAL robot may be refused because this dashboard does not
let an agent start physical motion on its own. That refusal is final for you:
say in one sentence that the operator has to allow it on screen (a card just
appeared) or press play themselves, and do NOT retry, reword or pick another
robot. A spoken yes cannot grant it - only their tap can. Stopping is never
refused, so always act on a stop request immediately."""

_DEFAULT_VOICES = {"openai": "marin", "nova_sonic": "tiffany", "gemini": "Kore"}


_refusal_listeners: list[Any] = []


def add_refusal_listener(cb: Any) -> Any:
    """Register ``cb(text)`` for every refusal the fleet tool raises; returns an unsubscribe."""
    _refusal_listeners.append(cb)

    def _off() -> None:
        with contextlib.suppress(ValueError):
            _refusal_listeners.remove(cb)

    return _off


def _notify_refusal(text: str) -> None:
    for cb in list(_refusal_listeners):
        with contextlib.suppress(Exception):
            cb(text)


def make_fleet_tool(bridge: Any) -> Any:
    """The one tool the voice agent holds: read the fleet, task a peer, stop it.

    Voice has no confirm rail (bidi cannot pause on an interrupt), so a task on a
    physical peer is decided by :func:`agent_motion.agent_motion_allowed` alone and
    fails closed; the refusal is spoken once through the listeners above.
    """
    import json as _json

    from strands import tool

    from strands_robots.dashboard.agent_motion import agent_motion_allowed
    from strands_robots.dashboard.mesh_bridge import route_task_target

    @tool
    def fleet(
        action: str,
        target: str = "",
        instruction: str = "",
        policy_provider: str = "mock",
        duration: float = 15.0,
        robot_name: str = "",
    ) -> dict[str, Any]:
        """Coordinate robots on the mesh (dashboard gateway). Actions: peers, task, stop, stop_all, status."""
        if bridge is None:
            return {"status": "error", "content": [{"text": "mesh bridge offline"}]}

        if action == "peers":
            snap = bridge.snapshot()
            lines = []
            for pid, p in sorted((snap.get("peers") or {}).items()):
                if p.get("stale"):
                    continue
                pres = p.get("presence") or {}
                st = p.get("state") or {}
                task = st.get("task") or {}
                cams = list((p.get("cameras") or {}).keys())
                lines.append(
                    f"- {pid}: type={pres.get('robot_type', '?')} hw_connected={pres.get('connected')} "
                    f"cameras={cams} joints={len(st.get('joints') or {})} task={task.get('status', 'idle')} "
                    f"instruction={task.get('instruction', '')!r}"
                )
            text = "Online peers:\n" + "\n".join(lines) if lines else "No live peers on the mesh."
            return {"status": "success", "content": [{"text": text}]}

        if action == "task":
            if not target or not instruction:
                return {"status": "error", "content": [{"text": "task requires target and instruction"}]}
            try:
                peers = bridge.snapshot().get("peers") or {}
            except Exception:  # noqa: BLE001 - an unreadable snapshot means UNKNOWN, i.e. metal
                peers = {}
            verdict = agent_motion_allowed("task", peer=peers.get(target), target=target)
            if not verdict["allowed"]:
                _notify_refusal(verdict["reason"])
                return {"status": "error", "content": [{"text": verdict["reason"]}]}
            cmd: dict[str, Any] = {
                "action": "execute",
                "instruction": instruction,
                "policy_provider": policy_provider,
                "duration": float(duration),
            }
            if robot_name:
                cmd["robot_name"] = robot_name
            target, cmd = route_task_target(target, cmd)
            res = bridge.send_cmd(target, cmd, timeout=float(duration) + 30.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:1500]}]}

        if action == "stop":
            if not target:
                return {"status": "error", "content": [{"text": "stop requires target"}]}
            res = bridge.send_cmd(target, {"action": "stop"}, timeout=10.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        if action == "stop_all":
            results = {}
            for pid, p in (bridge.snapshot().get("peers") or {}).items():
                if p.get("stale"):
                    continue
                results[pid] = bridge.send_cmd(pid, {"action": "stop"}, timeout=5.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(results)[:1500]}]}

        if action == "status":
            if not target:
                return {"status": "error", "content": [{"text": "status requires target"}]}
            res = bridge.send_cmd(target, {"action": "status"}, timeout=10.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        return {
            "status": "error",
            "content": [{"text": f"unknown action {action!r}. Valid: peers, task, stop, stop_all, status"}],
        }

    return fleet


def _build_bidi_model(provider: str, voice: str | None = None) -> Any:
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


def build_voice_agent(provider: str | None = None, voice: str | None = None, *, bridge: Any = None) -> Any:
    """BidiAgent with the fleet toolset. Caller supplies browser audio IO.

    Deliberately NO robot_mesh here, and NO touching STRANDS_MESH_HITL_ACTIONS:
    bidi cannot pause a tool for a human answer (the SDK's agent/loop.py raises
    "tool interrupts are not supported in bidi"), so a gated robot_mesh action
    would blow up the tool task instead of asking. An earlier version worked
    around that by setdefault-ing STRANDS_MESH_HITL_ACTIONS="none" - a
    PROCESS-WIDE write that silently disarmed the chat agent's robot_mesh
    confirm gate the moment one voice session was opened. Voice safety instead
    rests on the fleet tool's own backstop (agent_motion_allowed fail-closes
    physical tasks - voice has no confirm rail, so no grant can ever appear).
    """
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

    from strands.experimental.bidi import BidiAgent
    from strands.experimental.bidi.tools import stop_conversation

    provider = provider or os.getenv("VOICE_PROVIDER", "openai")
    voice = voice or os.getenv("VOICE_NAME") or None
    model = _build_bidi_model(provider or "openai", voice)
    return BidiAgent(
        model=model,
        tools=[make_fleet_tool(bridge), stop_conversation],
        system_prompt=os.getenv("DASHBOARD_VOICE_PROMPT", VOICE_PROMPT),
    )


async def run_voice_session(ws: Any, *, bridge: Any = None) -> None:
    """Bridge one /ws/voice websocket to a fresh BidiAgent session. Browser -> binary PCM16 frames (16
    kHz mono) or {"type":"stop"} text.
    """
    import asyncio
    import json
    import queue as _queue

    # The bidi event vocabulary is experimental and has been renamed between SDK
    # releases (``BidiAudioStreamEvent`` became ``BidiAudioDeltaEvent``, and so on).
    # Resolve the input class by candidate name and classify output events by the
    # ``type`` string they all carry, so a rename is a no-op here rather than an
    # ImportError at the first spoken word.
    import strands.experimental.bidi.types.events as _bidi_events
    from starlette.websockets import WebSocketDisconnect

    audio_input_cls: Any = next(
        (getattr(_bidi_events, n) for n in ("BidiAudioInputEvent",) if hasattr(_bidi_events, n)), None
    )
    if audio_input_cls is None:
        raise ImportError(
            "strands.experimental.bidi has no audio input event this dashboard knows how to send",
            name="strands.experimental.bidi.types.events",
        )

    def _event_type(event: Any) -> str:
        try:
            return str(event.get("type", "") if hasattr(event, "get") else getattr(event, "type", ""))
        except Exception:  # noqa: BLE001 - an event that cannot say its type is not one we forward
            return ""

    in_q: asyncio.Queue[bytes] = asyncio.Queue()
    stop_evt = asyncio.Event()

    # A refusal raised inside the fleet tool is spoken once and gone: no transcript rail carries a
    # decision, and the operator cannot grant a permission by talking.
    from strands_robots.dashboard.consent import classify_refusal

    need_q: _queue.Queue[dict] = _queue.Queue()

    def _on_refusal(text: str) -> None:
        need = classify_refusal(text)
        if need is not None:
            need_q.put({"type": "needs_consent", "need": need.as_dict(), "spoken": text[:400]})

    drop_listener = add_refusal_listener(_on_refusal)

    async def _drain_needs() -> None:
        while not stop_evt.is_set():
            try:
                frame = need_q.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.2)
                continue
            try:
                await ws.send_text(json.dumps(frame))
            except Exception:  # noqa: BLE001 - the session is going away; the refusal still held
                break

    class _BrowserInput:
        async def start(self, agent: Any) -> None:
            self._cfg = agent.model.config["audio"]

        async def stop(self) -> None:
            pass

        async def __call__(self) -> Any:
            data = await in_q.get()
            return audio_input_cls(
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
            kind = _event_type(event)
            if kind.startswith("bidi_audio") and hasattr(event, "get") and event.get("audio"):
                await ws.send_text(json.dumps({"type": "audio", "data": event["audio"]}))
            elif kind.startswith("bidi_transcript") and hasattr(event, "get") and event.get("text") is not None:
                try:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "transcript",
                                "role": event.get("role", ""),
                                "text": event.get("text", ""),
                            }
                        )
                    )
                except Exception:
                    pass

    try:
        agent = build_voice_agent(bridge=bridge)
    except Exception as e:
        # This return happens BEFORE the finally below exists, so the listener has to be dropped here
        # too: one left behind would outlive the session, push into a queue nobody drains and pin this
        # closure for every later turn on the machine.
        drop_listener()
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
    needs_task = _a.create_task(_drain_needs())
    runner = _a.create_task(agent.run(inputs=[_BrowserInput()], outputs=[_BrowserOutput()]))
    try:
        await stop_evt.wait()
    finally:
        # Unregister FIRST: a listener left behind would keep pushing into a queue nobody drains and
        # would hold this session's closure alive for every later turn on the machine.
        drop_listener()
        runner.cancel()
        reader_task.cancel()
        needs_task.cancel()
