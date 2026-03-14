#!/usr/bin/env python3
"""Strands Robots Dashboard — Zenoh-native web dashboard for all Robot() instances.

Every Robot("so100"), Robot("panda"), etc. auto-joins the Zenoh mesh.
This dashboard subscribes to ALL mesh traffic and renders it in a
real-time web UI:

- Live peer discovery (robots, sims, agents)
- Joint state visualization (sparklines + gauges)
- VLA execution stream (watch policy steps live)
- Camera feeds (JPEG frames from sim/real cameras over Zenoh)
- Action vector visualization (live bar charts)
- Command dispatch (tell any robot to do something)
- Emergency stop (broadcast to all)
- Agent grid layout — switch between peers, see everything at once

Architecture:
    Robot() ──► Zenoh mesh ◄── Dashboard (this)
    Robot() ──►              ◄── Agent
    Sim()   ──►              ◄── Other dashboards

Usage:
    python -m strands_robots.dashboard.server --port 7860
    # Opens http://localhost:7860

    # Or from Python:
    from strands_robots.dashboard.server import start_dashboard
    start_dashboard(port=7860)
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State — shared between Zenoh subscribers and WebSocket clients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_STATE: Dict[str, Any] = {
    "peers": {},           # peer_id → {type, hostname, last_seen, caps, ...}
    "states": {},          # peer_id → {joints, sim_time, task, ...}
    "streams": {},         # peer_id → [last N stream steps]
    "cameras": {},         # peer_id → {camera_name: base64_jpeg}
    "commands_sent": [],   # [{target, cmd, ts}, ...]
    "responses": [],       # [{from, turn, result, ts}, ...]
    "events": [],          # [{ts, type, data}, ...] ring buffer
}
_STATE_LOCK = threading.Lock()
_WS_CLIENTS: List[Any] = []
_WS_LOCK = threading.Lock()
_ZENOH_SESSION = None
_ZENOH_SUBS = []
_DASHBOARD_PEER_ID = f"dashboard-{uuid.uuid4().hex[:6]}"

MAX_EVENTS = 500
MAX_STREAM_STEPS = 200


def _add_event(event_type: str, data: dict):
    """Add event to the ring buffer."""
    with _STATE_LOCK:
        _STATE["events"].append({
            "ts": time.time(),
            "type": event_type,
            "data": data,
        })
        if len(_STATE["events"]) > MAX_EVENTS:
            _STATE["events"] = _STATE["events"][-MAX_EVENTS:]
    _broadcast_ws({"type": "event", "event_type": event_type, "data": data, "ts": time.time()})


def _broadcast_ws(msg: dict):
    """Send message to all connected WebSocket clients."""
    payload = json.dumps(msg, default=str)
    with _WS_LOCK:
        dead = []
        for ws in _WS_CLIENTS:
            try:
                ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _WS_CLIENTS.remove(ws)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zenoh Mesh Subscriber — listens to everything on strands/**
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _start_zenoh():
    """Connect to the Zenoh mesh and subscribe to all robot traffic."""
    global _ZENOH_SESSION

    try:
        import zenoh
    except ImportError:
        logger.warning("eclipse-zenoh not installed. Dashboard runs without live mesh data.")
        logger.warning("Install: pip install eclipse-zenoh")
        return False

    try:
        config = zenoh.Config()

        MESH_PORT = int(os.getenv("STRANDS_MESH_PORT", "7447"))
        MESH_EP = f"tcp/127.0.0.1:{MESH_PORT}"

        connect = os.getenv("ZENOH_CONNECT")
        listen = os.getenv("ZENOH_LISTEN")

        if connect:
            config.insert_json5("connect/endpoints", json.dumps([e.strip() for e in connect.split(",")]))
        if listen:
            config.insert_json5("listen/endpoints", json.dumps([e.strip() for e in listen.split(",")]))

        if not connect and not listen:
            try:
                cfg_try = zenoh.Config()
                cfg_try.insert_json5("listen/endpoints", json.dumps([MESH_EP]))
                cfg_try.insert_json5("connect/endpoints", json.dumps([MESH_EP]))
                _ZENOH_SESSION = zenoh.open(cfg_try)
                logger.info("Zenoh dashboard session opened (listener on %s)", MESH_EP)
            except Exception:
                config.insert_json5("mode", '"client"')
                config.insert_json5("connect/endpoints", json.dumps([MESH_EP]))
                _ZENOH_SESSION = zenoh.open(config)
                logger.info("Zenoh dashboard session opened (client)")
        else:
            _ZENOH_SESSION = zenoh.open(config)
            logger.info("Zenoh dashboard session opened")

        # Subscribe to all strands topics
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("strands/*/presence", _on_presence))
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("strands/*/state", _on_state))
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("strands/*/stream", _on_stream))
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("strands/*/camera", _on_camera))
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("strands/*/event", _on_sim_event))
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("strands/broadcast", _on_broadcast))
        _ZENOH_SUBS.append(_ZENOH_SESSION.declare_subscriber("reachy_mini/**", _on_external))

        # Publish dashboard presence
        threading.Thread(target=_dashboard_heartbeat, daemon=True).start()

        logger.info("Zenoh subscriptions active — listening for robots")
        return True

    except Exception as e:
        logger.error("Zenoh setup failed: %s", e)
        return False


def _dashboard_heartbeat():
    """Publish dashboard presence on the mesh."""
    while _ZENOH_SESSION:
        try:
            _ZENOH_SESSION.put(
                f"strands/{_DASHBOARD_PEER_ID}/presence",
                json.dumps({
                    "robot_id": _DASHBOARD_PEER_ID,
                    "robot_type": "dashboard",
                    "hostname": socket.gethostname(),
                    "timestamp": time.time(),
                    "ws_clients": len(_WS_CLIENTS),
                }).encode(),
            )
        except Exception:
            pass
        time.sleep(2.0)


def _on_presence(sample):
    """Handle presence heartbeats from robots/sims."""
    try:
        data = json.loads(sample.payload.to_bytes().decode())
        peer_id = data.get("robot_id", "")
        if not peer_id or peer_id == _DASHBOARD_PEER_ID:
            return

        with _STATE_LOCK:
            is_new = peer_id not in _STATE["peers"]
            _STATE["peers"][peer_id] = {
                **data,
                "last_seen": time.time(),
            }

        if is_new:
            _add_event("peer_join", {"peer_id": peer_id, **data})
            logger.info("🤖 New peer: %s (%s)", peer_id, data.get("robot_type", "?"))

            # Auto-request camera stream from sim and robot peers
            role = data.get("role", data.get("robot_type", ""))
            if role in ("sim", "robot"):
                threading.Thread(
                    target=lambda pid=peer_id: (
                        time.sleep(1),  # Wait for peer to be fully ready
                        _send_command(pid, {"action": "start_camera_stream", "camera": "default", "fps": 10 if role == "sim" else 5}),
                        logger.info("📹 Auto-requested camera stream from %s", pid),
                    ),
                    daemon=True,
                ).start()

        _broadcast_ws({"type": "peers", "peers": _get_peers()})

    except Exception as e:
        logger.debug("Presence parse error: %s", e)


def _on_state(sample):
    """Handle state updates (joint positions, sim time, task status)."""
    try:
        data = json.loads(sample.payload.to_bytes().decode())
        peer_id = data.get("peer_id", "")
        if not peer_id:
            return

        with _STATE_LOCK:
            _STATE["states"][peer_id] = {
                **data,
                "_received_at": time.time(),
            }

        _broadcast_ws({"type": "state", "peer_id": peer_id, "state": data})

    except Exception as e:
        logger.debug("State parse error: %s", e)


def _on_stream(sample):
    """Handle VLA execution stream steps."""
    try:
        data = json.loads(sample.payload.to_bytes().decode())
        peer_id = data.get("peer_id", "")
        if not peer_id:
            return

        with _STATE_LOCK:
            if peer_id not in _STATE["streams"]:
                _STATE["streams"][peer_id] = []
            _STATE["streams"][peer_id].append(data)
            if len(_STATE["streams"][peer_id]) > MAX_STREAM_STEPS:
                _STATE["streams"][peer_id] = _STATE["streams"][peer_id][-MAX_STREAM_STEPS:]

        _broadcast_ws({"type": "stream_step", "peer_id": peer_id, "step": data})

    except Exception as e:
        logger.debug("Stream parse error: %s", e)


def _on_camera(sample):
    """Handle camera frame data (base64 JPEG from sim/real cameras)."""
    try:
        data = json.loads(sample.payload.to_bytes().decode())
        peer_id = data.get("peer_id", "")
        cam_name = data.get("camera", "default")
        frame_b64 = data.get("frame", "")
        if not peer_id or not frame_b64:
            return

        with _STATE_LOCK:
            if peer_id not in _STATE["cameras"]:
                _STATE["cameras"][peer_id] = {}
            _STATE["cameras"][peer_id][cam_name] = frame_b64

        # Forward camera frame to WS clients that are watching this peer
        _broadcast_ws({
            "type": "camera_frame",
            "peer_id": peer_id,
            "camera": cam_name,
            "frame": frame_b64,
            "ts": data.get("ts", time.time()),
        })

    except Exception as e:
        logger.debug("Camera parse error: %s", e)


def _on_broadcast(sample):
    """Handle broadcast messages (e.g., emergency stop)."""
    try:
        data = json.loads(sample.payload.to_bytes().decode())
        _add_event("broadcast", data)
    except Exception:
        pass


def _on_sim_event(sample):
    """Handle discrete simulation events (object added/removed, policy started, etc.)."""
    try:
        data = json.loads(sample.payload.to_bytes().decode())
        peer_id = data.get("peer_id", "")
        event_type = data.get("event_type", "")
        event_data = data.get("data", {})
        if not peer_id or not event_type:
            return

        _add_event(f"sim:{event_type}", {"peer_id": peer_id, **event_data})

        # Forward to WS clients as a dedicated message type
        _broadcast_ws({
            "type": "sim_event",
            "peer_id": peer_id,
            "event_type": event_type,
            "data": event_data,
            "ts": data.get("ts", time.time()),
        })

    except Exception as e:
        logger.debug("Sim event parse error: %s", e)


def _on_external(sample):
    """Handle external robot topics (Reachy Mini, etc.)."""
    try:
        key = str(sample.key_expr)
        raw = sample.payload.to_bytes().decode()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw}

        _add_event("external", {"topic": key, "data": data})
    except Exception:
        pass


def _get_peers() -> list:
    """Get all peers with age calculation."""
    now = time.time()
    with _STATE_LOCK:
        peers = []
        stale = []
        for pid, info in _STATE["peers"].items():
            age = now - info.get("last_seen", 0)
            if age > 15.0:
                stale.append(pid)
                continue
            peers.append({
                "peer_id": pid,
                "type": info.get("robot_type", "unknown"),
                "hostname": info.get("hostname", "?"),
                "age": round(age, 1),
                "task_status": info.get("task_status", ""),
                "instruction": info.get("instruction", ""),
                "tool_name": info.get("tool_name", ""),
                "connected": info.get("connected", None),
                "sim_robots": info.get("sim_robots", []),
                "world": info.get("world", False),
                "action_keys": info.get("action_keys", []),
            })
        for pid in stale:
            del _STATE["peers"][pid]
            _add_event("peer_leave", {"peer_id": pid})
        return peers


def _send_command(target: str, command: dict):
    """Send a command to a peer via Zenoh."""
    if not _ZENOH_SESSION:
        return {"error": "Zenoh not connected"}

    turn_id = uuid.uuid4().hex[:8]
    msg = {
        "sender_id": _DASHBOARD_PEER_ID,
        "turn_id": turn_id,
        "command": command,
        "timestamp": time.time(),
    }

    try:
        if target == "broadcast":
            _ZENOH_SESSION.put("strands/broadcast", json.dumps(msg).encode())
        else:
            _ZENOH_SESSION.put(f"strands/{target}/cmd", json.dumps(msg).encode())

        with _STATE_LOCK:
            _STATE["commands_sent"].append({
                "target": target,
                "command": command,
                "turn_id": turn_id,
                "ts": time.time(),
            })
            if len(_STATE["commands_sent"]) > 100:
                _STATE["commands_sent"] = _STATE["commands_sent"][-100:]

        _add_event("command_sent", {"target": target, "command": command, "turn_id": turn_id})
        return {"status": "sent", "turn_id": turn_id}

    except Exception as e:
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocket Handler (using websockets library)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _ws_handler(websocket):
    """Handle a single WebSocket client connection."""
    class WSWrapper:
        def __init__(self, ws):
            self._ws = ws
            self._loop = asyncio.get_event_loop()

        def send(self, data):
            asyncio.run_coroutine_threadsafe(self._ws.send(data), self._loop)

    wrapper = WSWrapper(websocket)
    with _WS_LOCK:
        _WS_CLIENTS.append(wrapper)

    logger.info("WebSocket client connected (%d total)", len(_WS_CLIENTS))

    try:
        # Send initial state
        await websocket.send(json.dumps({
            "type": "init",
            "peers": _get_peers(),
            "dashboard_id": _DASHBOARD_PEER_ID,
            "states": {k: v for k, v in _STATE.get("states", {}).items()},
            "events": _STATE.get("events", [])[-50:],
        }, default=str))

        async for message in websocket:
            try:
                msg = json.loads(message)
                msg_type = msg.get("type", "")

                if msg_type == "command":
                    target = msg.get("target", "")
                    command = msg.get("command", {})
                    result = _send_command(target, command)
                    await websocket.send(json.dumps({"type": "command_result", **result}, default=str))

                elif msg_type == "tell":
                    target = msg.get("target", "")
                    instruction = msg.get("instruction", "")
                    policy = msg.get("policy_provider", "mock")
                    duration = msg.get("duration", 30.0)

                    # Handle special commands
                    if instruction == "__stop__" or msg.get("stop"):
                        result = _send_command(target, {"action": "stop"})
                        await websocket.send(json.dumps({"type": "command_result", "action": "stop", **result}, default=str))
                    elif instruction == "__teleop_start__" or msg.get("teleop"):
                        cmd = {
                            "action": "teleop_start",
                            "policy_provider": policy,
                            "duration": duration,
                        }
                        for k in ("pretrained_name_or_path", "model_path", "server_address", "policy_type"):
                            if k in msg:
                                cmd[k] = msg[k]
                        result = _send_command(target, cmd)
                        await websocket.send(json.dumps({"type": "command_result", "action": "teleop_start", **result}, default=str))
                    elif instruction == "__teleop_stop__" or msg.get("teleop_stop"):
                        result = _send_command(target, {"action": "teleop_stop"})
                        await websocket.send(json.dumps({"type": "command_result", "action": "teleop_stop", **result}, default=str))
                    elif instruction == "__teleop_delta__" and msg.get("teleop_delta"):
                        delta = msg["teleop_delta"]
                        result = _send_command(target, {"action": "teleop_delta", **delta})
                        # Don't send ack for delta (too noisy)
                    else:
                        cmd = {
                            "action": "execute",
                            "instruction": instruction,
                            "policy_provider": policy,
                            "duration": duration,
                        }
                        # Forward extra policy kwargs
                        for k in ("pretrained_name_or_path", "model_path", "server_address", "policy_type"):
                            if k in msg:
                                cmd[k] = msg[k]
                        result = _send_command(target, cmd)
                        await websocket.send(json.dumps({"type": "command_result", **result}, default=str))

                elif msg_type == "emergency_stop":
                    result = _send_command("broadcast", {"action": "stop"})
                    await websocket.send(json.dumps({"type": "command_result", "action": "emergency_stop", **result}, default=str))

                elif msg_type == "get_peers":
                    await websocket.send(json.dumps({"type": "peers", "peers": _get_peers()}, default=str))

                elif msg_type == "get_state":
                    peer_id = msg.get("peer_id", "")
                    state = _STATE.get("states", {}).get(peer_id, {})
                    await websocket.send(json.dumps({"type": "state", "peer_id": peer_id, "state": state}, default=str))

                elif msg_type == "get_stream":
                    peer_id = msg.get("peer_id", "")
                    steps = _STATE.get("streams", {}).get(peer_id, [])
                    await websocket.send(json.dumps({"type": "stream_history", "peer_id": peer_id, "steps": steps[-50:]}, default=str))

                elif msg_type == "get_camera":
                    peer_id = msg.get("peer_id", "")
                    cams = _STATE.get("cameras", {}).get(peer_id, {})
                    await websocket.send(json.dumps({"type": "camera_state", "peer_id": peer_id, "cameras": cams}, default=str))

                elif msg_type == "request_camera_stream":
                    # Ask a sim peer to start publishing camera frames
                    target = msg.get("peer_id", "")
                    cam = msg.get("camera", "default")
                    fps = msg.get("fps", 10)
                    result = _send_command(target, {
                        "action": "start_camera_stream",
                        "camera": cam,
                        "fps": fps,
                    })
                    await websocket.send(json.dumps({"type": "command_result", **result}, default=str))

                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except Exception:
        pass
    finally:
        with _WS_LOCK:
            if wrapper in _WS_CLIENTS:
                _WS_CLIENTS.remove(wrapper)
        logger.info("WebSocket client disconnected (%d remaining)", len(_WS_CLIENTS))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML Dashboard UI — Full Agent Grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strands Robots Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#07070d;--surface:#0e0e18;--surface2:#151524;--surface3:#1c1c30;
  --border:#252540;--border2:#35355a;
  --text:#e8e8f0;--text2:#8888aa;--text3:#555570;
  --accent:#7c6cf0;--accent2:#a89afe;--accent-dim:rgba(124,108,240,0.15);
  --green:#00d4a0;--green-dim:rgba(0,212,160,0.15);
  --red:#ff5252;--red-dim:rgba(255,82,82,0.15);
  --orange:#ffb74d;--orange-dim:rgba(255,183,77,0.15);
  --blue:#64b5f6;--blue-dim:rgba(100,181,246,0.15);
  --cyan:#4dd0e1;--yellow:#ffd54f;
  --radius:8px;--radius-lg:12px;
}
body{font-family:'Inter','SF Pro',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow:hidden}

/* ── Header ── */
.header{background:var(--surface);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:14px;height:48px;z-index:100}
.header .logo{font-size:16px;font-weight:700;display:flex;align-items:center;gap:8px}
.header .logo span{background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .status{font-size:11px;color:var(--text3);display:flex;align-items:center;gap:4px}
.header .status .dot{width:6px;height:6px;border-radius:50%;background:var(--text3)}
.header .status.connected .dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.header .status.connected{color:var(--green)}
.header .peer-count{font-size:11px;color:var(--text2);background:var(--surface2);padding:3px 10px;border-radius:12px;border:1px solid var(--border)}
.header .view-toggle{display:flex;gap:2px;background:var(--surface2);border-radius:6px;padding:2px;border:1px solid var(--border);margin-left:auto}
.header .view-toggle button{background:none;border:none;color:var(--text3);padding:4px 12px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:500}
.header .view-toggle button.active{background:var(--accent-dim);color:var(--accent2)}
.header .estop{background:var(--red);color:white;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:700;font-size:12px;letter-spacing:0.5px;transition:all 0.15s}
.header .estop:hover{background:#ff3333;box-shadow:0 0 20px rgba(255,82,82,0.4)}

/* ── Layout ── */
.app{display:flex;height:calc(100vh - 48px)}
.sidebar{width:240px;background:var(--surface);border-right:1px solid var(--border);overflow-y:auto;flex-shrink:0;padding:12px}
.sidebar h3{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;padding:0 4px}
.main-area{flex:1;overflow:auto;padding:16px}
.event-sidebar{width:280px;background:var(--surface);border-left:1px solid var(--border);overflow-y:auto;flex-shrink:0;padding:12px}
.event-sidebar h3{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px}

/* ── Peer Cards (sidebar) ── */
.peer-item{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:10px;margin-bottom:6px;cursor:pointer;transition:all 0.15s}
.peer-item:hover{border-color:var(--border2);background:var(--surface3)}
.peer-item.selected{border-color:var(--accent);background:var(--accent-dim)}
.peer-item .top{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.peer-item .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.peer-item .dot.alive{background:var(--green);box-shadow:0 0 4px var(--green)}
.peer-item .dot.stale{background:var(--orange)}
.peer-item .name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.peer-item .badge{font-size:9px;padding:1px 6px;border-radius:4px;font-weight:600;flex-shrink:0}
.peer-item .badge.sim{background:var(--blue-dim);color:var(--blue)}
.peer-item .badge.robot{background:var(--green-dim);color:var(--green)}
.peer-item .badge.agent{background:var(--accent-dim);color:var(--accent2)}
.peer-item .badge.dashboard{background:var(--surface3);color:var(--text3)}
.peer-item .meta{font-size:10px;color:var(--text3);display:flex;gap:8px}
.peer-item .task{font-size:10px;color:var(--orange);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ── Grid View ── */
.grid-view{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.grid-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden;display:flex;flex-direction:column}
.grid-card .card-header{padding:10px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;background:var(--surface2)}
.grid-card .card-header .name{font-size:13px;font-weight:600;flex:1}
.grid-card .card-header .pill{font-size:9px;padding:2px 8px;border-radius:10px;font-weight:600}
.grid-card .card-body{display:grid;grid-template-columns:1fr 1fr;gap:0;flex:1;min-height:0}

/* ── Camera Feed ── */
.camera-panel{background:#000;position:relative;min-height:180px;display:flex;align-items:center;justify-content:center;border-right:1px solid var(--border)}
.camera-panel img{max-width:100%;max-height:100%;object-fit:contain}
.camera-panel .no-feed{color:var(--text3);font-size:11px;text-align:center}
.camera-panel .no-feed .icon{font-size:28px;margin-bottom:6px;opacity:0.3}
.camera-panel .fps-badge{position:absolute;top:6px;right:6px;background:rgba(0,0,0,0.7);color:var(--green);font-size:9px;padding:2px 6px;border-radius:4px;font-family:monospace}

/* ── Joint State Panel ── */
.joints-panel{padding:10px;overflow-y:auto;max-height:300px}
.joints-panel h4{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.joint-row{display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:11px}
.joint-row .jname{color:var(--text2);min-width:70px;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.joint-row .jbar{flex:1;height:4px;background:var(--surface3);border-radius:2px;overflow:hidden;position:relative}
.joint-row .jbar .fill{height:100%;border-radius:2px;transition:width 0.1s ease}
.joint-row .jval{color:var(--accent2);font-family:monospace;font-size:10px;min-width:50px;text-align:right}

/* ── Action Vector Panel ── */
.action-panel{padding:10px;border-top:1px solid var(--border);grid-column:1/-1}
.action-panel h4{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.action-bars{display:flex;gap:3px;align-items:end;height:40px}
.action-bar{flex:1;background:var(--accent-dim);border-radius:2px 2px 0 0;transition:height 0.1s;position:relative;min-width:0}
.action-bar .tip{position:absolute;top:-14px;left:50%;transform:translateX(-50%);font-size:8px;color:var(--text3);white-space:nowrap;display:none}
.action-bars:hover .action-bar .tip{display:block}

/* ── Stream Panel ── */
.stream-panel{padding:10px;border-top:1px solid var(--border);grid-column:1/-1;max-height:150px;overflow-y:auto}
.stream-panel h4{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;display:flex;align-items:center;gap:6px}
.stream-panel h4 .live{width:6px;height:6px;border-radius:50%;background:var(--red);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.stream-line{font-size:10px;font-family:monospace;padding:2px 0;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stream-line .step{color:var(--accent);margin-right:6px}
.stream-line .instr{color:var(--orange)}

/* ── Detail View ── */
.detail-view{display:grid;grid-template-columns:1fr 340px;gap:16px;height:100%}
.detail-main{display:flex;flex-direction:column;gap:12px;overflow-y:auto}
.detail-sidebar{display:flex;flex-direction:column;gap:12px;overflow-y:auto}
.detail-camera{background:#000;border-radius:var(--radius-lg);min-height:300px;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden}
.detail-camera img{max-width:100%;max-height:100%;object-fit:contain}
.detail-joints{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:14px}
.detail-actions{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:14px}
.detail-stream{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:14px;max-height:300px;overflow-y:auto}

/* ── Command Bar ── */
.cmd-bar{padding:12px 14px;border-top:1px solid var(--border);grid-column:1/-1;background:var(--surface2);display:flex;flex-direction:column;gap:8px}
.cmd-bar .cmd-row{display:flex;gap:6px;align-items:center}
.cmd-bar select{background:var(--surface3);border:1px solid var(--border);color:var(--text);padding:6px 8px;border-radius:var(--radius);font-size:11px;cursor:pointer}
.cmd-bar input{flex:1;background:var(--surface3);border:1px solid var(--border);color:var(--text);padding:8px 10px;border-radius:var(--radius);font-size:12px;font-family:inherit}
.cmd-bar input:focus{outline:none;border-color:var(--accent)}
.cmd-bar input::placeholder{color:var(--text3)}
.cmd-bar .btn{border:none;padding:7px 14px;border-radius:var(--radius);cursor:pointer;font-size:11px;font-weight:600;transition:all 0.15s;white-space:nowrap;display:flex;align-items:center;gap:4px}
.cmd-bar .btn-primary{background:var(--accent);color:white}
.cmd-bar .btn-primary:hover{background:var(--accent2)}
.cmd-bar .btn-green{background:var(--green);color:#111}
.cmd-bar .btn-green:hover{background:#00e8b0;box-shadow:0 0 12px rgba(0,212,160,0.3)}
.cmd-bar .btn-orange{background:var(--orange);color:#111}
.cmd-bar .btn-orange:hover{background:#ffc76e}
.cmd-bar .btn-red{background:var(--red-dim);color:var(--red);border:1px solid var(--red)}
.cmd-bar .btn-red:hover{background:var(--red);color:white}
.cmd-bar .cmd-label{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;font-weight:600}

/* ── Events ── */
.evt{font-size:10px;padding:5px 0;border-bottom:1px solid rgba(37,37,64,0.5);line-height:1.4}
.evt .ts{color:var(--text3);margin-right:4px;font-family:monospace;font-size:9px}
.evt .tag{font-weight:600;margin-right:3px;font-size:9px}
.evt .tag.peer_join{color:var(--green)}.evt .tag.peer_leave{color:var(--red)}
.evt .tag.command_sent{color:var(--blue)}.evt .tag.broadcast{color:var(--orange)}
.evt .tag.stream_step{color:var(--accent2)}.evt .tag.external{color:var(--cyan)}
.evt .tag[class*="sim\:"]{color:var(--green)}

.no-data{color:var(--text3);font-size:12px;text-align:center;padding:40px 20px}
.no-data .big{font-size:32px;margin-bottom:8px;opacity:0.2}

/* ── Sparkline canvas ── */
.sparkline{width:100%;height:24px}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

@media(max-width:900px){.sidebar,.event-sidebar{display:none}.grid-view{grid-template-columns:1fr}}
.hf-search-wrap{position:relative;flex:0.8;min-width:160px}
.hf-search-wrap input{width:100%;box-sizing:border-box}
.hf-dropdown{position:absolute;top:100%;left:0;right:0;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);max-height:280px;overflow-y:auto;z-index:1000;display:none;box-shadow:0 8px 24px rgba(0,0,0,0.4)}
.hf-dropdown.open{display:block}
.hf-item{padding:8px 10px;cursor:pointer;border-bottom:1px solid var(--border);font-size:11px;transition:background 0.1s}
.hf-item:hover,.hf-item.selected{background:var(--surface3)}
.hf-item:last-child{border-bottom:none}
.hf-item .hf-name{color:var(--accent);font-weight:600;font-size:11px}
.hf-item .hf-meta{color:var(--text3);font-size:9px;margin-top:2px;display:flex;gap:8px}
.hf-item .hf-tag{background:var(--surface2);padding:1px 5px;border-radius:3px;font-size:8px;color:var(--text2)}
.hf-searching{padding:10px;color:var(--text3);font-size:11px;text-align:center}
</style>
</head>
<body>
<div class="header">
  <div class="logo">🤖 <span>Strands Robots</span></div>
  <div class="status" id="ws-status"><div class="dot"></div>connecting</div>
  <div class="peer-count" id="peer-count">0 peers</div>
  <div class="view-toggle">
    <button class="active" onclick="setView('grid')" id="btn-grid">Grid</button>
    <button onclick="setView('detail')" id="btn-detail">Detail</button>
  </div>
  <button class="estop" onclick="emergencyStop()">⛔ E-STOP</button>
</div>

<div class="app">
  <div class="sidebar">
    <h3>Peers</h3>
    <div id="peers-list"><div class="no-data"><div class="big">📡</div>Waiting for robots...</div></div>
  </div>
  <div class="main-area" id="main-area">
    <div class="grid-view" id="grid-view"></div>
    <div class="detail-view" id="detail-view" style="display:none"></div>
    <div id="empty-state" class="no-data"><div class="big">🤖</div>Create a Robot() to see it here<br><code style="color:var(--accent);font-size:11px;margin-top:8px;display:inline-block">from strands_robots import Robot; sim = Robot("so100")</code></div>
  </div>
  <div class="event-sidebar">
    <h3>Event Log</h3>
    <div id="events-list"></div>
  </div>
</div>

<script>
// ━━━ State ━━━
let ws, selectedPeer = null, currentView = 'grid', reconnectTimer = null;
const peers = new Map(); // peer_id → peer info
const states = new Map(); // peer_id → latest state
const streams = new Map(); // peer_id → [steps]
const jointHistory = new Map(); // peer_id → {joint_name → [values]}
const cameraFrames = new Map(); // peer_id → {cam → dataurl}
const lastActions = new Map(); // peer_id → {action dict}
const WS_PORT = location.port ? parseInt(location.port) + 1 : 7861;
const MAX_HISTORY = 60; // sparkline points

// ━━━ Connection ━━━
function connect() {
  ws = new WebSocket(`ws://${location.hostname}:${WS_PORT}`);
  ws.onopen = () => {
    document.getElementById('ws-status').innerHTML = '<div class="dot"></div>connected';
    document.getElementById('ws-status').className = 'status connected';
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };
  ws.onclose = () => {
    document.getElementById('ws-status').innerHTML = '<div class="dot"></div>disconnected';
    document.getElementById('ws-status').className = 'status';
    reconnectTimer = setTimeout(connect, 2000);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    switch(msg.type) {
      case 'init': handleInit(msg); break;
      case 'peers': handlePeers(msg.peers); break;
      case 'state': handleState(msg.peer_id, msg.state); break;
      case 'stream_step': handleStreamStep(msg.peer_id, msg.step); break;
      case 'camera_frame': handleCameraFrame(msg.peer_id, msg.camera, msg.frame); break;
      case 'sim_event': handleSimEvent(msg); break;
      case 'event': addEvent(msg); break;
    }
  };
}

function handleInit(msg) {
  handlePeers(msg.peers);
  if (msg.states) Object.entries(msg.states).forEach(([pid, s]) => handleState(pid, s));
  if (msg.events) msg.events.forEach(e => addEvent({event_type: e.type, data: e.data, ts: e.ts}));
}

// ━━━ Peer Management ━━━
const cameraRequested = new Set(); // track which peers we already asked for camera

function handlePeers(list) {
  const ids = new Set();
  list.forEach(p => { peers.set(p.peer_id, p); ids.add(p.peer_id); });
  // Remove stale
  for (const pid of peers.keys()) { if (!ids.has(pid)) { peers.delete(pid); cameraRequested.delete(pid); } }
  document.getElementById('peer-count').textContent = `${peers.size} peer${peers.size!==1?'s':''}`;
  // Skip full re-render if user is focused on an input/select (prevents losing focus & typed text)
  const ae = document.activeElement;
  const userBusy = ae && (ae.tagName === 'INPUT' || ae.tagName === 'SELECT' || ae.tagName === 'TEXTAREA' || ae.closest('.hf-dropdown.open'));
  if (!userBusy) {
    renderSidebar();
    renderMain();
  } else {
    // Still update peer count badge in sidebar without full rebuild
    document.querySelectorAll('.peer-item .dot').forEach(dot => {
      const item = dot.closest('.peer-item');
      if (!item) return;
      const pid = item.getAttribute('onclick')?.match(/'([^']+)'/)?.[1];
      const p = pid && peers.get(pid);
      if (p) dot.className = 'dot ' + (p.age < 5 ? 'alive' : 'stale');
    });
  }

  // Auto-request camera stream from sim and robot peers that we haven't asked yet
  list.forEach(p => {
    if ((p.type === 'sim' || p.type === 'robot') && !cameraRequested.has(p.peer_id)) {
      cameraRequested.add(p.peer_id);
      const fps = p.type === 'sim' ? 10 : 5;
      ws.send(JSON.stringify({type: 'request_camera_stream', peer_id: p.peer_id, camera: 'default', fps: fps}));
      console.log('📹 Auto-requested camera stream from', p.peer_id);
    }
  });
}

// ━━━ State Handling ━━━
function handleState(peerId, state) {
  states.set(peerId, state);
  // Track joint history for sparklines
  const joints = state.joints || {};
  if (!jointHistory.has(peerId)) jointHistory.set(peerId, {});
  const hist = jointHistory.get(peerId);
  for (const [k, v] of Object.entries(joints)) {
    const val = typeof v === 'object' ? (v.position ?? v) : v;
    if (typeof val !== 'number') continue;
    if (!hist[k]) hist[k] = [];
    hist[k].push(val);
    if (hist[k].length > MAX_HISTORY) hist[k].shift();
  }
  updateGridCard(peerId);
  const ae2 = document.activeElement;
  const busy2 = ae2 && (ae2.tagName === 'INPUT' || ae2.tagName === 'SELECT' || ae2.closest('.hf-dropdown.open'));
  if (currentView === 'detail' && selectedPeer === peerId && !busy2) renderDetailView();
}

function handleStreamStep(peerId, step) {
  if (!streams.has(peerId)) streams.set(peerId, []);
  const s = streams.get(peerId);
  s.push(step);
  if (s.length > MAX_STREAM_STEPS) s.splice(0, s.length - MAX_STREAM_STEPS);
  // Track last action
  if (step.action) lastActions.set(peerId, step.action);
  updateStreamPanel(peerId);
  updateActionPanel(peerId);
}

function handleCameraFrame(peerId, camera, frame) {
  if (!cameraFrames.has(peerId)) cameraFrames.set(peerId, {});
  cameraFrames.get(peerId)[camera] = 'data:image/jpeg;base64,' + frame;
  updateCameraPanel(peerId);
}

// ━━━ Views ━━━
function setView(v) {
  currentView = v;
  document.getElementById('btn-grid').classList.toggle('active', v === 'grid');
  document.getElementById('btn-detail').classList.toggle('active', v === 'detail');
  document.getElementById('grid-view').style.display = v === 'grid' ? '' : 'none';
  document.getElementById('detail-view').style.display = v === 'detail' ? '' : 'none';
  renderMain();
}

function selectPeer(pid) {
  selectedPeer = pid;
  renderSidebar();
  if (currentView === 'detail') renderDetailView();
  // Request state+stream
  if (ws?.readyState === 1) {
    ws.send(JSON.stringify({type: 'get_state', peer_id: pid}));
    ws.send(JSON.stringify({type: 'get_stream', peer_id: pid}));
  }
}

// ━━━ Sidebar ━━━
function renderSidebar() {
  const el = document.getElementById('peers-list');
  if (!peers.size) { el.innerHTML = '<div class="no-data"><div class="big">📡</div>Waiting for robots...</div>'; return; }
  el.innerHTML = [...peers.values()].map(p => {
    const icon = {robot:'🤖',sim:'🎮',agent:'🧠',dashboard:'📊'}[p.type]||'🔧';
    const dotCls = p.age < 5 ? 'alive' : 'stale';
    const sel = p.peer_id === selectedPeer ? ' selected' : '';
    const badgeCls = p.type || 'robot';
    const task = p.instruction ? `<div class="task">🎯 ${p.instruction}</div>` : '';
    return `<div class="peer-item${sel}" onclick="selectPeer('${p.peer_id}')">
      <div class="top"><div class="dot ${dotCls}"></div><div class="name">${icon} ${p.peer_id}</div><div class="badge ${badgeCls}">${p.type}</div></div>
      <div class="meta"><span>${p.hostname}</span><span>${p.age}s</span></div>
      ${task}
    </div>`;
  }).join('');
}

// ━━━ Grid View ━━━
function renderMain() {
  const empty = document.getElementById('empty-state');
  if (!peers.size) { empty.style.display = ''; return; }
  empty.style.display = 'none';
  if (currentView === 'grid') renderGridView();
  else renderDetailView();
}

function renderGridView() {
  const el = document.getElementById('grid-view');
  // Only re-render structure if peer count changed
  const existingIds = new Set(el.querySelectorAll('.grid-card').length ? [...el.querySelectorAll('.grid-card')].map(c => c.dataset.peer) : []);
  const currentIds = new Set(peers.keys());
  const needsRebuild = existingIds.size !== currentIds.size || [...currentIds].some(id => !existingIds.has(id));

  if (needsRebuild) {
    el.innerHTML = [...peers.values()].filter(p => p.type !== 'dashboard').map(p => {
      const icon = {robot:'🤖',sim:'🎮',agent:'🧠'}[p.type]||'🔧';
      return `<div class="grid-card" data-peer="${p.peer_id}">
        <div class="card-header">
          <div class="name">${icon} ${p.peer_id}</div>
          <div class="pill" style="background:var(--${p.type==='sim'?'blue':'green'}-dim);color:var(--${p.type==='sim'?'blue':'green'})">${p.type}</div>
        </div>
        <div class="card-body">
          <div class="camera-panel" id="cam-${CSS.escape(p.peer_id)}">
            <div class="no-feed"><div class="icon">📷</div>No camera feed</div>
          </div>
          <div class="joints-panel" id="joints-${CSS.escape(p.peer_id)}">
            <h4>Joint States</h4>
            <div class="no-data" style="padding:20px;font-size:11px">Waiting for data...</div>
          </div>
          <div class="action-panel" id="actions-${CSS.escape(p.peer_id)}">
            <h4>Actions</h4>
            <div class="action-bars" id="abars-${CSS.escape(p.peer_id)}"></div>
          </div>
          <div class="stream-panel" id="stream-${CSS.escape(p.peer_id)}">
            <h4><div class="live"></div>Policy Stream</h4>
          </div>
          <div class="action-panel" id="objects-${CSS.escape(p.peer_id)}" style="padding:10px;border-top:1px solid var(--border);grid-column:1/-1">
            <h4>Scene Objects</h4>
          </div>
          <div class="cmd-bar">
            <div class="cmd-row">
              <span class="cmd-label">Model</span>
              <div class="hf-search-wrap" style="flex:1">
                <input type="text" id="model-${CSS.escape(p.peer_id)}" placeholder="🔍 Search HuggingFace policies..." value="lerobot/act_aloha_sim_transfer_cube_human" autocomplete="off" onfocus="hfSearchFocus(this)" oninput="hfSearch(this)" onkeydown="hfKeydown(event,this)">
                <div class="hf-dropdown" id="hf-dd-${CSS.escape(p.peer_id)}"></div>
              </div>
            </div>
            <div class="cmd-row">
              <span class="cmd-label">Action</span>
              <select id="policy-${CSS.escape(p.peer_id)}">
                <option value="mock">mock</option>
                <option value="lerobot_local" selected>lerobot_local</option>
                <option value="groot">groot</option>
                <option value="lerobot_async">lerobot_async</option>
              </select>
              <input type="text" id="cmd-${CSS.escape(p.peer_id)}" placeholder="Instruction (e.g. pick up the cube)" style="flex:1" onkeydown="if(event.key==='Enter')sendTo('${p.peer_id}')">
              <button class="btn btn-green" onclick="sendTo('${p.peer_id}')">▶ Run</button>
              <button class="btn btn-orange" onclick="startTeleop('${p.peer_id}')">🎮 Teleop</button>
              <button class="btn btn-red" onclick="stopPeer('${p.peer_id}')">⏹ Stop</button>
            </div>
          </div>
        </div>
      </div>`;
    }).join('');
  }

  // Update all cards
  for (const pid of peers.keys()) {
    updateGridCard(pid);
  }
}

function updateGridCard(peerId) {
  updateJointsPanel(peerId);
  updateCameraPanel(peerId);
  updateActionPanel(peerId);
  updateObjectsPanel(peerId);
}

// ━━━ Joint State Rendering ━━━
function updateJointsPanel(peerId) {
  const el = document.getElementById('joints-' + CSS.escape(peerId));
  if (!el) return;
  const state = states.get(peerId);
  if (!state?.joints) return;

  const joints = state.joints;
  const entries = Object.entries(joints);
  if (!entries.length) return;

  let html = '<h4>Joint States</h4>';
  for (const [name, v] of entries) {
    const val = typeof v === 'object' ? (v.position ?? 0) : (typeof v === 'number' ? v : 0);
    // Normalize to 0-100% for bar (assuming range roughly -3.14 to 3.14)
    const pct = Math.min(100, Math.max(0, ((val + 3.14) / 6.28) * 100));
    const color = val >= 0 ? 'var(--accent)' : 'var(--cyan)';
    html += `<div class="joint-row">
      <span class="jname" title="${name}">${name}</span>
      <div class="jbar"><div class="fill" style="width:${pct}%;background:${color}"></div></div>
      <span class="jval">${val.toFixed(3)}</span>
    </div>`;
  }
  el.innerHTML = html;
}

// ━━━ Camera Panel ━━━
function updateCameraPanel(peerId) {
  const el = document.getElementById('cam-' + CSS.escape(peerId));
  if (!el) return;
  const cams = cameraFrames.get(peerId);
  if (!cams) return;
  const firstCam = Object.values(cams)[0];
  if (firstCam) {
    el.innerHTML = `<img src="${firstCam}" alt="camera"><div class="fps-badge">LIVE</div>`;
  }
}

// ━━━ Action Vector Bars ━━━
function updateActionPanel(peerId) {
  const el = document.getElementById('abars-' + CSS.escape(peerId));
  if (!el) return;
  const action = lastActions.get(peerId);
  if (!action) return;

  const vals = Object.entries(action);
  if (!vals.length) return;

  el.innerHTML = vals.map(([k, v]) => {
    const val = Array.isArray(v) ? v[0] : (typeof v === 'number' ? v : 0);
    const h = Math.min(40, Math.max(2, Math.abs(val) * 400));
    const color = val >= 0 ? 'var(--accent)' : 'var(--cyan)';
    return `<div class="action-bar" style="height:${h}px;background:${color}"><div class="tip">${k}: ${typeof val==='number'?val.toFixed(3):val}</div></div>`;
  }).join('');
}

// ━━━ Stream Panel ━━━
function updateStreamPanel(peerId) {
  const el = document.getElementById('stream-' + CSS.escape(peerId));
  if (!el) return;
  const s = streams.get(peerId);
  if (!s || !s.length) return;

  const recent = s.slice(-8);
  let html = '<h4><div class="live"></div>Policy Stream</h4>';
  for (const step of recent) {
    const obsKeys = Object.keys(step.observation || {}).slice(0, 3).join(', ');
    html += `<div class="stream-line"><span class="step">#${step.step}</span><span class="instr">${step.instruction||''}</span> ${step.policy||''} [${obsKeys}]</div>`;
  }
  el.innerHTML = html;
}

// ━━━ Detail View ━━━
function renderDetailView() {
  const el = document.getElementById('detail-view');
  if (!selectedPeer) {
    el.innerHTML = '<div class="no-data"><div class="big">👈</div>Select a peer from the sidebar</div>';
    return;
  }
  const p = peers.get(selectedPeer);
  if (!p) return;

  const state = states.get(selectedPeer) || {};
  const action = lastActions.get(selectedPeer) || {};
  const s = streams.get(selectedPeer) || [];

  // Joints
  let jointsHtml = '';
  if (state.joints) {
    for (const [name, v] of Object.entries(state.joints)) {
      const val = typeof v === 'object' ? (v.position ?? 0) : (typeof v === 'number' ? v : 0);
      const pct = Math.min(100, Math.max(0, ((val + 3.14) / 6.28) * 100));
      jointsHtml += `<div class="joint-row">
        <span class="jname">${name}</span>
        <div class="jbar"><div class="fill" style="width:${pct}%;background:var(--accent)"></div></div>
        <span class="jval">${val.toFixed(4)}</span>
      </div>`;
    }
  }

  // Actions
  let actHtml = '';
  const actEntries = Object.entries(action);
  if (actEntries.length) {
    actHtml = '<div class="action-bars" style="height:50px">' + actEntries.map(([k, v]) => {
      const val = Array.isArray(v) ? v[0] : (typeof v === 'number' ? v : 0);
      const h = Math.min(50, Math.max(2, Math.abs(val) * 500));
      return `<div class="action-bar" style="height:${h}px;background:var(--accent)"><div class="tip">${k}: ${typeof val==='number'?val.toFixed(4):val}</div></div>`;
    }).join('') + '</div>';
  }

  // Stream
  let streamHtml = s.slice(-20).map(step =>
    `<div class="stream-line"><span class="step">#${step.step}</span><span class="instr">${step.instruction||''}</span> ${step.policy||''}</div>`
  ).join('');

  // Camera
  const cams = cameraFrames.get(selectedPeer);
  const camImg = cams ? Object.values(cams)[0] : null;
  const camHtml = camImg ? `<img src="${camImg}" alt="camera"><div class="fps-badge">LIVE</div>` : '<div class="no-feed"><div class="icon">📷</div>No camera feed</div>';

  el.innerHTML = `
    <div class="detail-main">
      <div class="detail-camera">${camHtml}</div>
      <div class="detail-actions" style="padding:14px">
        <h4 style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Last Action Vector</h4>
        ${actHtml || '<div style="color:var(--text3);font-size:11px">No actions yet</div>'}
      </div>
      <div class="cmd-bar" style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg)">
        <div class="cmd-row">
          <span class="cmd-label">Model</span>
          <div class="hf-search-wrap" style="flex:1">
            <input type="text" id="detail-model" placeholder="🔍 Search HuggingFace policies..." value="lerobot/act_aloha_sim_transfer_cube_human" autocomplete="off" onfocus="hfSearchFocus(this)" oninput="hfSearch(this)" onkeydown="hfKeydown(event,this)">
            <div class="hf-dropdown" id="hf-dd-detail"></div>
          </div>
        </div>
        <div class="cmd-row">
          <span class="cmd-label">Action</span>
          <select id="detail-policy">
            <option value="mock">mock</option><option value="lerobot_local" selected>lerobot_local</option>
            <option value="groot">groot</option><option value="lerobot_async">lerobot_async</option>
          </select>
          <input type="text" id="detail-cmd" placeholder="Instruction (e.g. pick up the cube)" style="flex:1" onkeydown="if(event.key==='Enter')sendToDetail()">
          <button class="btn btn-green" onclick="sendToDetail()">▶ Run</button>
          <button class="btn btn-orange" onclick="startTeleop(selectedPeer)">🎮 Teleop</button>
          <button class="btn btn-red" onclick="stopPeer(selectedPeer)">⏹ Stop</button>
        </div>
      </div>
    </div>
    <div class="detail-sidebar">
      <div class="detail-joints">
        <h4 style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Joint States ${state.sim_time !== undefined ? `· t=${state.sim_time.toFixed(3)}s`:''}</h4>
        ${jointsHtml || '<div style="color:var(--text3);font-size:11px">No joint data</div>'}
      </div>
      <div class="detail-stream">
        <h4 style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:flex;align-items:center;gap:6px"><div class="live"></div>Policy Stream</h4>
        ${streamHtml || '<div style="color:var(--text3);font-size:11px">No stream data</div>'}
      </div>
    </div>
  `;
}

// ━━━ Sim Events ━━━
const simObjects = new Map(); // peer_id → {name → {shape, position, color}}

function handleSimEvent(msg) {
  const {peer_id, event_type, data} = msg;
  addEvent({ts: msg.ts, event_type: `sim:${event_type}`, data: {peer_id, ...data}});

  // Flash the grid card to indicate a change
  const card = document.querySelector(`.grid-card[data-peer="${peer_id}"]`);
  if (card) {
    card.style.borderColor = 'var(--green)';
    card.style.boxShadow = '0 0 12px rgba(0,212,160,0.3)';
    setTimeout(() => { card.style.borderColor = ''; card.style.boxShadow = ''; }, 1500);
  }

  // Update objects panel immediately
  updateObjectsPanel(peer_id);
}

// ━━━ Objects Panel ━━━
function updateObjectsPanel(peerId) {
  const el = document.getElementById('objects-' + CSS.escape(peerId));
  if (!el) return;
  const state = states.get(peerId);
  const objs = state?.objects;
  if (!objs || !Object.keys(objs).length) {
    el.innerHTML = '<h4>Scene Objects</h4><div style="color:var(--text3);font-size:10px;padding:4px 0">No objects</div>';
    return;
  }
  let html = `<h4>Scene Objects <span style="color:var(--text2);font-weight:normal">(${Object.keys(objs).length})</span></h4>`;
  for (const [name, obj] of Object.entries(objs)) {
    const c = obj.color || [0.5, 0.5, 0.5];
    const rgb = `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
    const pos = (obj.position || []).map(v => v.toFixed(2)).join(', ');
    html += `<div style="display:flex;align-items:center;gap:5px;padding:2px 0;font-size:10px">
      <div style="width:8px;height:8px;border-radius:2px;background:${rgb};flex-shrink:0"></div>
      <span style="color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${name}">${name}</span>
      <span style="color:var(--text3);font-size:9px">${obj.shape}</span>
    </div>`;
  }
  el.innerHTML = html;
}

// ━━━ Events ━━━
function addEvent(evt) {
  const el = document.getElementById('events-list');
  const div = document.createElement('div');
  div.className = 'evt';
  const ts = new Date((evt.ts || Date.now()/1000) * 1000).toLocaleTimeString();
  const evType = evt.event_type || evt.type || '?';
  const summary = evType === 'peer_join' ? `${evt.data?.peer_id} (${evt.data?.robot_type||'?'})` :
    evType === 'peer_leave' ? evt.data?.peer_id :
    evType === 'command_sent' ? `→ ${evt.data?.target}: ${JSON.stringify(evt.data?.command||{}).slice(0,60)}` :
    JSON.stringify(evt.data || {}).slice(0, 100);
  div.innerHTML = `<span class="ts">${ts}</span><span class="tag ${evType}">${evType}</span> ${summary}`;
  el.prepend(div);
  while (el.children.length > 200) el.lastChild.remove();
}

// ━━━ Commands ━━━
// --- HuggingFace Model Search ---
let _hfCache = {};
let _hfDebounce = null;
let _hfSelectedIdx = -1;
let _hfActiveInput = null;

async function hfFetch(query) {
  if (_hfCache[query]) return _hfCache[query];
  // Search HF Hub API for lerobot-compatible POLICY models (not datasets)
  const searchQ = query.includes('/') ? query : `lerobot ${query}`;
  // Use filter=lerobot to only get models tagged with 'lerobot' (policies have this tag)
  const params = new URLSearchParams({search: searchQ, limit: '20', sort: 'downloads', direction: '-1', filter: 'lerobot'});
  try {
    const resp = await fetch(`https://huggingface.co/api/models?${params}`);
    if (!resp.ok) return [];
    const models = await resp.json();
    // Filter: only show models (policies) not datasets — policies have config.json
    // Heuristic: exclude repos with 'dataset' in tags or pipeline_tag
    const results = models
      .filter(m => !(m.pipeline_tag || '').includes('dataset') && !(m.tags || []).some(t => t === 'dataset'))
      .map(m => ({
        id: m.modelId || m.id,
        downloads: m.downloads || 0,
        likes: m.likes || 0,
        tags: (m.tags || []).filter(t => ['act','diffusion','tdmpc','vla','pi0','smolvla','xvla','vqbet','sac'].some(k => t.toLowerCase().includes(k))).slice(0, 3),
        pipeline: m.pipeline_tag || '',
        updated: m.lastModified ? new Date(m.lastModified).toLocaleDateString() : '',
      }));
    _hfCache[query] = results;
    return results;
  } catch(e) { console.warn('HF search error:', e); return []; }
}

function hfGetDropdown(input) {
  const wrap = input.closest('.hf-search-wrap');
  return wrap ? wrap.querySelector('.hf-dropdown') : null;
}

function hfRender(dropdown, results, input) {
  if (!results.length) {
    dropdown.innerHTML = '<div class="hf-searching">No models found</div>';
    dropdown.classList.add('open');
    return;
  }
  dropdown.innerHTML = results.map((m, i) => `
    <div class="hf-item${i === _hfSelectedIdx ? ' selected' : ''}" data-id="${m.id}" onclick="hfSelect('${m.id}', this)">
      <div class="hf-name">${m.id}</div>
      <div class="hf-meta">
        <span>⬇ ${m.downloads.toLocaleString()}</span>
        <span>♥ ${m.likes}</span>
        ${m.pipeline ? `<span>${m.pipeline}</span>` : ''}
        ${m.updated ? `<span>${m.updated}</span>` : ''}
      </div>
      ${m.tags.length ? `<div class="hf-meta">${m.tags.map(t => `<span class="hf-tag">${t}</span>`).join('')}</div>` : ''}
    </div>
  `).join('');
  dropdown.classList.add('open');
}

function hfSelect(modelId, el) {
  // Find the input in the same wrapper
  const wrap = el.closest('.hf-search-wrap');
  const input = wrap.querySelector('input');
  input.value = modelId;
  const dropdown = wrap.querySelector('.hf-dropdown');
  dropdown.classList.remove('open');
  _hfSelectedIdx = -1;
  // Auto-set policy to lerobot_local when selecting a HF model
  const cmdBar = wrap.closest('.cmd-bar');
  if (cmdBar) {
    const policySelect = cmdBar.querySelector('select');
    if (policySelect) policySelect.value = 'lerobot_local';
  }
}

function hfSearch(input) {
  const query = input.value.trim();
  const dropdown = hfGetDropdown(input);
  if (!dropdown) return;
  _hfActiveInput = input;
  _hfSelectedIdx = -1;
  if (query.length < 2) { dropdown.classList.remove('open'); return; }
  clearTimeout(_hfDebounce);
  dropdown.innerHTML = '<div class="hf-searching">🔍 Searching HuggingFace...</div>';
  dropdown.classList.add('open');
  _hfDebounce = setTimeout(async () => {
    const results = await hfFetch(query);
    if (_hfActiveInput === input) hfRender(dropdown, results, input);
  }, 300);
}

function hfSearchFocus(input) {
  const query = input.value.trim();
  if (query.length >= 2) hfSearch(input);
}

function hfKeydown(e, input) {
  const dropdown = hfGetDropdown(input);
  if (!dropdown || !dropdown.classList.contains('open')) return;
  const items = dropdown.querySelectorAll('.hf-item');
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    _hfSelectedIdx = Math.min(_hfSelectedIdx + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle('selected', i === _hfSelectedIdx));
    if (items[_hfSelectedIdx]) items[_hfSelectedIdx].scrollIntoView({block: 'nearest'});
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    _hfSelectedIdx = Math.max(_hfSelectedIdx - 1, 0);
    items.forEach((el, i) => el.classList.toggle('selected', i === _hfSelectedIdx));
    if (items[_hfSelectedIdx]) items[_hfSelectedIdx].scrollIntoView({block: 'nearest'});
  } else if (e.key === 'Enter' && _hfSelectedIdx >= 0 && items[_hfSelectedIdx]) {
    e.preventDefault();
    const id = items[_hfSelectedIdx].dataset.id;
    hfSelect(id, items[_hfSelectedIdx]);
  } else if (e.key === 'Escape') {
    dropdown.classList.remove('open');
    _hfSelectedIdx = -1;
  }
}

// Close dropdowns on outside click
document.addEventListener('click', (e) => {
  if (!e.target.closest('.hf-search-wrap')) {
    document.querySelectorAll('.hf-dropdown').forEach(d => d.classList.remove('open'));
  }
});


function sendTo(peerId) {
  const input = document.getElementById('cmd-' + CSS.escape(peerId));
  const policy = document.getElementById('policy-' + CSS.escape(peerId));
  const model = document.getElementById('model-' + CSS.escape(peerId));
  if (!input?.value.trim()) return;
  const msg = {
    type: 'tell', target: peerId,
    instruction: input.value.trim(),
    policy_provider: policy?.value || 'mock',
    duration: 30.0
  };
  if (model?.value.trim()) msg.pretrained_name_or_path = model.value.trim();
  ws.send(JSON.stringify(msg));
  addEvent({ts: Date.now()/1000, event_type: 'command_sent', data: {target: peerId, command: {instruction: input.value.trim(), policy: policy?.value, model: model?.value}}});
  input.value = '';
}

function sendToDetail() {
  if (!selectedPeer) return;
  const input = document.getElementById('detail-cmd');
  const policy = document.getElementById('detail-policy');
  const model = document.getElementById('detail-model');
  if (!input?.value.trim()) return;
  const msg = {
    type: 'tell', target: selectedPeer,
    instruction: input.value.trim(),
    policy_provider: policy?.value || 'mock',
    duration: 30.0
  };
  if (model?.value.trim()) msg.pretrained_name_or_path = model.value.trim();
  ws.send(JSON.stringify(msg));
  addEvent({ts: Date.now()/1000, event_type: 'command_sent', data: {target: selectedPeer, command: {instruction: input.value.trim(), policy: policy?.value, model: model?.value}}});
  input.value = '';
}

// ━━━ Teleop & Stop ━━━
let _teleopActive = {};

function startTeleop(peerId) {
  if (!peerId || !ws || ws.readyState !== 1) return;
  if (_teleopActive[peerId]) { stopTeleop(peerId); return; }

  // Get selected model
  const modelEl = document.getElementById('model-' + CSS.escape(peerId)) || document.getElementById('detail-model');
  const modelId = modelEl?.value?.trim() || 'lerobot/act_aloha_sim_transfer_cube_human';

  // Send teleop start command via zenoh
  ws.send(JSON.stringify({
    type: 'tell', target: peerId,
    instruction: '__teleop_start__',
    policy_provider: 'lerobot_local',
    pretrained_name_or_path: modelId,
    teleop: true,
    duration: 300.0
  }));

  _teleopActive[peerId] = true;
  addEvent({ts: Date.now()/1000, event_type: 'command_sent', data: {target: peerId, command: {action: 'teleop_start', model: modelId}}});

  // Update button states
  updateTeleopButtons(peerId);

  // Enable keyboard teleop listener
  if (!window._teleopKeyHandler) {
    window._teleopKeyHandler = (e) => handleTeleopKey(e);
    document.addEventListener('keydown', window._teleopKeyHandler);
    document.addEventListener('keyup', (e) => handleTeleopKeyUp(e));
  }
  window._teleopPeer = peerId;
}

function stopTeleop(peerId) {
  if (!peerId) return;
  ws.send(JSON.stringify({type: 'tell', target: peerId, instruction: '__teleop_stop__', teleop_stop: true}));
  delete _teleopActive[peerId];
  updateTeleopButtons(peerId);
  addEvent({ts: Date.now()/1000, event_type: 'command_sent', data: {target: peerId, command: {action: 'teleop_stop'}}});
  if (window._teleopPeer === peerId) window._teleopPeer = null;
}

function stopPeer(peerId) {
  if (!peerId || !ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({type: 'tell', target: peerId, instruction: '__stop__', stop: true}));
  delete _teleopActive[peerId];
  updateTeleopButtons(peerId);
  addEvent({ts: Date.now()/1000, event_type: 'command_sent', data: {target: peerId, command: {action: 'stop'}}});
}

function updateTeleopButtons(peerId) {
  // Toggle button appearance based on active state
  const active = !!_teleopActive[peerId];
  document.querySelectorAll('.cmd-bar .btn-orange').forEach(btn => {
    const card = btn.closest('.grid-card') || btn.closest('.detail-main');
    if (!card) return;
    const cardPeer = card.dataset?.peer || selectedPeer;
    if (cardPeer === peerId) {
      btn.innerHTML = active ? '🎮 Stop Teleop' : '🎮 Teleop';
      btn.style.background = active ? 'var(--red)' : '';
      btn.style.color = active ? 'white' : '';
    }
  });
}

// Keyboard teleop: WASD + QE for joint control
const TELEOP_KEYS = {
  'w': {joint: 0, delta: 0.05},   // shoulder up
  's': {joint: 0, delta: -0.05},  // shoulder down
  'a': {joint: 1, delta: -0.05},  // base left
  'd': {joint: 1, delta: 0.05},   // base right
  'q': {joint: 2, delta: 0.05},   // elbow up
  'e': {joint: 2, delta: -0.05},  // elbow down
  'r': {joint: 3, delta: 0.05},   // wrist up
  'f': {joint: 3, delta: -0.05},  // wrist down
  'z': {joint: 4, delta: 0.05},   // wrist rotate
  'x': {joint: 4, delta: -0.05},  // wrist rotate back
  'c': {joint: 5, delta: 0.05},   // gripper open
  'v': {joint: 5, delta: -0.05},  // gripper close
};

let _teleopPressed = {};

function handleTeleopKey(e) {
  if (!window._teleopPeer || !_teleopActive[window._teleopPeer]) return;
  // Don't capture if typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (TELEOP_KEYS[k] && !_teleopPressed[k]) {
    e.preventDefault();
    _teleopPressed[k] = true;
    sendTeleopDelta(window._teleopPeer, TELEOP_KEYS[k].joint, TELEOP_KEYS[k].delta);
  }
}

function handleTeleopKeyUp(e) {
  const k = e.key.toLowerCase();
  delete _teleopPressed[k];
}

function sendTeleopDelta(peerId, jointIdx, delta) {
  if (!ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({
    type: 'tell', target: peerId,
    instruction: '__teleop_delta__',
    teleop_delta: {joint: jointIdx, delta: delta}
  }));
}

function emergencyStop() {
  if (!confirm('⛔ EMERGENCY STOP — Send to ALL robots?')) return;
  ws.send(JSON.stringify({type: 'emergency_stop'}));
  addEvent({ts: Date.now()/1000, event_type: 'broadcast', data: {action: 'emergency_stop'}});
}

// ━━━ Init ━━━
connect();
setInterval(() => { if (ws?.readyState === 1) ws.send(JSON.stringify({type: 'get_peers'})); }, 3000);
</script>
</body>
</html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTTP + WebSocket Server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a separate thread."""
    daemon_threads = True


class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard HTML and API endpoints."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif self.path == "/api/peers":
            self._json_response(_get_peers())
        elif self.path == "/api/state":
            self._json_response(_STATE.get("states", {}))
        elif self.path == "/api/events":
            self._json_response(_STATE.get("events", [])[-100:])
        elif self.path.startswith("/api/state/"):
            peer_id = self.path.split("/api/state/")[1]
            self._json_response(_STATE.get("states", {}).get(peer_id, {}))
        elif self.path.startswith("/api/cameras/"):
            peer_id = self.path.split("/api/cameras/")[1]
            self._json_response(_STATE.get("cameras", {}).get(peer_id, {}))
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        pass


def start_dashboard(port: int = 7860, open_browser: bool = True):
    """Start the Strands Robots Dashboard.

    Args:
        port: HTTP port (WebSocket will be port+1)
        open_browser: Auto-open browser
    """
    ws_port = port + 1

    print(f"🤖 Strands Robots Dashboard")
    print(f"=" * 50)

    zenoh_ok = _start_zenoh()
    if zenoh_ok:
        print(f"✅ Zenoh mesh connected (peer: {_DASHBOARD_PEER_ID})")
    else:
        print(f"⚠️  Zenoh not available — dashboard will show no live data")
        print(f"   Install: pip install eclipse-zenoh")

    async def _ws_server():
        import websockets
        async with websockets.serve(_ws_handler, "0.0.0.0", ws_port):
            logger.info("WebSocket server on :%d", ws_port)
            await asyncio.Future()

    ws_thread = threading.Thread(
        target=lambda: asyncio.new_event_loop().run_until_complete(_ws_server()),
        daemon=True,
    )
    ws_thread.start()
    print(f"✅ WebSocket server: ws://localhost:{ws_port}")

    httpd = ThreadingHTTPServer(("0.0.0.0", port), DashboardHTTPHandler)
    print(f"✅ Dashboard: http://localhost:{port}")
    print(f"=" * 50)
    print(f"📡 Listening for Robot() instances on Zenoh mesh...")
    print(f"   Create a robot to see it here: Robot('so100')")
    print()

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
        httpd.shutdown()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Strands Robots Dashboard")
    parser.add_argument("--port", "-p", type=int, default=7860, help="HTTP port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()
    start_dashboard(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
