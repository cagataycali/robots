"""Agent tool for direct Zenoh pub/sub/query interactions.

Provides low-level Zenoh access for communicating with robots that use
Zenoh natively (e.g. Pollen Robotics Reachy Mini, Reachy 2, or any
Zenoh-enabled device on the network).

Actions:
    discover     - Scout the Zenoh network for active key expressions
    get          - Get current value of a key expression
    put          - Publish a value to a key expression
    subscribe    - Subscribe to a key expression and buffer N messages
    query        - Perform a Zenoh query (queryable pattern)
    list_keys    - List known key expressions matching a pattern
    info         - Show Zenoh session info and router connections

Typical Reachy Mini keys:
    reachy/<serial>/joint/*/present_position
    reachy/<serial>/joint/*/goal_position
    reachy/<serial>/camera/*
    reachy/<serial>/state

Example:
    use_zenoh(action="discover")
    use_zenoh(action="get", key="reachy/**/present_position")
    use_zenoh(action="put", key="reachy/mini/joint/r_shoulder_pitch/goal_position", value="0.5")
    use_zenoh(action="subscribe", key="reachy/**/present_position", count=5)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from strands import tool

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

# Module-level session cache (reuse across calls)
_SESSION = None
_SESSION_CONFIG = None


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _get_session(config: str | None = None) -> Any:
    """Get or create a Zenoh session (cached)."""
    global _SESSION, _SESSION_CONFIG

    zenoh: Any = require_optional("zenoh", pip_install="eclipse-zenoh", extra="mesh")

    if _SESSION is not None and _SESSION_CONFIG == config:
        return _SESSION

    if config:
        cfg = zenoh.Config.from_json5(config)
    else:
        cfg = zenoh.Config()

    _SESSION = zenoh.open(cfg)
    _SESSION_CONFIG = config
    logger.info("Zenoh session opened")
    return _SESSION


def _format_sample(sample) -> dict[str, Any]:
    """Format a Zenoh sample into a readable dict."""
    payload = sample.payload.to_bytes()

    # Try JSON decode first
    try:
        value = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Try float/int
        try:
            if len(payload) == 4:
                import struct
                value = struct.unpack("<f", payload)[0]
            elif len(payload) == 8:
                import struct
                value = struct.unpack("<d", payload)[0]
            else:
                value = payload.hex()
        except Exception:
            value = payload.hex()

    return {
        "key": str(sample.key_expr),
        "value": value,
        "encoding": str(sample.encoding) if hasattr(sample, "encoding") else None,
    }


@tool
def use_zenoh(
    action: str,
    key: str = "",
    value: str = "",
    count: int = 10,
    timeout_ms: int = 3000,
    config: str = "",
) -> dict[str, Any]:
    """Interact with devices on the Zenoh network (pub/sub/query).

    Use this to communicate with Zenoh-native robots like Reachy Mini,
    or any device publishing/subscribing on the Zenoh bus.

    Args:
        action: One of: discover, get, put, subscribe, query, list_keys, info
        key: Zenoh key expression (supports wildcards: * and **)
        value: Value to publish (for put action). JSON string or raw.
        count: Max samples to collect (for subscribe, default 10)
        timeout_ms: Timeout in milliseconds (default 3000)
        config: Optional Zenoh config as JSON5 string

    Returns:
        Dict with status and content
    """
    try:
        zenoh: Any = require_optional("zenoh", pip_install="eclipse-zenoh", extra="mesh")
    except Exception as e:
        return _err(f"Zenoh not available: {e}. Install with: pip install eclipse-zenoh")

    cfg = config if config else None

    try:
        if action == "info":
            session = _get_session(cfg)
            zid = str(session.zid())
            # Get routers and peers info
            info_text = f"Zenoh Session Info:\n  ZID: {zid}\n"
            try:
                routers = session.info().routers_zid()
                peers = session.info().peers_zid()
                info_text += f"  Routers: {[str(r) for r in routers]}\n"
                info_text += f"  Peers: {[str(p) for p in peers]}\n"
            except Exception as e:
                info_text += f"  (info query failed: {e})\n"
            return _ok(info_text)

        elif action == "discover":
            session = _get_session(cfg)
            pattern = key if key else "**"
            samples = []

            # Use liveliness to discover active keys
            try:
                replies = session.get(
                    pattern,
                    timeout=timeout_ms / 1000.0,
                )
                for reply in replies:
                    if reply.ok is not None:
                        samples.append(_format_sample(reply.ok))
            except Exception:
                pass

            if not samples:
                # Fallback: try scout
                try:
                    hello_msgs = zenoh.scout(
                        zenoh.WhatAmI.ROUTER | zenoh.WhatAmI.PEER,
                        timeout=timeout_ms / 1000.0,
                    )
                    scout_results = []
                    for hello in hello_msgs:
                        scout_results.append({
                            "zid": str(hello.zid),
                            "whatami": str(hello.whatami),
                            "locators": [str(loc) for loc in hello.locators],
                        })
                    if scout_results:
                        return _ok(
                            f"Scouted {len(scout_results)} Zenoh node(s):\n"
                            + json.dumps(scout_results, indent=2)
                            + "\n\nUse action='get' with specific key patterns to read data."
                        )
                except Exception as e:
                    logger.debug(f"Scout failed: {e}")

                return _ok(
                    f"No responses for pattern '{pattern}' within {timeout_ms}ms.\n"
                    "Tip: try specific keys like 'reachy/**' or broader '**'."
                )

            text = f"Discovered {len(samples)} key(s) matching '{pattern}':\n"
            for s in samples[:50]:
                val_preview = str(s["value"])[:100]
                text += f"  {s['key']}: {val_preview}\n"
            if len(samples) > 50:
                text += f"  ... and {len(samples) - 50} more\n"
            return _ok(text)

        elif action == "get":
            if not key:
                return _err("'key' parameter required for get action")

            session = _get_session(cfg)
            replies = session.get(key, timeout=timeout_ms / 1000.0)

            samples = []
            for reply in replies:
                if reply.ok is not None:
                    samples.append(_format_sample(reply.ok))

            if not samples:
                return _ok(f"No data for key '{key}' (timeout {timeout_ms}ms)")

            if len(samples) == 1:
                return _ok(
                    f"Key: {samples[0]['key']}\nValue: {json.dumps(samples[0]['value'], indent=2)}"
                )

            text = f"Got {len(samples)} response(s) for '{key}':\n"
            for s in samples:
                text += f"  {s['key']}: {json.dumps(s['value'])}\n"
            return _ok(text)

        elif action == "put":
            if not key:
                return _err("'key' parameter required for put action")
            if not value:
                return _err("'value' parameter required for put action")

            session = _get_session(cfg)

            # Try to encode as JSON if it looks like it
            try:
                payload = json.dumps(json.loads(value)).encode("utf-8")
            except (json.JSONDecodeError, TypeError):
                payload = value.encode("utf-8")

            session.put(key, payload)
            return _ok(f"Published to '{key}': {value}")

        elif action == "subscribe":
            if not key:
                return _err("'key' parameter required for subscribe action")

            session = _get_session(cfg)
            samples = []
            deadline = time.time() + (timeout_ms / 1000.0)

            sub = session.declare_subscriber(key)

            while len(samples) < count and time.time() < deadline:
                try:
                    sample = sub.recv_timeout(timeout=0.5)
                    if sample is not None:
                        samples.append(_format_sample(sample))
                except Exception:
                    break

            sub.undeclare()

            if not samples:
                return _ok(
                    f"No messages received on '{key}' within {timeout_ms}ms.\n"
                    "The key may not be actively published."
                )

            text = f"Received {len(samples)} message(s) on '{key}':\n"
            for i, s in enumerate(samples):
                val_preview = json.dumps(s["value"])[:200]
                text += f"  [{i}] {s['key']}: {val_preview}\n"
            return _ok(text)

        elif action == "query":
            if not key:
                return _err("'key' parameter required for query action")

            session = _get_session(cfg)
            replies = session.get(
                key,
                value=value.encode("utf-8") if value else None,
                timeout=timeout_ms / 1000.0,
            )

            results = []
            for reply in replies:
                if reply.ok is not None:
                    results.append(_format_sample(reply.ok))
                elif reply.err is not None:
                    results.append({"error": str(reply.err.payload.to_bytes())})

            if not results:
                return _ok(f"No query responses for '{key}'")

            text = f"Query '{key}' returned {len(results)} result(s):\n"
            for r in results:
                if "error" in r:
                    text += f"  ERROR: {r['error']}\n"
                else:
                    text += f"  {r['key']}: {json.dumps(r['value'])[:200]}\n"
            return _ok(text)

        elif action == "list_keys":
            # Alias for discover with pattern
            pattern = key if key else "**"
            session = _get_session(cfg)
            keys = set()

            replies = session.get(pattern, timeout=timeout_ms / 1000.0)
            for reply in replies:
                if reply.ok is not None:
                    keys.add(str(reply.ok.key_expr))

            if not keys:
                return _ok(f"No keys found matching '{pattern}'")

            sorted_keys = sorted(keys)
            text = f"Found {len(sorted_keys)} key(s) matching '{pattern}':\n"
            for k in sorted_keys[:100]:
                text += f"  {k}\n"
            if len(sorted_keys) > 100:
                text += f"  ... and {len(sorted_keys) - 100} more\n"
            return _ok(text)

        else:
            return _err(
                f"Unknown action: '{action}'. "
                "Valid: discover, get, put, subscribe, query, list_keys, info"
            )

    except Exception as e:
        logger.error(f"use_zenoh error: {e}", exc_info=True)
        return _err(f"Zenoh error: {e}")
