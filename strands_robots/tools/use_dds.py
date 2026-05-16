"""Agent tool for ROS 2 / DDS communication.

Provides out-of-the-box ROS 2 capabilities via DDS (Data Distribution Service)
for interacting with ROS-based robots without requiring a full ROS workspace.

Uses the Zenoh-ROS2 bridge or direct rclpy when available, falling back to
Zenoh DDS plugin for environments without a ROS 2 installation.

Actions:
    topics         - List available ROS 2 topics
    echo           - Subscribe to a topic and print N messages
    publish        - Publish a message to a topic
    services       - List available ROS 2 services
    call_service   - Call a ROS 2 service
    nodes          - List active ROS 2 nodes
    info           - Get info about a specific topic/node

Supports two backends:
    1. rclpy (native) - When ROS 2 is installed
    2. zenoh-bridge-ros2dds - Via Zenoh DDS plugin (no ROS install needed)

Example:
    use_dds(action="topics")
    use_dds(action="echo", topic="/joint_states", count=3)
    use_dds(action="publish", topic="/cmd_vel", message='{"linear": {"x": 0.1}, "angular": {"z": 0.0}}')
    use_dds(action="call_service", service="/reset", message="{}")
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _has_rclpy() -> bool:
    """Check if rclpy is available."""
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ros2_cli() -> bool:
    """Check if ros2 CLI is available."""
    try:
        result = subprocess.run(
            ["ros2", "--help"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_zenoh_bridge() -> bool:
    """Check if zenoh-bridge-ros2dds is available or Zenoh DDS plugin."""
    try:
        import zenoh  # noqa: F401
        return True
    except ImportError:
        return False


def _ros2_cli(args: list[str], timeout: float = 10.0) -> str:
    """Run a ros2 CLI command and return output."""
    try:
        result = subprocess.run(
            ["ros2"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"ros2 command failed: {args}")
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("ros2 CLI not found. Source your ROS 2 workspace.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ros2 command timed out ({timeout}s): {args}")


def _zenoh_ros2_get(key_pattern: str, timeout_ms: int = 3000) -> list[dict]:
    """Query ROS 2 topics via Zenoh (when bridge is running)."""
    try:
        import zenoh
    except ImportError:
        raise RuntimeError("zenoh not available")

    # ROS 2 topics are mapped to Zenoh keys by the bridge:
    # /topic_name -> rt/topic_name (for data)
    # /service_name -> rs/service_name/request + rs/service_name/reply
    session = zenoh.open(zenoh.Config())
    try:
        replies = session.get(key_pattern, timeout=timeout_ms / 1000.0)
        results = []
        for reply in replies:
            if reply.ok is not None:
                payload = reply.ok.payload.to_bytes()
                try:
                    value = json.loads(payload.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    value = payload.hex()
                results.append({
                    "key": str(reply.ok.key_expr),
                    "value": value,
                })
        return results
    finally:
        session.close()


@tool
def use_dds(
    action: str,
    topic: str = "",
    service: str = "",
    node: str = "",
    message: str = "",
    msg_type: str = "",
    count: int = 5,
    timeout_ms: int = 5000,
    backend: str = "auto",
) -> dict[str, Any]:
    """Interact with ROS 2 / DDS network for robot communication.

    Provides topic pub/sub, service calls, and node discovery for
    ROS 2 robots. Works with native rclpy or via Zenoh DDS bridge.

    Args:
        action: One of: topics, echo, publish, services, call_service, nodes, info
        topic: ROS 2 topic name (e.g. /joint_states, /cmd_vel)
        service: ROS 2 service name (e.g. /reset_world)
        node: ROS 2 node name for info queries
        message: JSON message payload (for publish/call_service)
        msg_type: Message type (e.g. geometry_msgs/msg/Twist). Auto-detected if omitted.
        count: Number of messages to receive (for echo, default 5)
        timeout_ms: Timeout in milliseconds (default 5000)
        backend: Backend to use: 'auto', 'ros2', or 'zenoh'

    Returns:
        Dict with status and content
    """
    # Determine available backend
    has_ros2 = _has_ros2_cli()
    has_zenoh = _has_zenoh_bridge()

    if backend == "auto":
        if has_ros2:
            active_backend = "ros2"
        elif has_zenoh:
            active_backend = "zenoh"
        else:
            return _err(
                "No DDS backend available.\n"
                "Options:\n"
                "  1. Install ROS 2 and source the workspace\n"
                "  2. Install eclipse-zenoh + run zenoh-bridge-ros2dds\n"
                "  3. pip install eclipse-zenoh (for Zenoh-native robots)\n\n"
                "For Zenoh-native robots (Reachy), use the use_zenoh tool instead."
            )
    elif backend == "ros2":
        if not has_ros2:
            return _err("ROS 2 CLI not available. Source your workspace first.")
        active_backend = "ros2"
    elif backend == "zenoh":
        if not has_zenoh:
            return _err("Zenoh not available. pip install eclipse-zenoh")
        active_backend = "zenoh"
    else:
        return _err(f"Unknown backend '{backend}'. Valid: auto, ros2, zenoh")

    timeout_s = timeout_ms / 1000.0

    try:
        if action == "topics":
            if active_backend == "ros2":
                output = _ros2_cli(["topic", "list", "-t"], timeout=timeout_s)
                return _ok(f"ROS 2 Topics (via ros2 CLI):\n{output}")
            else:
                # Via Zenoh bridge - topics are under rt/**
                results = _zenoh_ros2_get("rt/**", timeout_ms)
                if not results:
                    # Try discovery
                    results = _zenoh_ros2_get("@ros2/topic/**", timeout_ms)

                if not results:
                    return _ok(
                        "No ROS 2 topics found via Zenoh.\n"
                        "Ensure zenoh-bridge-ros2dds is running."
                    )
                topics = sorted(set(r["key"] for r in results))
                text = f"ROS 2 Topics via Zenoh ({len(topics)}):\n"
                for t in topics:
                    # Strip rt/ prefix
                    name = t.replace("rt/", "/", 1) if t.startswith("rt/") else t
                    text += f"  {name}\n"
                return _ok(text)

        elif action == "echo":
            if not topic:
                return _err("'topic' parameter required for echo action")

            if active_backend == "ros2":
                # Use ros2 topic echo with timeout
                try:
                    result = subprocess.run(
                        ["ros2", "topic", "echo", topic, "--once"]
                        if count == 1
                        else ["ros2", "topic", "echo", topic, f"--max-wait-time={timeout_s}"],
                        capture_output=True,
                        text=True,
                        timeout=timeout_s + 2,
                    )
                    if result.stdout:
                        return _ok(f"Echo {topic}:\n{result.stdout[:5000]}")
                    return _ok(f"No messages on {topic} within timeout")
                except subprocess.TimeoutExpired:
                    return _ok(f"Timeout waiting for messages on {topic}")
            else:
                # Via Zenoh - subscribe to rt/<topic>
                import zenoh
                zenoh_key = f"rt{topic}" if topic.startswith("/") else f"rt/{topic}"
                session = zenoh.open(zenoh.Config())
                samples: list[dict] = []
                try:
                    sub = session.declare_subscriber(zenoh_key)
                    deadline = time.time() + timeout_s
                    while len(samples) < count and time.time() < deadline:
                        try:
                            sample = sub.recv_timeout(timeout=0.5)
                            if sample is not None:
                                payload = sample.payload.to_bytes()
                                try:
                                    val = json.loads(payload.decode("utf-8"))
                                except (json.JSONDecodeError, UnicodeDecodeError):
                                    val = payload.hex()
                                samples.append({"key": str(sample.key_expr), "value": val})
                        except Exception:
                            break
                    sub.undeclare()
                finally:
                    session.close()

                if not samples:
                    return _ok(f"No messages on {topic} (zenoh key: {zenoh_key})")

                text = f"Echo {topic} ({len(samples)} msg):\n"
                for i, s in enumerate(samples):
                    text += f"  [{i}] {json.dumps(s['value'])[:300]}\n"
                return _ok(text)

        elif action == "publish":
            if not topic:
                return _err("'topic' parameter required for publish action")
            if not message:
                return _err("'message' parameter required for publish action")

            if active_backend == "ros2":
                if not msg_type:
                    # Try to infer type
                    try:
                        type_output = _ros2_cli(["topic", "type", topic], timeout=5)
                        msg_type = type_output.strip()
                    except Exception:
                        return _err(
                            f"Cannot infer message type for {topic}. "
                            "Specify msg_type parameter."
                        )
                cmd = ["ros2", "topic", "pub", "--once", topic, msg_type, message]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if proc.returncode == 0:
                    return _ok(f"Published to {topic} ({msg_type}): {message}")
                return _err(f"Publish failed: {proc.stderr}")
            else:
                # Via Zenoh
                import zenoh
                zenoh_key = f"rt{topic}" if topic.startswith("/") else f"rt/{topic}"
                session = zenoh.open(zenoh.Config())
                try:
                    payload = message.encode("utf-8")
                    session.put(zenoh_key, payload)
                    return _ok(f"Published to {topic} (zenoh: {zenoh_key}): {message}")
                finally:
                    session.close()

        elif action == "services":
            if active_backend == "ros2":
                output = _ros2_cli(["service", "list", "-t"], timeout=timeout_s)
                return _ok(f"ROS 2 Services:\n{output}")
            else:
                results = _zenoh_ros2_get("rs/**", timeout_ms)
                if not results:
                    return _ok("No ROS 2 services found via Zenoh bridge.")
                services = sorted(set(r["key"] for r in results))
                text = f"ROS 2 Services via Zenoh ({len(services)}):\n"
                for s in services:
                    text += f"  {s}\n"
                return _ok(text)

        elif action == "call_service":
            if not service:
                return _err("'service' parameter required for call_service action")

            if active_backend == "ros2":
                cmd = ["ros2", "service", "call", service]
                if msg_type:
                    cmd.append(msg_type)
                if message:
                    cmd.append(message)
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_s + 5
                )
                if proc.returncode == 0:
                    return _ok(f"Service call {service}:\n{proc.stdout}")
                return _err(f"Service call failed: {proc.stderr}")
            else:
                return _err(
                    "Service calls via Zenoh bridge not yet supported. "
                    "Use ros2 backend or call via Zenoh queryable pattern."
                )

        elif action == "nodes":
            if active_backend == "ros2":
                output = _ros2_cli(["node", "list"], timeout=timeout_s)
                return _ok(f"ROS 2 Nodes:\n{output}")
            else:
                return _ok(
                    "Node listing via Zenoh bridge requires @ros2/node/** pattern.\n"
                    "Use: use_zenoh(action='discover', key='@ros2/**')"
                )

        elif action == "info":
            if active_backend == "ros2":
                if topic:
                    output = _ros2_cli(["topic", "info", topic, "-v"], timeout=timeout_s)
                    return _ok(f"Topic info {topic}:\n{output}")
                elif node:
                    output = _ros2_cli(["node", "info", node], timeout=timeout_s)
                    return _ok(f"Node info {node}:\n{output}")
                elif service:
                    output = _ros2_cli(["service", "type", service], timeout=timeout_s)
                    return _ok(f"Service type {service}: {output}")
                else:
                    return _err("Provide 'topic', 'node', or 'service' for info action")
            else:
                return _ok(
                    "Detailed info requires ros2 CLI.\n"
                    "Use use_zenoh(action='get', key='...') for direct queries."
                )

        else:
            return _err(
                f"Unknown action: '{action}'. "
                "Valid: topics, echo, publish, services, call_service, nodes, info"
            )

    except Exception as e:
        logger.error(f"use_dds error: {e}", exc_info=True)
        return _err(f"DDS error: {e}")
