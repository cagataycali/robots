#!/usr/bin/env python3
"""Universal ROS 2 bridge tool - one tool for the full ROS 2 surface.

Like ``use_lerobot`` wraps the lerobot module tree, ``use_ros`` gives a Strands
agent a single, structured entry point into any ROS 2 graph on the host or LAN
without shelling out to the ``ros2`` CLI by hand or hard-coding message types.

Backends (auto-detected, override with the ``ROS2_MODE`` env var):

* ``native`` - ``rclpy`` is importable in this interpreter; commands run in-process
  helpers via ``python3 -c`` and the ``ros2`` CLI directly.
* ``docker`` - no local ROS 2, but a running container (``ROS2_DOCKER_CONTAINER``,
  default ``ros-dev``) has it sourced; commands are forwarded via ``docker exec``.
* ``none`` - no backend; every action returns a clear error naming the ``[ros2]``
  extra and the docker fallback.

Message and service types are resolved dynamically through
``rosidl_runtime_py`` (``get_message`` / ``get_service``), so any type installed
in the ROS 2 environment works with no static registry. Field payloads are
passed as plain JSON dicts and applied with ``set_message_fields`` - the
standard ROS 2 idiom - with booleans and nulls preserved (the payload is
round-tripped through ``json.loads`` inside the helper rather than pasted into
source, which would turn ``true``/``false``/``null`` into Python ``NameError``s).

Actions:
    status         - report the active backend.
    list_topics    - list topics with their types.
    list_nodes     - list nodes.
    list_services  - list services with their types.
    info           - describe a topic, node, or service.
    echo           - subscribe to a topic and return N samples as JSON.
    publish        - publish N messages built from a JSON field dict.
    service_call   - call a service with a JSON request dict, return the response.
    exec_raw       - run an arbitrary ``ros2 <args>`` CLI command (escape hatch).

Examples:
    use_ros(action="status")
    use_ros(action="list_topics")
    use_ros(action="echo", topic="/turtle1/pose", timeout=2.0, count=2)
    use_ros(action="publish", topic="/turtle1/cmd_vel",
            type="geometry_msgs/msg/Twist",
            fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})
    use_ros(action="service_call", service="/spawn",
            type="turtlesim/srv/Spawn",
            fields={"x": 3.0, "y": 3.0, "name": "t2"})
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import textwrap
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)

ROS2_DOCKER_CONTAINER = os.getenv("ROS2_DOCKER_CONTAINER", "ros-dev")
ROS2_DOCKER_SETUP = os.getenv("ROS2_DOCKER_SETUP", "/opt/ros/jazzy/setup.bash")
ROS2_MODE_OVERRIDE = os.getenv("ROS2_MODE")  # "native" | "docker" | "none"

# Validation allowlists. ROS 2 graph names are alnum plus _ / ~ (and the {ns}
# substitution braces); interface types are pkg/(msg|srv)/Name. Rejecting
# everything else keeps untrusted, agent-supplied strings from carrying shell
# metacharacters or path-traversal into the subprocess / generated-helper layer.
_NAME_RE = re.compile(r"^[A-Za-z0-9_/~{}]+$")
_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+$")
# Characters that must never reach the CLI escape hatch, even though native
# mode passes argv without a shell and docker mode shlex-quotes every token.
_SHELL_META_RE = re.compile(r"[;|&$`<>\n\r\x00]")


def _detect_mode() -> str:
    """Return the active backend: ``native``, ``docker``, or ``none``."""
    if ROS2_MODE_OVERRIDE:
        return ROS2_MODE_OVERRIDE
    try:
        import rclpy  # noqa: F401

        return "native"
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", ROS2_DOCKER_CONTAINER],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip() == "true":
            return "docker"
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return "none"


_SNIPPET_HEADER = textwrap.dedent(
    """
    import json, sys, time
    import rclpy
    from rclpy.node import Node
    from rosidl_runtime_py.utilities import get_message, get_service
    from rosidl_runtime_py.set_message import set_message_fields
    from rosidl_runtime_py.convert import message_to_ordereddict

    def _out(ok, data=None, err=None):
        print(json.dumps({"ok": ok, "data": data, "err": err}))
        sys.stdout.flush()
    """
)


def _snippet_echo(topic: str, msg_type: str, timeout: float, count: int) -> str:
    return _SNIPPET_HEADER + textwrap.dedent(
        f"""
        rclpy.init()
        node = Node("strands_robots_echo")
        MsgCls = get_message({msg_type!r})
        received = []
        sub = node.create_subscription(MsgCls, {topic!r},
            lambda m: received.append(dict(message_to_ordereddict(m))), 10)
        deadline = time.time() + {timeout}
        while len(received) < {count} and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node(); rclpy.shutdown()
        _out(True, data={{"samples": received, "count": len(received)}})
        """
    )


def _snippet_publish(topic: str, msg_type: str, fields: dict[str, Any], count: int, rate: float) -> str:
    # Embed the field payload as a repr'd JSON *string* and parse it back with
    # json.loads inside the helper. Pasting json.dumps(...) straight into source
    # would emit bare true/false/null tokens that are NameErrors in Python.
    fields_json = json.dumps(fields)
    return _SNIPPET_HEADER + textwrap.dedent(
        f"""
        rclpy.init()
        node = Node("strands_robots_pub")
        MsgCls = get_message({msg_type!r})
        pub = node.create_publisher(MsgCls, {topic!r}, 10)
        msg = MsgCls()
        set_message_fields(msg, json.loads({fields_json!r}))
        time.sleep(0.3)
        for _ in range({count}):
            pub.publish(msg)
            time.sleep(1.0 / {rate} if {rate} > 0 else 0)
        node.destroy_node(); rclpy.shutdown()
        _out(True, data={{"published": {count}, "topic": {topic!r}}})
        """
    )


def _snippet_service_call(service: str, srv_type: str, fields: dict[str, Any], timeout: float) -> str:
    fields_json = json.dumps(fields)
    return _SNIPPET_HEADER + textwrap.dedent(
        f"""
        rclpy.init()
        node = Node("strands_robots_srv")
        SrvCls = get_service({srv_type!r})
        client = node.create_client(SrvCls, {service!r})
        if not client.wait_for_service(timeout_sec={timeout}):
            _out(False, err="service not available within timeout"); sys.exit(0)
        req = SrvCls.Request()
        set_message_fields(req, json.loads({fields_json!r}))
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec={timeout})
        if fut.result() is None:
            _out(False, err="service call timed out")
        else:
            _out(True, data=dict(message_to_ordereddict(fut.result())))
        node.destroy_node(); rclpy.shutdown()
        """
    )


def _docker_python(py_code: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", "-i", ROS2_DOCKER_CONTAINER, "bash", "-c", f"source {ROS2_DOCKER_SETUP} && python3 -"],
        input=py_code,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _docker_cli(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    cmd = f"source {ROS2_DOCKER_SETUP} && ros2 " + " ".join(shlex.quote(a) for a in args)
    return subprocess.run(
        ["docker", "exec", ROS2_DOCKER_CONTAINER, "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_python(py_code: str, timeout: float = 30.0) -> dict[str, Any]:
    """Run a generated rclpy helper and return its parsed ``{ok,data,err}`` dict."""
    mode = _detect_mode()
    if mode == "native":
        proc = subprocess.run(["python3", "-c", py_code], capture_output=True, text=True, timeout=timeout)
    elif mode == "docker":
        proc = _docker_python(py_code, timeout)
    else:
        return {"ok": False, "err": "no ROS2 backend (install the [ros2] extra or run a 'ros-dev' docker container)"}
    if proc.returncode != 0:
        return {"ok": False, "err": proc.stderr.strip() or proc.stdout.strip()}
    last = None
    for ln in proc.stdout.strip().splitlines():
        s = ln.strip()
        if s.startswith("{"):
            last = s
    if not last:
        return {"ok": False, "err": f"no JSON output from helper: {proc.stdout!r}"}
    try:
        return json.loads(last)
    except json.JSONDecodeError as exc:
        return {"ok": False, "err": f"helper returned non-JSON line: {last!r} ({exc})"}


def _run_cli(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
    """Run a ``ros2 <args>`` CLI command and return ``{ok, data|err}``."""
    mode = _detect_mode()
    if mode == "native":
        proc = subprocess.run(["ros2", *args], capture_output=True, text=True, timeout=timeout)
    elif mode == "docker":
        proc = _docker_cli(args, timeout)
    else:
        return {"ok": False, "err": "no ROS2 backend (install the [ros2] extra or run a 'ros-dev' docker container)"}
    if proc.returncode != 0:
        return {"ok": False, "err": proc.stderr.strip() or proc.stdout.strip()}
    return {"ok": True, "data": proc.stdout.strip()}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"use_ros: {text}"}]}


def _resolve_topic_type(topic: str) -> str | None:
    """Look up a topic's message type from ``ros2 topic list -t``."""
    listing = _run_cli(["topic", "list", "-t"])
    if not listing["ok"]:
        return None
    for ln in listing["data"].splitlines():
        if ln.startswith(topic + " "):
            return ln.split("[", 1)[1].rstrip("]").strip()
    return None


@tool
def use_ros(
    action: str,
    topic: str | None = None,
    service: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    command: str | None = None,
) -> dict[str, Any]:
    """Universal ROS 2 tool - native rclpy or docker backend, dynamic types.

    Args:
        action: One of ``status``, ``list_topics``, ``list_nodes``,
            ``list_services``, ``info``, ``echo``, ``publish``,
            ``service_call``, ``exec_raw``.
        topic: Topic name (``echo``, ``publish``, ``info``).
        service: Service name (``service_call``, ``info``).
        type: Fully-qualified interface type, e.g. ``geometry_msgs/msg/Twist``
            or ``turtlesim/srv/Spawn``. Auto-resolved for ``echo`` when omitted.
        fields: JSON field dict applied with ``set_message_fields`` (``publish``,
            ``service_call``). Booleans and nulls are preserved.
        timeout: Seconds to wait for samples / a service.
        count: Number of messages to echo or publish.
        rate: Publish rate in Hz.
        command: Raw ``ros2`` argument string for ``exec_raw``.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    fields = fields or {}

    # Validate agent-supplied names before they reach the subprocess / helper layer.
    if topic is not None and not _NAME_RE.match(topic):
        return _err(f"invalid topic name: {topic!r}")
    if service is not None and not _NAME_RE.match(service):
        return _err(f"invalid service name: {service!r}")
    if type is not None and not _TYPE_RE.match(type):
        return _err(f"invalid interface type: {type!r} (expected pkg/msg/Name or pkg/srv/Name)")

    if action == "status":
        mode = _detect_mode()
        suffix = f" / container={ROS2_DOCKER_CONTAINER}" if mode == "docker" else ""
        return _ok(f"backend: {mode}{suffix}")

    if action == "list_topics":
        r = _run_cli(["topic", "list", "-t"])
        return _ok(r["data"]) if r["ok"] else _err(r["err"])

    if action == "list_nodes":
        r = _run_cli(["node", "list"])
        return _ok(r["data"]) if r["ok"] else _err(r["err"])

    if action == "list_services":
        r = _run_cli(["service", "list", "-t"])
        return _ok(r["data"]) if r["ok"] else _err(r["err"])

    if action == "info":
        target = topic or service
        if not target:
            return _err("info requires topic or service")
        for kind in ("topic", "node", "service"):
            r = _run_cli([kind, "info", target])
            if r["ok"] and r["data"]:
                return _ok(f"{kind} info {target}:\n{r['data']}")
        return _err(f"no info for {target}")

    if action == "echo":
        if not topic:
            return _err("echo requires topic")
        msg_type = type or _resolve_topic_type(topic)
        if not msg_type:
            return _err(f"cannot resolve type for {topic}; pass type=pkg/msg/Name")
        r = _run_python(_snippet_echo(topic, msg_type, timeout, count), timeout=timeout + 5)
        if not r["ok"]:
            return _err(r["err"])
        return _ok(f"echo {topic} ({msg_type}):\n{json.dumps(r['data']['samples'], indent=2, default=str)}")

    if action == "publish":
        if not topic or not type:
            return _err("publish requires topic and type")
        r = _run_python(_snippet_publish(topic, type, fields, count, rate), timeout=count / max(rate, 0.1) + 10)
        return _ok(f"published {count} message(s) to {topic}") if r["ok"] else _err(r["err"])

    if action == "service_call":
        if not service or not type:
            return _err("service_call requires service and type")
        r = _run_python(_snippet_service_call(service, type, fields, timeout), timeout=timeout + 10)
        return _ok(f"response:\n{json.dumps(r['data'], indent=2, default=str)}") if r["ok"] else _err(r["err"])

    if action == "exec_raw":
        if not command:
            return _err("exec_raw requires command")
        if _SHELL_META_RE.search(command):
            return _err("command contains forbidden shell metacharacters")
        r = _run_cli(shlex.split(command), timeout=timeout)
        return _ok(r["data"]) if r["ok"] else _err(r["err"])

    return _err(f"unknown action: {action}")
