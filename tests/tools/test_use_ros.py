"""Behavior tests for the ``use_ros`` agent tool.

The tool bridges a Strands agent to a ROS 2 graph through one of two backends
(native ``rclpy`` or ``docker exec`` into a ROS 2 container). These tests run
with NO ROS 2 installed: the subprocess-facing helpers (``_run_cli`` /
``_run_python``) and ``_detect_mode`` are patched, so every action-dispatch
branch, the agent-input validation, and the no-backend error path are exercised
hardware- and ROS-free.

It also pins two contracts the reference sketch got wrong:

* No emoji / non-ASCII in any returned ``text`` (package-wide rule).
* Field payloads survive as JSON: a ``bool``/``None`` value must NOT be pasted
  into the generated helper as a bare ``true``/``false``/``null`` token (a
  Python ``NameError``); it is round-tripped via ``json.loads`` instead.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

import strands_robots.tools.use_ros as ros_mod

# Reference the tool via a module-local alias rather than a second `from`
# import: the tests monkeypatch module internals through `ros_mod`, so the
# module object is the single source of truth and a dual import of the same
# module is avoided.
use_ros = ros_mod.use_ros


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _ascii_only(result: dict[str, Any]) -> None:
    text = _texts(result)
    assert text.isascii(), f"non-ASCII in tool output: {text!r}"


# Validation ----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["/foo; rm -rf", "/a b", "/x|y", "../etc", "/a$(x)"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_ros(action="echo", topic=bad)
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)
    _ascii_only(result)


def test_invalid_type_rejected() -> None:
    result = use_ros(action="publish", topic="/cmd_vel", type="not_a_type")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


def test_invalid_service_rejected() -> None:
    result = use_ros(action="service_call", service="/spawn bad", type="turtlesim/srv/Spawn")
    assert result["status"] == "error"
    assert "invalid service" in _texts(result)


# Status / listings ---------------------------------------------------------


def test_status_reports_docker_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_detect_mode", lambda: "docker")
    result = use_ros(action="status")
    assert result["status"] == "success"
    assert "backend: docker" in _texts(result)
    assert f"container={ros_mod.ROS2_DOCKER_CONTAINER}" in _texts(result)
    _ascii_only(result)


def test_status_reports_none_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_detect_mode", lambda: "none")
    result = use_ros(action="status")
    assert "backend: none" in _texts(result)


def test_list_topics_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_cli(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
        captured["args"] = args
        return {"ok": True, "data": "/turtle1/cmd_vel [geometry_msgs/msg/Twist]"}

    monkeypatch.setattr(ros_mod, "_run_cli", fake_cli)
    result = use_ros(action="list_topics")
    assert result["status"] == "success"
    assert captured["args"] == ["topic", "list", "-t"]
    assert "/turtle1/cmd_vel" in _texts(result)


def test_list_nodes_and_services(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_cli(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
        seen.append(args)
        return {"ok": True, "data": "ok"}

    monkeypatch.setattr(ros_mod, "_run_cli", fake_cli)
    assert use_ros(action="list_nodes")["status"] == "success"
    assert use_ros(action="list_services")["status"] == "success"
    assert ["node", "list"] in seen
    assert ["service", "list", "-t"] in seen


def test_list_topics_surfaces_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_run_cli", lambda args, timeout=10.0: {"ok": False, "err": "boom"})
    result = use_ros(action="list_topics")
    assert result["status"] == "error"
    assert "boom" in _texts(result)


# info ----------------------------------------------------------------------


def test_info_requires_target() -> None:
    result = use_ros(action="info")
    assert result["status"] == "error"
    assert "requires topic or service" in _texts(result)


def test_info_tries_kinds_and_returns_first_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_cli(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
        # topic info has no data; node info has data.
        if args[0] == "node":
            return {"ok": True, "data": "node details"}
        return {"ok": True, "data": ""}

    monkeypatch.setattr(ros_mod, "_run_cli", fake_cli)
    result = use_ros(action="info", topic="/some_node")
    assert result["status"] == "success"
    assert "node info /some_node" in _texts(result)


# echo ----------------------------------------------------------------------


def test_echo_requires_topic() -> None:
    assert use_ros(action="echo")["status"] == "error"


def test_echo_autoresolves_type_and_returns_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ros_mod,
        "_run_cli",
        lambda args, timeout=10.0: {"ok": True, "data": "/turtle1/pose [turtlesim/msg/Pose]"},
    )
    samples = [{"x": 5.5, "y": 1.0}, {"x": 6.0, "y": 1.0}]

    def fake_py(code: str, timeout: float = 30.0) -> dict[str, Any]:
        assert "turtlesim/msg/Pose" in code  # auto-resolved type reached the helper
        return {"ok": True, "data": {"samples": samples, "count": 2}}

    monkeypatch.setattr(ros_mod, "_run_python", fake_py)
    result = use_ros(action="echo", topic="/turtle1/pose", count=2)
    assert result["status"] == "success"
    assert "turtlesim/msg/Pose" in _texts(result)
    assert "5.5" in _texts(result)


def test_echo_unresolvable_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_run_cli", lambda args, timeout=10.0: {"ok": True, "data": "/other [x/msg/Y]"})
    result = use_ros(action="echo", topic="/turtle1/pose")
    assert result["status"] == "error"
    assert "cannot resolve type" in _texts(result)


# publish / service_call ----------------------------------------------------


def test_publish_requires_topic_and_type() -> None:
    assert use_ros(action="publish", topic="/cmd_vel")["status"] == "error"


def test_publish_dispatches_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_run_python", lambda code, timeout=30.0: {"ok": True})
    result = use_ros(
        action="publish",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}},
        count=3,
    )
    assert result["status"] == "success"
    assert "published 3 message(s) to /turtle1/cmd_vel" in _texts(result)


def test_service_call_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_run_python", lambda code, timeout=30.0: {"ok": True, "data": {"name": "t2"}})
    result = use_ros(
        action="service_call",
        service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 3.0, "y": 3.0, "name": "t2"},
    )
    assert result["status"] == "success"
    assert "t2" in _texts(result)


# exec_raw ------------------------------------------------------------------


def test_exec_raw_rejects_shell_metacharacters() -> None:
    result = use_ros(action="exec_raw", command="topic list; rm -rf /")
    assert result["status"] == "error"
    assert "forbidden shell metacharacters" in _texts(result)


def test_exec_raw_splits_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_cli(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
        captured["args"] = args
        return {"ok": True, "data": "done"}

    monkeypatch.setattr(ros_mod, "_run_cli", fake_cli)
    result = use_ros(action="exec_raw", command="topic list -t")
    assert result["status"] == "success"
    assert captured["args"] == ["topic", "list", "-t"]


# unknown / no-backend ------------------------------------------------------


def test_unknown_action_errors() -> None:
    result = use_ros(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)


def test_no_backend_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_detect_mode", lambda: "none")
    # _run_cli consults _detect_mode; with no backend it must name the remedy.
    out = ros_mod._run_cli(["topic", "list"])
    assert out["ok"] is False
    assert "[ros2] extra" in out["err"] and "docker" in out["err"]


def test_mode_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "ROS2_MODE_OVERRIDE", "docker")
    assert ros_mod._detect_mode() == "docker"


# Regression: JSON field payload must not be pasted as Python source ---------


def _extract_set_fields_expr(snippet: str) -> str:
    """Pull the expression passed as the 2nd arg of set_message_fields(...)."""
    m = re.search(r"set_message_fields\([^,]+,\s*(.+)\)\s*$", snippet, re.MULTILINE)
    assert m, f"set_message_fields call not found in:\n{snippet}"
    return m.group(1)


@pytest.mark.parametrize(
    "fields",
    [
        {"data": True},
        {"enabled": False, "name": "t2"},
        {"value": None},
        {"linear": {"x": 2.0}, "angular": {"z": 1.5}},
    ],
)
def test_publish_snippet_preserves_json_types(fields: dict[str, Any]) -> None:
    """The payload round-trips through json.loads, not bare true/false/null.

    Pasting ``json.dumps(fields)`` directly into the helper source emits bare
    ``true``/``false``/``null`` tokens that raise ``NameError`` at runtime. The
    fix embeds the payload as a JSON *string* and parses it. Evaluating the
    extracted expression in a namespace that defines ONLY ``json`` reproduces
    the original dict; the broken form would raise ``NameError``.
    """
    snippet = ros_mod._snippet_publish("/t", "geometry_msgs/msg/Twist", fields, count=1, rate=10.0)
    expr = _extract_set_fields_expr(snippet)
    recovered = eval(expr, {"json": json})  # noqa: S307 - controlled expr from our own generator
    assert recovered == fields


def test_service_snippet_preserves_json_types() -> None:
    fields = {"x": 3.0, "active": True, "label": None}
    snippet = ros_mod._snippet_service_call("/spawn", "turtlesim/srv/Spawn", fields, timeout=5.0)
    expr = _extract_set_fields_expr(snippet)
    assert eval(expr, {"json": json}) == fields  # noqa: S307
