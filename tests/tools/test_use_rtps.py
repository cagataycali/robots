"""Behavior tests for the ``use_rtps`` tool.

``use_rtps`` is a pure-RTPS ROS 2 participant on cyclonedds. These tests run
with NO cyclonedds and NO ROS 2 present: the backend's ``available`` probe and
its writer/reader factories are monkeypatched, so every action-dispatch branch,
the agent-input validation, the no-backend error path, and the structured
error-return contract are exercised middleware-free.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import strands_robots.tools.use_rtps as rtps_mod

use_rtps = rtps_mod.use_rtps


# Module-level fake IDL dataclasses so typing.get_type_hints can resolve nested
# field types against module globals (mirrors the real module-level IDL bundle).
@dataclasses.dataclass
class _Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclasses.dataclass
class _Twist:
    linear: _Vec3 = dataclasses.field(default_factory=_Vec3)
    angular: _Vec3 = dataclasses.field(default_factory=_Vec3)


@dataclasses.dataclass
class _FlatTwist:
    linear: float = 0.0


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _ascii_only(result: dict[str, Any]) -> None:
    assert _texts(result).isascii(), f"non-ASCII in output: {_texts(result)!r}"


class _FakeWriter:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def write(self, sample: Any) -> None:
        self.written.append(sample)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeWriter:
    """Patch the backend to be available with a recording writer; no real DDS."""
    writer = _FakeWriter()
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rtps_mod._backend, "writer", lambda topic, type: writer)
    # publish sleeps for settle/rate; make it instant.
    monkeypatch.setattr(rtps_mod.time, "sleep", lambda *_: None)
    return writer


# Validation ----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["cmd_vel", "/bad name", "/x;y", "../etc"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_rtps(action="publish", topic=bad, type="geometry_msgs/msg/Twist")
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)
    _ascii_only(result)


def test_invalid_type_rejected() -> None:
    result = use_rtps(action="publish", topic="/cmd_vel", type="not_a_type")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


# Status / no-backend -------------------------------------------------------


def test_status_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    result = use_rtps(action="status")
    assert result["status"] == "success"
    assert "cyclonedds" in _texts(result)
    _ascii_only(result)


def test_status_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: False)
    result = use_rtps(action="status")
    assert result["status"] == "success"
    assert "backend: none" in _texts(result)


def test_action_without_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: False)
    result = use_rtps(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "error"
    assert "cyclonedds" in _texts(result)
    _ascii_only(result)


# types ---------------------------------------------------------------------


def test_types_lists_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    # REGISTRY is imported inside the function from the idl module; patch there.
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "REGISTRY", {"geometry_msgs/msg/Twist": object})
    result = use_rtps(action="types")
    assert result["status"] == "success"
    assert "geometry_msgs/msg/Twist" in _texts(result)


# publish (the headline "act as a robot" path) ------------------------------


def test_publish_builds_and_writes_count(fake_backend: _FakeWriter, monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: _Twist)
    monkeypatch.setattr(idl_mod, "REGISTRY", {"geometry_msgs/msg/Twist": _Twist})

    result = use_rtps(
        action="publish",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}},
        count=3,
    )
    assert result["status"] == "success"
    assert "published 3 message(s) to /turtle1/cmd_vel" in _texts(result)
    assert len(fake_backend.written) == 3
    # Nested field dict was built into real dataclass instances, types intact.
    sent = fake_backend.written[0]
    assert sent.linear.x == 2.0
    assert sent.angular.z == 1.5


def test_publish_unknown_field_is_structured_error(fake_backend: _FakeWriter, monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: _FlatTwist)
    result = use_rtps(
        action="publish",
        topic="/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"bogus": 1.0},
    )
    assert result["status"] == "error"
    assert "publish failed" in _texts(result)
    assert "unknown field" in _texts(result)


def test_publish_requires_topic_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    assert use_rtps(action="publish", topic="/cmd_vel")["status"] == "error"


# advertise / unknown -------------------------------------------------------


def test_advertise_creates_writer(fake_backend: _FakeWriter, monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: object)
    result = use_rtps(action="advertise", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "success"
    assert "advertised /turtle1/cmd_vel" in _texts(result)


def test_unknown_action_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    result = use_rtps(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)
