"""Behavior tests for the ``use_rtps`` tool.

``use_rtps`` is a pure-RTPS ROS 2 participant on cyclonedds. These tests run
with NO cyclonedds and NO ROS 2 present: the backend's ``available`` probe and
its writer/reader factories are monkeypatched, so every action-dispatch branch,
the agent-input validation, the no-backend error path, and the structured
error-return contract are exercised middleware-free.
"""

from __future__ import annotations

import dataclasses
import json
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


# subscribe / echo (the read-side participant path) -------------------------


class _FakeReader:
    """DataReader stub: yields a fixed batch per ``take`` call, then empties.

    Mirrors the cyclonedds ``DataReader.take(N=...)`` contract closely enough to
    drive the echo poll loop without a live DDS graph.
    """

    def __init__(self, batches: list[list[Any]]) -> None:
        self._batches = list(batches)

    def take(self, N: int) -> list[Any]:
        return self._batches.pop(0) if self._batches else []


@pytest.fixture
def with_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch the backend available + reader factory; return a setter for batches."""
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rtps_mod.time, "sleep", lambda *_: None)

    def _install(batches: list[list[Any]]) -> _FakeReader:
        reader = _FakeReader(batches)
        monkeypatch.setattr(rtps_mod._backend, "reader", lambda topic, type: reader)
        return reader

    return _install


def test_subscribe_creates_reader(with_reader) -> None:
    with_reader([])
    result = use_rtps(action="subscribe", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "success"
    assert "subscribed to /turtle1/cmd_vel" in _texts(result)
    _ascii_only(result)


def test_echo_returns_samples_as_dicts(with_reader) -> None:
    # Two single-sample batches force the partial-take + poll-again branch.
    with_reader([[_Twist(linear=_Vec3(x=2.0))], [_Twist(angular=_Vec3(z=1.5))]])
    result = use_rtps(
        action="echo",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        count=2,
        timeout=5.0,
    )
    assert result["status"] == "success"
    text = _texts(result)
    assert "echo /turtle1/cmd_vel" in text
    # Samples were recursively converted to nested plain dicts.
    payload = json.loads(text.split("):\n", 1)[1])
    assert payload[0]["linear"]["x"] == 2.0
    assert payload[1]["angular"]["z"] == 1.5
    _ascii_only(result)


def test_echo_times_out_with_empty_samples(with_reader) -> None:
    with_reader([])  # reader never yields
    result = use_rtps(
        action="echo",
        topic="/cmd_vel",
        type="geometry_msgs/msg/Twist",
        count=1,
        timeout=0.0,  # deadline already reached -> no spin, empty result
    )
    assert result["status"] == "success"
    text = _texts(result)
    assert json.loads(text.split("):\n", 1)[1]) == []


def test_echo_requires_topic_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    result = use_rtps(action="echo", topic="/cmd_vel")
    assert result["status"] == "error"
    assert "echo requires topic and type" in _texts(result)


def test_advertise_requires_topic_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    result = use_rtps(action="advertise", topic="/cmd_vel")
    assert result["status"] == "error"
    assert "advertise requires topic and type" in _texts(result)


# Pure helpers --------------------------------------------------------------


def test_sample_to_dict_handles_nested_lists_and_scalars() -> None:
    sample = _Twist(linear=_Vec3(x=1.0, y=2.0), angular=_Vec3(z=3.0))
    out = rtps_mod._sample_to_dict([sample, 7])
    assert out[0]["linear"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    assert out[0]["angular"]["z"] == 3.0
    assert out[1] == 7  # scalars pass through unchanged


def test_resolve_field_types_falls_back_when_hints_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_cls: Any) -> dict[str, Any]:
        raise NameError("unresolved forward ref")

    monkeypatch.setattr(rtps_mod.typing, "get_type_hints", _boom)
    resolved = rtps_mod._resolve_field_types(_Vec3)
    # Falls back to the raw dataclasses Field.type (a string under future-annotations).
    assert set(resolved) == {"x", "y", "z"}


# Real backend availability probe (no monkeypatch on available) -------------


def test_backend_available_false_without_cyclonedds() -> None:
    # cyclonedds is not installed in this environment; the real probe returns
    # False rather than raising, so the tool degrades to a clear status message.
    assert rtps_mod._RtpsBackend().available() is False


def test_backend_available_import_error_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fail(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "strands_robots.rtps.idl" or name.endswith("rtps.idl"):
            raise ImportError("simulated missing idl module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)
    assert rtps_mod._RtpsBackend().available() is False
