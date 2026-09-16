"""``use_rosbridge`` refuses a call missing its required names before dialing.

``echo`` needs ``topic``; ``publish`` and ``service_call`` need ``type`` as
well. The tool's own comment promises that a caller mistake is reported
identically whether or not roslibpy is installed and before the WebSocket is
dialed. A missing name is the same kind of mistake as a bad number, so it is
held to the same order: with nothing listening, the caller must not wait out
the connect timeout to be told the bridge "did not reconnect".
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

import strands_robots.tools.use_rosbridge as rb_mod

use_rosbridge = rb_mod.use_rosbridge


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


_INCOMPLETE_CALLS: tuple[tuple[dict[str, Any], str], ...] = (
    ({"action": "echo"}, "echo requires topic"),
    ({"action": "publish", "topic": "/demo/twist"}, "publish requires topic and type"),
    ({"action": "publish", "type": "geometry_msgs/Twist"}, "publish requires topic and type"),
    ({"action": "service_call", "service": "/demo/reset"}, "service_call requires service and type"),
    ({"action": "service_call", "type": "std_srvs/Empty"}, "service_call requires service and type"),
)


@pytest.fixture
def dial(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, float]]:
    """A backend that is installed and records every dial as an unreachable bridge."""
    dials: list[tuple[str, int, float]] = []

    def connect(host: str, port: int, timeout: float) -> Any:
        dials.append((host, port, timeout))
        raise TimeoutError(f"rosbridge at ws://{host}:{port} did not reconnect within {timeout}s")

    monkeypatch.setattr(rb_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rb_mod._backend, "connect", connect)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    return dials


@pytest.mark.parametrize(("call", "expected"), _INCOMPLETE_CALLS, ids=lambda v: v if isinstance(v, str) else "")
def test_a_missing_name_is_refused_without_dialing_the_bridge(
    call: dict[str, Any], expected: str, dial: list[tuple[str, int, float]]
) -> None:
    result = use_rosbridge(**call, timeout=2)

    assert result["status"] == "error"
    assert expected in _texts(result), _texts(result)
    assert dial == [], "the bridge was dialed for a call that could never run"


@pytest.mark.parametrize(("call", "expected"), _INCOMPLETE_CALLS, ids=lambda v: v if isinstance(v, str) else "")
def test_the_refusal_is_the_same_without_roslibpy(
    call: dict[str, Any], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "roslibpy", None)
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})

    result = use_rosbridge(**call)

    assert result["status"] == "error"
    assert expected in _texts(result), _texts(result)
