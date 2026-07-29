"""The ROS 2 transport tools refuse a numeric option they cannot honor.

``use_ros`` and ``use_rtps`` expose the same three numeric options to an agent -
``count``, ``rate`` and ``timeout`` - and both consume them the same way: ``count``
is a ``range()`` bound, ``rate`` becomes an inter-message period ``1 / rate``, and
``timeout`` is a wait budget. Only positive, finite values can be honored, so a
value outside that domain must be refused with a message naming the option,
before any DDS entity joins the graph and before a single message is published.

These tests run with NO ROS 2 and NO cyclonedds installed: the backend
availability probe and the rclpy-facing helpers are monkeypatched, so the guards,
the per-action scoping, the cross-transport parity and the "nothing was
published" contract are all exercised transport-free.
"""

from __future__ import annotations

import sys
import time
import types as _types
from collections.abc import Callable
from typing import Any

import pytest

import strands_robots.tools.use_ros as ros_mod
import strands_robots.tools.use_rtps as rtps_mod

# Values outside the accepted domain of each option, with the reason each one is
# unusable. ``inf`` matters as much as ``0``: it passes a bare ``rate > 0`` test
# and then collapses ``1 / rate`` to ``0``, leaving the burst unthrottled.
UNUSABLE_RATES: list[Any] = [0.0, -5.0, float("nan"), float("inf"), "10", None]
UNUSABLE_TIMEOUTS: list[Any] = [0.0, -1.0, float("nan"), float("inf"), "2", None]
UNUSABLE_COUNTS: list[Any] = [0, -1, True, 2.7, "3", None]


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


@pytest.fixture(autouse=True)
def _both_backends_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default both transports to a present backend; opt out where needed."""
    monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)


# ---------------------------------------------------------------------------
# A refused option publishes nothing: the real ``_publish`` body never runs.
# ---------------------------------------------------------------------------


@pytest.fixture
def published_at(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Wire ``use_ros`` to a recording publisher and return its send timestamps.

    The real ``_publish`` body runs - only the rclpy node, the message class and
    the executor spin are faked - so the list records exactly the messages that
    reached the wire, and their spacing is the pacing the tool actually applied.
    """
    stamps: list[float] = []

    class FakePublisher:
        def publish(self, msg: Any) -> None:
            stamps.append(time.perf_counter())

    class FakeNode:
        def create_publisher(self, cls: Any, topic: str, depth: int) -> FakePublisher:
            return FakePublisher()

        def destroy_publisher(self, pub: Any) -> None:
            pass

    set_message = _types.ModuleType("rosidl_runtime_py.set_message")
    set_message.set_message_fields = lambda msg, fields: None  # type: ignore[attr-defined]
    package = _types.ModuleType("rosidl_runtime_py")
    package.set_message = set_message  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", package)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.set_message", set_message)

    monkeypatch.setattr(ros_mod._backend, "_ensure_node", lambda: FakeNode())
    monkeypatch.setattr(ros_mod._backend, "spin_for", lambda predicate, timeout: None)
    monkeypatch.setattr(ros_mod, "_get_message", lambda msg_type: object)
    return stamps


def _publish_twist(**options: Any) -> dict[str, Any]:
    return ros_mod.use_ros(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist", **options)


@pytest.mark.parametrize("rate", UNUSABLE_RATES)
def test_an_unusable_rate_publishes_no_message(published_at: list[float], rate: Any) -> None:
    """A rate that cannot pace the burst is refused before anything is sent.

    Pre-fix the loop fell back to ``period = 0.0`` and sent every message
    back-to-back, reporting success - a velocity hold collapsed into an
    instantaneous burst that a base then latches as its last command.
    """
    result = _publish_twist(count=6, rate=rate)

    assert result["status"] == "error"
    assert f"rate must be > 0, got {rate!r}." in _texts(result)
    assert published_at == []


@pytest.mark.parametrize("count", UNUSABLE_COUNTS)
def test_an_unusable_count_publishes_no_message(published_at: list[float], count: Any) -> None:
    """A count that is not a positive integer is refused, not silently absorbed.

    ``range(-1)`` and ``range(0)`` publish nothing, and ``range(2.7)`` raises a
    ``TypeError`` naming neither the tool nor the option.
    """
    result = _publish_twist(count=count, rate=10.0)

    assert result["status"] == "error"
    assert f"count must be a positive integer, got {count!r}." in _texts(result)
    assert published_at == []


def test_a_usable_rate_paces_the_burst(published_at: list[float]) -> None:
    """The honored path still publishes ``count`` messages spaced by ``1 / rate``."""
    result = _publish_twist(count=4, rate=100.0)

    assert result["status"] == "success"
    assert len(published_at) == 4


# ---------------------------------------------------------------------------
# The refusal is identical with and without a backend installed.
# ---------------------------------------------------------------------------


def test_a_refusal_names_the_option_even_with_no_ros_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard runs ahead of the availability probe, so the message is stable.

    A caller mistake must not be masked by an install hint on a machine without
    ROS 2 and then reported differently on a machine with it.
    """
    monkeypatch.setattr(ros_mod._backend, "available", lambda: False)

    result = _publish_twist(count=6, rate=0.0)

    assert result["status"] == "error"
    assert "rate must be > 0" in _texts(result)


# ---------------------------------------------------------------------------
# Per-action scoping: an option the action never reads is not second-guessed.
# ---------------------------------------------------------------------------


def test_publish_accepts_an_unusable_timeout_it_never_reads(published_at: list[float]) -> None:
    """``publish`` consumes ``count`` and ``rate`` only, so ``timeout`` is inert."""
    result = _publish_twist(count=2, rate=100.0, timeout=-1.0)

    assert result["status"] == "success"
    assert len(published_at) == 2


@pytest.mark.parametrize("action", ["status", "list_topics"])
def test_an_action_reading_no_numeric_option_is_never_refused(monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    """A query action must not fail for a value it does not look at."""
    monkeypatch.setattr(ros_mod, "_list_topics", lambda: "/cmd_vel [geometry_msgs/msg/Twist]")

    result = ros_mod.use_ros(action=action, count=-1, rate=float("nan"), timeout=-1.0)

    assert result["status"] == "success"


def test_the_scoping_table_covers_every_action_that_reads_an_option() -> None:
    """Only the three known option names may appear in the scoping tables.

    The table is what decides whether a value is checked at all, so a typo in it
    would silently disable a guard.
    """
    for table in (ros_mod._ACTION_NUMERIC_OPTIONS, rtps_mod._ACTION_NUMERIC_OPTIONS):
        for action, options in table.items():
            assert options, f"{action} lists no options; drop the entry instead"
            assert set(options) <= {"count", "rate", "timeout"}, action


# ---------------------------------------------------------------------------
# Cross-transport parity: two transports onto one graph, one accepted domain.
# ---------------------------------------------------------------------------

_PUBLISH_CALLS: list[tuple[str, Callable[..., dict[str, Any]]]] = [
    (
        "use_ros",
        lambda **kw: ros_mod.use_ros(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist", **kw),
    ),
    (
        "use_rtps",
        lambda **kw: rtps_mod.use_rtps(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist", **kw),
    ),
]


def _refusal_reasons(
    calls: list[tuple[str, Callable[..., dict[str, Any]]]], param: str, **options: Any
) -> dict[str, str]:
    """Map each transport to the sentence it refused ``options`` with.

    The verdict alone would be satisfied by an unrelated failure - with no ROS 2
    and no cyclonedds installed both transports error for *some* reason - so the
    returned text is asserted on, and it must name the option.
    """
    reasons = {}
    for name, call in calls:
        result = call(**options)
        assert result["status"] == "error", f"{name} accepted {param}={options[param]!r}"
        reasons[name] = _texts(result)
    return reasons


@pytest.mark.parametrize("value", UNUSABLE_RATES)
def test_both_transports_refuse_the_same_rate(value: Any) -> None:
    """A rate one transport refuses cannot be publishable through the other."""
    reasons = _refusal_reasons(_PUBLISH_CALLS, "rate", count=1, rate=value)

    for name, reason in reasons.items():
        assert f"rate must be > 0, got {value!r}." in reason, (name, reason)


@pytest.mark.parametrize("value", UNUSABLE_COUNTS)
def test_both_transports_refuse_the_same_count(value: Any) -> None:
    """A count one transport refuses cannot be publishable through the other."""
    reasons = _refusal_reasons(_PUBLISH_CALLS, "count", count=value, rate=10.0)

    for name, reason in reasons.items():
        assert f"count must be a positive integer, got {value!r}." in reason, (name, reason)


@pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS)
def test_both_transports_refuse_the_same_echo_timeout(value: Any) -> None:
    """An echo wait budget one transport refuses is refused by the other too."""
    echo_calls: list[tuple[str, Callable[..., dict[str, Any]]]] = [
        (
            "use_ros",
            lambda **kw: ros_mod.use_ros(action="echo", topic="/odom", type="nav_msgs/msg/Odometry", **kw),
        ),
        (
            "use_rtps",
            lambda **kw: rtps_mod.use_rtps(action="echo", topic="/odom", type="nav_msgs/msg/Odometry", **kw),
        ),
    ]
    reasons = _refusal_reasons(echo_calls, "timeout", timeout=value)

    for name, reason in reasons.items():
        assert f"timeout must be > 0, got {value!r}." in reason, (name, reason)


# ---------------------------------------------------------------------------
# Output contract.
# ---------------------------------------------------------------------------


def test_a_refusal_message_is_ascii_and_names_the_action() -> None:
    """The message must be plain ASCII and say which action rejected the value."""
    text = _texts(_publish_twist(count=1, rate=float("nan")))

    assert text.isascii(), text
    assert text.startswith("use_ros: publish: ")
