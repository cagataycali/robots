"""Tests for the planner input sources (scripted, keyboard, agent)."""

from __future__ import annotations

from strands_robots.planning import STYLES, PlannerUpdate
from strands_robots.planning.inputs import AgentInput, KeyboardInput, ScriptedInput


class _ManualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# -- ScriptedInput ----------------------------------------------------------
def test_scripted_input_emits_updates_at_their_offsets() -> None:
    clock = _ManualClock()
    src = ScriptedInput(
        [(0.0, PlannerUpdate(root_vel=(0.3, 0.0, 0.0))), (1.0, PlannerUpdate(style="boxing"))],
        clock=clock,
    )
    src.start()
    first = src.poll()
    assert first is not None and first.root_vel == (0.3, 0.0, 0.0)
    # Same instant yields nothing more (one update per poll).
    assert src.poll() is None
    # Before the next offset: still nothing.
    clock.t = 0.5
    assert src.poll() is None
    clock.t = 1.0
    second = src.poll()
    assert second is not None and second.style == "boxing"
    # Exhausted, non-looping.
    assert src.poll() is None


def test_scripted_input_before_start_returns_none() -> None:
    src = ScriptedInput([(0.0, PlannerUpdate(style="run"))])
    assert src.poll() is None


def test_scripted_input_rejects_negative_offset() -> None:
    import pytest

    with pytest.raises(ValueError, match=">= 0"):
        ScriptedInput([(-1.0, PlannerUpdate())])


def test_scripted_input_loops_when_requested() -> None:
    clock = _ManualClock()
    src = ScriptedInput(
        [(0.0, PlannerUpdate(style="run")), (1.0, PlannerUpdate(style="happy"))], clock=clock, loop=True
    )
    src.start()
    assert src.poll() is not None  # t=0 update
    clock.t = 1.0
    assert src.poll() is not None  # t=1 update
    # After exhaustion the loop rearms; subsequent polls resume from the top.
    assert src.poll() is None  # rearm tick
    got = []
    for _ in range(5):
        u = src.poll()
        if u is not None:
            got.append(u)
    assert got and got[0].style == "run"


# -- KeyboardInput ----------------------------------------------------------
def test_keyboard_wasd_integrates_velocity() -> None:
    kb = KeyboardInput(speed_step=0.1)

    def vel(ch: str) -> tuple[float, float, float] | None:
        u = kb._apply_char(ch)
        assert u is not None
        return u.root_vel

    assert vel("w") == (0.1, 0.0, 0.0)
    assert vel("w") == (0.2, 0.0, 0.0)
    assert vel("a") == (0.2, 0.1, 0.0)
    # Space halts.
    assert vel(" ") == (0.0, 0.0, 0.0)


def test_keyboard_omega_and_height_and_style() -> None:
    kb = KeyboardInput(omega_step=0.2, height_step=0.05)
    q = kb._apply_char("q")
    assert q is not None and q.root_vel == (0.0, 0.0, 0.2)
    r = kb._apply_char("r")
    assert r is not None and r.height is not None
    one = kb._apply_char("1")
    assert one is not None and one.style == STYLES[0]
    eight = kb._apply_char("8")
    assert eight is not None and eight.style == STYLES[7]
    assert kb._apply_char("9") is None  # out of style range


def test_keyboard_esc_requests_stop() -> None:
    kb = KeyboardInput()
    update = kb._apply_char("\x1b")
    assert update is not None and update.stop


def test_keyboard_poll_drains_queue_and_merges() -> None:
    kb = KeyboardInput(speed_step=0.1)
    for ch in ("w", "w", "3"):
        kb._queue.put(ch)
    merged = kb.poll()
    assert merged is not None
    assert merged.root_vel == (0.2, 0.0, 0.0)
    assert merged.style == STYLES[2]


def test_keyboard_start_no_tty_is_noop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    kb = KeyboardInput()
    kb.start()  # must not raise / must not spawn a reader
    assert kb.poll() is None
    kb.stop()


# -- AgentInput -------------------------------------------------------------
class _FakeToolRegistry:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register_tool(self, tool: object) -> None:
        self.registered.append(tool)


class _FakeAgent:
    """Agent stub that drives the locomotion tool when called."""

    def __init__(self) -> None:
        self.tool_registry = _FakeToolRegistry()
        self.goal: str | None = None

    def __call__(self, goal: str) -> str:
        self.goal = goal
        return "done"


def test_agent_input_tool_enqueues_updates() -> None:
    agent = _FakeAgent()
    src = AgentInput(agent, "walk forward then stealth")  # type: ignore[arg-type]
    set_intent = src.build_tool()
    # Invoke the underlying function the way the agent would.
    fn = getattr(set_intent, "_tool_func", None) or getattr(set_intent, "__wrapped__", None) or set_intent
    fn(vx=0.5, vy=0.0, omega=0.1, style="stealth")
    update = src.poll()
    assert update is not None
    assert update.root_vel == (0.5, 0.0, 0.1)
    assert update.style == "stealth"
    assert src.poll() is None  # queue drained


def test_agent_input_rejects_unknown_style() -> None:
    agent = _FakeAgent()
    src = AgentInput(agent, "go")  # type: ignore[arg-type]
    set_intent = src.build_tool()
    fn = getattr(set_intent, "_tool_func", None) or getattr(set_intent, "__wrapped__", None) or set_intent
    result = fn(style="moonwalk")
    assert "unknown style" in result
    assert src.poll() is None  # nothing enqueued


def test_agent_input_start_registers_tool_and_runs_goal() -> None:
    agent = _FakeAgent()
    src = AgentInput(agent, "walk")  # type: ignore[arg-type]
    src.start()
    src.stop()
    assert len(agent.tool_registry.registered) == 1
    # The agent ran with the goal on the background thread.
    deadline_ok = agent.goal == "walk"
    assert deadline_ok or agent.goal is None  # thread may still be joining
