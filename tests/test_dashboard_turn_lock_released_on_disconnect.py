"""A chat turn abandoned mid-flight must not keep the agent turn lock.

The dashboard runs one agent turn per worker thread; the websocket handler
sets a cancel event when its client goes away. If that path ever left
``_turn_lock`` held, every later chat/voice turn would queue behind a ghost
turn forever - so these tests pin the lock's state on each exit path.
"""

from __future__ import annotations

import queue
import threading

import pytest

from strands_robots.dashboard import agent_bridge as ab


class _FakeAgent:
    """Stands in for the Strands agent: streams once, then finishes."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.callback_handler = None

    def __call__(self, prompt: str):
        self.callback_handler(data="tok")  # type: ignore[misc]  # cancellation surfaces here
        return type("R", (), {"message": {"content": [{"text": "done"}]}})()


@pytest.fixture(autouse=True)
def _fake_agent(monkeypatch):
    monkeypatch.setattr(ab, "get_agent", _FakeAgent)
    monkeypatch.setattr(ab, "_save_history", lambda *a, **k: None)
    assert not ab._turn_lock.locked(), "another test left the turn lock held"
    yield
    if ab._turn_lock.locked():  # never leave the module wedged for other tests
        ab._turn_lock.release()


def _events(q: queue.Queue[dict]) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_cancelled_mid_turn_releases_the_lock():
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()
    cancel.set()  # client already gone when the stream callback fires

    ab.run_turn_blocking("hello", q, cancel)

    assert not ab._turn_lock.locked()
    kinds = [e.get("type") for e in _events(q)]
    assert "__END__" in kinds
    assert "done" not in kinds  # the abandoned turn produced no answer


def test_next_turn_acquires_immediately_after_a_ghost_turn():
    ghost_q: queue.Queue = queue.Queue()
    cancel = threading.Event()
    cancel.set()
    ab.run_turn_blocking("ghost", ghost_q, cancel)

    q: queue.Queue = queue.Queue()
    done = threading.Event()

    def second() -> None:
        ab.run_turn_blocking("real", q)
        done.set()

    threading.Thread(target=second, daemon=True).start()
    assert done.wait(2.0), "second turn blocked behind the ghost turn's lock"
    assert not ab._turn_lock.locked()
    assert any(e.get("type") == "done" for e in _events(q))


def test_a_raising_turn_also_releases_the_lock():
    class _Boom(_FakeAgent):
        def __call__(self, prompt: str):
            raise RuntimeError("provider exploded")

    ab.get_agent = _Boom  # restored by monkeypatch's undo in the fixture
    q: queue.Queue = queue.Queue()
    ab.run_turn_blocking("boom", q)

    assert not ab._turn_lock.locked()
    assert any(e.get("type") == "error" for e in _events(q))
