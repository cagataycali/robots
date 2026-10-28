"""``Sequential.retarget``: the old instruction's plan is never played under the new one; one infer at a time."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot.remote_inference")

from lerobot.remote_inference import RequestCancelled  # noqa: E402

from strands_robots.policy.lerobot.sequential import Sequential  # noqa: E402


class _Contract:
    def to_keys_chunk(self, chunk):
        return np.asarray(chunk, dtype=float).reshape(-1, 2)


class _Chunk:
    def __init__(self, rows):
        self.canonical_actions = SimpleNamespace(
            detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: rows))
        )
        self.server_durations = {}


class _Client:
    """One ``infer`` at a time, like ``RemoteClient``; each call blocks until the test releases it or cancels it."""

    def __init__(self):
        self.descriptor = {"limits": {"max_input_chars": 200}}
        self.config = SimpleNamespace(deployment="d")
        self.requests: list = []
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)
        self.inside, self.max_inside, self.cancelled = 0, 0, 0
        self._lock = threading.Lock()

    def infer(self, req, *, cancelled=lambda: False):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        self.requests.append(req)
        self.started.release()
        try:
            while not self.gate.wait(0.005):
                if cancelled():
                    self.cancelled += 1
                    raise RequestCancelled("superseded")
            return _Chunk(np.full((3, 2), float(len(self.requests))))  # rows carry the request number
        finally:
            with self._lock:
                self.inside -= 1


def _drain(seq: Sequential) -> None:
    t = seq._inflight
    if t is not None:
        t.join(2.0)
        assert not t.is_alive()


def test_a_reply_that_landed_before_the_retarget_is_dropped_not_played_under_the_new_task():
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("old")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)  # request 1 sent for "old"
    client.gate.set()  # the chunk lands between ticks: it is in _reply now
    _drain(seq)
    assert seq._reply is not None
    client.gate.clear()
    seq.retarget("new")  # before the next tick
    assert seq._reply is None and seq.queued == 0
    out = seq.step(frame)  # must ask again for "new", not play request 1's rows
    assert out is None and client.started.acquire(timeout=2.0)
    assert [r.observation.task for r in client.requests] == ["old", "new"]
    client.gate.set()
    _drain(seq)
    row = seq.step(frame)
    assert row is not None and row.tolist() == [2.0, 2.0]  # request 2's rows, booked under "new"
    assert [c["instruction"] for c in seq.calls] == ["new"]
    seq.close()


def test_a_retarget_cancels_the_request_in_flight_and_never_runs_two_infers_at_once():
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("old")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)
    seq.retarget("new")  # while request 1 is still waiting on the server
    for _ in range(50):  # the next ticks hold until the superseded request has drained, then ask once for "new"
        if client.started.acquire(timeout=0.05):
            break
        assert seq.step(frame) is None
    else:
        pytest.fail("no request was sent for the new instruction")
    assert client.cancelled == 1 and client.max_inside == 1
    assert [r.observation.task for r in client.requests] == ["old", "new"]
    client.gate.set()
    _drain(seq)
    assert seq.step(frame).tolist() == [2.0, 2.0] and seq.calls[-1]["instruction"] == "new"  # no PolicyFault surfaced
    seq.close()
    assert seq.queued == 0


def test_close_drops_the_plan_under_the_lock():
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    seq.retarget("x")
    assert seq.step({"observation.state": np.zeros(2)}) is None and client.started.acquire(timeout=2.0)
    client.gate.set()
    _drain(seq)
    seq.close()
    assert seq._reply is None and seq.queued == 0 and seq.task is None
