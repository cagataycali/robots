"""Hold-for-observation: one whole chunk per still frame, asked for only when the queue is empty.

LeRobot's engine cannot express it (``ChunkRuntime.should_request`` fires one turnaround BEFORE the queue empties,
from a moving arm), so this is the loop a chunked model trained on whole chunks from still frames runs under.
``step`` is one tick: pick up a reply, refill the queue, ask again when empty, play one row — or hold (None).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from ..base import PolicyFault
from .contract import Contract


class Sequential:
    """The runtime of one admitted ``RemoteClient`` session; ``calls`` lists every chunk it received."""

    def __init__(self, client: Any, contract: Contract, who: str, calls: list[dict[str, Any]] | None = None) -> None:
        self.client, self.contract, self.who = client, contract, who
        self.calls: list[dict[str, Any]] = [] if calls is None else calls  # the policy's ledger outlives the session
        self.task: str | None = None
        self._queue: deque[np.ndarray] = deque()
        self._inflight: threading.Thread | None = None
        self._reply: tuple[Any, float, float] | None = None  # (chunk | exception, t_sent, t_received)
        self._lock = threading.Lock()
        self._gen = 0  # bumps on retarget/close: a reply for an old plan is dropped

    @property
    def queued(self) -> int:
        return len(self._queue)

    def retarget(self, task: str | None) -> None:
        """A new instruction (or a reset): whatever was planned for the old one is dropped."""
        limit = self.client.descriptor["limits"]["max_input_chars"]
        if task is not None and len(task) > limit:
            raise ValueError(
                f"{self.who} refused the instruction: {len(task)} characters; the deployment allows {limit}"
            )
        self._supersede(task)

    def close(self) -> None:
        self._supersede(None)

    def _supersede(self, task: str | None) -> None:
        """Drop the old plan: the queue, a reply that landed but was not consumed yet, and the request in flight.

        All under the lock the worker takes before it stores a reply, so a chunk that lands in the same instant is
        either dropped here or dropped by the worker's generation check: never played under the new instruction.
        The worker thread is kept in ``_inflight`` until it has exited (``cancelled`` ends its wait promptly), so the
        next request waits for the superseded one to drain instead of racing it for the one ``/act`` subscriber.
        """
        with self._lock:
            self.task = task
            self._gen += 1
            self._queue.clear()
            self._reply = None

    def _request(self, features: dict[str, np.ndarray]) -> None:
        """Send one frame for a whole chunk on a worker thread; a later ``step`` picks the reply up."""
        from uuid import uuid4

        from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot
        from lerobot.policies import QueueSnapshot

        now = time.monotonic()
        snap = ObservationSnapshot(features, now, self.task or "", 0, uuid4().hex)
        empty = QueueSnapshot(
            generation=0, cursor=0, index=0, model_actions=None, canonical_actions=None, provenance=()
        )
        req = ChunkRequest(snap, empty, uuid4().hex, 0, ExecutionMode.CHUNK, 0, now, 0.0)
        gen, client = self._gen, self.client
        superseded = lambda: gen != self._gen  # noqa: E731  (retarget/close end this request's wait promptly)

        def go() -> None:
            t0 = time.monotonic()
            try:
                reply: Any = client.infer(req, cancelled=superseded)
            except Exception as e:  # noqa: BLE001  (raised on the loop's thread as a PolicyFault, by name)
                reply = e
            with self._lock:
                if gen == self._gen:  # a reply for a superseded plan (or its RequestCancelled) is dropped
                    self._reply = (reply, t0, time.monotonic())

        self._inflight = threading.Thread(target=go, name=f"zenoh-infer-{client.config.deployment}", daemon=True)
        self._inflight.start()

    def step(self, frame: dict[str, np.ndarray]) -> np.ndarray | None:
        """One tick: the next row of the plan, or None (hold) while the chunk for this frame is on its way."""
        with self._lock:
            reply, self._reply = self._reply, None
        if self._inflight is not None and not self._inflight.is_alive():
            self._inflight = None  # its reply is in ``reply`` now, or was dropped as superseded
        if reply is not None:
            chunk, t0, t1 = reply
            if isinstance(chunk, Exception):
                raise PolicyFault(f"{self.who}: {type(chunk).__name__}: {chunk}")
            rows = self.contract.to_keys_chunk(chunk.canonical_actions.detach().cpu().numpy())
            self._queue.extend(rows)
            self.calls.append(
                {
                    "wall_s": round(t1 - t0, 4),
                    "rows": len(rows),
                    "server": dict(getattr(chunk, "server_durations", None) or {}),
                    "instruction": self.task,
                }
            )
        if not self._queue and self._inflight is None:  # the queue is EMPTY: this still frame is the observation
            self._request(frame)  # (never while a superseded request is still draining: one infer per client)
        if self._queue:
            return np.asarray(self._queue.popleft(), dtype=float)
        return None  # waiting for the chunk: hold (the loop writes nothing, counts a hold, never a miss)
