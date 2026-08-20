"""A small bounded TTL cache - the type-ahead's memory, with an actual end to it.

Both Hub searches (checkpoints and datasets) memoised their answers in a plain dict
keyed by the query, with a TTL checked on read. That TTL made an entry USELESS but
never collected it: nothing in either module ever removed a key. A type-ahead writes
one entry per keystroke - "s", "so", "so1", "so10", "so101" - each holding up to 50
rows, and a dashboard left open for days keeps every prefix anyone ever typed, expired
or not.

So the store itself is responsible for its own size: an expired entry is deleted the
moment it is looked at, every write first drops what has expired, and when the cap is
still reached the OLDEST entry goes. Insertion order is the right eviction order here
because a type-ahead's older keys are its abandoned prefixes.

Deliberately NOT here: caching failures. Both callers cache only successful rows so the
next keystroke retries a hub outage, and that judgement stays at the call site where
the failure is visible.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Generic, TypeVar

V = TypeVar("V")

#: Enough for a long type-ahead session (each keystroke is a key) without letting a
#: day-long page keep every prefix ever typed.
DEFAULT_MAX_ENTRIES = 64


class TTLCache(Generic[V]):
    """Thread-safe, size-bounded, self-pruning cache of values with an age."""

    def __init__(
        self,
        ttl_s: float,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._ttl_s = float(ttl_s)
        self._max = int(max_entries)
        self._clock = clock
        self._lock = threading.Lock()
        # dicts preserve insertion order, which is exactly the eviction order wanted
        self._data: dict[str, tuple[float, V]] = {}

    def get(self, key: str) -> V | None:
        """The value if it is still fresh, else None - and an expired entry is dropped."""
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            if self._clock() - hit[0] >= self._ttl_s:
                del self._data[key]
                return None
            return hit[1]

    def put(self, key: str, value: V) -> None:
        with self._lock:
            now = self._clock()
            # a re-written key must move to the END of the eviction order: it is the
            # most recently useful, and leaving it in place would evict it first
            self._data.pop(key, None)
            self._data = {k: v for k, v in self._data.items() if now - v[0] < self._ttl_s}
            while len(self._data) >= self._max:
                self._data.pop(next(iter(self._data)))
            self._data[key] = (now, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
