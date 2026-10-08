"""A lock wrapper that reports the moment another thread asks for a held lock.

A "does X wait for the lock" cell used to hold ``sim._lock`` for a fixed budget
and conclude from the timeout: a correct verb blocks, so every passing cell sat
out the whole budget. Wrapping the lock answers the same question at once - the
other thread has asked for the lock while it is held, so it is blocked - and
everything it did before asking has already happened.
"""

from __future__ import annotations

import threading
from typing import Any


class ContendedLock:
    """Delegate to *inner*; set *signal* when a second thread asks while it is held.

    Args:
        inner: The real lock (an ``RLock``, so the holder may re-enter).
        signal: Set on the first contention. Pass one the test also sets for
            "the other thread finished without asking", so a single wait covers
            both outcomes.
    """

    def __init__(self, inner: Any, signal: threading.Event) -> None:
        self._inner = inner
        self._signal = signal
        self._holder: int | None = None
        self._depth = 0
        #: How many times another thread asked for the lock while it was held.
        self.contentions = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        holder = self._holder
        if holder is not None and holder != threading.get_ident():
            self.contentions += 1
            self._signal.set()
        acquired = self._inner.acquire(blocking, timeout)
        if acquired:
            self._holder = threading.get_ident()
            self._depth += 1
        return acquired

    def release(self) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._holder = None
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
