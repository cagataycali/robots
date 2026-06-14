"""Tests for ``strands_robots._async_utils._resolve_coroutine``.

The helper bridges async results into sync call sites and has three
branches:

    1. plain (non-coroutine) value -> returned unchanged
    2. coroutine, no running event loop -> ``asyncio.run``
    3. coroutine, inside a running event loop -> offloaded to the reused
       single-worker thread executor

Branch 3 (the running-loop offload) was previously uncovered. These
tests pin all three deterministically without touching any robot or
simulation surface.
"""

from __future__ import annotations

import asyncio

from strands_robots._async_utils import _resolve_coroutine


class TestResolveCoroutine:
    """_resolve_coroutine() - the three resolution branches."""

    def test_plain_value_returned_unchanged(self) -> None:
        """A non-coroutine value is passed straight through."""
        sentinel = object()
        assert _resolve_coroutine(sentinel) is sentinel

    def test_plain_falsy_value_passthrough(self) -> None:
        """Falsy non-coroutine values must not be mistaken for 'no result'."""
        assert _resolve_coroutine(0) == 0
        assert _resolve_coroutine(None) is None
        assert _resolve_coroutine("") == ""

    def test_coroutine_no_running_loop_uses_asyncio_run(self) -> None:
        """With no running loop, the coroutine is resolved via asyncio.run."""

        async def produce() -> str:
            return "resolved-sync"

        assert _resolve_coroutine(produce()) == "resolved-sync"

    def test_coroutine_inside_running_loop_offloads_to_thread(self) -> None:
        """Inside a running loop, resolution is offloaded to the executor thread.

        Calling asyncio.run() directly here would raise 'cannot be called
        from a running event loop'; the helper must instead run the
        coroutine on its worker thread and return the value synchronously.
        """

        async def inner() -> int:
            return 42

        async def driver() -> int:
            # We are inside a running loop here, so this exercises the
            # get_running_loop()-succeeds (offload) branch.
            return _resolve_coroutine(inner())

        assert asyncio.run(driver()) == 42

    def test_coroutine_inside_running_loop_propagates_exception(self) -> None:
        """Exceptions raised inside the offloaded coroutine surface to the caller."""

        async def boom() -> None:
            raise ValueError("kaboom")

        async def driver() -> None:
            return _resolve_coroutine(boom())

        try:
            asyncio.run(driver())
        except ValueError as exc:
            assert str(exc) == "kaboom"
        else:  # pragma: no cover - explicit failure if no exception
            raise AssertionError("expected ValueError to propagate")
