"""Shared async-to-sync helpers used by robot.py and simulation.py."""

import asyncio
import concurrent.futures


def _resolve_coroutine(coro_or_result):
    """Safely resolve a potentially-async result to a sync value.

    If *coro_or_result* is not a coroutine it is returned as-is.
    When called from inside a running event loop a one-shot
    ``ThreadPoolExecutor`` is used to avoid nested-loop errors.
    """
    if not asyncio.iscoroutine(coro_or_result):
        return coro_or_result
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro_or_result).result()
    except RuntimeError:
        return asyncio.run(coro_or_result)


def _run_async(coro_fn):
    """Run an async callable from sync context, handling nested loops."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro_fn()).result()
    except RuntimeError:
        return asyncio.run(coro_fn())
