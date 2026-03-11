"""Shared async utilities for strands_robots.

Extracted to avoid duplicating _resolve_coroutine() in both robot.py and simulation.py.
"""

import asyncio


def _resolve_coroutine(coro_or_result):
    """Safely resolve a potentially-async result to a sync value.

    If the value is already a plain result, returns it unchanged.
    If it's a coroutine, runs it in the current event loop (or creates one).
    When called from within an existing event loop, delegates to a new thread
    to avoid nesting errors.
    """
    if not asyncio.iscoroutine(coro_or_result):
        return coro_or_result
    try:
        asyncio.get_running_loop()
        # Already in an event loop — run in a new thread to avoid nesting
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro_or_result).result()
    except RuntimeError:
        return asyncio.run(coro_or_result)


def _run_async(coro_fn):
    """Run an async function from sync context, handling nested loops."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro_fn()).result()
    except RuntimeError:
        return asyncio.run(coro_fn())
