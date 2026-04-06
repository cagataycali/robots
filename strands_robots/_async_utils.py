"""Async-to-sync helper for resolving coroutines in sync contexts."""

import asyncio
import concurrent.futures

# Module-level executor reused across calls to avoid creating threads at high frequency.
# A single worker is sufficient — we only need to offload one asyncio.run() at a time.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="strands_async")


def _resolve_coroutine(coro_or_result):  # type: ignore[no-untyped-def]
    """Safely resolve a potentially-async result to a sync value.

    Handles three cases:
        1. Already a plain value → return as-is
        2. Coroutine, no running loop → asyncio.run()
        3. Coroutine, inside running loop → offload to reused thread

    Args:
        coro_or_result: Either a coroutine or an already-resolved value.

    Returns:
        The resolved (sync) value.
    """
    if not asyncio.iscoroutine(coro_or_result):
        return coro_or_result
    try:
        asyncio.get_running_loop()
        return _EXECUTOR.submit(asyncio.run, coro_or_result).result()
    except RuntimeError:
        return asyncio.run(coro_or_result)
