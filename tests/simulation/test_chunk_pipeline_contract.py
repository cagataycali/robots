"""Behavior contract for the action-chunk pipeline used by policy rollouts.

The pipeline (``_ChunkPipeline`` in
``strands_robots.simulation.policy_runner``) turns a policy's ``query_chunk``
callable into a flat stream of ``(observation, action)`` pairs for both the
synchronous and the asynchronous real-time-chunking (RTC) rollout paths. It is
exercised end to end by the MuJoCo rollout tests, but its standalone contract -
empty-chunk handling (the "never silently emit a zero/no-op action" rule), the
async prefetch hit/block/timeout accounting, and the drop-and-requery degrade -
is pinned here directly so a regression in any single branch is caught without
needing a full simulation.

The pipeline is backend-agnostic: it only ever calls the supplied
``query_chunk`` and ``observation_fn``, so these tests drive it with plain
stubs. Async-path timing is made deterministic with short real sleeps (a
sleeping prefetch is reliably "not ready" at the swap point; an instant one is
reliably ready when the consumer yields the lock between actions).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from strands_robots.simulation.policy_runner import _ChunkPipeline


def _const_obs() -> dict[str, int]:
    return {"step": 0}


def _scripted_query(
    steps: list[Callable[[], list[dict[str, int]]]],
    default: list[dict[str, int]] | None = None,
) -> Callable[[dict[str, int], int], list[dict[str, int]]]:
    """Build a ``query_chunk`` that returns one scripted chunk per call.

    Each entry in ``steps`` is a zero-arg callable producing the chunk for that
    call (so a step can sleep before returning, to control async timing). Once
    the script is exhausted, ``default`` is returned on every further call;
    if ``default`` is None an assertion fires (the test over-consumed).
    """
    calls = {"n": 0}

    def query(_obs: dict[str, int], _delay: int) -> list[dict[str, int]]:
        i = calls["n"]
        calls["n"] += 1
        if i < len(steps):
            return steps[i]()
        assert default is not None, f"query_chunk called {i + 1} times; script exhausted"
        return list(default)

    return query


def test_sync_path_flattens_chunks_in_order() -> None:
    """Synchronous path yields every action of every chunk, in order, and
    counts one acquisition per chunk pulled."""
    query = _scripted_query(
        [
            lambda: [{"a": 1}, {"a": 2}],
            lambda: [{"a": 3}],
        ],
        default=[{"a": 9}],
    )
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=False, rtc_inference_timeout_s=None)
    actions = []
    with pipe as stream:
        for _obs, action in stream:
            actions.append(action)
            if len(actions) == 3:
                break

    assert actions == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert pipe.chunks_acquired == 2


def test_sync_path_empty_chunk_raises() -> None:
    """An empty chunk on the sync path is a hard error, never a silent no-op."""
    query = _scripted_query([lambda: []])
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=False, rtc_inference_timeout_s=None)
    with pytest.raises(RuntimeError, match="empty action chunk; cannot run rollout"):
        with pipe as stream:
            for _ in stream:
                pass


def test_async_initial_empty_chunk_raises() -> None:
    """The async path rejects an empty very-first chunk just like the sync path."""
    query = _scripted_query([lambda: []])
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=None)
    with pytest.raises(RuntimeError, match="empty action chunk; cannot run rollout"):
        with pipe as stream:
            for _ in stream:
                pass


def test_async_short_chunks_use_synchronous_requery() -> None:
    """Length-1 chunks never trigger a prefetch, so the pipeline falls back to
    a synchronous re-query for each subsequent chunk and still streams in order."""
    query = _scripted_query(
        [
            lambda: [{"a": 1}],
            lambda: [{"a": 2}],
            lambda: [{"a": 3}],
        ],
        default=[{"a": 9}],
    )
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=None)
    actions = []
    with pipe as stream:
        for _obs, action in stream:
            actions.append(action)
            if len(actions) == 3:
                break

    assert actions == [{"a": 1}, {"a": 2}, {"a": 3}]
    # No prefetch was ever issued (chunks too short to reach the trigger).
    assert pipe.prefetch_hits == 0
    assert pipe.prefetch_blocks == 0


def test_async_prefetch_hit_when_inference_outpaces_consumption() -> None:
    """When a prefetched chunk is ready by the swap point it counts as a hit
    and the seam is invisible (actions stream straight through)."""
    query = _scripted_query(
        [
            lambda: [{"a": 1}, {"a": 2}],
            lambda: [{"a": 3}, {"a": 4}],
        ],
        default=[{"a": 9}],
    )
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=None)
    actions = []
    with pipe as stream:
        for _obs, action in stream:
            actions.append(action)
            # Yield the GIL so the (instant) prefetch worker finishes before the
            # next swap point -> deterministic hit.
            time.sleep(0.02)
            if len(actions) == 3:
                break

    assert actions[:3] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert pipe.prefetch_hits >= 1
    assert pipe.prefetch_blocks == 0


def test_async_prefetch_block_when_inference_lags() -> None:
    """A prefetch that is still running at the swap point is counted as a block
    (seam starvation) and the consumer waits on it rather than stalling silently."""

    def _slow() -> list[dict[str, int]]:
        time.sleep(0.25)
        return [{"a": 3}, {"a": 4}]

    query = _scripted_query(
        [
            lambda: [{"a": 1}, {"a": 2}],
            _slow,
        ],
        default=[{"a": 9}],
    )
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=None)
    actions = []
    with pipe as stream:
        for _obs, action in stream:
            actions.append(action)
            if len(actions) == 3:
                break

    assert actions[:3] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert pipe.prefetch_blocks >= 1


def test_async_prefetch_timeout_raises_structured_error() -> None:
    """A prefetch that exceeds rtc_inference_timeout_s becomes a structured
    RuntimeError instead of an unbounded hang."""

    def _stuck() -> list[dict[str, int]]:
        time.sleep(0.5)
        return [{"a": 3}, {"a": 4}]

    query = _scripted_query([lambda: [{"a": 1}, {"a": 2}], _stuck])
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=0.05)
    with pytest.raises(RuntimeError, match="exceeded rtc_inference_timeout_s"):
        with pipe as stream:
            for _ in stream:
                pass


def test_async_empty_prefetch_degrades_to_one_requery() -> None:
    """A prefetched chunk arriving empty degrades to a single synchronous
    re-query rather than killing an otherwise-healthy rollout."""
    query = _scripted_query(
        [
            lambda: [{"a": 1}, {"a": 2}],
            lambda: [],  # prefetched chunk arrives empty
            lambda: [{"a": 3}],  # synchronous re-query recovers
        ],
        default=[{"a": 9}],
    )
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=None)
    actions = []
    with pipe as stream:
        for _obs, action in stream:
            actions.append(action)
            if len(actions) == 3:
                break

    assert actions[:3] == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_async_empty_prefetch_twice_raises() -> None:
    """If both the prefetch AND the fallback re-query come back empty, the
    rollout fails loudly."""
    query = _scripted_query(
        [
            lambda: [{"a": 1}, {"a": 2}],
            lambda: [],  # prefetch empty
            lambda: [],  # re-query also empty
        ]
    )
    pipe = _ChunkPipeline(query, _const_obs, async_rtc=True, rtc_inference_timeout_s=None)
    with pytest.raises(RuntimeError, match="empty action chunk twice"):
        with pipe as stream:
            for _ in stream:
                pass
