"""Persistent, reusable policy handles - load the model once, reuse everywhere.

Loading a VLA / LeRobot checkpoint (a MolmoAct2 SO-100/101 build reads ~1300
weight files into GPU memory) costs on the order of a minute or two. A naive
multi-episode loop that calls :func:`create_policy` per rollout pays that cost
every episode; the dominant fix already exists - a process-level model cache in
:mod:`strands_robots.policies.lerobot_local.policy` shares the resident weights
across instances - but two ergonomic gaps remained:

1. The win was implicit. There was no first-class "load this once and hand me a
   handle I reuse" object, so an LLM harness driving the API blind had no
   obvious way to express the intent and would re-call ``create_policy``.
2. There was no provider-agnostic way to *warm* the cache ahead of a run, *see*
   what is resident, or *free* a checkpoint between runs of different policies.

This module closes both. :class:`PersistentPolicy` is a thin, thread-safe
wrapper that builds the underlying policy ONCE at construction and is meant to
be passed to every ``run_policy``/``eval_policy`` call via ``policy_object=``::

    from strands_robots.policies import PersistentPolicy

    policy = PersistentPolicy("lerobot_local", pretrained_name_or_path="...")
    for _ in range(20):
        sim.run_policy(robot_name="arm", policy_object=policy)  # zero reload
        sim.save_episode()
        sim.reset()

The :func:`preload`, :func:`list_cached`, and :func:`evict` helpers are the
agent-facing cache controls: warm before a run, introspect what is hot, free
memory before switching checkpoints.

This is a SYNCHRONOUS persistent worker: the model lives in-process and is
shared via the module-level cache. It deliberately does not spawn a background
daemon or expose cross-process IPC - inference is GIL- and GPU-serialised, so a
per-call lock gives correct concurrent reuse without the complexity (and races)
of a separate worker process. Cross-process sharing is a separable concern.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from strands_robots.policies.base import Policy
from strands_robots.policies.factory import create_policy
from strands_robots.utils import process_rss_mb

__all__ = ["PersistentPolicy", "preload", "list_cached", "evict"]


class PersistentPolicy(Policy):
    """A persistent, reusable handle around an underlying policy.

    Builds the wrapped policy ONCE (warming the process-level model cache) and
    delegates every :class:`Policy` operation to it, so the same object can be
    passed to many ``run_policy``/``eval_policy`` calls without ever reloading
    weights. Inference calls are serialised by a per-call lock, so two threads
    sharing one handle never corrupt the wrapped model's per-episode state.

    The wrapper is transparent: chunk-shape introspection (``execution_horizon``,
    ``is_chunk_emitting``, ``actions_per_step``, ``supports_rtc``), RTC delay /
    control-frequency hooks, ``reset``, and load telemetry (``load_time_s``,
    ``load_cache_hit``) all forward to the wrapped policy, so the runtime drives
    it exactly as it would the bare policy.

    Args:
        provider: Provider name or smart string forwarded to
            :func:`create_policy` (e.g. ``"lerobot_local"``, ``"mock"``).
        policy_object: An already-constructed policy to wrap instead of building
            a new one. When given, ``provider`` is recorded for identification
            only and ``**config`` is ignored.
        **config: Provider-specific keyword arguments forwarded to
            :func:`create_policy`.
    """

    def __init__(
        self,
        provider: str,
        *,
        policy_object: Policy | None = None,
        **config: Any,
    ) -> None:
        self._provider_arg = provider
        self._config = dict(config)
        # Serialise inference across threads (sync path) and coroutines (async
        # path). The wrapped model holds per-episode state, so concurrent
        # get_actions on one shared handle must not interleave.
        self._call_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        if policy_object is not None:
            self._inner: Policy = policy_object
        else:
            self._inner = create_policy(provider, **config)
        # Snapshot the wrapped policy's load telemetry so it surfaces on the
        # wrapper too (the runtime reads these off whatever object it is given).
        self.load_time_s: float = float(getattr(self._inner, "load_time_s", 0.0))
        self.load_cache_hit: bool = bool(getattr(self._inner, "load_cache_hit", False))

    @property
    def inner(self) -> Policy:
        """The wrapped policy instance."""
        return self._inner

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        async with self._async_lock:
            return await self._inner.get_actions(observation_dict, instruction, **kwargs)

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        with self._call_lock:
            return self._inner.get_actions_sync(observation_dict, instruction, **kwargs)

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._inner.set_robot_state_keys(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        self._inner.reset(seed)

    def set_control_frequency(self, hz: float) -> None:
        self._inner.set_control_frequency(hz)

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        self._inner.set_rtc_observed_delay(steps)

    @property
    def requires_images(self) -> bool:
        return self._inner.requires_images

    @property
    def execution_horizon(self) -> int:
        return self._inner.execution_horizon

    def is_chunk_emitting(self) -> bool:
        return self._inner.is_chunk_emitting()

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    def __getattr__(self, name: str) -> Any:
        # Forward any attribute not defined on the wrapper (e.g.
        # ``actions_per_step``, ``supports_rtc``, ``control_frequency``) to the
        # wrapped policy. Only invoked when normal lookup fails, so it never
        # shadows the explicit delegations above. ``object.__getattribute__``
        # avoids re-entering __getattr__ while fetching ``_inner`` itself.
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, name)


def preload(provider: str, **config: Any) -> dict[str, Any]:
    """Warm the model cache for a provider and report the load cost.

    Builds a :class:`PersistentPolicy` (which loads the model once into the
    process-level cache) and measures the wall time and resident-memory delta.
    Call this before a multi-episode run so every subsequent ``run_policy`` with
    the returned ``policy`` is a zero-reload cache hit.

    Args:
        provider: Provider name or smart string (see :func:`create_policy`).
        **config: Provider-specific keyword arguments.

    Returns:
        A dict with:
            ``policy``: the ready :class:`PersistentPolicy` to pass as
                ``policy_object=`` to ``run_policy``/``eval_policy``.
            ``load_time_s``: wall-clock seconds spent constructing the policy.
            ``load_cache_hit``: whether the heavy weight load was served from
                the process-level cache (a warm preload).
            ``resident_rss_mb``: process RSS in MB after the load (``None`` if
                unmeasurable on this platform).
            ``rss_delta_mb``: change in process RSS across the load (``None`` if
                unmeasurable); ~0 on a cache hit, large on a cold load.
    """
    import time

    rss_before = process_rss_mb()
    t0 = time.perf_counter()
    policy = PersistentPolicy(provider, **config)
    elapsed = time.perf_counter() - t0
    rss_after = process_rss_mb()
    # Prefer the wrapped policy's own measured load time (it isolates the weight
    # read from wrapper overhead); fall back to the wall clock around construction.
    load_time_s = float(getattr(policy, "load_time_s", 0.0)) or round(elapsed, 3)
    rss_delta = None
    if rss_before is not None and rss_after is not None:
        rss_delta = round(rss_after - rss_before, 1)
    return {
        "policy": policy,
        "load_time_s": round(load_time_s, 3),
        "load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
        "resident_rss_mb": None if rss_after is None else round(rss_after, 1),
        "rss_delta_mb": rss_delta,
    }


def list_cached() -> list[dict[str, Any]]:
    """List the heavy models currently resident in the process-level cache.

    Provider-agnostic introspection for deciding whether to evict before
    loading a different checkpoint. Currently the ``lerobot_local`` provider is
    the only one with a process-level weight cache; the list is empty when its
    optional dependencies are not installed.

    Returns:
        One dict per cached entry (see
        :func:`strands_robots.policies.lerobot_local.list_cached_models`), or an
        empty list when no cache is available.
    """
    try:
        from strands_robots.policies.lerobot_local import list_cached_models
    except ImportError:
        return []
    return list_cached_models()


def evict(pretrained_name_or_path: str | None = None) -> int:
    """Free cached models, returning held GPU/CPU memory.

    Args:
        pretrained_name_or_path: When ``None`` (default), evict every cached
            model. When set, evict only the entries loaded from that checkpoint
            - free one policy before switching to another without dropping the
            rest.

    Returns:
        Number of cache entries evicted (``0`` when no cache is available).
    """
    try:
        from strands_robots.policies.lerobot_local import clear_model_cache
    except ImportError:
        return 0
    return clear_model_cache(pretrained_name_or_path)
