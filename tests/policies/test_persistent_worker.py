"""PersistentPolicy and the preload/list_cached/evict cache controls.

Loading a VLA checkpoint is the dominant cost in a multi-episode rollout. The
process-level model cache (tested in
``tests/policies/lerobot_local/test_model_cache.py``) already shares resident
weights across instances; these tests pin the ergonomic layer on top of it:

* :class:`PersistentPolicy` builds the wrapped policy ONCE and reuses it across
  many calls, delegating the full Policy contract transparently.
* Concurrent inference on one shared handle is serialised (no interleaving of
  the wrapped model's per-episode state).
* :func:`preload` warms the cache and reports honest load/RSS telemetry.
* :func:`list_cached` / :func:`evict` introspect and free the cache.

The tests use the dependency-free ``mock`` provider so they run without torch /
lerobot, exercising the wrapper contract rather than any specific backend.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from strands_robots.policies import (
    PersistentPolicy,
    evict,
    list_cached,
    preload,
)
from strands_robots.policies.base import Policy
from strands_robots.policies.mock import MockPolicy


class _CountingPolicy(MockPolicy):
    """MockPolicy that counts constructions to prove single-load reuse."""

    build_count = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).build_count += 1
        self.load_time_s = 0.0
        self.load_cache_hit = False


def _obs():
    return {"observation.state": [0.0] * 6}


class TestPersistentPolicyWrapper:
    def test_builds_once_and_reuses_inner(self):
        _CountingPolicy.build_count = 0
        inner = _CountingPolicy()
        assert _CountingPolicy.build_count == 1
        wrapper = PersistentPolicy("mock", policy_object=inner)
        # Wrapping an existing object never rebuilds it.
        assert _CountingPolicy.build_count == 1
        assert wrapper.inner is inner

    def test_is_a_policy(self):
        wrapper = PersistentPolicy("mock", policy_object=MockPolicy())
        assert isinstance(wrapper, Policy)

    def test_delegates_inference(self):
        inner = MockPolicy()
        inner.set_robot_state_keys(["j1", "j2", "j3", "j4", "j5", "j6"])
        wrapper = PersistentPolicy("mock", policy_object=inner)
        wrapper.set_robot_state_keys(["j1", "j2", "j3", "j4", "j5", "j6"])
        actions = wrapper.get_actions_sync(_obs(), "")
        assert isinstance(actions, list) and len(actions) >= 1
        assert all(isinstance(a, dict) for a in actions)

    def test_forwards_introspection_and_telemetry(self):
        inner = MockPolicy()
        inner.load_time_s = 12.5
        inner.load_cache_hit = True
        wrapper = PersistentPolicy("mock", policy_object=inner)
        # Provider identity, chunk-shape introspection, and image need all
        # reflect the wrapped policy so the runtime drives it identically.
        assert wrapper.provider_name == inner.provider_name
        assert wrapper.requires_images == inner.requires_images
        assert wrapper.execution_horizon == inner.execution_horizon
        assert wrapper.is_chunk_emitting() == inner.is_chunk_emitting()
        # Snapshotted load telemetry surfaces on the wrapper.
        assert wrapper.load_time_s == 12.5
        assert wrapper.load_cache_hit is True

    def test_reset_and_control_hooks_delegate(self):
        inner = MockPolicy()
        wrapper = PersistentPolicy("mock", policy_object=inner)
        wrapper.set_control_frequency(50.0)
        wrapper.set_rtc_observed_delay(3)
        wrapper.reset(seed=7)
        # Hooks land on the wrapped object (the one the runtime's get_actions
        # ultimately reads), not on the wrapper's inherited class defaults.
        assert inner.control_frequency == 50.0
        assert inner.rtc_observed_delay_steps == 3

    def test_concurrent_inference_is_serialised(self):
        # Two threads share one handle; the per-call lock must serialise them so
        # no two inferences run on the wrapped model's mutable state at once.
        inner = MockPolicy()
        inner.set_robot_state_keys(["j1", "j2", "j3", "j4", "j5", "j6"])
        wrapper = PersistentPolicy("mock", policy_object=inner)

        overlap = {"max": 0}
        active = {"n": 0}
        guard = threading.Lock()
        orig = inner.get_actions_sync

        def instrumented(*a, **k):
            with guard:
                active["n"] += 1
                overlap["max"] = max(overlap["max"], active["n"])
            try:
                return orig(*a, **k)
            finally:
                with guard:
                    active["n"] -= 1

        inner.get_actions_sync = instrumented  # type: ignore[method-assign]

        results: list[int] = []
        rlock = threading.Lock()

        def worker():
            for _ in range(20):
                acts = wrapper.get_actions_sync(_obs(), "")
                with rlock:
                    results.append(len(acts))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 40
        # The wrapper's lock guarantees at most one inference in flight.
        assert overlap["max"] == 1

    def test_async_get_actions_delegates_to_inner(self):
        # The async entry point (used by async runtimes) must delegate to the
        # wrapped policy's coroutine under the async lock and return its
        # per-tick action dicts - not a separately rebuilt or zeroed result.
        inner = MockPolicy()
        inner.set_robot_state_keys(["j1", "j2", "j3", "j4", "j5", "j6"])
        wrapper = PersistentPolicy("mock", policy_object=inner)
        wrapper.set_robot_state_keys(["j1", "j2", "j3", "j4", "j5", "j6"])
        actions = asyncio.run(wrapper.get_actions(_obs(), ""))
        assert isinstance(actions, list) and len(actions) >= 1
        assert all(isinstance(a, dict) for a in actions)

    def test_forwards_unknown_attribute_to_inner(self):
        # Attributes not explicitly delegated on the wrapper (e.g. provider
        # extras the runtime may probe) fall through __getattr__ to the wrapped
        # policy rather than raising AttributeError.
        inner = MockPolicy()
        inner.custom_runtime_hint = 42  # type: ignore[attr-defined]
        wrapper = PersistentPolicy("mock", policy_object=inner)
        assert wrapper.custom_runtime_hint == 42

    def test_unknown_attribute_absent_on_inner_raises(self):
        # A name missing on BOTH wrapper and inner still raises AttributeError
        # (transparent forwarding, not silent None).
        wrapper = PersistentPolicy("mock", policy_object=MockPolicy())
        with pytest.raises(AttributeError):
            _ = wrapper.no_such_attribute_anywhere


class TestPreload:
    def test_returns_policy_and_telemetry(self):
        out = preload("mock")
        assert isinstance(out["policy"], PersistentPolicy)
        assert isinstance(out["load_time_s"], float)
        assert out["load_cache_hit"] in (True, False)
        # RSS is a positive float when measurable, else None on a bare platform.
        rss = out["resident_rss_mb"]
        assert rss is None or (isinstance(rss, float) and rss > 0.0)
        assert "rss_delta_mb" in out

    def test_preloaded_policy_is_reusable(self):
        policy = preload("mock")["policy"]
        policy.set_robot_state_keys(["j1", "j2", "j3", "j4", "j5", "j6"])
        first = policy.get_actions_sync(_obs(), "")
        second = policy.get_actions_sync(_obs(), "")
        assert len(first) >= 1 and len(second) >= 1


class TestCacheControls:
    def test_list_cached_returns_list(self):
        # No model loaded through the cache here; the call must still be safe and
        # return a list (empty when the lerobot cache is unavailable/empty).
        assert isinstance(list_cached(), list)

    def test_evict_returns_int(self):
        # Evicting a checkpoint that is not resident frees nothing but reports a
        # count rather than raising - safe to call defensively before a run.
        assert evict("definitely/not-loaded") == 0
        assert isinstance(evict(), int)

    def test_list_cached_degrades_when_cache_backend_unavailable(self, monkeypatch):
        # When the lerobot_local cache backend cannot be imported (optional deps
        # absent), list_cached degrades to an empty list instead of raising.
        monkeypatch.setitem(sys.modules, "strands_robots.policies.lerobot_local", None)
        assert list_cached() == []

    def test_evict_degrades_when_cache_backend_unavailable(self, monkeypatch):
        # Same optional-dep degrade for evict: report zero entries freed rather
        # than raising, so a defensive pre-run evict is always safe to call.
        monkeypatch.setitem(sys.modules, "strands_robots.policies.lerobot_local", None)
        assert evict() == 0
        assert evict("some/checkpoint") == 0
