"""#331: shared client-side RNG reseed helper + provider parity.

Pins that ``reseed_client_rngs`` reseeds Python ``random`` + NumPy (and torch
when present) deterministically, and that both Gr00tPolicy and Cosmos3Policy
route their reset reseed through it so they behave identically for #187
reproducibility.
"""

from __future__ import annotations

import random

import numpy as np

from strands_robots.policies._rng import reseed_client_rngs


def test_reseed_is_deterministic_across_python_and_numpy():
    reseed_client_rngs(1234)
    py_a = [random.random() for _ in range(3)]
    np_a = np.random.rand(3).tolist()

    reseed_client_rngs(1234)
    py_b = [random.random() for _ in range(3)]
    np_b = np.random.rand(3).tolist()

    assert py_a == py_b, "Python random must be reproducible after reseed"
    assert np_a == np_b, "NumPy RNG must be reproducible after reseed"


def test_reseed_none_is_noop():
    # Establish a known state, draw once, then call with None and confirm the
    # stream is NOT reset (the next draw differs from a fresh-seed draw).
    reseed_client_rngs(7)
    first = random.random()
    reseed_client_rngs(None)  # must not reset
    second = random.random()
    reseed_client_rngs(7)
    fresh = random.random()
    assert first == fresh, "reseed(7) must reproduce the first draw"
    assert second != first, "reseed(None) must be a no-op, not a reset"


def test_distinct_seeds_diverge():
    reseed_client_rngs(1)
    a = np.random.rand(4).tolist()
    reseed_client_rngs(2)
    b = np.random.rand(4).tolist()
    assert a != b


def test_both_providers_route_reset_through_shared_helper():
    """Source-level parity pin: both reset() methods call reseed_client_rngs
    so they cannot drift apart again (the #331 root cause)."""
    import inspect

    from strands_robots.policies.cosmos3 import policy as cosmos_mod
    from strands_robots.policies.groot import policy as groot_mod

    cosmos_src = inspect.getsource(cosmos_mod.Cosmos3Policy.reset)
    groot_src = inspect.getsource(groot_mod.Gr00tPolicy.reset)
    assert "reseed_client_rngs" in cosmos_src, "Cosmos3Policy.reset must use the shared reseed helper (#331)"
    assert "reseed_client_rngs" in groot_src, "Gr00tPolicy.reset must use the shared reseed helper (#331)"
    # The old global-only mutation must be gone from cosmos3.
    assert "np.random.seed(seed)" not in cosmos_src, (
        "Cosmos3Policy.reset must not reseed only the global NumPy RNG anymore (#331)"
    )


def test_reseed_seeds_torch_cpu_and_cuda_when_available(monkeypatch):
    """When torch is importable and reports CUDA, the helper must seed the CPU
    and CUDA generators and pin cuDNN into deterministic mode -- the per-episode
    reproducibility contract on GPU hosts (#331)."""
    import sys
    import types

    calls: dict[str, object] = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda s: calls.__setitem__("manual_seed", s)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda s: calls.__setitem__("manual_seed_all", s),
    )
    fake_torch.cuda = fake_cuda
    fake_torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reseed_client_rngs(99)

    assert calls["manual_seed"] == 99, "torch CPU generator must be seeded"
    assert calls["manual_seed_all"] == 99, "CUDA generators must be seeded when CUDA is available"
    assert fake_torch.backends.cudnn.deterministic is True, "cuDNN must be pinned deterministic"
    assert fake_torch.backends.cudnn.benchmark is False, "cuDNN autotuner must be disabled for determinism"


def test_reseed_skips_cuda_when_unavailable(monkeypatch):
    """On a CPU-only host the CUDA reseed must be skipped, not attempted."""
    import sys
    import types

    calls: dict[str, object] = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda s: calls.__setitem__("manual_seed", s)

    def _boom(_s):
        raise AssertionError("manual_seed_all must not be called when CUDA is unavailable")

    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=_boom)
    fake_torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reseed_client_rngs(5)

    assert calls["manual_seed"] == 5
    assert fake_torch.backends.cudnn.deterministic is True


def test_reseed_tolerates_missing_torch(monkeypatch):
    """torch is an optional dependency; a service-only / mock install without it
    must reseed Python + NumPy and silently skip torch (no ImportError leak)."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError

    reseed_client_rngs(11)
    a = np.random.rand(3).tolist()
    reseed_client_rngs(11)
    b = np.random.rand(3).tolist()
    assert a == b, "Python+NumPy reseed must still be deterministic without torch"


def test_reseed_swallows_unexpected_failures(monkeypatch, caplog):
    """reset() is a soft reproducibility hint: an unexpected reseed failure must
    be logged and swallowed, never propagated to the caller."""
    import logging

    def _boom(_seed):
        raise RuntimeError("rng backend exploded")

    monkeypatch.setattr(np.random, "seed", _boom)

    with caplog.at_level(logging.INFO, logger="strands_robots.policies._rng"):
        reseed_client_rngs(3)  # must not raise

    assert any("reseed failed" in r.message for r in caplog.records), "the swallowed failure must be logged"
