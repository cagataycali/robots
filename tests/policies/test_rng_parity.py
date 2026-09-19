"""#331: shared client-side RNG reseed helper + provider parity.

Pins that ``reseed_client_rngs`` reseeds Python ``random`` + NumPy (and torch
when present) deterministically, and that every provider holding its sampler in
this process - Gr00tPolicy, Cosmos3Policy, LerobotLocalPolicy - routes its
reset reseed through it, so they behave identically for #187 reproducibility
whether the rollout drives them in-process or over a ``PolicyServer``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

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


#: Every provider that samples in the process ``reset`` runs in, so the seed it
#: is handed is the only seeding that process gets. A rollout reaches them
#: through ``set_eval_seed`` when the policy is local, and through nothing at
#: all when the policy is served by a
#: :class:`~strands_robots.inference.server.PolicyServer` - the client's
#: reseed cannot cross a socket - so ``reset`` is where reproducibility is won
#: or lost for all three alike.
RESEEDING_PROVIDERS = [
    ("strands_robots.policies.cosmos3.policy", "Cosmos3Policy"),
    ("strands_robots.policies.groot.policy", "Gr00tPolicy"),
    ("strands_robots.policies.lerobot_local.policy", "LerobotLocalPolicy"),
]


@pytest.mark.parametrize(("module_path", "class_name"), RESEEDING_PROVIDERS, ids=lambda v: v.rsplit(".", 1)[-1])
def test_every_in_process_sampler_routes_reset_through_shared_helper(module_path: str, class_name: str) -> None:
    """Parity pin: one reseed path for every provider, so they cannot drift (#331)."""
    import importlib
    import inspect

    reset_src = inspect.getsource(getattr(importlib.import_module(module_path), class_name).reset)
    assert "reseed_client_rngs" in reset_src, f"{class_name}.reset must use the shared reseed helper (#331)"
    # Neither the old global-only NumPy mutation nor a discarded seed.
    assert "np.random.seed(seed)" not in reset_src, (
        f"{class_name}.reset must not reseed only the global NumPy RNG (#331)"
    )
    assert "del seed" not in reset_src, (
        f"{class_name}.reset must apply the seed, not discard it: a policy whose sampler draws from the "
        "process-global torch RNG is reproducible only if reset seeds that RNG, and over a PolicyServer "
        "this reset is the only seeding the inference process receives"
    )


def test_lerobot_local_reset_reseeds_the_process_it_samples_in() -> None:
    """Behaviour behind the parity pin: the same seed replays the same stream.

    A lerobot policy takes its flow-matching / diffusion noise from the
    process-global torch RNG and exposes no seed kwarg of its own, so a seeded
    episode is reproducible only when ``reset`` seeds that process. Pinned
    without a checkpoint because the reseed is not a property of any weights.
    """
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    policy = LerobotLocalPolicy()
    draws = []
    for _ in range(2):
        policy.reset(seed=4242)
        draws.append(([random.random() for _ in range(3)], np.random.rand(3).tolist()))
    assert draws[0] == draws[1], "reset(seed) must replay one stream, or a seeded episode is not reproducible"

    policy.reset(seed=4243)
    assert ([random.random() for _ in range(3)], np.random.rand(3).tolist()) != draws[0], (
        "a different seed must give a different stream"
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
