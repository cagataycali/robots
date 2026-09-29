"""The provider picks NATTEN's pure-torch backend when the installed kernels skip this GPU.

The published ``natten`` wheels (0.21.6, aarch64 cu130) carry kernel images for
sm_75, 80, 86, 89, 90, 100, 103 and 120 - not sm_110 (Jetson AGX Thor). NATTEN's
own probe only checks "compute capability >= 6.0", so on Thor it selects
``cutlass-fna`` and dies mid-rollout with ``no kernel image is available``. The
provider reads the arch list off ``libnatten`` with ``cuobjdump`` and pre-selects
``flex-fna`` when this device is missing. These tests pin the two pure pieces of
that decision (the arch parser and the na2d/na3d forwarding shim) without a GPU
or the flux_action install.
"""

from __future__ import annotations

import re
import sys
import types

import pytest

from strands_robots.policies.flux3_action import policy as module


def test_cuobjdump_arch_listing_parses_two_and_three_digit_sm_names() -> None:
    listing = "ELF file 1: libnatten.1.sm_75.cubin\nELF file 2: x.sm_100.cubin\nELF file 3: x.sm_120.cubin\nELF 4: x.sm_90.cubin"
    arches = {divmod(int(n), 10) for n in re.findall(r"\bsm_(\d+)\b", listing)}
    assert arches == {(7, 5), (10, 0), (12, 0), (9, 0)}
    assert (11, 0) not in arches  # Thor


def test_forwarding_shim_passes_the_chosen_fna_backend_as_the_backend_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def na2d(*args, attention_kwargs=None, backend=None, **kwargs):
        calls.append({"attention_kwargs": attention_kwargs, "backend": backend})
        return "na2d"

    def na3d(*args, attention_kwargs=None, backend=None, **kwargs):
        calls.append({"attention_kwargs": attention_kwargs, "backend": backend})
        return "na3d"

    fake_natten = types.ModuleType("natten")
    fake_functional = types.ModuleType("natten.functional")
    fake_flux = types.ModuleType("flux_action")
    fake_models = types.ModuleType("flux_action.models")
    fake_vae = types.ModuleType("flux_action.models.video_vae")
    # ModuleType carries no attribute table for mypy; the fakes are built by name.
    for mod, attrs in (
        (fake_functional, {"na2d": na2d, "na3d": na3d}),
        (fake_natten, {"functional": fake_functional}),
        (fake_vae, {"na2d": na2d, "na3d": na3d}),
        (fake_models, {"video_vae": fake_vae}),
        (fake_flux, {"models": fake_models}),
    ):
        for attr, value in attrs.items():
            setattr(mod, attr, value)
    for name, mod in {
        "natten": fake_natten,
        "natten.functional": fake_functional,
        "flux_action": fake_flux,
        "flux_action.models": fake_models,
        "flux_action.models.video_vae": fake_vae,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    module._forward_natten_backend_to_neighborhood_calls()
    module._forward_natten_backend_to_neighborhood_calls()  # idempotent

    assert fake_vae.na2d("q", "k", "v", kernel_size=[7, 7], attention_kwargs={"backend": "flex-fna"}) == "na2d"
    assert calls[-1] == {"attention_kwargs": {"backend": "flex-fna"}, "backend": "flex-fna"}
    # A dense (-fmha) choice is left to attention_kwargs, where NATTEN reads it.
    fake_vae.na3d("q", "k", "v", kernel_size=[1, 7, 7], attention_kwargs={"backend": "flex-fmha"})
    assert calls[-1] == {"attention_kwargs": {"backend": "flex-fmha"}, "backend": None}
    # An explicit backend= wins.
    fake_vae.na2d("q", "k", "v", attention_kwargs={"backend": "flex-fna"}, backend="cutlass-fna")
    assert calls[-1]["backend"] == "cutlass-fna"
    assert fake_vae._strands_natten_backend_forwarded is True


def test_shim_is_a_no_op_without_flux_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flux_action", None)
    monkeypatch.setitem(sys.modules, "flux_action.models", None)
    monkeypatch.setitem(sys.modules, "flux_action.models.video_vae", None)
    module._forward_natten_backend_to_neighborhood_calls()  # must not raise


def test_construction_does_not_load_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails before the fix: ``create_policy("flux3_action")`` pulled 3.9B parameters onto the GPU.

    A name-only construction (the trust-gate control grader does exactly that for every
    provider) must stay cheap; the weights arrive on ``load()`` / first ``reset()``.
    """
    calls: list[str] = []
    so101 = types.ModuleType("flux_action.inference.so101")

    def load_policy(name: str, revision: str | None = None) -> object:  # noqa: ARG001
        calls.append(name)
        raise RuntimeError("load_policy must not run at construction")

    so101.load_policy = load_policy  # type: ignore[attr-defined]
    for name in ("flux_action", "flux_action.inference", "flux_action.models", "flux_action.models.video_vae"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "flux_action.inference.so101", so101)
    # No torch, no natten: the optional imports and the shim are stood in for, so
    # the test runs on a box without either and never touches a GPU.
    monkeypatch.setattr(module, "require_optional", lambda name, **kwargs: types.ModuleType(name))
    monkeypatch.setattr(module, "_forward_natten_backend_to_neighborhood_calls", lambda: None)
    monkeypatch.setattr(module.Flux3ActionPolicy, "_select_natten_backend", lambda self, override: "flex-fna")
    policy = module.Flux3ActionPolicy(device="cpu")
    assert not policy.loaded and policy.load_s is None and calls == []
    with pytest.raises(RuntimeError, match="must not run at construction"):
        policy.load()
    assert calls == [module.DEFAULT_CHECKPOINT]


@pytest.mark.parametrize(("mode", "execute_steps", "expected"), [("queued", 32, 1), ("chunk", 32, 32), ("chunk", 8, 8)])
def test_chunk_mode_declares_the_chunk_it_returns(
    monkeypatch: pytest.MonkeyPatch, mode: str, execute_steps: int, expected: int
) -> None:
    """Fails before the fix: ``resolve_chunk_length`` handed the runner 8 of the 32 actions.

    ``"chunk"`` mode returns ``execute_steps`` actions from one stateless call, so
    that is the trained chunk the consumer executes before re-querying; ``"queued"``
    mode is one action per tick.
    """
    from strands_robots.policies.base import resolve_chunk_length

    for name in ("flux_action", "flux_action.inference", "flux_action.inference.so101"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(module, "require_optional", lambda name, **kwargs: types.ModuleType(name))
    monkeypatch.setattr(module, "_forward_natten_backend_to_neighborhood_calls", lambda: None)
    monkeypatch.setattr(module.Flux3ActionPolicy, "_select_natten_backend", lambda self, override: "flex-fna")
    policy = module.Flux3ActionPolicy(device="cpu", mode=mode, execute_steps=execute_steps)
    assert policy.actions_per_step == expected
    assert policy.execution_horizon == expected
    assert resolve_chunk_length(policy, 8) == max(8, expected)
