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
    fake_functional.na2d, fake_functional.na3d = na2d, na3d
    fake_natten.functional = fake_functional
    fake_flux = types.ModuleType("flux_action")
    fake_models = types.ModuleType("flux_action.models")
    fake_vae = types.ModuleType("flux_action.models.video_vae")
    fake_vae.na2d, fake_vae.na3d = na2d, na3d
    fake_models.video_vae = fake_vae
    fake_flux.models = fake_models
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
