# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gsplat_rasterizer_available`` answers by rasterizing, not by importing.

A plain ``pip install gsplat`` is importable on a host whose CUDA kernels can
never build: gsplat JIT-compiles them through ``nvcc`` on first use, and a GPU
image that ships the CUDA runtime without the toolkit disables the backend
silently. The first ``gsplat.rasterization`` call then raises
``AttributeError: 'NoneType' object has no attribute 'CameraModelType'``. That
is why
:func:`~strands_robots.rendering.backgrounds.gsplat_rasterizer_available`
performs a one-gaussian trial rasterization instead of reporting on the import,
and why ``examples/isaac_gs/background.py`` can choose
:class:`~strands_robots.rendering.PanoramaBackground` up front rather than
erroring on every frame.

The suite already pins that the probe never raises on the host it runs on, but
a host without the ``sim-gs`` extra answers from the import guard alone, so the
three outcomes the probe exists to distinguish - no CUDA device, an importable
rasterizer that cannot rasterize, and a working one - were never reached. These
tests reach all three by standing ``torch`` and ``gsplat`` in, which also pins
the part a caller depends on and an import check cannot provide: a ``True``
verdict means a rasterization really ran.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.rendering.backgrounds import gsplat_rasterizer_available

# The documented failure of an importable-but-uncompiled gsplat: its CUDA
# backend is None, so the first rasterization attribute lookup fails.
_DISABLED_BACKEND = AttributeError("'NoneType' object has no attribute 'CameraModelType'")


class _Tensor:
    """Stand-in for a ``torch`` tensor: records how it was built, slices to itself."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __getitem__(self, item: Any) -> _Tensor:
        return _Tensor(f"{self.label}[{item!r}]")


def _torch_stand_in(*, cuda_available: bool) -> types.ModuleType:
    """A ``torch`` module exposing only what the probe reaches for."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]
    torch.tensor = lambda data, device=None: _Tensor(f"tensor@{device}")  # type: ignore[attr-defined]
    torch.full = lambda shape, value, device=None: _Tensor(f"full@{device}")  # type: ignore[attr-defined]
    torch.ones = lambda *shape, device=None: _Tensor(f"ones@{device}")  # type: ignore[attr-defined]
    torch.eye = lambda n, device=None: _Tensor(f"eye@{device}")  # type: ignore[attr-defined]
    return torch


def _gsplat_stand_in(raises: BaseException | None) -> tuple[types.ModuleType, list[dict[str, Any]]]:
    """A ``gsplat`` module whose ``rasterization`` records each call.

    Args:
        raises: Exception the rasterization raises, or ``None`` for a working one.

    Returns:
        The module and the list its ``rasterization`` appends a record to.
    """
    calls: list[dict[str, Any]] = []

    def rasterization(
        means: Any,
        quats: Any,
        scales: Any,
        opacities: Any,
        colors: Any,
        viewmats: Any,
        Ks: Any,  # noqa: N803 -- gsplat's own parameter name for the intrinsics batch
        width: int = 0,
        height: int = 0,
    ) -> str:
        calls.append({"width": width, "height": height, "means": means.label})
        if raises is not None:
            raise raises
        return "rasterized"

    module = types.ModuleType("gsplat")
    module.rasterization = rasterization  # type: ignore[attr-defined]
    return module, calls


@pytest.fixture
def probe(monkeypatch):
    """Build the probe's dependencies, returning ``(verdict, rasterization calls)``."""

    def _probe(*, cuda_available: bool = True, raises: BaseException | None = None):
        gsplat, calls = _gsplat_stand_in(raises)
        monkeypatch.setitem(sys.modules, "torch", _torch_stand_in(cuda_available=cuda_available))
        monkeypatch.setitem(sys.modules, "gsplat", gsplat)
        return gsplat_rasterizer_available(), calls

    return _probe


class TestTheProbeAnswersByRasterizing:
    """The three outcomes an import check cannot tell apart."""

    def test_a_working_rasterizer_reports_ok_and_really_rasterized(self, probe) -> None:
        """``True`` is earned by a completed rasterization, not by the import.

        This is the half no import check can supply: a caller that skips
        ``PanoramaBackground`` on this verdict is relying on a frame having
        actually been produced.
        """
        (ok, reason), calls = probe()

        assert (ok, reason) == (True, "ok")
        assert len(calls) == 1, "an ok verdict must come from exactly one trial rasterization"
        assert calls[0]["width"] > 0 and calls[0]["height"] > 0, (
            f"the trial must rasterize a real image, got {calls[0]['width']}x{calls[0]['height']}"
        )
        assert calls[0]["means"] == "tensor@cuda", "the trial gaussian must be built on the CUDA device"

    def test_an_importable_rasterizer_that_cannot_rasterize_is_reported_not_raised(self, probe) -> None:
        """The uncompiled-kernel case: importable, and unusable."""
        (ok, reason), calls = probe(raises=_DISABLED_BACKEND)

        assert ok is False
        assert "gsplat CUDA rasterizer unavailable" in reason
        assert "AttributeError" in reason and "CameraModelType" in reason, (
            f"the reason must name the failure a caller has to act on, got {reason!r}"
        )
        assert len(calls) == 1, "the verdict must come from an attempted rasterization"

    def test_no_cuda_device_is_reported_before_anything_is_rasterized(self, probe) -> None:
        """A host with no device is answered from the device check, not a failed trial."""
        (ok, reason), calls = probe(cuda_available=False)

        assert (ok, reason) == (False, "no CUDA device available to torch")
        assert calls == [], "the probe must not attempt a rasterization without a CUDA device"

    def test_an_absent_dependency_is_reported_as_an_import_failure(self, monkeypatch) -> None:
        """The control: no ``sim-gs`` extra, so the import guard answers."""
        monkeypatch.setitem(sys.modules, "torch", _torch_stand_in(cuda_available=True))
        monkeypatch.setitem(sys.modules, "gsplat", None)

        ok, reason = gsplat_rasterizer_available()

        assert ok is False
        assert "not importable" in reason, f"the reason must name the missing dependency, got {reason!r}"
