"""``strands-robots doctor`` tells "no GPU" from "no torch".

Measured on a Jetson AGX Thor (L4T R38, CUDA 13.0, ``/dev/nvidia0`` present,
``nvidia-smi`` reporting "NVIDIA Thor") with the ``dev`` venv and no torch, at
0fa5ded90: the CUDA line said ``WARN torch not installed`` with the remedy
``uv pip install torch`` and the arch line said ``SKIP torch arch: no CUDA
device to compare against`` - a false claim, because the only CUDA probe the
command had was torch itself. The driver's own library answers without torch.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError


def _cpu_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.version = types.SimpleNamespace(cuda=None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)


def _driver_reports(monkeypatch: pytest.MonkeyPatch, devices: int | None) -> None:
    import strands_robots.doctor as doctor

    monkeypatch.setattr(doctor, "cuda_devices_per_driver", lambda: devices)


class TestTheCudaLine:
    def test_torch_missing_with_a_gpu_names_the_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_cuda

        _no_torch(monkeypatch)
        _driver_reports(monkeypatch, 1)
        line = check_cuda()
        assert "WARN" in line
        assert "1 CUDA device" in line
        assert "torch not installed" in line

    def test_torch_missing_without_a_gpu_does_not_invent_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_cuda

        _no_torch(monkeypatch)
        _driver_reports(monkeypatch, None)
        line = check_cuda()
        assert "CUDA device" not in line
        assert "uv pip install torch" in line

    def test_a_cpu_torch_next_to_a_gpu_is_called_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_cuda

        _cpu_torch(monkeypatch)
        _driver_reports(monkeypatch, 2)
        line = check_cuda()
        assert "CPU-only build" in line
        assert "2 CUDA device" in line

    def test_a_cpu_torch_with_no_gpu_says_so_instead_of_selling_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_cuda

        _cpu_torch(monkeypatch)
        _driver_reports(monkeypatch, None)
        line = check_cuda()
        assert "no CUDA device found" in line
        assert "UV_TORCH_BACKEND" not in line


class TestTheArchLine:
    def test_a_present_gpu_torch_cannot_see_is_not_reported_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_torch_arch

        _no_torch(monkeypatch)
        _driver_reports(monkeypatch, 1)
        line = check_torch_arch()
        assert "SKIP" in line
        assert "no CUDA device" not in line
        assert "present" in line

    def test_no_gpu_is_still_no_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_torch_arch

        _no_torch(monkeypatch)
        _driver_reports(monkeypatch, None)
        assert "no CUDA device to compare against" in check_torch_arch()


class TestTheRemedyFitsTheMachine:
    """PyPI's aarch64 torch carries CUDA 13 since 2.10 (2.11.0: 420 MB, pulls
    nvidia-cuda-runtime 13); a JetPack 6 driver (12.6) cannot load it."""

    @pytest.fixture
    def tegra(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
        import strands_robots.doctor as doctor

        release = tmp_path / "nv_tegra_release"
        real_path = doctor.Path

        def fake_path(arg: Any = ".", *rest: Any) -> Any:
            return release if str(arg) == "/etc/nv_tegra_release" else real_path(arg, *rest)

        monkeypatch.setattr(doctor, "Path", fake_path)
        monkeypatch.setattr(doctor.platform, "machine", lambda: "aarch64")
        return release

    def test_thor_jetpack_7_uses_pypi(self, tegra: Path) -> None:
        from strands_robots.doctor import _torch_cuda_remedy

        tegra.write_text("# R38 (release), REVISION: 2.2, GCID: 42205042, BOARD: generic, EABI: aarch64\n")
        remedy = _torch_cuda_remedy()
        assert remedy.startswith("uv pip install torch")
        assert "jetson-ai-lab" not in remedy

    def test_orin_jetpack_6_uses_the_jetson_index(self, tegra: Path) -> None:
        from strands_robots.doctor import _torch_cuda_remedy

        tegra.write_text("# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64\n")
        remedy = _torch_cuda_remedy()
        assert "https://pypi.jetson-ai-lab.io/jp6/cu126" in remedy
        assert "R36" in remedy

    def test_an_x86_box_gets_the_generic_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.doctor as doctor

        monkeypatch.setattr(doctor.platform, "machine", lambda: "x86_64")
        assert doctor._torch_cuda_remedy() == "UV_TORCH_BACKEND=auto uv pip install torch"


def test_the_driver_probe_answers_none_where_there_is_no_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a machine without libcuda and without the nvidia kernel module the
    answer is None - a real absence, which is what keeps the old SKIP honest."""
    import ctypes

    import strands_robots.doctor as doctor

    def no_lib(name: str) -> Any:
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", no_lib)
    real_path = doctor.Path
    monkeypatch.setattr(
        doctor,
        "Path",
        lambda a=".", *r: real_path("/nonexistent/gpus") if str(a).startswith("/proc/driver") else real_path(a, *r),
    )
    assert doctor.cuda_devices_per_driver() is None
