"""The Warp verdict is about the arch table Warp reports, not the wheel installed.

Warp chooses a device's architecture out of the table its binary was built with.
A build older than the GPU therefore compiles for a *different* architecture
instead of refusing: on an NVIDIA Thor, whose driver reports ``sm_110``, the
CUDA-12 build offers a table with no ``110`` in it and settles on ``sm_101``. A
simple kernel still returns the right answer under that substitution, so nothing
surfaces until a kernel needs an instruction ``sm_101`` cannot express - which is
why the doctor has to ask rather than waiting for a runtime error.

A wheel tag cannot answer the question. PyPI's ``warp_lang`` filenames carry no
local version segment, so one release's CUDA-12 and CUDA-13 builds are
indistinguishable by name, while the arch table is the value Warp itself reads.

The archs below are written out here rather than imported, so these cases grade
the shipped behaviour against a second opinion. The pair (a table missing the
device's arch, and one containing it) is taken from the two builds of Warp
1.16.0: the CUDA-12.9 build reports ``101`` and ``103`` but not ``110``, and the
CUDA-13.0 build reports ``110``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

# The device architecture an NVIDIA Thor's driver reports, and the arch Warp's
# CUDA-12 build settles on there. 103 is in that build's table too and sits
# below 110, so it is what a "nearest supported arch below the device's" rule
# would name - and Warp does not follow that rule.
THOR_ARCH = 110
CUDA12_CHOSEN_ARCH = 101
CUDA12_NEAREST_BELOW = 103
CUDA12_TABLE = (50, 52, 70, 80, 89, 90, 100, CUDA12_CHOSEN_ARCH, CUDA12_NEAREST_BELOW, 120, 121)
CUDA13_TABLE = (75, 80, 89, 90, 100, 103, THOR_ARCH, 120, 121)


def _install_torch(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int] | None) -> None:
    """Make the driver report ``capability`` for device 0, or no CUDA device at all."""
    cuda = SimpleNamespace(
        is_available=lambda: capability is not None,
        get_device_capability=lambda _index: capability,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))


def _install_warp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    table: tuple[int, ...],
    chosen: int,
    toolkit: tuple[int, int],
    driver: tuple[int, int] = (13, 0),
) -> None:
    """Make Warp report ``table`` as its build's archs and ``chosen`` for device 0."""
    warp: Any = SimpleNamespace(
        is_cuda_available=lambda: True,
        get_cuda_device_count=lambda: 1,
        get_cuda_supported_archs=lambda: list(table),
        get_device=lambda _alias: SimpleNamespace(arch=chosen),
        get_cuda_toolkit_version=lambda: toolkit,
        get_cuda_driver_version=lambda: driver,
    )
    monkeypatch.setitem(sys.modules, "warp", warp)


def _thor_on_cuda12(monkeypatch: pytest.MonkeyPatch) -> str:
    """The verdict for a Thor running Warp's CUDA-12 build."""
    from strands_robots.doctor import check_warp_arch

    _install_torch(monkeypatch, (11, 0))
    _install_warp(monkeypatch, table=CUDA12_TABLE, chosen=CUDA12_CHOSEN_ARCH, toolkit=(12, 9))
    return check_warp_arch()


class TestABuildOlderThanTheGpuIsRefused:
    """A build whose arch table cannot name the device is reported, not tolerated."""

    def test_a_table_without_the_device_arch_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _thor_on_cuda12(monkeypatch)
        assert "FAIL" in result
        assert f"sm_{THOR_ARCH}" in result

    def test_the_refusal_names_the_arch_warp_settled_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The substitution is the number a user saw in Warp's own banner."""
        assert f"sm_{CUDA12_CHOSEN_ARCH}" in _thor_on_cuda12(monkeypatch)

    def test_the_refusal_does_not_name_the_nearest_supported_arch_below(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Warp's choice is read from Warp, not derived from its table.

        Deriving it would name ``103`` here - the highest entry below ``110`` -
        and describe a rule Warp does not follow.
        """
        assert str(CUDA12_NEAREST_BELOW) not in _thor_on_cuda12(monkeypatch)

    def test_the_refusal_names_the_toolkit_the_build_was_made_against(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The toolkit version is what distinguishes two builds of one release."""
        assert "CUDA 12.9" in _thor_on_cuda12(monkeypatch)

    def test_the_remedy_names_the_cuda_major_the_driver_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wheel to fetch is the one matching the driver, not the one PyPI serves."""
        result = _thor_on_cuda12(monkeypatch)
        assert "+cu13" in result
        assert "github.com/NVIDIA/warp/releases" in result


class TestABuildThatCoversTheDeviceIsAccepted:
    """The verdict turns on the table alone, so a matching build passes."""

    def test_a_table_containing_the_device_arch_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_warp_arch

        _install_torch(monkeypatch, (11, 0))
        _install_warp(monkeypatch, table=CUDA13_TABLE, chosen=THOR_ARCH, toolkit=(13, 0))
        result = check_warp_arch()
        assert "PASS" in result
        assert f"sm_{THOR_ARCH}" in result

    def test_an_older_gpu_on_the_cuda12_build_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refusal is about this device, not about the build's age."""
        from strands_robots.doctor import check_warp_arch

        _install_torch(monkeypatch, (8, 9))
        _install_warp(monkeypatch, table=CUDA12_TABLE, chosen=89, toolkit=(12, 9))
        assert "PASS" in check_warp_arch()


class TestTheCheckDeclinesWhenThereIsNothingToCompare:
    """No device, or no Warp, is not a failure - there is no disagreement to report."""

    def test_no_cuda_device_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_warp_arch

        _install_torch(monkeypatch, None)
        _install_warp(monkeypatch, table=CUDA12_TABLE, chosen=CUDA12_CHOSEN_ARCH, toolkit=(12, 9))
        assert "SKIP" in check_warp_arch()

    def test_torch_absent_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch is this module's only driver query, so without it there is no truth to compare to."""
        from strands_robots.doctor import check_warp_arch

        monkeypatch.setitem(sys.modules, "torch", None)
        _install_warp(monkeypatch, table=CUDA12_TABLE, chosen=CUDA12_CHOSEN_ARCH, toolkit=(12, 9))
        assert "SKIP" in check_warp_arch()

    def test_warp_absent_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Warp ships in the ``sim-newton`` extra, so most installs have nothing to check."""
        from strands_robots.doctor import check_warp_arch

        _install_torch(monkeypatch, (11, 0))
        monkeypatch.setitem(sys.modules, "warp", None)
        assert "SKIP" in check_warp_arch()

    def test_warp_seeing_no_cuda_device_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_warp_arch

        _install_torch(monkeypatch, (11, 0))
        warp: Any = SimpleNamespace(is_cuda_available=lambda: False, get_cuda_device_count=lambda: 0)
        monkeypatch.setitem(sys.modules, "warp", warp)
        assert "SKIP" in check_warp_arch()


class TestTheDoctorRunsTheCheckAndFailsOnIt:
    """A verdict nothing runs is decoration, so the wiring is graded too."""

    def test_a_refusal_from_the_check_fails_the_doctor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from strands_robots import doctor

        sentinel = "  FAIL  warp-arch-sentinel"
        monkeypatch.setattr(doctor, "check_warp_arch", lambda: sentinel)
        exit_code = doctor.run_doctor()
        assert sentinel in capsys.readouterr().out
        assert exit_code == 1


class TestAgainstTheInstalledWarp:
    """The stand-ins above agree with a real Warp about what it reports."""

    def test_the_installed_build_can_target_this_devices_arch(self) -> None:
        pytest.importorskip("warp", reason="warp ships in the sim-newton extra")
        from strands_robots.doctor import _driver_compute_arch, _warp_cuda_report

        device_arch = _driver_compute_arch()
        report = _warp_cuda_report()
        if device_arch is None or report is None:
            pytest.skip("no CUDA device for Warp to report on")
        supported, chosen, _toolkit, _driver = report
        assert device_arch in supported, (
            f"the installed Warp build targets {sorted(supported)} and settled on sm_{chosen} "
            f"for this sm_{device_arch} device"
        )
        assert chosen == device_arch

    def test_a_real_warp_answers_every_call_the_stand_ins_make(self) -> None:
        """The stand-ins are only faithful while these remain Warp's public surface."""
        warp = pytest.importorskip("warp", reason="warp ships in the sim-newton extra")
        for name in (
            "is_cuda_available",
            "get_cuda_device_count",
            "get_cuda_supported_archs",
            "get_device",
            "get_cuda_toolkit_version",
            "get_cuda_driver_version",
        ):
            assert callable(getattr(warp, name)), name
