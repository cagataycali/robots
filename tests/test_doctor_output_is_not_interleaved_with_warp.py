"""``strands-robots doctor`` asks Warp for its devices without printing Warp's init banner.

The Warp probe called the device API without ``warp.config.quiet``, so Warp's
8-line "Warp 1.19.0 initialized: ... Devices ... Kernel cache" block landed in
the middle of the doctor's report, between the torch and Warp checks.
"""

from __future__ import annotations

import sys
import types

import pytest

from strands_robots import doctor


@pytest.mark.parametrize("modern", [True, False])
def test_the_warp_probe_is_quiet_before_its_first_device_query(monkeypatch: pytest.MonkeyPatch, modern: bool) -> None:
    """Warp 1.19+ gates the banner on ``config.log_level``; older builds on ``config.quiet``."""
    seen: dict[str, bool] = {}
    fake = types.ModuleType("warp")
    if modern:
        fake.LOG_INFO, fake.LOG_WARNING = 2, 3  # type: ignore[attr-defined]
        fake.config = types.SimpleNamespace(log_level=2)  # type: ignore[attr-defined]
    else:
        fake.config = types.SimpleNamespace(quiet=False)  # type: ignore[attr-defined]

    def is_cuda_available() -> bool:
        cfg = fake.config  # type: ignore[attr-defined]
        seen["quiet_at_first_query"] = cfg.log_level > 2 if modern else cfg.quiet
        return False

    fake.is_cuda_available = is_cuda_available  # type: ignore[attr-defined]
    fake.get_cuda_device_count = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "warp", fake)
    assert doctor._warp_cuda_report() is None
    assert seen == {"quiet_at_first_query": True}
