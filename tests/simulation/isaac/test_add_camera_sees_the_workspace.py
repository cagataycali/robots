"""An ``add_camera`` camera renders what is closer than one metre.

USD's ``Camera`` default ``clippingRange`` is ``(1, 1e6)`` metres, and the
``Camera`` sensor keeps it: measured on Isaac Sim 6.1, a camera 0.8 m from a
tabletop scene rendered only the far floor - the robot and both cubes were
culled - so every manipulation camera (and any policy or recording reading it)
saw nothing. The viewport camera uses a 1 cm near plane; ``_create_camera_prim``
now sets the same.
"""

from __future__ import annotations

from typing import Any

import pytest

from .test_add_camera_fov_is_vertical import _FakeCamera, _make_camera


def test_the_near_plane_is_a_centimetre(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _record(self: Any, near_distance: float | None = None, far_distance: float | None = None) -> None:
        calls.append({"near": near_distance, "far": far_distance})

    monkeypatch.setattr(_FakeCamera, "set_clipping_range", _record, raising=False)
    _make_camera(monkeypatch)

    assert calls, "add_camera left USD's 1 m default near plane in place"
    assert calls[-1]["near"] == pytest.approx(0.01)
    assert calls[-1]["far"] >= 1.0e3
