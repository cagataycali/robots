"""``Robot(..., ros2_bridge=True, ros2_transport="rtps")`` in sim mode names the real problem.

The pure-RTPS transport is a hardware-bridge option; the simulation bridge is
rclpy-only. Before this guard the sim backend absorbed ``ros2_transport`` via
``**kwargs`` and the caller was told "'rclpy' is required" - the one dependency
the rtps choice was documented to avoid.
"""

from __future__ import annotations

import pytest

from strands_robots import Robot


def test_sim_rtps_transport_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"ros2_transport='rtps' is a hardware option") as exc:
        Robot("so100", ros2_bridge=True, ros2_transport="rtps")
    assert "mode='real'" in str(exc.value)
    assert "rclpy is required" not in str(exc.value)


def test_sim_rtps_transport_refused_even_without_bridge_flag() -> None:
    # The kwarg alone would be silently dropped; it is refused whether or not
    # the bridge is on, so the caller learns the option does not apply here.
    with pytest.raises(ValueError, match="hardware option"):
        Robot("so100", ros2_transport="rtps")


def test_sim_explicit_rclpy_transport_still_reaches_the_rclpy_check() -> None:
    # ``ros2_transport="rclpy"`` is the sim bridge's own transport; the only
    # error left is the honest one about rclpy itself (when it is not installed).
    pytest.importorskip("mujoco")
    try:
        import rclpy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="rclpy"):
            Robot("so100", ros2_bridge=True, ros2_transport="rclpy")
    else:
        pytest.skip("rclpy present: the sim bridge would start for real")
