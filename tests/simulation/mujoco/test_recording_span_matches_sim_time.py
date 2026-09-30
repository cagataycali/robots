"""A 30 fps recording on MuJoCo defaults covers the sim time it declares (#4392).

Measured on main d9b78b45: ``start_recording(fps=30)`` then ``run_policy(duration=2,
control_frequency=30)`` wrote 60 frames stamped 0 .. 1.967 s while
``mj_data.time`` reached 2.040 s (17 steps of 0.002 s per action). The rollout
must now leave the clock within one physics step of ``frames / fps``.
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def test_sixty_frames_at_30_fps_leave_the_clock_at_two_seconds() -> None:
    sim = Simulation()
    try:
        sim.create_world(ground_plane=True)
        sim.add_robot("so101")
        dt = sim.physics_timestep()
        assert dt == pytest.approx(0.002)
        with tempfile.TemporaryDirectory() as root:
            started = sim.start_recording(repo_id="local/span-4392", task="span", fps=30, root=root, cameras=[])
            assert started["status"] == "success", started
            t0 = float(sim.mj_data.time)
            result = sim.run_policy(
                robot_name="so101", policy_provider="mock", n_steps=60, control_frequency=30, fast_mode=True
            )
            assert result["status"] == "success", result
            advanced = float(sim.mj_data.time) - t0
            stopped = sim.stop_recording()
            assert stopped["status"] == "success", stopped
        [saved] = [block["json"] for block in stopped["content"] if isinstance(block, dict) and "json" in block]
        frames = int(saved["frame_count"])
        assert frames == 60, saved
        # 1.967 s of timestamps over 2.040 s of sim time was the bug: 2 percent early.
        assert advanced == pytest.approx(frames / 30.0, abs=dt), advanced
    finally:
        sim.cleanup()
