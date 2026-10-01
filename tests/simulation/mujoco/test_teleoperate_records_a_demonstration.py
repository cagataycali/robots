"""A teleoperated MuJoCo session keeps sim time with the session and records.

``teleoperate`` drives the follower through ``send_action``, which takes one
physics step. At 30 Hz that advanced the world 2 ms per 33 ms tick, and because
only ``step`` feeds an open recording, a leader-to-follower demonstration saved
nothing: ``stop_recording`` refused the empty session. Each tick now steps the
world to the end of its control period, so the recording holds one frame per
tick and the world's clock reads the session's length.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot.datasets.lerobot_dataset")

HZ = 30


class _ScriptedLeader:
    """A leader arm that speaks lerobot keys and sweeps two joints."""

    is_connected = False

    def connect(self) -> None:
        self.is_connected, self._t0 = True, time.monotonic()

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        phase = math.sin(2 * math.pi * 0.5 * (time.monotonic() - self._t0))
        return {"shoulder_pan.pos": 0.4 * phase, "elbow_flex.pos": 0.3 * phase}


def test_a_teleoperated_session_records_one_frame_per_tick(tmp_path: Path) -> None:
    from strands_robots import Robot

    sim = Robot("so101", mode="sim")
    try:
        sim.attach_teleop(
            _ScriptedLeader(), name="leader", map_fn=lambda a: {k[: -len(".pos")]: v for k, v in a.items()}
        )
        opened = sim.start_recording(
            repo_id="local/teleop",
            task="follow",
            fps=HZ,
            root=str(tmp_path / "ds"),
            overwrite=True,
            cameras=["default"],
        )
        assert opened["status"] == "success", opened
        ran = sim.teleoperate(hz=HZ, duration=1.0, block=True)
        frames = ran["content"][1]["json"]["frames"]
        clock = re.search(r"t=([0-9.]+)s", sim.step(0)["content"][0]["text"])
        assert clock is not None, "step(0) did not report the sim clock"
        sim_time = float(clock.group(1))
        stopped = sim.stop_recording()
    finally:
        sim.destroy()

    assert ran["status"] == "success" and frames > 0, ran
    assert stopped["status"] == "success", stopped
    assert stopped["content"][1]["json"]["frame_count"] == frames
    assert sim_time == pytest.approx(frames / HZ, abs=2 * 0.002)
