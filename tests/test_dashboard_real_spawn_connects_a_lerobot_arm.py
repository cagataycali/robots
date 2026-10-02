"""The dashboard's real-mode spawner must bring a lerobot-backed arm UP.

``device_manager._SPAWNER`` is the script every ``mode="real"`` spawn runs. It
connected the arm eagerly through ``robot.connect_eagerly()`` so that the mesh
would publish joints and camera frames right away -- ``Mesh._publish_cameras_once``
reads hardware frames only while the inner lerobot robot ``is_connected``.

But ``connect_eagerly`` is a contract of the native drivers (g1, ur, robotiq,
reachy). The lerobot-backed :class:`HardwareRobot`, which is what ``Robot("so101",
mode="real")`` builds, never had it: its way up is the ``_connect_robot()``
coroutine the teleop and task paths use. So on every SO-10x spawned from the
Devices sheet the script hit its except-branch, printed ``eager connect failed
(will retry on first task): 'Robot' object has no attribute 'connect_eagerly'``,
and the arm sat on the fleet with ``connected: false`` and no frame until a task
was run on it. Measured 2026-10-01 on a Mac with an SO-101 on
``/dev/cu.usbmodem5AB01584281`` and a UVC camera at index 1.

These tests run the real branch of the script against a fake ``strands_robots``
whose ``Robot`` is shaped like each driver family, and read what the script
printed. ``time.sleep`` is made to raise so the ``while True`` loop ends.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from strands_robots.dashboard import deploy, device_manager

_CFG = {
    "mode": "real",
    "robot_name": "so101",
    "port": "/dev/cu.usbmodemFAKE",
    "peer_id": "so101-test",
    "cameras": {"main": {"index_or_path": 1}},
}


def _run_spawner(tmp_path, robot_src: str) -> str:
    """Run ``_SPAWNER`` with a fake ``strands_robots.Robot`` defined by ``robot_src``."""
    pkg = tmp_path / "strands_robots"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent(robot_src), encoding="utf-8")
    harness = tmp_path / "run.py"
    harness.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            class _Stop(Exception):
                pass
            def _sleep(_):
                raise _Stop()
            time.sleep = _sleep
            sys.argv = ["spawner", {json.dumps(json.dumps(_CFG))}]
            try:
                exec(compile({device_manager._SPAWNER!r}, "<spawner>", "exec"), {{"__name__": "__main__"}})
            except _Stop:
                pass
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(harness)],
        cwd=tmp_path,  # the fake package shadows the real one
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


_LEROBOT_SHAPED = """
    # Shaped like HardwareRobot: no connect_eagerly, an async _connect_robot.
    class Robot:
        calls = []
        def __init__(self, *a, **kw):
            self.kw = kw
        async def _connect_robot(self):
            Robot.calls.append("_connect_robot")
            print("CONNECT_ROBOT_CALLED", flush=True)
            return True, ""
"""

_LEROBOT_REFUSES = """
    class Robot:
        def __init__(self, *a, **kw):
            pass
        async def _connect_robot(self):
            return False, "Robot so101 is not calibrated. Please calibrate the robot manually first"
"""

_NATIVE_SHAPED = """
    # Shaped like the native drivers (g1, ur, robotiq, reachy).
    class Robot:
        def __init__(self, *a, **kw):
            pass
        def connect_eagerly(self):
            print("CONNECT_EAGERLY_CALLED", flush=True)
            return True, {"wrist": "no camera at index 3"}, None
        async def _connect_robot(self):
            raise AssertionError("a driver with connect_eagerly must be brought up through it")
"""


def test_a_lerobot_shaped_robot_is_connected_through_connect_robot(tmp_path):
    out = _run_spawner(tmp_path, _LEROBOT_SHAPED)
    assert "CONNECT_ROBOT_CALLED" in out, out
    assert "hardware connected" in out, out
    assert "eager connect failed" not in out, out
    assert "has no attribute 'connect_eagerly'" not in out, out
    assert "so101-test (real @ /dev/cu.usbmodemFAKE) online" in out


def test_a_refused_connect_is_reported_with_the_drivers_reason(tmp_path):
    out = _run_spawner(tmp_path, _LEROBOT_REFUSES)
    assert "eager connect failed (will retry on first task): Robot so101 is not calibrated" in out, out
    assert "hardware connected" not in out
    # The refusal does not kill the peer: it still joins the mesh.
    assert "online" in out


def test_a_native_driver_keeps_connect_eagerly_and_its_degraded_cameras(tmp_path):
    out = _run_spawner(tmp_path, _NATIVE_SHAPED)
    assert "CONNECT_EAGERLY_CALLED" in out, out
    assert "camera 'wrist' unavailable, dropped: no camera at index 3" in out, out
    assert "hardware connected WITHOUT camera(s): wrist" in out, out


@pytest.mark.parametrize("has_eager", [True, False])
def test_the_deploy_snippet_handles_both_connect_contracts(tmp_path, has_eager):
    """The Deploy tab's snippet runs on the robot host; same two contracts, same fix."""
    payload = {"robot_name": "so101", "mode": "real", "port": "/dev/ttyACM0", "peer_id": "bench"}
    snippet = deploy.render_snippet(payload)["snippet"]
    assert "connect_eagerly" in snippet
    assert "_connect_robot" in snippet, "the lerobot contract is missing from the snippet"
    assert "hasattr(robot, 'connect_eagerly')" in snippet
