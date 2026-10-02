"""An SO-arm with no calibration file comes up on the native driver, says so, and writes nothing.

The owner's bench arm has never been calibrated (no file under lerobot's
calibration store), and until the native default (2026-10-01) the lerobot path
refused it at connect: "Robot so101 is not calibrated. Please calibrate the
robot manually first". The native :class:`FeetechDriver` takes the servo's
full travel instead, which is the right fallback for a bench - the arm reads
and publishes - as long as nobody mistakes those degrees for calibrated ones.

Three things are graded here, each one a sentence a dashboard needs:

* the connect path neither refuses for a missing calibration nor writes to a
  servo - the fake bus records every write, and the only traffic is the open;
* status and presence carry ``calibration``: ``none (raw servo counts ...)``
  until a file is given, the file's path after;
* the dashboard spawner reads the native ``connect_eagerly`` contract
  (``str | None`` plus ``camera_failures``) instead of unpacking a 3-tuple -
  measured before the fix with a native-shaped fake returning ``None``::

      eager connect failed (will retry on first task): cannot unpack non-iterable NoneType object

  which is what every shipped native driver would have printed on a
  SUCCESSFUL connect.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from strands_robots import Robot
from strands_robots.dashboard import device_manager
from strands_robots.drivers.feetech.bus import FeetechBus, full_travel_calibration
from strands_robots.drivers.feetech.driver import FeetechDriver


class _RecordingBus(FeetechBus):
    """A bus whose port always opens and whose every write is recorded."""

    writes: list[tuple[str, Any]] = []

    def connect(self) -> None:
        if self.is_connected:
            return
        self._conn = _OpenPort()

    def set_torque(self, enabled: bool, motor_ids: Any = None) -> list[int]:
        self.writes.append(("set_torque", enabled))
        return []

    def write_goal_positions(self, targets: Any) -> None:
        self.writes.append(("write_goal_positions", dict(targets)))

    def sync_read(self, register: str = "Present_Position", num_retry: int = 0) -> dict[str, float]:
        return {name: 180.0 for name in self.motors}


class _OpenPort:
    is_open = True

    def close(self) -> None:
        self.is_open = False


@pytest.fixture(autouse=True)
def _recording_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FeetechDriver, "BUS", _RecordingBus)
    _RecordingBus.writes = []


def _arm(**kwargs: Any) -> FeetechDriver:
    robot: Any = Robot("so101", mode="real", port="/dev/cu.usbmodemFAKE", **kwargs)
    assert isinstance(robot, FeetechDriver), f"premise: the bare so101 call builds the native driver, got {type(robot)}"
    return robot


class TestAnUncalibratedArmComesUp:
    def test_connect_does_not_refuse_for_a_missing_calibration(self) -> None:
        arm = _arm()
        assert arm.connect_eagerly() is None
        assert arm.is_connected

    def test_connect_writes_nothing_to_the_servos(self) -> None:
        """Torque is left exactly as the operator left it: no enable sweep, no goal."""
        arm = _arm()
        arm.connect_eagerly()
        assert _RecordingBus.writes == []

    def test_the_bus_reads_the_full_travel(self) -> None:
        arm = _arm()
        assert arm.bus.calibration == full_travel_calibration(arm.bus.motors)

    def test_status_says_the_degrees_are_raw_counts(self) -> None:
        arm = _arm()
        payload = asyncio.run(arm.get_status())["content"][0]["json"]
        assert payload["calibration_source"] is None
        assert payload["calibration"].startswith("none (raw servo counts")

    def test_a_calibration_file_replaces_the_fact(self, tmp_path: Path) -> None:
        records = {
            name: {
                "id": spec.motor_id,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 100,
                "range_max": 4000,
            }
            for name, spec in FeetechDriver.MOTORS.items()
        }
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        arm = _arm(calibration=str(path))
        payload = asyncio.run(arm.get_status())["content"][0]["json"]
        assert payload["calibration"] == str(path)

    def test_presence_carries_the_fact(self) -> None:
        from strands_robots.mesh.core import Mesh

        arm = _arm()
        arm.connect_eagerly()
        payload = Mesh(arm, peer_id="so101-bench", peer_type="robot")._build_presence()
        assert payload["connected"] is True
        assert payload["calibration"].startswith("none (raw servo counts")


_CFG = {
    "robot_name": "so101",
    "mode": "real",
    "port": "/dev/cu.usbmodemFAKE",
    "peer_id": "so101-test",
    "cameras": {"main": {"index_or_path": 1}},
}

_SHIPPED_NATIVE_SHAPE = """
    # Shaped like every shipped native driver: connect_eagerly() -> str | None,
    # camera failures on their own attribute, a calibration sentence.
    class Robot:
        camera_failures = {"main": "camera 'main': could not open 1"}
        calibration_fact = "none (raw servo counts over the full travel)"
        def __init__(self, *a, **kw):
            pass
        def connect_eagerly(self):
            print("CONNECT_EAGERLY_CALLED", flush=True)
            return None
"""

_SHIPPED_NATIVE_REFUSES = """
    class Robot:
        camera_failures = {}
        def __init__(self, *a, **kw):
            pass
        def connect_eagerly(self):
            return "FeetechBus: could not open /dev/cu.usbmodemFAKE"
"""


def _run_spawner(tmp_path: Path, robot_src: str) -> str:
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
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestTheSpawnerReadsTheNativeContract:
    def test_a_successful_native_connect_is_reported_as_connected(self, tmp_path: Path) -> None:
        out = _run_spawner(tmp_path, _SHIPPED_NATIVE_SHAPE)
        assert "CONNECT_EAGERLY_CALLED" in out, out
        assert "cannot unpack" not in out, out
        assert "camera 'main' unavailable, dropped: camera 'main': could not open 1" in out, out
        assert "hardware connected WITHOUT camera(s): main" in out, out
        assert "calibration: none (raw servo counts over the full travel)" in out, out

    def test_a_refused_native_connect_is_reported_with_the_bus_reason(self, tmp_path: Path) -> None:
        out = _run_spawner(tmp_path, _SHIPPED_NATIVE_REFUSES)
        assert "eager connect failed (will retry on first task): FeetechBus: could not open" in out, out
        assert "hardware connected" not in out
        assert "online" in out
