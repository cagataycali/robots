"""Hardware degrees are exact: counts -> rad (driver) -> degrees (units) equals LeRobot's own MotorsBus to 1e-6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots.units import calibrated, lerobot_units

# Imported at collection, before any test purges sys.modules: torch cannot be imported twice in one process.
motors_mod = pytest.importorskip("lerobot.motors")
FeetechMotorsBus = pytest.importorskip("lerobot.motors.feetech").FeetechMotorsBus

CAL = json.loads((Path(__file__).parent / "fixtures" / "so101_follower_calibration.json").read_text())


def test_feetech_degrees_match_lerobot_motors_bus():
    norm = {"*": "degrees", "gripper": "range_0_100"}
    motors = {
        n: motors_mod.Motor(
            c["id"],
            "sts3215",
            motors_mod.MotorNormMode.RANGE_0_100 if n == "gripper" else motors_mod.MotorNormMode.DEGREES,
        )
        for n, c in CAL.items()
    }
    cal = {n: motors_mod.MotorCalibration(**c) for n, c in CAL.items()}
    bus = FeetechMotorsBus(port="/dev/null", motors=motors, calibration=cal)
    driver, _ = make_driver(cal=CAL)
    units = calibrated(lerobot_units(driver.description, norm), driver.calibration, resolution=4096)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        counts = {n: int(rng.integers(c["range_min"], c["range_max"] + 1)) for n, c in CAL.items()}
        theirs = bus._normalize({CAL[n]["id"]: v for n, v in counts.items()})
        rad = np.array([driver.counts_to_rad(n, counts[n]) for n in driver.names])
        ours = units.from_rad(rad)
        for i, n in enumerate(driver.names):
            worst = max(worst, abs(ours[i] - theirs[CAL[n]["id"]]))
    assert worst < 1e-6, worst
    np.testing.assert_allclose(units.to_rad(units.from_rad(rad)), rad, atol=1e-12)


def test_calibrated_leaves_range_modes_and_sim_untouched():
    driver, _ = make_driver(cal=CAL)
    sim_units = lerobot_units(driver.description, {"*": "range_m100_100", "gripper": "range_0_100"})
    assert calibrated(sim_units, driver.calibration, resolution=4096) is sim_units


def test_a_hardware_run_binds_the_calibrated_degrees():
    from strands_robots import Robot
    from strands_robots.robot.run.bind import bind

    class DegreesPolicy:
        def units_for(self, description):
            return lerobot_units(description, {"*": "degrees", "gripper": "range_0_100"})

        def act(self, obs, instruction=None):
            return obs.joints.positions

        def reset(self):
            pass

    driver, _ = make_driver(cal=CAL)
    real = Robot("so101", driver=driver, twin=False)
    conv = bind(real, DegreesPolicy(), "auto").units
    want = calibrated(DegreesPolicy().units_for(real.description), driver.calibration, resolution=4096)
    np.testing.assert_array_equal(conv.lo_u, want.lo_u)
    sim = bind(Robot("so101"), DegreesPolicy(), "auto").units
    assert not np.array_equal(sim.lo_u, conv.lo_u)  # sim keeps rad2deg about the joint-range midpoint
    real.close()
