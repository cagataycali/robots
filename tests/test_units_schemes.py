"""Units schemes are registry data (part 3.5): gripper keys, LeRobot norm modes, counts round-trip.

The 'lerobot' scheme reads its per-joint modes from ``hardware.lerobot_norm`` and its gripper keys from
``gripper.keys`` -- both registry data now, not Python constants. A robot LeRobot does not define is
refused by name, and the counts the feetech driver maps match LeRobot's own normalisation to 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.description import Description, registry
from strands_robots.robot.drivers.calibration import Calibration, JointCalibration
from strands_robots.robot.drivers.feetech import FeetechDriver
from strands_robots.units import SCHEMES, Units


def _gripper_robots():
    seen, out = set(), []
    for entry in registry().values():
        if entry["name"] in seen or not entry.get("gripper"):
            continue
        seen.add(entry["name"])
        out.append(entry["name"])
    return out


@pytest.mark.parametrize("name", _gripper_robots())
def test_gripper_keys_are_a_subset_of_action_keys(name):
    d = Description.load(name)
    assert d.gripper is not None
    assert set(d.gripper.keys) <= set(d.action_keys), f"{name}: {d.gripper.keys} not in {d.action_keys}"


def test_an_unknown_scheme_is_refused_naming_the_known_ones():
    d = Description.load("so101")
    with pytest.raises(ValueError, match=r"unknown units 'furlongs'; known: lerobot, rad"):
        Units.for_robot(d, "furlongs")
    assert set(SCHEMES) == {"rad", "lerobot"}


def test_a_robot_without_lerobot_norm_is_refused_by_name():
    d = Description.load("panda")  # driver 'strands', no hardware.lerobot_norm
    assert d.hardware.lerobot_norm is None
    with pytest.raises(ValueError, match=r"panda declares no hardware.lerobot_norm"):
        Units.for_robot(d, "lerobot")


class _FakeBus:
    port = "fake"

    def ping(self, _id):
        return True

    def close(self):
        pass


def test_counts_to_rad_to_lerobot_equals_lerobots_own_normalisation():
    d = Description.load("so101")
    cmin, cmax = 100, 3900
    cal = Calibration({k: JointCalibration(i + 1, 0, 0, cmin, cmax) for i, k in enumerate(d.action_keys)})
    drv = FeetechDriver(d, port=None, bus=_FakeBus(), calibration=cal)
    u = Units.for_robot(d, "lerobot")
    for counts in (cmin, 1000, 2000, cmax):
        rad = np.array([drv.counts_to_rad(k, counts) for k in d.action_keys])
        got = u.from_rad(rad)
        frac = (counts - cmin) / (cmax - cmin)
        for i, k in enumerate(d.action_keys):
            expected = frac * 100.0 if k in d.gripper.keys else frac * 200.0 - 100.0
            assert abs(got[i] - expected) < 1e-9, f"{k} at {counts}: {got[i]} vs {expected}"
