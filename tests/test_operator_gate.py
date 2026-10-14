"""The operator gate: a real body never moves because of a default (#4749 part 5)."""

from __future__ import annotations

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots import Robot
from strands_robots.drivers import feetech as ft


def goal_writes(serial: object) -> list[bytes]:
    """Frames that would move a servo: a SYNC_WRITE of Goal_Position or of Torque_Enable."""
    return [
        f
        for f in serial.frames  # type: ignore[attr-defined]
        if f[4] == ft.SYNC_WRITE and f[5] in (ft.REG_GOAL_POSITION, ft.REG_TORQUE_ENABLE)
    ]


def test_direct_act_is_the_developers_call_and_needs_no_gate():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    real.act({"shoulder_pan": 0.2})  # plain Python: the developer is the operator
    assert real.observe()["shoulder_pan"] == pytest.approx(0.2, abs=1e-2)


@pytest.mark.parametrize(
    "confirm, match",
    [(None, "pass confirm="), (lambda action: False, "refused")],
    ids=["no-callable", "operator-says-no"],
)
def test_a_refused_hold_writes_nothing(confirm, match):
    d, s = make_driver()
    real = Robot("so101", driver=d, confirm=confirm)
    before = real.observe().joints.positions
    s.frames.clear()
    with pytest.raises(PermissionError, match=match):
        real.confirm_act(before)
    assert goal_writes(s) == [] and d.extras()["torque"] is False
    np.testing.assert_array_equal(real.observe().joints.positions, before)
    real.close()


def test_an_allowed_hold_writes_and_the_twin_mirrors_it():
    d, s = make_driver()
    asked: list[np.ndarray] = []
    real = Robot("so101", driver=d, confirm=lambda action: asked.append(np.array(action)) or True)
    measured = real.observe().joints
    real.confirm_act(measured.positions)
    assert len(asked) == 1 and len(goal_writes(s)) == 2  # torque on, then one Goal_Position write
    for name in measured.names:  # the hold lands on the bus within one count of where it was read
        goal = int.from_bytes(s.regs[d.ids[measured.names.index(name)]][ft.REG_GOAL_POSITION :][:2], "little")
        assert abs(goal - d.rad_to_counts(name, measured[name])) <= 1
    twin = real.twin
    assert twin is not None and twin.sim is not None
    twin.sim.step(200)
    np.testing.assert_allclose(twin.observe().joints.positions, measured.positions, atol=0.05)
    real.close()


def test_confirmed_out_of_band_and_sim_bodies_are_ungated():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    real.confirm_act({"shoulder_pan": 0.3}, confirmed=True)  # an operator said yes elsewhere (a dashboard dialog)
    assert real.observe()["shoulder_pan"] == pytest.approx(0.3, abs=1e-2)
    Robot("so101").confirm_act({"shoulder_pan": 0.3})  # simulated: no gate
    with pytest.raises(TypeError, match="confirm"):
        Robot("so101", confirm=lambda a: True)  # a gate on a simulated body is a mistake worth naming
