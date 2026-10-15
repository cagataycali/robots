"""One safety chain for every write, an e-stop that always wins, a watchdog that applies on_disconnect."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots import Robot
from strands_robots.robot.device import manifest_for
from strands_robots.robot.safety import RequestRefused, Watchdog


def test_out_of_range_write_is_refused_naming_joint_value_and_bound():
    arm = Robot("so101")
    lo, hi = arm.description.limits["elbow_flex"]
    with pytest.raises(
        RequestRefused, match=rf"elbow_flex={hi + 1:.4f} rad is outside its range \[{lo:.4f}, {hi:.4f}\]"
    ):
        arm.confirm_act({"elbow_flex": hi + 1.0})


def test_validator_refusal_reaches_the_caller_verbatim():
    arm = Robot("so101")

    def no_wrist(robot, request):
        if abs(dict(request.items())["wrist_roll"]) > 0.1:
            raise RequestRefused("wrist_roll must stay within 0.1 rad of zero while the cup is held")

    arm.validators.append(no_wrist)
    with pytest.raises(RequestRefused) as e:
        arm.confirm_act({"wrist_roll": 0.5})
    assert str(e.value) == "wrist_roll must stay within 0.1 rad of zero while the cup is held"
    arm.confirm_act({"wrist_roll": 0.05})


def test_estop_latches_refuses_act_and_resume_needs_an_operator():
    arm = Robot("so101")
    arm.estop()
    assert arm.estopped and arm.describe()["estop"] is True
    with pytest.raises(RequestRefused, match="e-stop is engaged"):
        arm.act({"shoulder_pan": 0.2})
    with pytest.raises(RequestRefused, match="approved_by"):
        arm.resume()
    with pytest.raises(RequestRefused, match="approved_by"):
        arm.resume(approved_by="  ")
    assert arm.estopped
    arm.resume(approved_by="cagatay")
    arm.act({"shoulder_pan": 0.2})


def test_run_stops_with_estop_within_one_tick():
    arm = Robot("so101")
    ticks: list[int] = []

    def pol(obs):
        ticks.append(len(ticks))
        if len(ticks) == 5:
            arm.estop()
        return {"shoulder_pan": 0.4}

    res = arm.run(pol, hz=50, steps=100)
    assert res.stopped_by == "estop" and res.steps == 4 and len(ticks) == 5


def test_estop_from_another_thread_stops_a_background_run():
    arm = Robot("so101")
    task = arm.start(lambda o: {"shoulder_pan": 0.5}, hz=100, seconds=30, realtime=True)
    time.sleep(0.15)
    t0 = time.perf_counter()
    threading.Thread(target=arm.estop).start()
    while task.running and time.perf_counter() - t0 < 2.0:
        time.sleep(0.001)
    assert not task.running and task.result.stopped_by == "estop"
    assert time.perf_counter() - t0 < 0.1  # one tick at 100 Hz is 10 ms; generous for CI


def test_hold_and_watchdog_lapse_apply_on_disconnect_in_manual_time():
    arm = Robot("so101")
    arm.act({"shoulder_pan": 0.6})
    with arm.sim.manual_time():
        dog = Watchdog(arm, timeout_ms=100)
        dog.beat()
        arm.sim.step(int(0.05 / arm.sim.physics.timestep))
        assert dog.poll() is False
        arm.sim.step(int(0.06 / arm.sim.physics.timestep))
        q = arm.observe().joints.positions
        assert dog.poll() is True and dog.poll() is False  # applied once per lapse
        np.testing.assert_allclose(arm.sim.backend.data.ctrl[: len(q)], q)


def test_hardware_robot_manifest_and_estop_on_a_faked_bus():
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    m = manifest_for(real)
    assert m.real and m.world is None and len(m.controls) == 6
    real.estop()  # torque was never on: a limp arm stays limp, nothing written
    assert d.extras()["torque"] is False
    real.resume(approved_by="op")
    real.act({"shoulder_pan": 0.1})
    real.estop()  # torque on: hold where it is
    assert d.extras()["torque"] is True
    real.close()


def _narrow_safe(robot, key, safe):
    from dataclasses import replace

    m = robot.manifest
    controls = tuple(replace(c, safe=safe) if c.name == f"{key}.pos" else c for c in m.controls)
    robot.__dict__["manifest"] = replace(m, controls=controls)


def test_outside_safe_inside_range_needs_the_operators_yes():
    d, _ = make_driver()
    asked: list[np.ndarray] = []
    answer = {"yes": False}
    real = Robot("so101", driver=d, twin=False, confirm=lambda a: asked.append(a) or answer["yes"])
    _narrow_safe(real, "shoulder_pan", (-0.5, 0.5))
    with pytest.raises(RequestRefused, match=r"shoulder_pan=0\.8000 rad outside safe \[-0\.5000, 0\.5000\]"):
        real.confirm_act({"shoulder_pan": 0.8})
    assert len(asked) == 1
    answer["yes"] = True
    real.confirm_act({"shoulder_pan": 0.8})
    assert len(asked) == 2  # asked once for this write, not again by the gate
    sim = Robot("so101")
    _narrow_safe(sim, "shoulder_pan", (-0.5, 0.5))
    with pytest.raises(RequestRefused, match="outside safe"):
        sim.confirm_act({"shoulder_pan": 0.8})
    sim.confirm_act({"shoulder_pan": 0.8}, confirmed=True)  # an operator said yes out of band
    real.close()


def test_a_write_already_past_the_latch_check_is_refused_after_estop():
    arm = Robot("so101")
    coerce = arm.coerce

    def estop_mid_command(action, **kw):
        out = coerce(action, **kw)
        arm.estop()  # another thread presses stop between the check and the write
        return out

    arm.coerce = estop_mid_command  # type: ignore[method-assign]
    before = np.array(arm.sim.backend.data.ctrl[:6])
    with pytest.raises(RequestRefused, match="e-stop is engaged"):
        arm.act({"shoulder_pan": 0.5})
    np.testing.assert_allclose(arm.sim.backend.data.ctrl[:6][0], arm.observe().joints.positions[0])
    assert arm.sim.backend.data.ctrl[0] != 0.5 and before.shape == (6,)


def test_the_latch_exists_before_any_thread_touches_it():
    arm = Robot("so101")
    assert "estop_latch" in arm.__dict__ and arm.estop_latch is arm.estop_latch


def test_lockstep_estop_names_the_robot_that_latched():
    from strands_robots import Simulation

    sim = Simulation()
    a, b = Robot("so101", sim=sim, id="a"), Robot("so101", sim=sim, id="b", position=(0.5, 0, 0))
    n = {"k": 0}

    def pol(obs):
        n["k"] += 1
        if n["k"] == 3:
            a.estop()
        return obs.joints.positions

    res = sim.run({a: pol, b: lambda o: o.joints.positions}, steps=20)
    assert {r.stopped_by for r in res.values()} == {"estop"}
    assert res["a"].fault == res["b"].fault == "e-stop engaged on a"
    sim.close()


def test_an_estop_resumed_before_the_next_tick_still_stops_the_run():
    arm = Robot("so101")
    task = arm.start(lambda o: {"shoulder_pan": 0.5}, hz=50, seconds=30, realtime=True)
    time.sleep(0.1)
    arm.estop()
    arm.resume(approved_by="op")  # released at once: the run must still have seen the stop
    task.halt(2.0)
    assert task.result.stopped_by == "estop"


def test_estop_from_another_thread_never_collides_on_the_half_duplex_wire():
    from test_operator_gate import OneWire

    d, s = make_driver(serial=OneWire())
    real = Robot("so101", driver=d, twin=False)
    real.act(real.observe().joints.positions)  # torque on, so estop() is a hold on the bus
    errors: list[BaseException] = []

    def observe():
        try:
            for _ in range(60):
                real.observe(images=False)
        except Exception as exc:  # the test reports what a thread raised
            errors.append(exc)

    def stop():
        for _ in range(20):
            real.estop()

    threads = [threading.Thread(target=observe), threading.Thread(target=stop)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and s.collisions == 0
    real.close()


def test_a_watchdog_refuses_a_real_body_with_no_declared_on_disconnect():
    from test_drivers import FakeLowState, FakeSub

    from strands_robots import Description
    from strands_robots.robot.drivers.unitree import UnitreeDriver

    g1 = Robot(
        "g1", driver=UnitreeDriver(Description.load("g1"), port="en7", subscriber=FakeSub(FakeLowState())), twin=False
    )
    with pytest.raises(RequestRefused, match="no proven safe on_disconnect"):
        Watchdog(g1, timeout_ms=200)
    g1.close()
    d, _ = make_driver()
    so = Robot("so101", driver=d, twin=False)
    Watchdog(so, timeout_ms=200)  # feetech declares hold in the registry
    so.close()


def test_workspace_bounds_validator_refuses_a_target_whose_tool_leaves_the_box():
    from strands_robots.robot.safety import WorkspaceBoundsValidator

    arm = Robot("so101")
    reach = WorkspaceBoundsValidator(
        lambda q: np.array([0.1 + 0.2 * q[0], 0.0, 0.2]), lo=(-0.2, -0.2, 0.0), hi=(0.2, 0.2, 0.4)
    )
    arm.validators.append(reach)
    arm.confirm_act({"shoulder_pan": 0.2})
    with pytest.raises(RequestRefused, match=r"tool x=0\.3000 m is outside the workspace \[-0\.2000, 0\.2000\]"):
        arm.confirm_act({"shoulder_pan": 1.0})
