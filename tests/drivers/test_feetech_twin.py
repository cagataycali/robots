"""The Feetech twin bus: the SO arm's serial bus, answered by a sim engine.

Everything here runs with no MuJoCo. One double stands in for the sim engine
and is faithful in the two respects that matter: it **records** every
``send_action`` (the actuator targets and the substeps) and answers
``get_observation`` from a mutable state, and it carries a small ``mj_model``
with the named ``joint`` / ``actuator`` views the twin binds through - the
SO-101's ``1``..``6`` with unlimited ``ctrlrange`` and the joint ``range``, or
the SO-100's ``Rotation``..``Jaw`` with a declared ``ctrlrange`` - so the tests
can say what the twin wrote to the model, in radians, and for how long.

The driver is the one under ``tests/drivers/test_feetech_driver.py``; what is
graded here is that the SAME driver, on ``transport="twin"``, puts the same
agent verbs onto the model in the model's units: degrees through the arm's own
calibration to counts to radians, percent onto the gripper's travel with ``0``
at the registry's closed end, torque as gain, one read period per write. The
real-MuJoCo half is ``tests_integ/simulation/test_feetech_twin.py``.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path
from typing import Any

import pytest

from strands_robots.bus_access import read_joints
from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.drivers.feetech.bus import DEFAULT_TIMEOUT_S, SO_ARM_MOTORS, MotorCalibration
from strands_robots.drivers.feetech.driver import TRANSPORTS
from strands_robots.drivers.feetech.twin import TWIN_REGISTERS, FeetechTwinBus, twin_endpoint

# ``test_feetech_module_load.py`` drops the feetech modules from ``sys.modules``
# to prove an import, so a class imported at the top of this module is not the
# class the driver builds after that test ran. Identity is graded by name.
_TWIN_BUS = "FeetechTwinBus"

_DT = 0.002

#: The SO-101 asset as the twin sees it: joints ``1``..``6``, actuators of the
#: same name, ``ctrlrange="0 0"`` (unlimited) and a joint ``range``.
_SO101 = {
    "1": (-1.91986218, 1.91986218),
    "2": (-1.74532925, 1.74532925),
    "3": (-1.74532925, 1.57079633),
    "4": (-1.6580628, 1.65806278),
    "5": (-2.7925269, 2.79252671),
    "6": (-0.17453293, 1.74532925),
}
#: The SO-100 asset: CAD-named joints, actuators with a declared ``ctrlrange``.
_SO100 = {
    "Rotation": (-1.92, 1.92),
    "Pitch": (-3.32, 0.174),
    "Elbow": (-0.174, 3.14),
    "Wrist_Pitch": (-1.66, 1.66),
    "Wrist_Roll": (-2.79, 2.79),
    "Jaw": (-0.174, 1.75),
}


class _View:
    """One ``mj_model.joint(...)`` / ``mj_model.actuator(...)`` view - the attributes the twin reads."""

    def __init__(self, index: int, name: str, low: float, high: float, *, ctrl_limited: bool, kp: float) -> None:
        self.id = index
        self.name = name
        self.range = [low, high]
        self.limited = [1]
        self.trntype = 0
        self.trnid = [index, -1]
        self.ctrllimited = 1 if ctrl_limited else 0
        self.ctrlrange = [low, high] if ctrl_limited else [0.0, 0.0]
        self.gainprm = [kp, 0.0, 0.0]
        self.biasprm = [0.0, -kp, -0.5]


class _RecordsWhetherHeld(list):  # type: ignore[type-arg]
    """A gain array that records, per write, whether the engine's lock was held."""

    def __init__(self, values: list[float], lock: threading.Lock, log: list[bool]) -> None:
        super().__init__(values)
        self._lock = lock
        self._log = log

    def __setitem__(self, index: Any, value: Any) -> None:
        self._log.append(self._lock.locked())
        super().__setitem__(index, value)


class _Model:
    """The named-access surface of an ``MjModel`` the twin uses, over one robot's joints."""

    def __init__(self, robot: str, joints: dict[str, tuple[float, float]], *, ctrl_limited: bool, kp: float) -> None:
        self.nu = len(joints)
        self._views = {
            f"{robot}/{name}": _View(index, f"{robot}/{name}", low, high, ctrl_limited=ctrl_limited, kp=kp)
            for index, (name, (low, high)) in enumerate(joints.items())
        }
        self._by_index = list(self._views.values())

    def joint(self, key: str | int) -> _View:
        return self._lookup(key)

    def actuator(self, key: str | int) -> _View:
        return self._lookup(key)

    def _lookup(self, key: str | int) -> _View:
        if isinstance(key, int):
            return self._by_index[key]
        if key not in self._views:
            raise KeyError(key)
        return self._views[key]

    def with_a_robot_added_ahead(self, robot: str, joints: dict[str, tuple[float, float]], *, kp: float) -> _Model:
        """The model a recompile yields when another arm lands ahead of ours: same names, shifted indices.

        ``add_robot`` reallocates the model, so every actuator this one carries
        moves down the list by the count the new arm brought.
        """
        merged = _Model(robot, joints, ctrl_limited=True, kp=kp)
        merged._views |= self._views
        merged._by_index = list(merged._views.values())
        merged.nu = len(merged._by_index)
        for index, view in enumerate(merged._by_index):
            view.id = index
        return merged


class _FakeEngine:
    """A sim engine double carrying one SO arm: records writes, position servos arrive instantly."""

    def __init__(self, robot: str = "so101", *, lock: Any | None = None) -> None:
        joints = _SO101 if robot == "so101" else _SO100
        self.robot = robot
        if lock is not None:
            # In-tree backends expose one; the attribute is absent when a
            # backend does not, which is the default here on purpose.
            self._lock = lock
        self.mj_model = _Model(robot, joints, ctrl_limited=robot == "so100", kp=17.8 if robot == "so101" else 50.0)
        self.state: dict[str, float] = dict.fromkeys(joints, 0.0)
        self.vel: dict[str, float] = dict.fromkeys(joints, 0.0)
        self.writes: list[tuple[dict[str, float], int]] = []
        self.destroyed = False
        self.refuse: str | None = None

    def list_robots(self) -> list[str]:
        return [self.robot]

    def create_world(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "world"}]}

    def add_robot(self, name: str, **kwargs: Any) -> dict[str, Any]:
        assert name == self.robot
        return {"status": "success", "content": [{"text": "added"}]}

    def physics_timestep(self) -> float:
        return _DT

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        assert robot_name == self.robot and skip_images
        obs: dict[str, Any] = dict(self.state)
        obs.update({f"{k}.vel": v for k, v in self.vel.items()})
        obs["wrist"] = object()  # an image, which the twin must skip
        return obs

    def send_action(self, action: dict[str, float], robot_name: str, n_substeps: int) -> dict[str, Any]:
        assert robot_name == self.robot
        if self.refuse:
            return {"status": "error", "content": [{"text": self.refuse}]}
        self.writes.append((dict(action), n_substeps))
        for key, value in action.items():
            if key not in self.state:
                return {"status": "error", "content": [{"text": f"keys ['{key}'] could not be resolved"}]}
            # A position servo whose gain is zero holds nothing; otherwise it arrives.
            if self.mj_model.actuator(f"{self.robot}/{key}").gainprm[0] > 0:
                self.state[key] = value
        return {"status": "success", "content": [{"text": f"Action applied to '{self.robot}' ({len(action)} keys)."}]}

    def destroy(self) -> None:
        self.destroyed = True


def _invoke(driver: FeetechDriver, **request: Any) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        tool_use: Any = {"toolUseId": "t1", "name": driver.tool_name, "input": request}
        return [event async for event in driver.stream(tool_use, {})][0]

    return asyncio.run(_run())


def _joints(driver: FeetechDriver) -> dict[str, float]:
    reply = _invoke(driver, action="sensors")
    assert reply["status"] == "success", reply
    return reply["content"][0]["json"]["joint_state"]


@pytest.fixture
def engine() -> _FakeEngine:
    return _FakeEngine("so101")


@pytest.fixture
def twin(engine: _FakeEngine) -> FeetechDriver:
    driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine)
    assert driver.connect_eagerly() is None
    return driver


# --------------------------------------------------------------------------- #
# Construction: the knobs, the default, the refusals.                          #
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_the_default_transport_is_the_serial_bus_and_nothing_about_it_moved(self) -> None:
        driver = FeetechDriver(tool_name="so101", port="/dev/tty.usbserial-1")
        assert TRANSPORTS == ("serial", "twin")
        assert driver.transport == "serial"
        assert driver.endpoint == "/dev/tty.usbserial-1"
        assert driver.sim is None
        assert type(driver.bus).__name__ == "FeetechBus"
        body = asyncio.run(driver.get_status())["content"][0]["json"]
        assert body["transport"] == "serial" and body["endpoint"] == "/dev/tty.usbserial-1"

    def test_the_twin_builds_the_twin_bus_and_reports_a_sim_endpoint(self, twin: FeetechDriver, engine) -> None:
        assert type(twin.bus).__name__ == _TWIN_BUS
        assert twin.transport == "twin"
        assert twin.endpoint == twin_endpoint("so101") == "sim://so101"
        assert twin.sim is engine
        assert twin.is_connected
        body = asyncio.run(twin.get_status())["content"][0]["json"]
        assert body["connected"] is True
        assert body["transport"] == "twin" and body["endpoint"] == "sim://so101"
        assert body["motors"] == {name: spec.motor_id for name, spec in SO_ARM_MOTORS.items()}

    def test_an_unknown_transport_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"transport must be one of \['serial', 'twin'\], got 'wifi'"):
            FeetechDriver(tool_name="so101", transport="wifi")

    def test_sim_without_the_twin_transport_is_refused(self, engine) -> None:
        with pytest.raises(ValueError, match="sim= is the twin transport's engine"):
            FeetechDriver(tool_name="so101", sim=engine)

    def test_realtime_must_be_a_boolean(self, engine) -> None:
        with pytest.raises(ValueError, match="realtime"):
            FeetechDriver(tool_name="so101", transport="twin", sim=engine, realtime="yes")  # type: ignore[arg-type]

    def test_a_robot_without_a_sim_asset_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="needs a simulation asset.*'hope_jr'"):
            FeetechDriver(tool_name="hope_jr", transport="twin")

    def test_a_tool_name_that_is_not_a_registry_arm_is_refused_with_the_fix(self) -> None:
        with pytest.raises(ValueError, match="'left_arm'.*not one of.*pass tool_name as the arm's registry name"):
            FeetechDriver(tool_name="left_arm", transport="twin")

    def test_a_callers_engine_names_the_robot_so_the_tool_name_is_free(self) -> None:
        """``sim=`` carries so100; the agent may still call the tool ``left_arm``."""
        engine = _FakeEngine("so100")
        driver = FeetechDriver(tool_name="left_arm", transport="twin", sim=engine)
        assert driver.connect_eagerly() is None
        assert driver.tool_name == "left_arm"
        assert driver.endpoint == "sim://so100"

    def test_an_alias_resolves_to_the_canonical_arm(self) -> None:
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101_follower", transport="twin", sim=engine)
        assert driver.connect_eagerly() is None
        bus: Any = driver.bus
        assert bus.robot == "so101"


# --------------------------------------------------------------------------- #
# Binding: labels -> joints -> actuators -> travel, for both SO arms.           #
# --------------------------------------------------------------------------- #


class TestBinding:
    @pytest.mark.parametrize(
        ("robot", "expected"),
        [
            (
                "so101",
                {
                    "shoulder_pan": "1",
                    "shoulder_lift": "2",
                    "elbow_flex": "3",
                    "wrist_flex": "4",
                    "wrist_roll": "5",
                    "gripper": "6",
                },
            ),
            (
                "so100",
                {
                    "shoulder_pan": "Rotation",
                    "shoulder_lift": "Pitch",
                    "elbow_flex": "Elbow",
                    "wrist_flex": "Wrist_Pitch",
                    "wrist_roll": "Wrist_Roll",
                    "gripper": "Jaw",
                },
            ),
        ],
    )
    def test_every_motor_binds_to_its_labelled_joint(self, robot: str, expected: dict[str, str]) -> None:
        engine = _FakeEngine(robot)
        driver = FeetechDriver(tool_name=robot, transport="twin", sim=engine)
        assert driver.connect_eagerly() is None
        bus: Any = driver.bus
        assert type(bus).__name__ == _TWIN_BUS
        assert {name: binding.joint for name, binding in bus._bindings.items()} == expected
        assert {name: binding.actuator for name, binding in bus._bindings.items()} == expected

    def test_so101_travel_is_the_joint_range_because_its_ctrlrange_is_unlimited(
        self, twin: FeetechDriver, engine
    ) -> None:
        bus: Any = twin.bus
        assert type(bus).__name__ == _TWIN_BUS
        assert (bus._bindings["shoulder_pan"].low, bus._bindings["shoulder_pan"].high) == _SO101["1"]
        assert engine.mj_model.actuator("so101/1").ctrlrange == [0.0, 0.0], (
            "a (0, 0) ctrlrange is unlimited, not zero-width"
        )

    def test_so100_travel_is_the_declared_ctrlrange(self) -> None:
        engine = _FakeEngine("so100")
        driver = FeetechDriver(tool_name="so100", transport="twin", sim=engine)
        assert driver.connect_eagerly() is None
        bus: Any = driver.bus
        assert type(bus).__name__ == _TWIN_BUS
        assert (bus._bindings["shoulder_lift"].low, bus._bindings["shoulder_lift"].high) == _SO100["Pitch"]

    def test_a_joint_with_neither_ctrlrange_nor_range_is_refused_by_name(self) -> None:
        engine = _FakeEngine("so101")
        view = engine.mj_model.joint("so101/3")
        view.limited = [0]
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine)
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "joint '3' (motor 'elbow_flex') declares neither an actuator ctrlrange nor a joint range" in reason
        assert not driver.is_connected

    def test_a_joint_with_no_actuator_is_refused_by_name(self) -> None:
        engine = _FakeEngine("so101")
        engine.mj_model.actuator("so101/5").trnid = [99, -1]
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine)
        reason = driver.connect_eagerly()
        assert reason is not None and "no actuator drives joint '5' (motor 'wrist_roll')" in reason

    def test_a_robot_whose_registry_entry_has_no_labels_is_refused_at_connect(self) -> None:
        engine = _FakeEngine("so101")
        engine.robot = "lekiwi"
        driver = FeetechDriver(tool_name="lekiwi", transport="twin", sim=engine)
        reason = driver.connect_eagerly()
        assert reason is not None and "declares no joint_labels" in reason and "'lekiwi'" in reason

    def test_a_motor_the_labels_do_not_name_is_refused_at_connect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.registry as registry

        labels = dict(registry.joint_labels("so101"))
        labels.pop("5")
        monkeypatch.setattr(registry, "joint_labels", lambda name: labels if name == "so101" else {})
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine)
        reason = driver.connect_eagerly()
        assert reason is not None and "no joint_labels entry names motor 'wrist_roll'" in reason


# --------------------------------------------------------------------------- #
# Units: degrees and percent, both ways, through the arm's calibration.          #
# --------------------------------------------------------------------------- #


class TestUnits:
    def test_zero_degrees_with_no_calibration_is_the_middle_of_the_models_travel(
        self, twin: FeetechDriver, engine
    ) -> None:
        reply = _invoke(twin, action="move_to", targets={"shoulder_pan": 0.0})
        assert reply["status"] == "success", reply
        action, _ = engine.writes[-1]
        low, high = _SO101["1"]
        assert action == {"1": pytest.approx((low + high) / 2, abs=1e-3)}

    def test_degrees_map_linearly_over_the_servos_full_travel_onto_the_joints_travel(self, twin, engine) -> None:
        """+90 deg = a quarter turn of the 4096-count servo = a quarter of the model's travel above the middle."""
        assert _invoke(twin, action="move_to", targets={"shoulder_pan": 90.0})["status"] == "success"
        action, _ = engine.writes[-1]
        low, high = _SO101["1"]
        middle, span = (low + high) / 2, high - low
        assert action["1"] == pytest.approx(middle + span / 4, abs=2e-3)

    def test_a_reading_is_the_models_angle_back_through_the_same_map(self, twin, engine) -> None:
        low, high = _SO101["2"]
        engine.state["2"] = low + (high - low) * 0.75  # three quarters up the travel = +90 deg
        assert _joints(twin)["shoulder_lift"] == pytest.approx(90.0, abs=0.1)

    def test_a_write_reads_back_within_one_encoder_count(self, twin) -> None:
        assert _invoke(twin, action="move_to", targets={"wrist_flex": -37.25, "gripper": 42.0})["status"] == "success"
        joints = _joints(twin)
        assert joints["wrist_flex"] == pytest.approx(-37.25, abs=360.0 / 4096)
        assert joints["gripper"] == pytest.approx(42.0, abs=100.0 / 4096 + 1e-9)

    def test_gripper_percent_spans_the_jaw_end_to_end_with_zero_at_the_closed_end(self, twin, engine) -> None:
        low, high = _SO101["6"]
        assert _invoke(twin, action="move_to", targets={"gripper": 0.0})["status"] == "success"
        assert engine.writes[-1][0] == {"6": pytest.approx(low, abs=1e-3)}, "registry gripper.closed is 'low'"
        assert _invoke(twin, action="move_to", targets={"gripper": 100.0})["status"] == "success"
        assert engine.writes[-1][0] == {"6": pytest.approx(high, abs=1e-3)}
        engine.state["6"] = low
        assert _joints(twin)["gripper"] == pytest.approx(0.0, abs=0.05)
        engine.state["6"] = high
        assert _joints(twin)["gripper"] == pytest.approx(100.0, abs=0.05)

    def test_a_percent_past_the_domain_is_the_bus_refusal(self, twin, engine) -> None:
        reply = _invoke(twin, action="move_to", targets={"gripper": 130.0})
        assert reply["status"] == "error"
        assert "gripper target 130.0 is outside 0..100 percent open" in reply["content"][0]["text"]
        assert engine.writes == [], "refused before the model was reached"

    def test_a_degree_target_the_encoder_cannot_hold_is_the_bus_refusal(self, twin, engine) -> None:
        reply = _invoke(twin, action="move_to", targets={"shoulder_pan": 400.0})
        assert reply["status"] == "error"
        assert "outside the travel the encoder can hold" in reply["content"][0]["text"]
        assert engine.writes == []

    def test_a_calibration_file_places_the_twin_where_it_places_the_arm(self, tmp_path: Path) -> None:
        """A shoulder_pan calibrated to counts 1024..3072: 0 deg is the middle of the model travel, -90 deg its low stop."""
        records = {
            name: {"id": spec.motor_id, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
            for name, spec in SO_ARM_MOTORS.items()
        }
        records["shoulder_pan"].update(range_min=1024, range_max=3072)
        path = tmp_path / "so101.json"
        path.write_text(json.dumps(records))
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine, calibration=path)
        assert driver.connect_eagerly() is None
        low, high = _SO101["1"]
        assert _invoke(driver, action="move_to", targets={"shoulder_pan": 0.0})["status"] == "success"
        assert engine.writes[-1][0]["1"] == pytest.approx((low + high) / 2, abs=1e-3)
        # 2048 counts of measured travel = 180 deg, so -90 deg is range_min = the model's low stop.
        assert _invoke(driver, action="move_to", targets={"shoulder_pan": -90.0})["status"] == "success"
        assert engine.writes[-1][0]["1"] == pytest.approx(low, abs=1e-3)
        # And a real arm's status names the file, as on the serial bus.
        body = asyncio.run(driver.get_status())["content"][0]["json"]
        assert body["calibration_source"] == str(path)

    def test_a_reversed_gripper_still_puts_zero_percent_on_the_closed_end(self) -> None:
        """``drive_mode=1`` flips which count is open on the wire; the model's closed end is still 0 percent."""
        records = {
            name: MotorCalibration(id=spec.motor_id, range_min=0, range_max=4095)
            for name, spec in SO_ARM_MOTORS.items()
        }
        records["gripper"] = MotorCalibration(id=6, drive_mode=1, range_min=500, range_max=3500)
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine, calibration=records)
        assert driver.connect_eagerly() is None
        low, high = _SO101["6"]
        assert _invoke(driver, action="move_to", targets={"gripper": 0.0})["status"] == "success"
        assert engine.writes[-1][0]["6"] == pytest.approx(low, abs=1e-3)
        engine.state["6"] = high
        assert _joints(driver)["gripper"] == pytest.approx(100.0, abs=0.05)

    def test_a_target_past_the_models_travel_is_clamped_and_reported(self, tmp_path: Path, caplog) -> None:
        """Calibrated to a narrow travel, 170 deg maps to a count past range_max: the model clamps, and says so."""
        records = {
            name: MotorCalibration(id=spec.motor_id, range_min=0, range_max=4095)
            for name, spec in SO_ARM_MOTORS.items()
        }
        records["shoulder_pan"] = MotorCalibration(id=1, range_min=1024, range_max=3072)
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine, calibration=records)
        assert driver.connect_eagerly() is None
        with caplog.at_level("WARNING"):
            reply = _invoke(driver, action="move_to", targets={"shoulder_pan": 170.0})
        assert reply["status"] == "success", reply
        body = reply["content"][0]["json"]
        assert "model travel clamped: shoulder_pan" in body["note"]
        assert engine.writes[-1][0]["1"] == pytest.approx(_SO101["1"][1], abs=1e-9)
        assert any("clamped" in record.message for record in caplog.records)
        # The next in-range write carries no note.
        reply = _invoke(driver, action="move_to", targets={"shoulder_pan": 0.0})
        assert "note" not in reply["content"][0]["json"]

    def test_value_bounds_are_the_bus_own(self, twin) -> None:
        assert twin.bus.value_bounds("shoulder_pan") == (-180.0, 180.0)
        assert twin.bus.value_bounds("gripper") == (0.0, 100.0)


# --------------------------------------------------------------------------- #
# Timing, torque, registers.                                                    #
# --------------------------------------------------------------------------- #


class TestTiming:
    def test_a_write_steps_one_read_period_so_the_servo_arrives_by_the_next_read(self, twin, engine) -> None:
        assert _invoke(twin, action="move_to", targets={"elbow_flex": 10.0})["status"] == "success"
        _, substeps = engine.writes[-1]
        assert substeps == round(DEFAULT_TIMEOUT_S / _DT) == 500

    def test_the_callers_timeout_is_the_read_period(self) -> None:
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine, timeout=0.25)
        assert driver.connect_eagerly() is None
        assert _invoke(driver, action="move_to", targets={"elbow_flex": 10.0})["status"] == "success"
        assert engine.writes[-1][1] == 125

    def test_realtime_sleeps_the_read_period_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.drivers.feetech.twin as twin_module

        slept: list[float] = []
        monkeypatch.setattr(twin_module.time, "sleep", slept.append)
        engine = _FakeEngine("so101")
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine, timeout=0.5, realtime=True)
        assert driver.connect_eagerly() is None
        assert _invoke(driver, action="move_to", targets={"elbow_flex": 10.0})["status"] == "success"
        assert slept == [0.5]

    def test_the_default_does_not_sleep(self, twin, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.drivers.feetech.twin as twin_module

        slept: list[float] = []
        monkeypatch.setattr(twin_module.time, "sleep", slept.append)
        assert _invoke(twin, action="move_to", targets={"elbow_flex": 10.0})["status"] == "success"
        assert slept == []


class TestTorque:
    def test_releasing_zeroes_the_gains_and_the_arm_goes_limp(self, twin, engine) -> None:
        reply = _invoke(twin, action="set_torque", enabled=False)
        assert reply == {"toolUseId": "t1", "status": "success", "content": [{"json": {"torque_enabled": False}}]}
        for index in range(engine.mj_model.nu):
            view = engine.mj_model.actuator(index)
            assert view.gainprm[0] == 0.0 and view.biasprm[1] == 0.0 and view.biasprm[2] == 0.0
        assert twin.bus.sync_read("Torque_Enable") == dict.fromkeys(SO_ARM_MOTORS, 0.0)

    def test_a_write_with_torque_off_is_refused_with_the_bus_sentence(self, twin, engine) -> None:
        assert _invoke(twin, action="set_torque", enabled=False)["status"] == "success"
        reply = _invoke(twin, action="move_to", targets={"shoulder_pan": 10.0})
        assert reply["status"] == "error"
        assert reply["content"][0]["text"] == (
            "send_action: FeetechBus: writing goal positions needs torque on; call set_torque(True) first (port='sim://so101')"
        )
        assert engine.writes == []

    def test_energizing_restores_the_gains_and_retargets_the_present_position(self, twin, engine) -> None:
        assert _invoke(twin, action="set_torque", enabled=False)["status"] == "success"
        engine.state["2"] = 0.9  # gravity took the shoulder while it was limp
        reply = _invoke(twin, action="set_torque", enabled=True)
        assert reply["status"] == "success"
        view = engine.mj_model.actuator("so101/2")
        assert view.gainprm[0] == 17.8 and view.biasprm[1] == -17.8 and view.biasprm[2] == -0.5
        action, substeps = engine.writes[-1]
        assert action["2"] == 0.9 and substeps == 1, "holds where it is, not where it was told last"
        assert twin.bus.sync_read("Torque_Enable") == dict.fromkeys(SO_ARM_MOTORS, 1.0)

    def test_stop_releases_the_arm(self, twin, engine) -> None:
        assert _invoke(twin, action="stop")["status"] == "success"
        assert engine.mj_model.actuator("so101/1").gainprm[0] == 0.0
        assert twin.bus.torque_enabled is False

    def test_every_gain_write_happens_while_the_engine_lock_is_held(self) -> None:
        """The gain arrays are the live model's, shared with the stepping and render threads.

        A write outside the engine's lock can be read half-applied by a
        concurrent ``mj_step``, so every one of them is made under it.
        """
        lock = threading.Lock()
        engine = _FakeEngine("so101", lock=lock)
        driver = FeetechDriver(tool_name="so101", transport="twin", sim=engine)
        assert driver.connect_eagerly() is None
        held: list[bool] = []
        for index in range(engine.mj_model.nu):
            view = engine.mj_model.actuator(index)
            view.gainprm = _RecordsWhetherHeld(view.gainprm, lock, held)
            view.biasprm = _RecordsWhetherHeld(view.biasprm, lock, held)

        assert _invoke(driver, action="set_torque", enabled=False)["status"] == "success"

        assert len(held) == 3 * len(_SO101), "expected a gain and two bias writes per motor"
        assert all(held), f"{held.count(False)} of {len(held)} gain writes ran with the engine lock free"

    def test_the_gains_follow_the_actuator_name_when_a_recompile_shifts_the_indices(self, twin, engine) -> None:
        """Releasing this arm must not limp an arm added after it.

        A scene recompile renumbers the actuators, so a gain write addressed by
        the index seen at connect time lands on whatever now sits there.
        """
        engine.mj_model = engine.mj_model.with_a_robot_added_ahead("so100", _SO100, kp=50.0)

        assert _invoke(twin, action="set_torque", enabled=False)["status"] == "success"

        assert [engine.mj_model.actuator(f"so101/{name}").gainprm[0] for name in _SO101] == [0.0] * len(_SO101)
        assert [engine.mj_model.actuator(f"so100/{name}").gainprm[0] for name in _SO100] == [50.0] * len(_SO100), (
            "released the actuators of the arm that was added, not its own"
        )


class TestRegisters:
    def test_present_velocity_is_the_models_velocity_in_counts_per_second(self, twin, engine) -> None:
        low, high = _SO101["1"]
        engine.vel["1"] = (high - low) / 2  # half the travel per second = 2048 counts/s with no calibration
        assert twin.bus.sync_read("Present_Velocity")["shoulder_pan"] == pytest.approx(2047.5, abs=0.01)

    def test_present_load_is_zero_and_says_why(self, twin) -> None:
        assert twin.bus.sync_read("Present_Load") == dict.fromkeys(SO_ARM_MOTORS, 0.0)
        assert "no load sensor" in TWIN_REGISTERS["Present_Load"]

    def test_an_unknown_register_is_refused_by_name(self, twin) -> None:
        with pytest.raises(ValueError, match=r"cannot read 'Present_Current'; the twin answers \["):
            twin.bus.sync_read("Present_Current")

    def test_the_mesh_joint_read_source_reads_the_model(self, twin, engine) -> None:
        """``bus_access.read_joints`` resolves ``bus.sync_read`` + ``is_connected`` - the same seam, so the mesh gets the model."""
        low, high = _SO101["1"]
        engine.state["1"] = (low + high) / 2
        joints = read_joints(twin)
        assert joints is not None
        assert joints["shoulder_pan.pos"] == pytest.approx(0.0, abs=0.1)
        assert set(joints) == {f"{name}.pos" for name in SO_ARM_MOTORS}


# --------------------------------------------------------------------------- #
# Lifecycle and the whole agent surface.                                        #
# --------------------------------------------------------------------------- #


class TestLifecycle:
    def test_cleanup_leaves_a_callers_engine_alone(self, twin, engine) -> None:
        twin.cleanup()
        assert engine.destroyed is False
        assert twin.is_connected is False
        # And the same engine can be bound again.
        assert twin.connect_eagerly() is None

    def test_cleanup_destroys_an_engine_the_twin_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.simulation as sim_module

        built: list[_FakeEngine] = []

        def _fake_create(backend: str, **kwargs: Any) -> _FakeEngine:
            assert (backend, kwargs) == ("mujoco", {"tool_name": "so101_twin"})
            built.append(_FakeEngine("so101"))
            return built[-1]

        monkeypatch.setattr(sim_module, "create_simulation", _fake_create)
        driver = FeetechDriver(tool_name="so101", transport="twin")
        assert driver.sim is None, "built lazily, not at construction"
        assert driver.connect_eagerly() is None
        assert driver.sim is built[0]
        driver.cleanup()
        assert built[0].destroyed is True
        assert driver.sim is None

    def test_a_build_failure_is_a_named_reason_not_a_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.simulation as sim_module

        def _boom(backend: str, **kwargs: Any) -> Any:
            raise ImportError("No module named 'mujoco'")

        monkeypatch.setattr(sim_module, "create_simulation", _boom)
        driver = FeetechDriver(tool_name="so101", transport="twin")
        reason = driver.connect_eagerly()
        assert reason == "FeetechTwinBus: could not build the so101 twin (ImportError: No module named 'mujoco')"
        assert driver.is_connected is False
        body = asyncio.run(driver.get_status())["content"][0]["json"]
        assert body["connect_error"] == reason

    def test_a_refusal_from_the_model_is_the_drivers_refusal(self, twin, engine) -> None:
        engine.refuse = "No robots in the world."
        reply = _invoke(twin, action="move_to", targets={"shoulder_pan": 1.0})
        assert reply["status"] == "error"
        assert (
            reply["content"][0]["text"]
            == "send_action: FeetechTwinBus: the model refused the write: No robots in the world."
        )

    def test_every_declared_verb_runs_on_the_twin(self, twin) -> None:
        """Every ``action`` in the tool spec answers success on the twin - the acceptance bar, per family."""
        verbs = twin.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert verbs == ["status", "sensors", "move_to", "set_torque", "stop"]
        requests: dict[str, dict[str, Any]] = {
            "status": {},
            "sensors": {},
            "move_to": {"targets": {"shoulder_pan": 5.0, "gripper": 50.0}},
            "set_torque": {"enabled": True},
            "stop": {},
        }
        for verb in verbs:
            reply = _invoke(twin, action=verb, **requests[verb])
            assert reply["status"] == "success", (verb, reply)

    def test_an_undeclared_verb_is_the_drivers_own_refusal(self, twin) -> None:
        reply = _invoke(twin, action="home")
        assert reply["status"] == "error"
        assert "home" in reply["content"][0]["text"]

    def test_the_tool_spec_is_identical_on_both_transports(self, twin) -> None:
        serial = FeetechDriver(tool_name="so101", port="/dev/tty.usbserial-1")
        assert twin.tool_spec == serial.tool_spec

    def test_the_twin_bus_own_domain(self) -> None:
        with pytest.raises(ValueError, match="realtime"):
            FeetechTwinBus("so101", realtime=1)  # type: ignore[arg-type]
        bus = FeetechTwinBus("so101", sim=_FakeEngine("so101"))
        assert bus.port == "sim://so101"
        assert bus.is_connected is False
        with pytest.raises(RuntimeError, match="needs an open bus; call connect\\(\\) first"):
            bus.sync_read()
        bus.connect()
        assert bus.is_connected
        assert bus.counts_to_model("shoulder_pan", 0) == pytest.approx(_SO101["1"][0])
        assert bus.model_to_counts("shoulder_pan", _SO101["1"][1]) == 4095
        assert bus.model_to_counts("gripper", _SO101["6"][0]) == 0
        assert math.isclose(bus.counts_to_model("gripper", 4095), _SO101["6"][1])
