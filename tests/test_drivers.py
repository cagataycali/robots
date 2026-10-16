"""Drivers against simulated buses: no hardware is touched here."""

import json
import math
import os
import socket
import tempfile
import threading

import numpy as np
import pytest

from strands_robots import Description, Robot, Simulation
from strands_robots.robot.drivers import Calibration
from strands_robots.robot.drivers import feetech as ft
from strands_robots.robot.drivers import microduck as md
from strands_robots.robot.drivers.unitree import G1_ORDER, H1_SLOTS, UnitreeDriver


class FakeSerial:
    """Six STS3215 servos on a half-duplex wire: answers PING/READ/WRITE/SYNC_READ, applies SYNC_WRITE."""

    def __init__(self, ids=(1, 2, 3, 4, 5, 6)):
        self.regs = {i: bytearray(0x50) for i in ids}
        self.actual = dict.fromkeys(ids, 2048)  # the encoder; the firmware reports Present = Actual - Homing_Offset
        self.rx = bytearray()
        self.frames = []
        self.closed = False
        self.waited = 0  # bytes asked for that never came: on a real wire each one costs the whole timeout

    def reset_input_buffer(self):
        self.rx.clear()

    def read(self, n):
        self.waited += max(0, n - len(self.rx))
        out, self.rx = bytes(self.rx[:n]), self.rx[n:]
        return out

    def close(self):
        self.closed = True

    def _status(self, mid, params=b""):
        payload = bytes([mid, len(params) + 2, 0]) + params
        return ft.HEADER + payload + bytes([ft.checksum(payload)])

    def _homing(self, i):
        return ft.decode_sm(bytes(self.regs[i][ft.REG_HOMING_OFFSET : ft.REG_HOMING_OFFSET + 2]), 11)

    def write(self, data):
        self.frames.append(bytes(data))
        for i, r in self.regs.items():
            present = (self.actual[i] - self._homing(i)) % 4096
            r[ft.REG_PRESENT_POSITION : ft.REG_PRESENT_POSITION + 2] = ft.word(present)
        mid, length, instr = data[2], data[3], data[4]
        params = data[5 : 5 + length - 2]
        assert data[5 + length - 2] == ft.checksum(data[2 : 5 + length - 2]), "host sent a bad checksum"
        if instr == ft.PING and mid in self.regs:
            self.rx += self._status(mid)
        elif instr == ft.READ and mid in self.regs:
            reg, n = params
            self.rx += self._status(mid, bytes(self.regs[mid][reg : reg + n]))
        elif instr == ft.WRITE and mid in self.regs:
            reg = params[0]
            self.regs[mid][reg : reg + len(params) - 1] = params[1:]
            self.rx += self._status(mid)
        elif instr == ft.SYNC_READ:
            reg, n, *ids = params
            for i in ids:
                if i in self.regs:
                    self.rx += self._status(i, bytes(self.regs[i][reg : reg + n]))
        elif instr == ft.SYNC_WRITE:
            reg, per = params[0], params[1]
            body = params[2:]
            for k in range(0, len(body), per + 1):
                i, chunk = body[k], body[k + 1 : k + 1 + per]
                self.regs[i][reg : reg + per] = chunk
                if reg == ft.REG_GOAL_POSITION:  # a perfect servo: it is where it was told to be
                    self.actual[i] = (int.from_bytes(chunk, "little") + self._homing(i)) % 4096


CAL = {
    n: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1024, "range_max": 3072}
    for i, n in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
}
CAL["wrist_roll"]["drive_mode"] = 1


def make_driver(cal=CAL, serial=None):
    serial = serial or FakeSerial()
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    return ft.FeetechDriver(Description.load("so101"), port="/dev/fake", calibration=cal, bus=bus), serial


def test_protocol_codec():
    assert ft.frame(1, ft.PING) == bytes.fromhex("ff ff 01 02 01 fb")
    assert ft.decode_sm(ft.word(0x8064), 15) == -100 and ft.decode_sm(ft.word(100), 15) == 100
    assert ft.encode_sm(-5, 11) == ft.word(0x805) and ft.encode_sm(5, 11) == ft.word(5)
    with pytest.raises(ft.ProtocolError, match="checksum"):
        ft.parse_status(bytes.fromhex("ff ff 01 02 00 00"), 1, 0)
    with pytest.raises(ft.ProtocolError, match="expected 1"):
        ft.parse_status(bytes.fromhex("ff ff 02 02 00 fb"), 1, 0)


class EchoSerial(FakeSerial):
    """An adapter that hears its own request before the servo answers."""

    def write(self, data):
        self.rx += bytes(data)
        super().write(data)


class NoisySerial(FakeSerial):
    """A stray byte on the line ahead of every reply."""

    def write(self, data):
        self.rx += b"\x00"
        super().write(data)


BUS_CALLS = {
    "ping": lambda bus: bus.ping(1) is True,
    "read": lambda bus: bus.read(1, ft.REG_PRESENT_POSITION, 2),
    "write": lambda bus: bus.write(1, ft.REG_TORQUE_ENABLE, b"\x01") is None,
    "sync_read": lambda bus: bus.sync_read(ft.REG_PRESENT_POSITION, 6, [1, 2, 3, 4, 5, 6]),
}


@pytest.mark.parametrize("wire", [FakeSerial, EchoSerial, NoisySerial])
@pytest.mark.parametrize("call", BUS_CALLS)
def test_bus_reads_exactly_the_reply_and_never_waits(wire, call):
    serial = wire()
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    for _ in range(3):
        assert BUS_CALLS[call](bus)
    assert serial.waited == 0 and not serial.rx


class BreakSerial(FakeSerial):
    """A line held low (a pinched cable): the adapter delivers 0x00 for as long as anyone reads."""

    def read(self, n):
        return b"\x00" * n


@pytest.mark.parametrize("call", BUS_CALLS)
def test_a_line_that_never_sends_a_header_is_refused_and_frees_the_wire(call):
    bus = ft.FeetechBus("/dev/fake", serial=BreakSerial())
    if call == "ping":
        assert bus.ping(1) is False
    else:
        with pytest.raises(ft.ProtocolError):
            BUS_CALLS[call](bus)
    freed = []
    t = threading.Thread(target=lambda: freed.append(bus.wire.acquire(timeout=0.5)))
    t.start()
    t.join()
    assert freed == [True]  # an e-stop from another thread still gets the bus


def test_driver_cycle_never_waits_on_the_wire():
    d, serial = make_driver(serial=EchoSerial())
    d.write(d.read().positions)
    assert serial.waited == 0


def test_driver_reads_through_calibration_and_writes_goals():
    d, s = make_driver()
    js = d.read()
    assert js.names == Description.load("so101").action_keys
    lo, hi = d.description.limits["shoulder_pan"]
    assert js["shoulder_pan"] == pytest.approx((lo + hi) / 2)  # 2048 is the middle of [1024, 3072]
    d.write(np.array([hi, 0, 0, 0, 0, 0.5]))
    assert d.extras()["torque"] is True
    assert s.regs[1][ft.REG_TORQUE_ENABLE] == 1
    assert int.from_bytes(s.regs[1][ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little") == 3072
    assert d.read()["shoulder_pan"] == pytest.approx(hi, abs=1e-3)
    assert d.read()["gripper"] == pytest.approx(0.5, abs=2e-3)
    # drive_mode=1 reverses: a target at the joint's low end goes to range_max
    rlo, rhi = d.description.limits["wrist_roll"]
    assert d.rad_to_counts("wrist_roll", rlo) == 3072 and d.rad_to_counts("wrist_roll", rhi) == 1024
    d.close()
    assert s.regs[1][ft.REG_TORQUE_ENABLE] == 0 and s.closed


def test_uncalibrated_driver_uses_full_turn_counts():
    d, _ = make_driver(cal=None)
    assert d.calibration is None and "raw" in d.extras()["calibration"]
    assert d.read()["elbow_flex"] == pytest.approx(0.0)  # 2048 → 0 rad
    assert d.rad_to_counts("elbow_flex", math.pi / 2) == 3072


def test_missing_servo_is_a_connection_error():
    with pytest.raises(ConnectionError, match=r"servo id\(s\) \[6\]"):
        make_driver(serial=FakeSerial(ids=(1, 2, 3, 4, 5)))


def test_calibration_loads_from_lerobot_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    p = tmp_path / "robots" / "so101_follower" / "so101_follower.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(CAL))
    d, _ = make_driver(cal=None)
    assert d.calibration_source == str(p) and d.calibration == Calibration.from_json(CAL)


def test_calibrate_writes_lerobot_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    d, s = make_driver(cal=None)
    with pytest.raises(RuntimeError, match="interactive=True"):
        d.calibrate(interactive=False)  # no file yet: recording one needs a human
    for i in s.regs:  # the encoder reads 2100 at "middle"; the EEPROM still carries an old calibration's offset
        s.actual[i] = 2100
        s.regs[i][ft.REG_HOMING_OFFSET : ft.REG_HOMING_OFFSET + 2] = ft.encode_sm(-300, 11)
    cal = d.calibrate(interactive=True, prompt=lambda msg: None, sweep_seconds=0.05)
    # LeRobot's Feetech convention: Homing_Offset = middle - 2047, so the firmware then reports the middle at 2047
    assert cal.joints["shoulder_pan"].homing_offset == 2100 - 2047 == s._homing(1)
    assert ft.decode_sm(d.bus.read(1, ft.REG_PRESENT_POSITION, 2), 15) == 2047
    assert (cal.joints["shoulder_pan"].range_min, cal.joints["shoulder_pan"].range_max) == (2047, 2047)
    saved = json.loads((tmp_path / "robots" / "so101_follower" / "so101_follower.json").read_text())
    assert saved["gripper"]["id"] == 6
    assert set(saved["gripper"]) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    assert d.calibrate(interactive=False) == cal  # reads what was saved, moves nothing
    assert make_driver(cal=None)[0].calibration == cal  # the next connect loads the file it wrote


def test_hardware_robot_gets_a_twin_in_the_default_world():
    d, s = make_driver()
    real = Robot("so101", driver=d)
    assert real.is_real and real.sim is None and real.twin is not None
    assert real.twin.sim is Simulation.default() and real.twin.id == "so101_twin"
    hi = d.description.limits["shoulder_pan"][1]
    real.act({"shoulder_pan": hi})
    assert real.observe()["shoulder_pan"] == pytest.approx(hi, abs=1e-3)
    # the twin was told the same target and now shows the measured joint
    for _ in range(30):
        real.twin.sim.step(10)
    assert real.twin.observe()["shoulder_pan"] == pytest.approx(hi, abs=0.05)
    with pytest.raises(NotImplementedError, match="hardware"):
        real.set_pose(position=[1, 0, 0])
    real.twin.set_pose(position=[1, 0, 0])
    assert real.calibrate(interactive=False) == Calibration.from_json(
        CAL
    )  # the loaded file; True walks a human through it
    real.close()
    assert "so101_twin" not in Simulation.default().robots


def test_hardware_robot_without_twin_and_bad_kwargs():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    assert real.twin is None and not Simulation.default().robots
    with pytest.raises(TypeError, match="not sim="):
        Robot("so101", driver=d, sim=Simulation.default())
    with pytest.raises(TypeError, match="hardware argument"):
        Robot("so101", cameras={"wrist": 0})
    with pytest.raises(RuntimeError, match="for hardware"):
        Robot("so101").calibrate()


class FakeMotor:
    def __init__(self, q):
        self.q, self.dq, self.tau_est = q, 0.1, 0.5


class FakeImu:
    quaternion = (1.0, 0.0, 0.0, 0.0)
    gyroscope = (0.0, 0.0, 0.0)
    accelerometer = (0.0, 0.0, 9.81)


class FakeLowState:
    def __init__(self):
        self.motor_state = [FakeMotor(i * 0.01) for i in range(35)]
        self.imu_state = FakeImu()
        self.power_v = 52.0
        self.mode_machine = 4


class FakeSub:
    def __init__(self, msg=None):
        self.msg = msg
        self.closed = False

    def read(self):
        return self.msg

    def close(self):
        self.closed = True


def test_unitree_read_path_maps_sdk_order_to_action_keys():
    desc = Description.load("g1")
    d = UnitreeDriver(desc, port="en7", subscriber=FakeSub(FakeLowState()))
    js = d.read()
    assert js.names == desc.action_keys and len(js) == 29
    assert js["right_elbow_joint"] == pytest.approx(G1_ORDER.index("right_elbow") * 0.01)
    assert js.efforts[0] == 0.5
    ex = d.extras()
    assert ex["imu"]["quaternion"] == [1, 0, 0, 0] and ex["power_v"] == 52.0 and ex["mode_machine"] == 4
    with pytest.raises(NotImplementedError, match="observe-only"):
        d.write(np.zeros(29))
    g1 = Robot("g1", driver=d, twin=False)
    assert g1.is_real and g1.observe().joints.names[0] == "left_hip_pitch_joint"
    g1.close()
    assert d.sub.closed


def test_unitree_without_samples_fails_fast():
    with pytest.raises(ConnectionError, match="no rt/lowstate"):
        UnitreeDriver(Description.load("g1"), port="en7", subscriber=FakeSub(None), timeout=0.05)


def test_unitree_h1_reads_the_go_wire_in_h1_slot_order():
    desc = Description.load("unitree_h1")
    d = UnitreeDriver(desc, port="en7", subscriber=FakeSub(FakeLowState()))
    js = d.read()
    assert js.names == desc.action_keys and len(js) == 19
    for joint in ("left_hip_yaw", "right_hip_roll", "torso", "left_elbow"):
        assert js[joint] == pytest.approx(H1_SLOTS[joint] * 0.01)
    assert 9 not in d.index  # the H1 has no motor in slot 9


class FakeRobotd:
    """robotd on a real unix socket: answers hello, then streams robot.state after robot.subscribe."""

    def __init__(self, joints):
        self.path = os.path.join(tempfile.mkdtemp(), "robotd.sock")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        self.joints = joints
        self.seen: list[dict] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.server.accept()
        for raw in conn.makefile("rb"):
            msg = json.loads(raw)
            self.seen.append(msg)
            if msg["method"] == "hello":
                conn.sendall(md.line({"jsonrpc": "2.0", "id": msg["id"], "result": {"api_version": 31}}))
            elif msg["method"] == "robot.subscribe":
                conn.sendall(md.line({"jsonrpc": "2.0", "id": msg["id"], "result": {}}))
                state = {"t": 1.5, "joints": self.joints, "odom": {"position": [0.1, 0, 0], "yaw": 0.2}}
                conn.sendall(md.line({"jsonrpc": "2.0", "method": "robot.state", "params": state}))


def test_microduck_reads_robotd_state_and_drops_the_mouth():
    joints = [i * 0.1 for i in range(15)]  # robotd's wire order, mouth at index 9
    robotd = FakeRobotd(joints)
    duck = Robot("microduck", port=robotd.path, twin=False)
    obs = duck.observe()
    assert len(obs.joints) == 14 and md.HARDWARE_JOINT_NAMES[md.MOUTH_INDEX] == "mouth"
    assert obs.joints["right_hip_yaw"] == pytest.approx(1.0) and obs.joints["head_roll"] == pytest.approx(0.8)
    assert obs.extras["odom"]["yaw"] == 0.2
    assert [m["method"] for m in robotd.seen] == ["hello", "robot.subscribe"]
    assert robotd.seen[0]["params"] == {"api_version": md.API_VERSION}
    with pytest.raises(NotImplementedError, match="observe-only"):
        duck.act(obs.joints.positions)
    duck.close()


def test_microduck_without_robotd_names_the_forward():
    with pytest.raises(ConnectionError, match="ssh -N -L"):
        Robot("microduck", port="/nonexistent/robotd.sock", twin=False)
