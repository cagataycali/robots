"""The twin follows the real arm: a read-only bus reader drives a session that never steps.

Graded on a fake bus and the fake engine from ``test_dashboard_sim_routes``, so
no serial port is opened; ``test_hardware_*`` at the bottom reads the real
SO-101 and is skipped when its port is not on this machine.
"""

from __future__ import annotations

import math
import os
import threading
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import mirror, routes_sim, settings, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests.test_dashboard_sim_routes import FakeEngine  # noqa: E402

REAL_PORT = "/dev/cu.usbmodem5AB01818061"


class FakeBus:
    """Answers ``connect`` / ``sync_read`` / ``disconnect`` and records every call, so a write shows up."""

    def __init__(self, port, motors, *, fail_connect=False, fail_after=None):
        self.port = port
        self.motors = motors
        self.calls: list[tuple] = []
        self.ticks = {name: 2048 + 512 * i for i, name in enumerate(motors)}
        self._fail_connect = fail_connect
        self._fail_after = fail_after
        self.reads = 0

    def connect(self, handshake=True):
        self.calls.append(("connect", handshake))
        if self._fail_connect:
            raise ConnectionError("Failed to open the port")

    def sync_read(self, data_name, motors=None, *, normalize=True):
        self.calls.append(("sync_read", data_name, normalize))
        self.reads += 1
        if self._fail_after is not None and self.reads > self._fail_after:
            raise OSError(6, "Device not configured")
        return dict(self.ticks)

    def disconnect(self, disable_torque=True):
        self.calls.append(("disconnect", disable_torque))

    # anything else is a write
    def write(self, *a, **k):
        raise AssertionError("the mirror wrote to the bus")

    sync_write = write
    enable_torque = write
    disable_torque = write
    configure_motors = write


class BusFactory:
    """Builds :class:`FakeBus` instances through the ``bus_factory`` seam, and keeps every one it built."""

    def __init__(self, **failure):
        self.made: list[FakeBus] = []
        self._failure = failure

    def __call__(self, port, motors):
        bus = FakeBus(port, motors, **self._failure)
        self.made.append(bus)
        return bus

    def __getitem__(self, index):
        return self.made[index]


@pytest.fixture()
def fake_bus(monkeypatch):
    """Every mirror in a test - built here or by the create route - reads a fake bus, not a port."""
    factory = BusFactory()
    monkeypatch.setattr(mirror, "_lerobot_bus", factory)
    return factory


def _wait(pred, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


class TestTicks:
    def test_centre_is_zero_and_a_quarter_turn_is_half_pi(self):
        assert mirror.ticks_to_rad(2048) == 0.0
        assert mirror.ticks_to_rad(3072) == pytest.approx(math.pi / 2)
        assert mirror.ticks_to_rad(1024) == pytest.approx(-math.pi / 2)

    def test_the_folded_arm_from_the_bench_lands_inside_the_models_ranges(self):
        # h02's reading of the real arm at rest. so101/2 range is +-1.745, so101/3 is [-1.745, 1.571].
        assert mirror.ticks_to_rad(953) == pytest.approx(-1.680, abs=1e-3)
        assert mirror.ticks_to_rad(2986) == pytest.approx(1.439, abs=1e-3)


class TestBusMirror:
    def test_reads_present_position_raw_and_never_writes(self, fake_bus):
        m = mirror.BusMirror("/dev/fake")
        assert m.wait_ready()
        assert m.error is None
        assert _wait(lambda: m.reading is not None)
        bus = fake_bus[0]
        assert bus.calls[0] == ("connect", True)
        assert all(c[0] in ("connect", "sync_read") for c in bus.calls)
        assert all(c[1:] == ("Present_Position", False) for c in bus.calls if c[0] == "sync_read")
        m.close()
        assert bus.calls[-1] == ("disconnect", False), "disconnect(disable_torque=True) is a write"

    def test_qpos_is_radians_in_motor_order(self, fake_bus):
        m = mirror.BusMirror("/dev/fake", {"a": 1, "b": 2})
        m.wait_ready()
        assert _wait(lambda: m.qpos() is not None)
        assert m.qpos() == pytest.approx((0.0, math.pi / 4))
        m.close()

    def test_health_says_what_it_does_not_apply(self, fake_bus):
        m = mirror.BusMirror("/dev/fake")
        m.wait_ready()
        assert _wait(lambda: m.reading is not None)
        h = m.health()
        assert h["connected"] is True and h["error"] is None and h["port"] == "/dev/fake"
        assert "no calibration" in h["angles"] and "none" in h["writes"]
        assert h["ticks"]["shoulder_pan"] == 2048
        m.close()

    def test_a_port_that_will_not_open_is_an_error_not_a_pose(self):
        m = mirror.BusMirror("/dev/fake", bus_factory=BusFactory(fail_connect=True))
        assert m.wait_ready()
        assert m.error and "could not open /dev/fake" in m.error and "Failed to open the port" in m.error
        assert m.qpos() is None
        assert m.health()["connected"] is False
        m.close()

    def test_a_bus_that_dies_mid_read_reports_it_and_stops(self):
        dying = BusFactory(fail_after=2)
        m = mirror.BusMirror("/dev/fake", bus_factory=dying)
        m.wait_ready()
        assert _wait(lambda: m.error is not None)
        assert "lost /dev/fake" in m.error and "Device not configured" in m.error
        assert dying[0].calls[-1] == ("disconnect", False)
        time.sleep(mirror.STALE_AFTER + 0.1)
        assert m.qpos() is None
        m.close()

    def test_a_reading_goes_stale(self, fake_bus, monkeypatch):
        m = mirror.BusMirror("/dev/fake")
        m.wait_ready()
        assert _wait(lambda: m.qpos() is not None)
        monkeypatch.setattr(mirror, "STALE_AFTER", -1.0)  # every reading is now too old
        assert m.qpos() is None
        assert m.reading is not None, "the reading is kept; only qpos() withholds it"
        m.close()


class FakeSource:
    """A ``BusMirror`` stand-in the session can be driven with, from the test's side."""

    def __init__(self, port="/dev/fake"):
        self.port = port
        self.error = None
        self._q: tuple | None = (0.1, 0.2, 0.3)
        self.closed = threading.Event()
        self.lock = threading.Lock()

    def qpos(self):
        with self.lock:
            return self._q

    def set(self, q):
        with self.lock:
            self._q = q

    def health(self):
        return {"port": self.port, "hz": 20.0, "age_ms": 10, "error": self.error}

    def close(self):
        self.closed.set()


class TestMirrorSession:
    def test_never_steps_and_poses_the_model_from_the_source(self):
        engines: list[FakeEngine] = []

        def factory(robot):
            e = FakeEngine(robot)
            engines.append(e)
            return e

        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=factory, source=src)
        assert s.wait_ready()
        assert _wait(lambda: s.snapshot.bus is not None and getattr(engines[0], "last_hold", None) is True)
        assert s.snapshot.state == "mirroring"
        assert engines[0].steps == 0
        assert s.snapshot.source == "real:/dev/fake"
        assert s.snapshot.bus["hz"] == 20.0
        assert s.snapshot.as_dict()["source"] == "real:/dev/fake"
        time.sleep(0.2)
        assert engines[0].steps == 0, "a mirror must not step physics"
        s.stop()
        assert src.closed.is_set()
        assert engines[0].closed

    def test_stale_source_shows_as_stale_and_a_dead_one_as_error(self):
        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=FakeEngine, source=src)
        s.wait_ready()
        assert _wait(lambda: s.snapshot.state == "mirroring")
        src.set(None)
        assert _wait(lambda: s.snapshot.state == "stale")
        src.error = "lost /dev/fake: OSError: Device not configured"
        assert _wait(lambda: s.snapshot.state == "error")
        assert s.snapshot.error == src.error
        s.stop()

    def test_a_pose_the_model_refuses_is_not_reported_as_mirroring(self):
        # set_joint_positions refuses the WHOLE write when one angle is outside the
        # joint's range, so the twin stays on the last accepted pose. Measured on the
        # real so101: joint 2 at -1.80 rad (range +-1.745) left qpos on the previous
        # pose while the state read "mirroring" and the bus read 20 Hz, age 10 ms.
        class Fussy(FakeEngine):
            REFUSAL = "set_joint_positions: position outside the joint's range, nothing written: j1=-1.8 outside [-1.745, 1.745] rad"

            def set_joint_positions(self, positions, robot_name=None, hold=False):
                if any(abs(float(v)) > 1.745 for v in positions.values()):
                    return {"status": "error", "content": [{"text": self.REFUSAL}]}
                return super().set_joint_positions(positions, robot_name=robot_name, hold=hold)

        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=Fussy, source=src)
        s.wait_ready()
        assert _wait(lambda: s.snapshot.state == "mirroring")
        src.set((0.1, -1.8, 0.3))
        assert _wait(lambda: s.snapshot.state == "refused"), f"reported {s.snapshot.state} for a refused write"
        assert "nothing written" in s.snapshot.error
        assert s.snapshot.bus["error"] is None, "the bus is healthy; the model refused the pose"
        src.set((0.1, -1.0, 0.3))  # the arm comes back inside the range
        assert _wait(lambda: s.snapshot.state == "mirroring" and s.snapshot.error is None)
        s.stop()

    def test_the_arm_decides_the_pose(self):
        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=FakeEngine, source=src)
        s.wait_ready()
        for kind, payload in (("set_joints", {"positions": {"j0": 0.0}}), ("reset", {})):
            r = s.command(kind, **payload)
            assert r["status"] == "error"
            assert "mirrors the real arm on /dev/fake" in r["content"][0]["text"]
        assert s.command("state")["status"] == "success", "reading is still allowed"
        s.stop()

    def test_the_estop_freezes_a_mirror_too_and_thaw_returns_to_mirroring(self):
        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=FakeEngine, source=src)
        s.wait_ready()
        s.freeze()
        assert _wait(lambda: s.snapshot.state == "frozen")
        s.thaw()
        assert _wait(lambda: s.snapshot.state == "mirroring")
        s.stop()

    def test_an_engine_that_will_not_build_releases_the_port(self):
        # The source holds the serial device exclusively. A session whose engine
        # never exists (missing asset, MuJoCo init error) has no engine to close,
        # and the port must not stay held until the process restarts.
        def refusing(robot):
            raise FileNotFoundError(f"no model for {robot}")

        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=refusing, source=src)
        assert s.wait_ready()
        assert s.snapshot.state == "error"
        assert "FileNotFoundError" in s.snapshot.error
        assert src.closed.wait(2.0), "the mirror source is still open after the engine failed to build"
        s.stop()

    def test_an_engine_that_will_not_render_releases_the_port(self):
        class Blind(FakeEngine):
            def get_frame(self, camera_name="default", width=None, height=None):
                raise RuntimeError("no GL context")

        src = FakeSource()
        s = sim_session.SimSession("so101", engine_factory=Blind, source=src)
        assert s.wait_ready()
        assert s.snapshot.state == "error"
        assert src.closed.wait(2.0), "the mirror source is still open after the first render failed"
        s.stop()


@pytest.fixture()
def client(tmp_path, monkeypatch, fake_bus):
    monkeypatch.setattr(sim_session, "_default_factory", FakeEngine)
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.state.safety.store.shutdown()


class TestMirrorRoutes:
    def test_a_path_that_is_not_a_device_is_refused_before_anything_opens(self, client, fake_bus):
        for bad in ("/tmp", "/dev/../etc/passwd", "COM3", "/dev/does-not-exist-xyz"):
            r = client.post("/api/sim", json={"robot": "so101", "mirror": {"port": bad}})
            assert r.status_code == 400, (bad, r.text)
        r = client.post("/api/sim", json={"robot": "so101", "mirror": "yes"})
        assert r.status_code == 400
        assert fake_bus.made == []

    def test_a_mirror_session_reports_its_source_and_refuses_joints(self, client, fake_bus, tmp_path, monkeypatch):
        dev = tmp_path / "cu.fake"
        dev.touch()
        monkeypatch.setattr(routes_sim, "_is_serial_device", lambda p: p == str(dev))
        r = client.post("/api/sim", json={"robot": "so101", "mirror": {"port": str(dev)}})
        assert r.status_code == 201, r.text
        snap = r.json()
        assert snap["source"] == f"real:{dev}"
        sid = snap["id"]
        assert _wait(lambda: client.get(f"/api/sim/{sid}").json()["state"] == "mirroring")
        got = client.get(f"/api/sim/{sid}").json()
        assert got["bus"]["port"] == str(dev) and got["bus"]["error"] is None
        r = client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": 0.5}})
        assert r.status_code == 400 and "mirrors the real arm" in r.text
        r = client.post(f"/api/sim/{sid}/reset")
        assert r.status_code == 200 and r.json()["status"] == "error"
        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws:
            m = ws.receive_json()
            assert m["source"] == f"real:{dev}" and m["bus"]["hz"] >= 0
        stopped = client.delete(f"/api/sim/{sid}")
        assert stopped.status_code == 200
        assert fake_bus[0].calls[-1] == ("disconnect", False)

    def test_the_telemetry_socket_lives_through_a_refused_sweep(self, client, fake_bus, tmp_path, monkeypatch):
        # "refused" is transient, so it belongs to no terminal set: the telemetry loop
        # breaks on ("stopped", "error") only. An arm a few degrees outside the model's
        # range would otherwise close the socket on its first sweep and leave the page's
        # twin, joint bars and bus line dead until a reload - after it comes back, too.
        class Fussy(FakeEngine):
            def set_joint_positions(self, positions, robot_name=None, hold=False):
                if any(abs(float(v)) > 1.745 for v in positions.values()):
                    return {
                        "status": "error",
                        "content": [{"text": "j0=2.99 outside [-1.745, 1.745] rad, nothing written"}],
                    }
                return super().set_joint_positions(positions, robot_name=robot_name, hold=hold)

        dev = tmp_path / "cu.fake"
        dev.touch()
        monkeypatch.setattr(routes_sim, "_is_serial_device", lambda p: p == str(dev))
        monkeypatch.setattr(sim_session, "_default_factory", Fussy)
        sid = client.post("/api/sim", json={"robot": "so101", "mirror": {"port": str(dev)}}).json()["id"]
        assert _wait(lambda: client.get(f"/api/sim/{sid}").json()["state"] == "mirroring")

        def until(ws, state, reads=40):
            """Read snapshots up to ``state``, failing on the first terminal one."""
            for _ in range(reads):
                m = ws.receive_json()
                assert m["state"] != "error", f"a transient refusal ended the socket: {m['error']}"
                if m["state"] == state:
                    return m
            raise AssertionError(f"{state} never arrived on the socket")

        def ticks(value):
            for motor in fake_bus[0].ticks:
                fake_bus[0].ticks[motor] = value

        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws:
            ticks(4000)  # 2.99 rad, past every so101 range
            refused = until(ws, "refused")
            assert "nothing written" in refused["error"]
            ticks(2048)  # the arm comes back inside
            recovered = until(ws, "mirroring")  # the same socket, never reopened
            assert recovered["error"] is None
        stopped = client.delete(f"/api/sim/{sid}")
        assert stopped.status_code == 200

    def test_a_bus_that_will_not_open_is_502_with_the_reason(self, client, fake_bus, tmp_path, monkeypatch):
        dev = tmp_path / "cu.fake"
        dev.touch()
        monkeypatch.setattr(routes_sim, "_is_serial_device", lambda p: p == str(dev))
        monkeypatch.setattr(mirror, "_lerobot_bus", BusFactory(fail_connect=True))
        r = client.post("/api/sim", json={"robot": "so101", "mirror": {"port": str(dev)}})
        assert r.status_code == 502
        assert "could not open" in r.text and "Failed to open the port" in r.text
        assert client.get("/api/sim").json()["sessions"] == []

    def test_ports_lists_servo_buses_first_and_opens_nothing(self, client, monkeypatch):
        from types import SimpleNamespace

        import strands_robots._serial_discovery as disc

        monkeypatch.setattr(
            disc,
            "scan_serial_devices",
            lambda: [
                SimpleNamespace(port="/dev/cu.usbserial-A", stable_id="A1", likely_servo_bus=False),
                SimpleNamespace(port="/dev/cu.usbmodem5AB0", stable_id="5AB0", likely_servo_bus=True),
            ],
        )
        r = client.get("/api/sim/ports")
        assert r.status_code == 200
        assert [p["port"] for p in r.json()["ports"]] == ["/dev/cu.usbmodem5AB0", "/dev/cu.usbserial-A"]
        assert r.json()["ports"][0]["likely_servo_bus"] is True


@pytest.mark.hardware
@pytest.mark.skipif(not os.path.exists(REAL_PORT), reason="the SO-101 is not on this machine")
def test_hardware_the_real_arm_is_read_and_left_alone():
    """Reads the real bus for a second; asserts the port is released and torque was never touched."""
    import subprocess

    m = mirror.BusMirror(REAL_PORT)
    assert m.wait_ready(15), "bus did not open in 15 s"
    assert m.error is None, m.error
    assert _wait(lambda: m.qpos() is not None, 3.0)
    time.sleep(1.5)
    h = m.health()
    assert h["hz"] > 10 and h["age_ms"] < 200 and len(h["ticks"]) == 6
    q = m.qpos()
    assert all(-math.pi <= x <= math.pi for x in q)
    m.close()
    held = subprocess.run(["lsof", REAL_PORT], capture_output=True, text=True).stdout
    assert held == "", f"port still held after close:\n{held}"
