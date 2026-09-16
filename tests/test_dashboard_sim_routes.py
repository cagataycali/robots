"""Sim sessions, MJPEG, telemetry and the e-stop, on a fake engine so the routes are graded alone.

The fake answers the five engine calls the session uses (``robot_joint_names``,
``list_cameras``, ``mj_model.opt.timestep``, ``mj_data.time/qpos``, ``step``,
``get_frame``, ``reset``, ``set_joint_positions``, ``get_robot_state``) with
the shapes the MuJoCo engine returns. ``test_real_engine_*`` at the bottom
runs the same session on the real engine and is skipped without ``mujoco``.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import settings, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402


class FakeEngine:
    def __init__(self, robot: str, joints: int = 3):
        self.robot = robot
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.mj_data = SimpleNamespace(time=0.0, qpos=np.zeros(joints))
        self.steps = 0
        self.closed = False

    def robot_joint_names(self, robot):
        return [f"j{i}" for i in range(len(self.mj_data.qpos))]

    def list_cameras(self):
        return ["default"]

    def step(self, n=1):
        self.steps += n
        self.mj_data.time += n * 0.002
        self.mj_data.qpos = self.mj_data.qpos + 0.001 * n
        return {"status": "success", "content": [{"text": f"+{n}"}]}

    def get_frame(self, camera_name="default", width=None, height=None):
        return np.full((height or 8, width or 8, 3), 128, dtype=np.uint8), np.zeros((8, 8))

    def reset(self):
        self.mj_data.time = 0.0
        self.mj_data.qpos = np.zeros_like(self.mj_data.qpos)
        return {"status": "success", "content": [{"text": "reset"}]}

    def set_joint_positions(self, positions, robot_name=None, hold=False):
        if isinstance(positions, dict) and any(k not in self.robot_joint_names(robot_name) for k in positions):
            return {"status": "error", "content": [{"text": "unknown joint"}]}
        return {"status": "success", "content": [{"text": "set"}]}

    def get_robot_state(self, robot_name=None):
        return {"status": "success", "content": [{"json": {"state": {}}}]}

    def close(self):
        self.closed = True


class ExplodingEngine:
    def __init__(self, robot):
        raise RuntimeError("no GL here")


@pytest.fixture()
def fake_factory(monkeypatch):
    made: list[FakeEngine] = []

    def factory(robot: str):
        e = FakeEngine(robot)
        made.append(e)
        return e

    monkeypatch.setattr(sim_session, "_default_factory", factory)  # looked up at call time
    return made


@pytest.fixture()
def client(tmp_path, monkeypatch, fake_factory):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.state.safety.store.shutdown()


def _create(client, robot="so101"):
    r = client.post("/api/sim", json={"robot": robot})
    assert r.status_code == 201, r.text
    return r.json()


# -- session -------------------------------------------------------------------


class TestSimSession:
    def test_starts_steps_and_renders(self, fake_factory):
        s = sim_session.SimSession("so101")
        assert s.wait_ready(5)
        time.sleep(0.25)
        snap = s.snapshot
        assert snap.state == "running"
        assert snap.joint_names == ("j0", "j1", "j2")
        assert snap.steps > 0 and snap.sim_time > 0
        assert s.latest_frame() is not None
        s.stop()
        assert s.snapshot.state == "stopped"
        assert fake_factory[0].closed

    def test_freeze_stops_stepping_but_keeps_answering(self, fake_factory):
        s = sim_session.SimSession("so101")
        s.wait_ready(5)
        time.sleep(0.1)
        s.freeze()
        time.sleep(0.1)
        t1 = s.snapshot.sim_time
        time.sleep(0.15)
        assert s.snapshot.sim_time == t1
        assert s.snapshot.state == "frozen"
        assert s.latest_frame() is not None, "a frozen robot is still drawn where it stopped"
        s.thaw()
        time.sleep(0.15)
        assert s.snapshot.sim_time > t1
        s.stop()

    def test_commands_run_on_the_worker(self, fake_factory):
        s = sim_session.SimSession("so101")
        s.wait_ready(5)
        assert s.command("reset")["status"] == "success"
        assert s.command("set_joints", positions={"j0": 0.1})["status"] == "success"
        assert s.command("set_joints", positions={"zz": 0.1})["status"] == "error"
        assert s.command("bogus")["status"] == "error"
        s.stop()
        with pytest.raises(RuntimeError):
            s.command("reset")

    def test_a_factory_failure_is_an_error_state_not_a_hang(self, monkeypatch):
        s = sim_session.SimSession("so101", engine_factory=ExplodingEngine)
        assert s.wait_ready(5)
        assert s.snapshot.state == "error"
        assert "no GL here" in (s.snapshot.error or "")

    def test_store_caps_live_sessions(self, fake_factory):
        store = sim_session.SessionStore(limit=2)
        a = store.create("so101")
        store.create("so101")
        with pytest.raises(RuntimeError, match="2 sessions"):
            store.create("so101")
        assert store.remove(a.id) and not store.remove(a.id)
        store.create("so101")  # room again
        store.shutdown()
        assert all(s.snapshot.state == "stopped" for s in store.all())


# -- routes --------------------------------------------------------------------


class TestSimRoutes:
    def test_create_get_delete(self, client):
        snap = _create(client)
        assert snap["state"] == "running" and snap["joint_names"] == ["j0", "j1", "j2"]
        assert client.get(f"/api/sim/{snap['id']}").json()["robot"] == "so101"
        assert client.get("/api/sim").json()["sessions"][0]["id"] == snap["id"]
        assert client.delete(f"/api/sim/{snap['id']}").status_code == 200
        assert client.get(f"/api/sim/{snap['id']}").status_code == 404
        assert client.delete(f"/api/sim/{snap['id']}").status_code == 404

    def test_only_registry_robots_with_a_sim_asset(self, client):
        assert client.post("/api/sim", json={"robot": "not-a-robot"}).status_code == 400
        assert client.post("/api/sim", json={"robot": 3}).status_code == 400
        assert client.post("/api/sim", json=[]).status_code == 400

    def test_an_alias_resolves_to_the_canonical_name(self, client):
        assert _create(client, "so-101")["robot"] == "so101"

    def test_engine_failure_is_500_and_the_session_is_not_kept(self, client, monkeypatch):
        monkeypatch.setattr(sim_session, "_default_factory", ExplodingEngine)
        r = client.post("/api/sim", json={"robot": "so101"})
        assert r.status_code == 500 and "no GL here" in r.json()["error"]
        assert client.get("/api/sim").json()["sessions"] == []

    def test_joints_validate_before_reaching_the_engine(self, client):
        sid = _create(client)["id"]
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {}}).status_code == 400
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": "x"}}).status_code == 400
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"zz": 0.1}}).status_code == 400
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": 0.1}}).status_code == 200
        assert client.post(f"/api/sim/{sid}/reset").status_code == 200

    def test_stream_is_multipart_mjpeg_and_bounded_on_request(self, client):
        sid = _create(client)["id"]
        time.sleep(0.2)
        r = client.get(f"/api/sim/{sid}/stream.mjpg?frames=2")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert r.content.count(b"Content-Type: image/jpeg") == 2
        assert client.get(f"/api/sim/{sid}/stream.mjpg?frames=0").status_code == 400

    def test_telemetry_websocket_carries_the_lockout(self, client):
        sid = _create(client)["id"]
        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws:
            m = ws.receive_json()
        assert m["id"] == sid and m["lockout"]["state"] == "clear"
        assert len(m["qpos"]) == 3

    def test_telemetry_for_an_unknown_session_closes(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws/telemetry/nope"):
            pass
        assert exc.value.code == 4404

    def test_telemetry_from_another_origin_is_closed_before_it_is_accepted(self, client):
        """WebSockets have no CORS: a page anywhere can open ws://127.0.0.1:8090.
        Its Origin is the one thing it cannot hide, and admission reads it."""
        from starlette.websockets import WebSocketDisconnect

        sid = client.post("/api/sim", json={"robot": "so101"}).json()["id"]
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(f"/ws/telemetry/{sid}", headers={"origin": "http://evil.example"}),
        ):
            pass
        assert exc.value.code == 4401


class TestEstop:
    def test_estop_freezes_and_gates_then_resume_needs_proof(self, client):
        sid = _create(client)["id"]
        assert client.get("/api/safety").json()["lockout"]["state"] == "clear"
        e = client.post("/api/safety/estop").json()
        assert e["lockout"]["state"] == "locked" and e["frozen"] == [sid]
        time.sleep(0.1)
        t1 = client.get(f"/api/sim/{sid}").json()["sim_time"]
        time.sleep(0.1)
        assert client.get(f"/api/sim/{sid}").json()["sim_time"] == t1
        assert client.get(f"/api/sim/{sid}").json()["state"] == "frozen"
        for path, body in (
            (f"/api/sim/{sid}/joints", {"positions": {"j0": 0.1}}),
            (f"/api/sim/{sid}/reset", None),
            ("/api/sim", {"robot": "so101"}),
        ):
            r = client.post(path, json=body)
            assert r.status_code == 423, path
            assert "e-stop engaged" in r.json()["error"]
        assert client.delete(f"/api/sim/{sid}").status_code == 200, "stopping is never refused"
        sid = None
        r = client.post("/api/safety/resume").json()
        assert r["lockout"]["state"] == "unknown", "a resume is a request, not proof"
        snap = _create(client)
        assert client.get("/api/safety").json()["lockout"]["state"] == "clear", "an accepted command is the proof"
        assert snap["state"] == "running"

    def test_an_estop_reaches_a_session_whose_engine_is_still_building(self, fake_factory):
        """The window an e-stop exists for: Start pressed, engine not built yet, then E-STOP.

        Building the engine takes real time (a model compile plus a renderer),
        so a session sits in ``starting`` for that long. It has not stepped yet
        and starts the instant the build returns, so the e-stop must reach it.
        """
        from strands_robots.dashboard.routes_sim import Safety

        building, release = threading.Event(), threading.Event()

        def slow(robot):
            building.set()
            release.wait(5)
            return FakeEngine(robot)

        store = sim_session.SessionStore()
        safety = Safety(store)
        session = store.create("so101", engine_factory=slow)
        assert building.wait(5) and session.snapshot.state == "starting"

        assert safety.estop(by="operator")["frozen"] == [session.id]
        release.set()
        assert session.wait_ready(5)
        time.sleep(0.2)
        snap = session.snapshot
        assert snap.state == "frozen", "the engine arrived into an e-stop, so it reports frozen"
        assert (snap.steps, snap.sim_time) == (0, 0.0), "no physics ran after the e-stop"

        safety.resume(by="operator")
        time.sleep(0.2)
        assert session.snapshot.steps > 0, "a resume thaws the session frozen while it was starting"
        store.shutdown()

    def test_an_estop_during_the_build_refuses_that_create_and_stays_latched(self, client, monkeypatch):
        """The create was admitted before the e-stop, so it is neither served nor taken as proof."""
        building, release = threading.Event(), threading.Event()

        def slow(robot):
            building.set()
            release.wait(5)
            return FakeEngine(robot)

        monkeypatch.setattr(sim_session, "_default_factory", slow)
        reply: dict = {}

        def create():
            r = client.post("/api/sim", json={"robot": "so101"})
            reply.update(status=r.status_code, body=r.json())

        worker = threading.Thread(target=create)
        worker.start()
        assert building.wait(5), "the create never reached the engine build"
        safety = client.app.state.safety
        session = safety.store.all()[0]
        assert safety.estop(by="operator")["frozen"] == [session.id]
        release.set()
        worker.join(10)

        assert reply["status"] == 423 and "e-stop engaged" in reply["body"]["error"]
        assert safety.lockout.state == "locked", "an in-flight create is not proof that the lockout lifted"
        assert safety.store.all() == [], "the refused session is not left running"
        assert session.snapshot.steps == 0

    def test_estop_is_never_refused(self, client):
        client.post("/api/safety/estop")
        assert client.post("/api/safety/estop").status_code == 200

    def test_everything_is_guarded(self, client, monkeypatch):
        sid = _create(client)["id"]
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "1")
        for method, path in (
            ("get", "/api/fleet"),
            ("get", "/api/robots/so101"),
            ("get", "/api/sim"),
            ("post", "/api/sim"),
            ("get", f"/api/sim/{sid}"),
            ("get", f"/api/sim/{sid}/stream.mjpg?frames=1"),
            ("post", f"/api/sim/{sid}/joints"),
            ("delete", f"/api/sim/{sid}"),
            ("get", "/api/safety"),
            ("post", "/api/safety/estop"),
            ("post", "/api/safety/resume"),
        ):
            assert getattr(client, method)(path).status_code == 401, (method, path)
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(f"/ws/telemetry/{sid}"):
            pass
        assert exc.value.code == 4401


class TestFleet:
    def test_fleet_lists_the_registry_and_says_mesh_off(self, client):
        f = client.get("/api/fleet").json()
        names = {r["name"] for r in f["robots"]}
        assert {"so101", "unitree_g1", "panda"} <= names
        assert f["count"] == len(f["robots"])
        assert f["mesh"]["status"] == "off"
        assert all("model_local" in r for r in f["robots"] if r["has_sim"])

    def test_fleet_mode_filter(self, client):
        assert client.get("/api/fleet?mode=nope").status_code == 400
        both = client.get("/api/fleet?mode=both").json()["robots"]
        assert both and all(r["has_sim"] and r["has_real"] for r in both)

    def test_robot_detail_and_alias(self, client):
        r = client.get("/api/robots/so-101").json()
        assert r["name"] == "so101" and r["entry"]["category"]
        assert client.get("/api/robots/nope").status_code == 404


# -- the real engine -----------------------------------------------------------


@pytest.mark.skipif(pytest.importorskip("mujoco", reason="mujoco not installed") is None, reason="mujoco")
def test_real_engine_session_steps_and_renders(monkeypatch):
    s = sim_session.SimSession("so101")
    assert s.wait_ready(60), "engine did not start"
    if s.snapshot.state == "error":
        pytest.skip(f"no renderer here: {s.snapshot.error}")
    time.sleep(1.0)
    snap = s.snapshot
    assert snap.joint_names == ("1", "2", "3", "4", "5", "6")
    assert snap.sim_time > 0.2
    frame = s.latest_frame()
    assert frame is not None and frame.shape == (384, 512, 3)
    assert s.command("set_joints", positions={"2": 0.3})["status"] == "success"
    s.stop()
