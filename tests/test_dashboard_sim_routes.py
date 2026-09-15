"""Sim sessions, MJPEG, telemetry and the e-stop, on a fake engine so the routes are graded alone.

The fake answers the five engine calls the session uses (``robot_joint_names``,
``list_cameras``, ``mj_model.opt.timestep``, ``mj_data.time/qpos``, ``step``,
``get_frame``, ``reset``, ``set_joint_positions``, ``get_robot_state``) with
the shapes the MuJoCo engine returns. ``test_real_engine_*`` at the bottom
runs the same session on the real engine and is skipped without ``mujoco``.
"""

from __future__ import annotations

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
        self.mj_data = SimpleNamespace(
            time=0.0, qpos=np.zeros(joints), geom_xpos=np.zeros((2, 3)), geom_xmat=np.tile(np.eye(3).ravel(), (2, 1))
        )
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
        self.last_hold = hold
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


class FakeModel:
    """Two geoms (a plane and a mesh), one 4-vertex / 2-face mesh - the fields scene.py reads."""

    ngeom, nmesh, ncam, nlight, nbody = 2, 1, 0, 1, 1
    geom_type = np.array([0, 7])
    geom_size = np.array([[5.0, 5.0, 0.01], [0.1, 0.1, 0.1]])
    geom_rgba = np.array([[0.5, 0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 0.5]])
    geom_matid = np.array([-1, -1])
    geom_group = np.array([0, 3])
    geom_dataid = np.array([-1, 0])
    geom_bodyid = np.array([0, 0])
    mat_rgba = np.zeros((0, 4))
    mesh_vertadr = np.array([0])
    mesh_vertnum = np.array([4])
    mesh_faceadr = np.array([0])
    mesh_facenum = np.array([2])
    mesh_vert = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mesh_face = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cam_fovy = np.zeros(0)


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
        assert fake_factory[0].last_hold is True, "servo setpoints must move with the pose"
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


class TestTwinGeometry:
    def test_scene_describes_the_compiled_model(self, monkeypatch):
        from strands_robots.dashboard import scene

        monkeypatch.setattr(scene, "_name", lambda model, kind, i: f"{kind.lower()}{i}")
        d = scene.describe(FakeModel())
        assert d["ngeom"] == 2 and d["pose_row_floats"] == 12
        plane, mesh = d["geoms"]
        assert plane["type"] == "plane" and plane["mesh"] is None and plane["rgba"] == [0.5, 0.5, 0.5, 1.0]
        assert mesh["type"] == "mesh" and mesh["mesh"] == 0 and mesh["group"] == 3 and mesh["body"] == "body0"
        assert d["meshes"] == [{"id": 0, "name": "mesh0", "vertices": 4, "faces": 2, "url": "mesh/0"}]

    def test_mesh_bytes_round_trip(self):
        import struct

        from strands_robots.dashboard import scene

        b = scene.mesh_bytes(FakeModel(), 0)
        assert b[:4] == b"SRM1"
        nvert, nface = struct.unpack("<II", b[4:12])
        assert (nvert, nface) == (4, 2)
        verts = np.frombuffer(b[12 : 12 + nvert * 12], dtype="<f4").reshape(4, 3)
        faces = np.frombuffer(b[12 + nvert * 12 :], dtype="<u4").reshape(2, 3)
        assert verts[3].tolist() == [0.0, 0.0, 1.0] and faces[1].tolist() == [0, 2, 3]
        with pytest.raises(IndexError):
            scene.mesh_bytes(FakeModel(), 1)

    def test_poses_are_packed_as_12_float32_per_geom(self, fake_factory):
        s = sim_session.SimSession("so101")
        s.wait_ready(5)
        time.sleep(0.15)
        poses = np.frombuffer(s.snapshot.poses, dtype="<f4").reshape(-1, 12)
        assert poses.shape == (2, 12)
        assert poses[0, 3:].tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 1]
        s.stop()

    def test_scene_and_mesh_routes(self, client, monkeypatch):
        sid = _create(client)["id"]
        monkeypatch.setattr(sim_session.SimSession, "model", property(lambda self: FakeModel()))
        from strands_robots.dashboard import scene

        monkeypatch.setattr(scene, "_name", lambda model, kind, i: None)
        assert client.get(f"/api/sim/{sid}/scene").json()["ngeom"] == 2
        r = client.get(f"/api/sim/{sid}/mesh/0")
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        assert r.content[:4] == b"SRM1" and "max-age" in r.headers["cache-control"]
        assert client.get(f"/api/sim/{sid}/mesh/7").status_code == 404
        assert client.get("/api/sim/nope/scene").status_code == 404

    def test_scene_before_the_engine_exists_is_409(self, client, monkeypatch):
        sid = _create(client)["id"]
        monkeypatch.setattr(sim_session.SimSession, "model", property(lambda self: None))
        assert client.get(f"/api/sim/{sid}/scene").status_code == 409

    def test_telemetry_sends_binary_poses_only_when_asked(self, client):
        sid = _create(client)["id"]
        time.sleep(0.15)
        with client.websocket_connect(f"/ws/telemetry/{sid}?poses=1") as ws:
            snap = ws.receive_json()
            raw = ws.receive_bytes()
        assert snap["id"] == sid and len(raw) == 2 * 12 * 4
        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws:
            ws.receive_json()
            ws.receive_json()  # two JSON frames in a row: no binary interleaved


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
            ("get", f"/api/sim/{sid}/scene"),
            ("get", f"/api/sim/{sid}/mesh/0"),
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
    from strands_robots.dashboard import scene

    d = scene.describe(s.model)
    assert d["ngeom"] == 31 and len(d["meshes"]) == 13 and d["geoms"][0]["type"] == "plane"
    assert len(s.snapshot.poses) == 31 * 12 * 4
    assert scene.mesh_bytes(s.model, 0)[:4] == b"SRM1"
    assert s.command("set_joints", positions={"2": 0.3})["status"] == "success"
    s.stop()
