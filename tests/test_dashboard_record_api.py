"""RecordController + /api/record routes: fleet choreography around a session.

What must hold, beyond the worker's own state machine:

* open() parks BOTH managed peers (despawn, autospawn suspended) and close()
  brings them back with their original spawn configs - even when finalize or
  the hub upload fails.
* a failed open leaves the fleet exactly as it found it.
* the HTTP surface speaks the FRONTEND_HANDOFF.md contract: 200 empty
  session when idle, 404 unknown peer, 409 double-open, 422 validation.

Everything runs on fakes - no serial ports, no lerobot, no threads.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from strands_robots.dashboard.record_api import RecordController, build_router


class FakeBackend:
    def __init__(self):
        self.camera_keys = ["top"]
        self.closed = False
        self._n = 0

    def leader_action(self):
        self._n += 1
        return {"pan": float(self._n)}

    def follower_apply(self, action):
        return action

    def follower_observation(self):
        return {"pan": float(self._n), "top": np.zeros((8, 8, 3), dtype=np.uint8)}

    def close(self):
        self.closed = True


class FakeRecorder:
    def __init__(self):
        self.saved = 0
        self.finalized = False

    # Q56 guard: the real parameter is `observation`, not `obs`.
    def add_frame(self, observation, action, task=None): ...

    def save_episode(self):
        self.saved += 1
        return {"status": "ok"}

    def clear_episode_buffer(self):
        return True

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, tags=None, private=False):
        """Q56: mirrors the REAL signature (tags, private) and returns the real answer SHAPE.

        This fake used to declare ``repo_id=None`` -- a parameter DatasetRecorder has never had --
        which is how a production branch that could only raise TypeError kept a green suite.
        """
        return {"status": "success", "repo_id": "cagatay/so101-pick"}


class FakeDevices:
    """Just enough DeviceManager: robots dict + spawn/despawn ledger."""

    def __init__(self):
        self.robots = {}
        self.despawned = []
        self.spawned = []
        self.autospawn = SimpleNamespace(suspended=False)

    def add(self, peer_id, mode="real", port="/dev/cu.usb1", robot_name="so101"):
        self.robots[peer_id] = SimpleNamespace(
            peer_id=peer_id, robot_name=robot_name, mode=mode, port=port,
            cameras={"top": {"index_or_path": 0}},
        )

    def despawn(self, peer_id):
        self.despawned.append(peer_id)
        self.robots.pop(peer_id, None)
        return {"peer_id": peer_id, "stopped": True}

    def spawn(self, **cfg):
        self.spawned.append(cfg)
        self.add(cfg["peer_id"], mode=cfg["mode"], port=cfg["port"],
                 robot_name=cfg["robot_name"])
        return {"peer_id": cfg["peer_id"], "pid": 123, "mode": cfg["mode"]}


def make_controller(tmp_path, devices=None, backend_factory=None):
    devices = devices or FakeDevices()
    devices.add("arm-leader")
    devices.add("arm-follower", port="/dev/cu.usb2")
    backends = []

    def default_backend_factory(**kw):
        b = FakeBackend()
        backends.append(b)
        return b

    ctl = RecordController(
        devices,
        backend_factory=backend_factory or default_backend_factory,
        recorder_factory_factory=lambda backend: (lambda **kw: FakeRecorder()),
        thumb_root=str(tmp_path / "thumbs"),
    )
    return ctl, devices, backends


OPEN = {
    "dataset": "cagatay/so101-pick", "task": "pick the cube",
    "leader": "arm-leader", "follower": "arm-follower", "target_episodes": 2,
}


def test_open_parks_both_peers_and_suspends_autospawn(tmp_path):
    ctl, dev, _ = make_controller(tmp_path)
    s = ctl.open(dict(OPEN))
    assert s["phase"] == "idle" and s["dataset"] == "cagatay/so101-pick"
    assert dev.despawned == ["arm-leader", "arm-follower"]
    assert dev.autospawn.suspended is True


def test_close_respawns_with_original_configs_and_resumes_autospawn(tmp_path):
    ctl, dev, backends = make_controller(tmp_path)
    ctl.open(dict(OPEN))
    r = ctl.close({})
    assert r["ok"] is True
    respawned = {c["peer_id"]: c for c in dev.spawned}
    assert set(respawned) == {"arm-leader", "arm-follower"}
    assert respawned["arm-follower"]["port"] == "/dev/cu.usb2"
    assert respawned["arm-follower"]["cameras"] == {"top": {"index_or_path": 0}}
    assert dev.autospawn.suspended is False
    assert backends[0].closed
    # session is empty again; a new open works
    assert ctl.session()["dataset"] is None
    ctl.open(dict(OPEN))


def test_failed_open_leaves_the_fleet_as_it_found_it(tmp_path):
    def boom(**kw):
        raise RuntimeError("serial port said no")

    ctl, dev, _ = make_controller(tmp_path, backend_factory=boom)
    with pytest.raises(HTTPException) as e:
        ctl.open(dict(OPEN))
    assert e.value.status_code == 500
    # both peers respawned, watcher resumed, no session
    assert {c["peer_id"] for c in dev.spawned} == {"arm-leader", "arm-follower"}
    assert dev.autospawn.suspended is False
    assert ctl.session()["dataset"] is None


def test_open_refusals(tmp_path):
    ctl, dev, _ = make_controller(tmp_path)
    # unknown peer
    with pytest.raises(HTTPException) as e:
        ctl.open({**OPEN, "follower": "ghost"})
    assert e.value.status_code == 404
    # sim peer
    dev.add("simmy", mode="sim", port=None)
    with pytest.raises(HTTPException) as e:
        ctl.open({**OPEN, "follower": "simmy"})
    assert e.value.status_code == 422
    # unknown leader robot type without an override
    dev.add("weird", robot_name="unobtainium")
    with pytest.raises(HTTPException) as e:
        ctl.open({**OPEN, "leader": "weird"})
    assert e.value.status_code == 422 and "leader_type" in e.value.detail
    # nothing was parked by any refusal
    assert dev.despawned == [] and dev.autospawn.suspended is False
    # double open
    ctl.open(dict(OPEN))
    with pytest.raises(HTTPException) as e:
        ctl.open(dict(OPEN))
    assert e.value.status_code == 409


def make_client(tmp_path):
    ctl, dev, backends = make_controller(tmp_path)
    app = FastAPI()
    app.include_router(build_router(ctl))
    return TestClient(app), ctl, dev


def test_http_surface_speaks_the_contract(tmp_path):
    client, ctl, dev = make_client(tmp_path)
    # idle: 200 with dataset null (the mock-detection probe relies on this)
    r = client.get("/api/record/session")
    assert r.status_code == 200 and r.json()["dataset"] is None
    # steps without a session: 409
    assert client.post("/api/record/episode/start").status_code == 409
    # open -> start -> stop over HTTP
    r = client.post("/api/record/open", json=OPEN)
    assert r.status_code == 200 and r.json()["phase"] == "idle"
    r = client.post("/api/record/episode/start")
    assert r.json()["phase"] == "recording"
    # the worker's own control loop is live here (the controller does not
    # pass autostart_loop=False) - wait for it to capture a frame or two
    import time as _t

    deadline = _t.time() + 3.0
    while _t.time() < deadline:
        if ctl.session()["episodes"][-1]["frames"] >= 1:
            break
        _t.sleep(0.02)
    r = client.post("/api/record/episode/stop")
    body = r.json()
    assert body["phase"] == "idle" and body["episodes"][0]["frames"] >= 1
    # discard validation
    assert client.post("/api/record/episode/discard", json={}).status_code == 422
    assert client.post(
        "/api/record/episode/discard", json={"index": 99}
    ).status_code == 404
    # thumbnail written by the first frame is served; traversal shapes 404
    r = client.get("/api/record/thumb/0/top")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/record/thumb/0/..%2F..%2Fetc").status_code == 404
    # close reports and empties the session
    r = client.post("/api/record/close", json={})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/api/record/session").json()["dataset"] is None


# ── Q101: a failed open that could not put the arms back must SAY SO ───────────────────────────────
# open() despawns both arms before it builds the recorder, so every failure after that point restores
# the fleet - best effort. Restoring can fail (a port still held, a spawn refused), and that used to go
# only to logger.warning: the refusal still read "no session was opened", the frontend added "the arms
# are untouched and still in the fleet", and both robot cards were gone. The operator cannot see the
# server log; they can see the sentence the click produced.

class RefusingSpawnDevices(FakeDevices):
    """Despawns fine, refuses to bring one arm back - the shape of a port still held."""

    def __init__(self, refuse: str, *, raises: bool = False):
        super().__init__()
        self._refuse = refuse
        self._raises = raises

    def spawn(self, **cfg):
        if cfg.get("peer_id") == self._refuse:
            if self._raises:
                raise RuntimeError("port /dev/cu.usb2 is in use")
            self.spawned.append(cfg)
            return {"peer_id": cfg["peer_id"], "error": "port /dev/cu.usb2 is in use"}
        return super().spawn(**cfg)


def _boom_factory(**kw):
    raise RuntimeError("no calibration for 'leader'")


def test_failed_open_that_cannot_restore_an_arm_says_which_one(tmp_path):
    dev = RefusingSpawnDevices("arm-follower")
    ctl, dev, _ = make_controller(tmp_path, devices=dev, backend_factory=_boom_factory)
    with pytest.raises(HTTPException) as ei:
        ctl.open(dict(OPEN))
    assert ei.value.status_code == 500
    detail = ei.value.detail
    # the original cause survives - it is why the session failed
    assert "no calibration for 'leader'" in detail
    # ...and the fleet loss is IN THE SENTENCE, naming the arm and the remedy
    assert "arm-follower" in detail
    assert "port /dev/cu.usb2 is in use" in detail
    assert "Devices" in detail
    # the arm that DID come back is not accused
    assert "arm-leader (" not in detail


def test_a_spawn_that_raises_is_reported_too(tmp_path):
    dev = RefusingSpawnDevices("arm-follower", raises=True)
    ctl, dev, _ = make_controller(tmp_path, devices=dev, backend_factory=_boom_factory)
    with pytest.raises(HTTPException) as ei:
        ctl.open(dict(OPEN))
    assert "arm-follower" in ei.value.detail and "in use" in ei.value.detail


def test_a_4xx_after_parking_admits_the_fleet_loss_too(tmp_path):
    """The dangerous one: a 4xx reads as 'refused, nothing happened' on the client."""

    def refuse_422(**kw):
        raise ValueError("fps must be positive")

    dev = RefusingSpawnDevices("arm-follower")
    ctl, dev, _ = make_controller(tmp_path, devices=dev, backend_factory=refuse_422)
    with pytest.raises(HTTPException) as ei:
        ctl.open(dict(OPEN))
    assert ei.value.status_code == 422
    assert "fps must be positive" in ei.value.detail
    assert "DID NOT FULLY RECOVER" in ei.value.detail and "arm-follower" in ei.value.detail


def test_a_restored_fleet_adds_nothing_to_the_refusal(tmp_path):
    """No news is no sentence: the ordinary failure must not grow a scary paragraph."""
    ctl, dev, _ = make_controller(tmp_path, backend_factory=_boom_factory)
    with pytest.raises(HTTPException) as ei:
        ctl.open(dict(OPEN))
    assert ei.value.detail == "could not open the arms: no calibration for 'leader'"
    assert {c["peer_id"] for c in dev.spawned} == {"arm-leader", "arm-follower"}
    assert dev.autospawn.suspended is False, "and the watcher is handed back either way"
