"""A recording must not silently depend on a camera that stopped publishing (Q45).

MEASURED 2026-08-20: so101-arm-1 advertised `top` (publishing at 4fps) and `wrist`, whose
last capture was 10.4 HOURS old - its reader thread had died that morning ("exceeded
maximum consecutive read failures") and the arm carried on. The camera TILE says so
honestly; nothing stood between that camera and /api/record/open, which takes the
follower's camera list straight from its profile.

A dataset is the expensive artifact in this product: an hour of hand-guiding, and the
frozen stream is discovered at training time with the arm long since put away.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import camera_liveness
from strands_robots.dashboard.record_api import RecordController

NOW = 1_787_200_000.0


class TestTheVerdictItself:
    def test_the_incident_a_ten_hour_old_capture_is_dead(self) -> None:
        dead = camera_liveness.dead_cameras(
            {"top": 2, "wrist": 1},
            {"top": {"t": NOW - 0.2}, "wrist": {"t": NOW - 37327}},
            now=NOW,
        )
        assert [d["camera"] for d in dead] == ["wrist"]
        assert dead[0]["age_s"] == 37327.0

    def test_no_frame_history_is_NOT_death(self) -> None:
        # The peer may have just spawned, or nothing may have subscribed yet. Refusing
        # here would block the legitimate first recording of the day.
        assert camera_liveness.dead_cameras(["top", "wrist"], {}, now=NOW) == []
        assert camera_liveness.dead_cameras(["top"], {"top": {}}, now=NOW) == []
        assert camera_liveness.dead_cameras(["top"], {"top": {"t": None}}, now=NOW) == []
        assert camera_liveness.dead_cameras(["top"], {"top": "nonsense"}, now=NOW) == []

    def test_a_normal_gap_between_frames_is_not_death(self) -> None:
        # 1fps cameras, a busy USB bus and a slow encode are all normal.
        assert camera_liveness.dead_cameras(["top"], {"top": {"t": NOW - 30}}, now=NOW) == []

    def test_a_future_capture_is_clock_skew_not_freshness_and_not_death(self) -> None:
        assert camera_liveness.camera_age({"t": NOW + 500}, NOW) is None
        assert camera_liveness.dead_cameras(["top"], {"top": {"t": NOW + 500}}, now=NOW) == []

    def test_a_camera_not_in_this_session_is_none_of_its_business(self) -> None:
        dead = camera_liveness.dead_cameras(["top"], {"wrist": {"t": NOW - 40000}}, now=NOW)
        assert dead == []

    def test_the_refusal_names_the_camera_the_age_and_the_way_out(self) -> None:
        msg = camera_liveness.refusal(
            [{"camera": "wrist", "age_s": 37327.0}], peer_id="so101-arm-1"
        )
        assert "so101-arm-1" in msg and "wrist" in msg
        assert "10.4h ago" in msg
        assert "frozen or missing" in msg, "say the consequence, not just the fact"
        assert "ignore_dead_cameras" in msg, "every gate here is continuable"


class _Managed:
    def __init__(self, peer_id: str, cameras: dict) -> None:
        self.peer_id = peer_id
        self.robot_name = "so101"
        self.mode = "real"
        self.port = f"/dev/{peer_id}"
        self.cameras = cameras


class _Devices:
    """Just enough DeviceManager to park and unpark two arms."""

    def __init__(self) -> None:
        self.despawned: list[str] = []
        self.spawned: list[dict] = []
        self.autospawn = type("W", (), {"suspended": False})()

    def despawn(self, peer_id: str) -> None:
        self.despawned.append(peer_id)

    def spawn(self, **cfg):
        self.spawned.append(cfg)
        return {"ok": True}


class _Bridge:
    def __init__(self, snap: dict) -> None:
        self._snap = snap
        self.calls = 0

    def snapshot(self) -> dict:
        self.calls += 1
        return self._snap


def _controller(bridge, *, cameras=None, backend=None, devices=None):
    devices = devices if devices is not None else _Devices()
    c = RecordController(devices, bridge=bridge,
                         backend_factory=backend or (lambda **k: (_ for _ in ()).throw(
                             RuntimeError("no real arms in this test"))))
    managed = {
        "leader": _Managed("leader", {}),
        "follower": _Managed("follower", cameras if cameras is not None else {"top": 0, "wrist": 1}),
    }
    c._managed = lambda peer_id, *, role: managed[peer_id]  # type: ignore[assignment]
    return c


BODY = {"dataset": "d", "task": "t", "leader": "leader", "follower": "follower"}


class TestOpenRefusesBeforeTouchingTheFleet:
    def test_a_dead_camera_refuses_with_409(self) -> None:
        bridge = _Bridge({"peers": {"follower": {"cameras": {
            "top": {"t": time.time()}, "wrist": {"t": time.time() - 37327},
        }}}})
        devices = _Devices()
        c = _controller(bridge, devices=devices)
        with pytest.raises(HTTPException) as e:
            c.open(dict(BODY))
        assert e.value.status_code == 409
        assert "wrist" in str(e.value.detail)
        # THE POINT: refused BEFORE the arms are parked. A refusal that has already
        # torn the fleet down is not a refusal, it is an outage.
        assert devices.despawned == []
        assert devices.autospawn.suspended is False

    def test_the_override_lets_the_operator_proceed(self) -> None:
        bridge = _Bridge({"peers": {"follower": {"cameras": {"wrist": {"t": time.time() - 37327}}}}})
        c = _controller(bridge)
        # Past the gate it fails later for unrelated reasons (no real devices here);
        # what matters is that the refusal is no longer the reason.
        with pytest.raises(HTTPException) as e:
            c.open({**BODY, "ignore_dead_cameras": True})
        assert "stopped publishing" not in str(e.value.detail)

    def test_a_live_camera_set_is_not_refused(self) -> None:
        bridge = _Bridge({"peers": {"follower": {"cameras": {"top": {"t": time.time()}}}}})
        c = _controller(bridge, cameras={"top": 0})
        with pytest.raises(HTTPException) as e:
            c.open(dict(BODY))
        assert "stopped publishing" not in str(e.value.detail)


class TestNoEvidenceNeverRefuses:
    @pytest.mark.parametrize("bridge", [
        None,
        _Bridge({}),
        _Bridge({"peers": {}}),
        _Bridge({"peers": {"follower": {}}}),
        _Bridge({"peers": {"follower": {"cameras": "not a mapping"}}}),
    ])
    def test_missing_or_odd_evidence_is_silent(self, bridge) -> None:
        c = _controller(bridge)
        with pytest.raises(HTTPException) as e:
            c.open(dict(BODY))
        assert "stopped publishing" not in str(e.value.detail)

    def test_a_bridge_that_raises_cannot_break_recording(self) -> None:
        class _Boom:
            def snapshot(self):
                raise RuntimeError("mesh is having a moment")

        c = _controller(_Boom())
        with pytest.raises(HTTPException) as e:
            c.open(dict(BODY))
        assert "stopped publishing" not in str(e.value.detail)
        assert "mesh is having a moment" not in str(e.value.detail)


class TestTheRosterRailIsWiredAndCannotInventAFault:
    """The second rail at the record gate: an index this machine does not list.

    Its evidence is a roster the devices screen ALREADY took. That choice is deliberate on both
    sides: a fresh name scan shells out to ffmpeg with a 10s timeout and would sit in front of the
    record button, and a probe that opens cameras to enumerate them could take the very index the
    arm is about to use. So the only question these tests ask is whether an ALREADY-KNOWN absence
    stops a session, and whether anything weaker than that stays out of the way.
    """

    @staticmethod
    def _controller(roster: list[dict[str, object]] | None, *, age_s: float = 1.0) -> RecordController:
        class Devices:
            _camera_names_cache = roster
            _camera_names_cache_t = time.time() - age_s

        return RecordController(Devices())  # type: ignore[arg-type]

    def test_a_fresh_roster_yields_its_indices(self) -> None:
        ctrl = self._controller([{"listing_index": 0, "name": "Logi"}, {"listing_index": 2, "name": "top"}])
        assert ctrl._present_camera_indices() == (0, 2)

    def test_a_stale_roster_is_not_evidence(self) -> None:
        """This morning's roster could omit a camera plugged in since lunch.

        Refusing a session over that would be the gate inventing a fault, so age alone disqualifies
        the evidence rather than being tolerated with a warning.
        """
        ctrl = self._controller([{"listing_index": 0}], age_s=RecordController.ROSTER_MAX_AGE_S + 1)
        assert ctrl._present_camera_indices() == ()

    def test_an_empty_or_never_taken_roster_is_not_evidence(self) -> None:
        assert self._controller([]) ._present_camera_indices() == ()
        assert self._controller(None)._present_camera_indices() == ()
        never = self._controller([{"listing_index": 1}], age_s=0.0)
        never._devices._camera_names_cache_t = 0.0  # type: ignore[attr-defined]
        assert never._present_camera_indices() == ()

    def test_a_devices_object_that_raises_is_not_evidence_either(self) -> None:
        """Evidence gathering must never be the thing that breaks a recording."""

        class Hostile:
            @property
            def _camera_names_cache(self):  # noqa: ANN202
                raise RuntimeError("no")

        assert RecordController(Hostile())._present_camera_indices() == ()  # type: ignore[arg-type]

    def test_garbage_roster_entries_are_skipped_not_fatal(self) -> None:
        ctrl = self._controller([{"listing_index": 0}, {"name": "no index"}, "junk", {"listing_index": "1"}])
        assert ctrl._present_camera_indices() == (0,)

    def test_the_two_rails_produce_DIFFERENT_refusals(self) -> None:
        """An operator acts differently on each: one is a cable, the other a dead reader thread."""
        stale = camera_liveness.refusal([{"camera": "wrist", "age_s": 37327.0}], peer_id="arm-1")
        gone = camera_liveness.missing_refusal([{"camera": "wrist", "index": 1}], peer_id="arm-1")
        assert "stopped publishing" in stale and "ignore_dead_cameras" in stale
        assert "not listed by this machine at all" in gone and "ignore_missing_cameras" in gone
        assert "RESCAN" in gone and "RESCAN" not in stale
